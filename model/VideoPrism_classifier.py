"""Zero-shot VideoPrism classifier used by the attribution experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


VIDEOPRISM_CHECKPOINT = "google/videoprism-lvt-base-f16r288"
# Converted PyTorch weights on the official Google checkpoint repository.
VIDEOPRISM_REVISION = "fb6de9f0eb7bc285be86bdca1cf7daa3e3ef51ff"


def _require_videoprism_transformers():
    try:
        from transformers import AutoTokenizer, VideoPrismClipModel
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "VideoPrism requires the Transformers/timm versions listed in "
            "docs/joint_st.md. Run attribution from that environment."
        ) from exc
    return AutoTokenizer, VideoPrismClipModel


class VideoPrismClassifier(nn.Module):
    """Classify videos with the released LvT video/text embedding pair."""

    image_size = 288

    def __init__(
        self,
        keywords,
        model_name=VIDEOPRISM_CHECKPOINT,
        revision=VIDEOPRISM_REVISION,
        prompt_template="{}",
        model_dtype=torch.float32,
    ):
        super().__init__()
        AutoTokenizer, VideoPrismClipModel = _require_videoprism_transformers()

        self.keywords = list(keywords)
        self.model_name = model_name
        self.checkpoint_revision = revision
        self.prompt_template = prompt_template
        self.model_dtype = model_dtype

        self.videoprism_model = VideoPrismClipModel.from_pretrained(
            model_name,
            revision=revision,
            attn_implementation="eager",
            dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        self.videoprism_model.eval()
        if hasattr(self.videoprism_model, "gradient_checkpointing_disable"):
            self.videoprism_model.gradient_checkpointing_disable()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

        prompts = [prompt_template.format(keyword) for keyword in self.keywords]
        with torch.no_grad():
            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.videoprism_model.config.text_config.max_position_embeddings,
                truncation=True,
                return_tensors="pt",
            )
            text_outputs = self.videoprism_model.get_text_features(**text_inputs)
            self.register_buffer("text_embeds", text_outputs.pooler_output)
        # The fixed class embeddings make the text transformer unnecessary
        # during every target-gradient video forward.
        del self.videoprism_model.text_model

    @property
    def video_model(self):
        return self.videoprism_model.video_model

    def preprocess(self, raw_frames):
        """Apply VideoPrism's released full-frame preprocessing.

        VideoPrism expects RGB ``[B,T,C,288,288]`` values in ``[0,1]`` and
        does not apply channel mean/std normalization.
        """
        if raw_frames.dim() != 5:
            raise ValueError(
                f"VideoPrism raw input must be [B,T,C,H,W], got {raw_frames.shape}"
            )
        if raw_frames.shape[2] != 3:
            raise ValueError(
                f"VideoPrism expects RGB in axis 2, got {raw_frames.shape}"
            )

        frames = raw_frames.float()
        if raw_frames.dtype == torch.uint8 or float(frames.max().item()) > 1.0:
            frames = frames / 255.0
        batch_size, num_frames, channels, height, width = frames.shape
        frames = F.interpolate(
            frames.reshape(batch_size * num_frames, channels, height, width),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return frames.reshape(
            batch_size,
            num_frames,
            channels,
            self.image_size,
            self.image_size,
        ).to(dtype=next(self.videoprism_model.parameters()).dtype)

    def encode_video(self, pixel_values_videos):
        video_embeds = self.videoprism_model.get_video_features(
            pixel_values_videos=pixel_values_videos,
            interpolate_pos_encoding=True,
        ).pooler_output
        if video_embeds.dim() == 3 and video_embeds.shape[1] == 1:
            video_embeds = video_embeds[:, 0]
        if video_embeds.dim() != 2:
            raise RuntimeError(
                f"Unexpected VideoPrism pooled shape {video_embeds.shape}"
            )
        return video_embeds

    def forward(self, pixel_values_videos):
        # Use the pre-normalized video/text embeddings' cosine similarities.
        # VideoPrismClipModel.forward exponentiates and normalizes retrieval
        # matrices across the batch, which degenerates for B=1 classification.
        video_embeds = self.encode_video(pixel_values_videos)
        return video_embeds @ self.text_embeds.t()

    @staticmethod
    def get_types(logits):
        k = min(5, logits.shape[1])
        return torch.topk(logits, k=k, dim=1, largest=True, sorted=True).indices
