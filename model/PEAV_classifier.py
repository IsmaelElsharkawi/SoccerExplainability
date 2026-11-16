"""Zero-shot PE-AV video classifier used by the attribution experiments."""

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F


PEAV_CHECKPOINT = "facebook/pe-av-base"
PEAV_REVISION = "1c4c329b78d80a18d860ebfb05daebff4bc44518"


def _require_peav_transformers():
    try:
        from transformers import AutoTokenizer, PeAudioVideoModel
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "PE-AV requires the Transformers/timm versions listed in "
            "docs/joint_st.md. Run attribution from that environment."
        ) from exc
    return AutoTokenizer, PeAudioVideoModel


class PEAVVideoClassifier(nn.Module):
    """Classify a video by PE-AV video/text embedding similarity.

    The released composite checkpoint also contains audio and audio-video
    branches. They are discarded after loading; only its released video/text
    submodel is retained on the module and moved to the GPU.
    """

    image_size = 336

    def __init__(
        self,
        keywords,
        model_name=PEAV_CHECKPOINT,
        revision=PEAV_REVISION,
        prompt_template="{}",
        model_dtype=torch.float32,
    ):
        super().__init__()
        AutoTokenizer, PeAudioVideoModel = _require_peav_transformers()

        self.keywords = list(keywords)
        self.model_name = model_name
        self.checkpoint_revision = revision
        self.prompt_template = prompt_template
        self.model_dtype = model_dtype

        composite = PeAudioVideoModel.from_pretrained(
            model_name,
            revision=revision,
            attn_implementation="eager",
            dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        self.peav_model = composite.video_model
        # The composite loader does not propagate its requested attention
        # implementation to the extracted video submodel in current
        # Transformers. Force the executable video path to return explicit
        # attention probabilities, which attribution hooks require.
        self.peav_model.set_attn_implementation("eager")
        video_attn_implementation = getattr(
            self.peav_model.video_encoder.config,
            "_attn_implementation",
            None,
        )
        if video_attn_implementation != "eager":
            raise RuntimeError(
                "PE-AV video attention must use the eager implementation for "
                f"Chefer attribution; got {video_attn_implementation!r}"
            )
        del composite
        gc.collect()

        self.peav_model.eval()
        if hasattr(self.peav_model, "gradient_checkpointing_disable"):
            self.peav_model.gradient_checkpointing_disable()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

        prompts = [prompt_template.format(keyword) for keyword in self.keywords]
        with torch.no_grad():
            text_inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_outputs = self.peav_model.text_model(
                **text_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            text_hidden = text_outputs.hidden_states[-1][:, 0]
            text_embeds = self.peav_model.text_video_head(text_hidden)
            self.register_buffer("text_embeds", text_embeds)
        # Text embeddings are fixed for this label set; retaining the full
        # ModernBERT branch would waste GPU memory during video attribution.
        del self.peav_model.text_model
        gc.collect()

    @property
    def temporal_encoder(self):
        return self.peav_model.video_encoder

    @property
    def spatial_encoder(self):
        return self.peav_model.video_encoder.embedder.vision_model.timm_model

    def preprocess(self, raw_frames):
        """Apply the released PE-AV video geometry and pixel normalization.

        ``raw_frames`` is ``[B,T,C,H,W]`` uint8 or float RGB. The released
        processor directly resizes each full frame to 336x336, rescales to
        [0,1], then normalizes every channel with mean/std 0.5.
        """
        if raw_frames.dim() != 5:
            raise ValueError(
                f"PE-AV raw input must be [B,T,C,H,W], got {raw_frames.shape}"
            )
        if raw_frames.shape[2] != 3:
            raise ValueError(f"PE-AV expects RGB in axis 2, got {raw_frames.shape}")

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
        frames = (frames - 0.5) / 0.5
        return frames.reshape(
            batch_size,
            num_frames,
            channels,
            self.image_size,
            self.image_size,
        ).to(dtype=next(self.peav_model.parameters()).dtype)

    def encode_video(self, pixel_values_videos):
        video_outputs = self.peav_model.video_encoder(
            pixel_values_videos=pixel_values_videos,
            return_dict=True,
        )
        return self.peav_model.video_head(video_outputs.pooler_output)

    def forward(self, pixel_values_videos):
        video_embeds = self.encode_video(pixel_values_videos)
        logits = video_embeds @ self.text_embeds.t()
        return (
            logits * self.peav_model.text_video_logit_scale
            + self.peav_model.text_video_logit_bias
        )

    @staticmethod
    def get_types(logits):
        k = min(5, logits.shape[1])
        return torch.topk(logits, k=k, dim=1, largest=True, sorted=True).indices
