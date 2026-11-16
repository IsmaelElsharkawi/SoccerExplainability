import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPModel


class CLIP_Classifier(nn.Module):
    """Zero-shot video classifier built from an off-the-shelf CLIP checkpoint.

    CLIP encodes frames independently. Projected frame embeddings are pooled
    over time, L2-normalized, and compared with normalized text embeddings
    using CLIP's learned logit scale.
    """

    def __init__(
        self,
        keywords,
        model_name="openai/clip-vit-base-patch16",
        temporal_aggregation="mean",
    ):
        super().__init__()
        if temporal_aggregation not in {"mean", "max"}:
            raise ValueError(
                f"Unsupported temporal_aggregation={temporal_aggregation!r}; use 'mean' or 'max'"
            )

        self.keywords = list(keywords)
        self.model_name = model_name
        self.temporal_aggregation = temporal_aggregation
        self.clip_model = CLIPModel.from_pretrained(
            model_name,
            attn_implementation="eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        with torch.no_grad():
            text_inputs = self.tokenizer(
                self.keywords,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_embeds = self.clip_model.get_text_features(**text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            self.register_buffer("text_embeds", text_embeds)

    def _aggregate_temporal_embeddings(self, frame_embeds):
        if self.temporal_aggregation == "mean":
            return frame_embeds.mean(dim=1)
        return frame_embeds.max(dim=1).values

    def encode_video(self, x):
        if x.dim() == 4:
            return self.clip_model.get_image_features(pixel_values=x)
        if x.dim() != 5:
            raise ValueError(f"Unexpected input rank {x.dim()} for CLIP_Classifier")

        if x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)
        batch_size, num_frames, channels, height, width = x.shape
        frame_inputs = x.reshape(
            batch_size * num_frames,
            channels,
            height,
            width,
        )
        frame_embeds = self.clip_model.get_image_features(pixel_values=frame_inputs)
        frame_embeds = frame_embeds.reshape(batch_size, num_frames, -1)
        return self._aggregate_temporal_embeddings(frame_embeds)

    def forward(self, x):
        image_embeds = self.encode_video(x)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return self.clip_model.logit_scale.exp() * image_embeds @ self.text_embeds.t()

    @staticmethod
    def get_types(logits):
        k = min(5, logits.shape[1])
        return torch.topk(logits, k=k, dim=1, largest=True, sorted=True).indices
