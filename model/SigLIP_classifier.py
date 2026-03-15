import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipTextModel, AutoTokenizer


class SigLIP_Classifier(nn.Module):
    def __init__(
        self,
        keywords=None,
        feature_dim=768,
        model_name="google/siglip-base-patch16-224",
        temporal_aggregation="mean",
    ):
        super(SigLIP_Classifier, self).__init__()
        if keywords is None:
            self.keywords = [
                'corner', 'goal', 'injury', 'own goal', 'penalty', 'penalty missed', 
                'red card', 'second yellow card', 'substitution', 'start of game(half)', 
                'end of game(half)', 'yellow card', 'throw in', 'free kick', 
                'saved by goal-keeper', 'shot off target', 'clearance', 'lead to corner', 
                'off-side', 'var', 'foul (no card)', 'statistics and summary', 
                'ball possession', 'ball out of play'
            ]
        else:
            self.keywords = keywords
        self.siglip_model = SiglipVisionModel.from_pretrained(model_name)
        self.text_model = SiglipTextModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_dim = feature_dim
        self.temporal_aggregation = temporal_aggregation

        if self.temporal_aggregation not in {"mean", "max"}:
            raise ValueError(
                f"Unsupported temporal_aggregation: {self.temporal_aggregation}. Use 'mean' or 'max'."
            )

        # Precompute text embeddings for all class names
        with torch.no_grad():
            inputs = self.tokenizer(self.keywords, padding="max_length", return_tensors="pt", truncation=True)
            text_outputs = self.text_model(**inputs)
            self.register_buffer("text_embeds", text_outputs.pooler_output)  # (num_classes, hidden_size)

    def _aggregate_temporal_embeddings(self, frame_embeds):
        # frame_embeds: (B, T, D)
        if self.temporal_aggregation == "mean":
            return frame_embeds.mean(dim=1)
        if self.temporal_aggregation == "max":
            return frame_embeds.max(dim=1).values
        raise RuntimeError(f"Unsupported temporal_aggregation at runtime: {self.temporal_aggregation}")

    def forward(self, x):
        # x can be (B, C, T, H, W), (B, T, C, H, W), or (B, C, H, W)
        if x.dim() == 5:
            if x.shape[1] == 3:
                # (B, C, T, H, W) -> (B, T, C, H, W)
                x = x.permute(0, 2, 1, 3, 4)
            # Flatten temporal dimension so each frame is encoded independently.
            bsz, timesteps, channels, height, width = x.shape
            x = x.reshape(bsz * timesteps, channels, height, width)
            outputs = self.siglip_model(pixel_values=x)
            frame_embeds = outputs.pooler_output.reshape(bsz, timesteps, -1)
            image_embeds = self._aggregate_temporal_embeddings(frame_embeds)
        elif x.dim() == 4:
            outputs = self.siglip_model(pixel_values=x)
            image_embeds = outputs.pooler_output  # (B, hidden_size)
        else:
            raise ValueError(f"Unexpected input rank {x.dim()} for SigLIP_Classifier")

        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = self.text_embeds
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # Compute similarity (dot product)
        logits = image_embeds @ text_embeds.t()  # (B, num_classes)
        return logits

    def get_types(self, logits):
        # Return predicted class indices
        return torch.argmax(logits, dim=-1, keepdim=True)
