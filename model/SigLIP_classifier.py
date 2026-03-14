import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipTextModel, AutoTokenizer

class SigLIP_Classifier(nn.Module):
    def __init__(self, keywords=None, feature_dim=768, model_name="google/siglip-base-patch16-224"):
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
        # Precompute text embeddings for all class names
        with torch.no_grad():
            inputs = self.tokenizer(self.keywords, padding="max_length", return_tensors="pt", truncation=True)
            text_outputs = self.text_model(**inputs)
            self.text_embeds = text_outputs.pooler_output  # (num_classes, hidden_size)

    def forward(self, x):
        # x: (B, C, T, H, W) or (B, T, C, H, W)
        if x.dim() == 5:
            # Take mean over time dimension (T)
            x = x.mean(dim=2) if x.shape[1] == 3 else x.mean(dim=1)
        outputs = self.siglip_model(pixel_values=x)
        image_embeds = outputs.pooler_output  # (B, hidden_size)
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = self.text_embeds.to(image_embeds.device)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # Compute similarity (dot product)
        logits = image_embeds @ text_embeds.t()  # (B, num_classes)
        return logits

    def get_types(self, logits):
        # Return predicted class indices
        return torch.argmax(logits, dim=-1, keepdim=True)
