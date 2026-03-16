# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
import os
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List

from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipTextConfig, SiglipTextModel, AutoTokenizer
from timm.models.layers import DropPath
from einops import rearrange
import torch.utils.checkpoint as checkpoint

class ResidualAttentionBlock(nn.Module):
    def __init__(self, spatial_encoder, d_model, n_head, drop_path=0., attn_mask=None, dropout=0., use_temporal=True):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_temporal = use_temporal
        self.attn_mask = attn_mask
        
        self.encoder = spatial_encoder

        if use_temporal:
            self.temporal_norm1 = nn.LayerNorm(d_model)
            self.temporal_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
            self.temporal_fc = nn.Linear(d_model, d_model)
            self.register_parameter('temporal_alpha_attn', nn.Parameter(torch.tensor(0.)))
    
    def temporal_attention(self, x):
        return self.temporal_attn(x, x, x)[0]

    def forward(self, x, B, T):
        if self.use_temporal:
            # Temporal 
            xt = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            res_temporal = self.drop_path(self.temporal_attention(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m', b=B, t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x + self.temporal_alpha_attn.tanh() * res_temporal

            # Spatial
            xs = xt
            res_spatial = self.encoder(xs, self.attn_mask)[0]
        else:
            # Spatial only (no temporal attention)
            res_spatial = self.encoder(x, self.attn_mask)[0]
        
        return res_spatial
    

class VisionBackbone(nn.Module):
    def __init__(self, ckpt_path: str, num_frames: int, temporal_start_layer: int = 8, drop_path: float = 0., checkpoint_num: int = 0, dropout: float = 0.):
        super().__init__()

        # Load pretrained model once
        model = SiglipVisionModel.from_pretrained(ckpt_path, device_map="cpu")
        siglip_vision_model = model.vision_model
        config = SiglipVisionConfig.from_pretrained(ckpt_path)
        
        self.vision_model_embedding = siglip_vision_model.embeddings
        self.temporal_start_layer = temporal_start_layer
        self.num_layers = config.num_hidden_layers
        self.checkpoint_num = checkpoint_num
        hidden_size = config.hidden_size
        
        pretrained_encoder_layers = siglip_vision_model.encoder.layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.num_layers)]
        
        self.encoder_blocks = nn.ModuleList()
        for idx in range(self.num_layers):
            # Only enable temporal attention for layers >= temporal_start_layer
            use_temporal = (idx >= temporal_start_layer)
            self.encoder_blocks.append(
                ResidualAttentionBlock(
                    spatial_encoder=pretrained_encoder_layers[idx],
                    d_model=hidden_size, 
                    n_head=config.num_attention_heads, 
                    drop_path=dpr[idx], 
                    dropout=dropout, 
                    use_temporal=use_temporal
                )
            )
        
        self.post_norm = siglip_vision_model.post_layernorm
        self.head = siglip_vision_model.head
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, hidden_size))
        
    def forward(self, images: torch.Tensor, temporal_attention_mask: Optional[torch.Tensor] = None, text: Optional[List[str]] = None):
        # Video mode: input is [B, T, C, H, W]
        B, T, _, _, _ = images.shape
        images = rearrange(images, 'b t c h w -> (b t) c h w')
        x = self.vision_model_embedding(images)
        
        for idx in range(self.temporal_start_layer):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(self.encoder_blocks[idx], x, B, T)
            else:
                x = self.encoder_blocks[idx](x, B, T)
        
        # Save early local features before temporal layers
        local_features_early = x  # [B*T, N, D]
        local_features_early = rearrange(local_features_early, '(b t) n m -> b t n m', b=B, t=T)
        
        # Add temporal embedding before temporal attention layers
        x = rearrange(x, '(b t) n m -> b n t m', b=B, t=T)
        x = x + self.temporal_embedding
        x = rearrange(x, 'b n t m -> (b t) n m')
        
        for idx in range(self.temporal_start_layer, self.num_layers):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(self.encoder_blocks[idx], x, B, T)
            else:
                x = self.encoder_blocks[idx](x, B, T)
        
        # local_features_late = x  # [B*T, N, D]
        # local_features_late = rearrange(local_features_late, '(b t) n m -> b t n m', b=B, t=T)
        
        # Generate global features
        x2 = self.post_norm(x)  # [B*T, N, D]
        x2 = self.head(x2)
        x2 = rearrange(x2, '(b t) m -> b t m', b=B, t=T)
        
        return local_features_early, x2

class SoccerMaster(nn.Module):
    def __init__(self, num_frames: int,
                 ckpt_path: str,
                 text_encoder_ckpt_path: str,
                 use_lora: bool,
                 use_temporal_gate: bool,
                 freeze_vision_encoder: bool = False,
                 freeze_text_encoder: bool = True,
                 temporal_start_layer: int = 8):
        super().__init__()
        
        self.vision_model = VisionBackbone(ckpt_path, num_frames, temporal_start_layer)
        
        text_config = SiglipTextConfig.from_pretrained(text_encoder_ckpt_path)
        self.text_hidden_size = text_config.hidden_size
        self.text_model = TextEncoder(text_encoder_ckpt_path)
        
        if freeze_vision_encoder:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        if freeze_text_encoder:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
    def forward(self, images: torch.Tensor, temporal_attention_mask: Optional[torch.Tensor] = None, text: Optional[List[str]] = None):
        local_features, pooled_output = self.vision_model(images, temporal_attention_mask, text)
        
        if text is not None:
            valid_texts = []
            valid_indices = []
            for i, t in enumerate(text):
                if t is not None:
                    valid_texts.append(t)
                    valid_indices.append(i)
            
            batch_size = len(text)
            text_pooled_output = torch.zeros(batch_size, self.text_hidden_size, device=images.device, dtype=images.dtype)
            
            if valid_texts:
                text_pooled_output_valid = self.text_model(valid_texts)[0]

                for valid_idx, original_idx in enumerate(valid_indices):
                    text_pooled_output[original_idx] = text_pooled_output_valid[valid_idx]
        else:
            text_pooled_output = None
        
        output = {
            'global_features': pooled_output,
            'local_features': local_features,
            'text_features': text_pooled_output
        }
        return output
    
class TextEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        
        self.model_name = model_name
        self.model = SiglipTextModel.from_pretrained(model_name, device_map="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cpu", use_fast=False)

    def forward(self, text):
        # important: make sure to set padding="max_length" as that's how the model was trained
        if 'siglip2' in self.model_name:
            inputs = self.tokenizer(text=text, padding="max_length", max_length=64, return_tensors="pt", truncation=True)
        else:
            inputs = self.tokenizer(text=text, padding="max_length", return_tensors="pt", truncation=True)
        inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        return pooled_output, last_hidden_state