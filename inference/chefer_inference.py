"""
Chefer explainability inference for MatchVision classification (per-frame spatial-only).

This is the ORIGINAL Chefer method applied to VisionTimesformer without the
temporal fix. Each frame has an independent R_pp [N, N] relevance matrix
that only accumulates spatial attention. Temporal attention is NOT included
in the relevance propagation.

For the temporal-fix version that builds a joint R [T*N, T*N] matrix
including both spatial and temporal attention, see chefer_inference.py.

================================================================================
CHEFER METHOD (per-frame spatial-only)
================================================================================

Based on:
    "Generic Attention-model Explainability for Interpreting Bi-Modal and
     Encoder-Decoder Transformers" (Chefer et al., ICCV 2021)

1. Forward pass through VisionTimesformer + classification head
2. Backward pass from the target CLASS LOGIT
3. For each spatial attention layer:
       cam = (grad * attn).clamp(min=0).mean(heads)    [Rule 5]
       R_pp[t] += cam @ R_pp[t]                         [Rules 6+7]
   (Temporal attention is NOT propagated into R)
4. Extract heatmap via pooling head probe (CLS-token equivalent):
       heatmap[t] = cam_cross[t] @ R_pp[t]             [Rule 10 analog]
5. Min-max normalize per frame

================================================================================
ARCHITECTURE ADAPTATION
================================================================================

- SigLIP's vision encoder has no CLS token (unlike ViT-B/16).  Its pooling
  head uses a learnable probe that cross-attends to all 196 patches, so
  cam_cross @ R_pp is the direct analog of R[cls, 1:] in original Chefer implementation.
- Temporal attention (Timesformer) is present in the forward pass but is
  NOT wrapped and NOT included in the relevance propagation.
- All spatial layers are used (no layer skipping).

================================================================================
LABEL MAPPING (LABEL_NAMES, alphabetical)
================================================================================

Use target_label / target_label_name to specify class:

    0:  var                  8:  shot off target     16: goal
    1:  end of half game     9:  start of half game  17: penalty
    2:  clearance           10:  substitution        18: yellow card
    3:  second yellow card  11:  saved by goal-keeper 19: foul lead to penalty
    4:  injury              12:  red card            20: corner
    5:  ball possession     13:  lead to corner      21: free kick
    6:  throw in            14:  ball out of play    22: foul with no card
    7:  show added time     15:  off side

Usage:
    python chefer_inference.py \\
        --config_path ../config/pretrain_classification_ibex.py \\
        --checkpoint_path /path/to/pretrained_classification.pth \\
        --output_dir /path/to/output
"""

import argparse
import math
import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference_utils import (
    load_config, create_test_dataloader, load_classifier,
    setup_attribution_evaluator, match_video_id,
    evaluate_and_print_video, print_and_save_eval_summary,
)
from visualization_video import save_lowres_visualization_video

# ============================================================================
# Chefer Method Constants
# ============================================================================

# Label mapping for MatchVision_Classifier — must match the keyword order
# used during training (config/pretrain_classification.py) so that indices
# align with the pretrained_classification.pth weight rows.
LABEL_NAMES = [
    'var', 'end of half game', 'clearance', 'second yellow card',
    'injury', 'ball possession', 'throw in', 'show added time',
    'shot off target', 'start of half game', 'substitution',
    'saved by goal-keeper', 'red card', 'lead to corner',
    'ball out of play', 'off side', 'goal', 'penalty',
    'yellow card', 'foul lead to penalty', 'corner', 'free kick',
    'foul with no card'
]

LABEL_TO_IDX = {name: idx for idx, name in enumerate(LABEL_NAMES)}

# Toggle verbose debug prints for shape/value verification at every critical step
DEBUG = False


# ============================================================================
# Chefer LRP Rules (from Transformer-MM-Explainability paper)
# ============================================================================

def avg_heads(cam, grad):
    """Rule 5 from paper: Average attention heads weighted by gradients."""
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# ============================================================================
# Attention Weight Extraction Wrappers
# ============================================================================

def wrap_siglip_attention_module(attn_module):
    """
    Wrap a SigLIP Attention module to capture attention weights AND gradients.
    
    This replaces the forward method to properly capture attention for Chefer.
    Gradients are captured via hooks on the attention tensor.
    
    Args:
        attn_module: SiglipAttention module
        
    Returns:
        The same module, now with get_attn() and get_attn_gradients() methods
    """
    # Idempotency guard: skip if already wrapped
    if hasattr(attn_module, '_chefer_wrapped'):
        return attn_module

    # Get dimensions from projection layers
    embed_dim = attn_module.q_proj.out_features  # 768
    num_heads = attn_module.num_heads if hasattr(attn_module, 'num_heads') else 12
    head_dim = embed_dim // num_heads
    scale = 1.0 / math.sqrt(head_dim)
    
    # Storage for attention weights and gradients
    attn_storage = {}
    grad_storage = {}
    
    def wrapped_forward(hidden_states, attention_mask=None, output_attentions=False):
        """Wrapped forward that captures attention weights AND gradients."""
        B, N, C = hidden_states.shape
        
        # Compute Q, K, V separately
        query = attn_module.q_proj(hidden_states)
        key = attn_module.k_proj(hidden_states)
        value = attn_module.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(B, N, num_heads, head_dim).transpose(1, 2)
        key = key.view(B, N, num_heads, head_dim).transpose(1, 2)
        value = value.view(B, N, num_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn = (query @ key.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        # Store attention weights (detached copy for storage)
        attn_storage['attn'] = attn.detach()
        if DEBUG:
            print(f"  [DEBUG wrap_siglip_spatial] Stored attn: shape={list(attn.shape)}, "
                  f"requires_grad={attn.requires_grad}, "
                  f"row_sum[0,0,0]={attn[0,0,0].sum().item():.4f} (expect ~1.0)")
        
        # Register gradient hook to capture gradients during backward pass
        if attn.requires_grad:
            def grad_hook(grad):
                grad_storage['grad'] = grad.detach()
                if DEBUG:
                    print(f"  [DEBUG wrap_siglip_spatial] Grad hook fired: shape={list(grad.shape)}, "
                          f"norm={grad.norm().item():.6e}")
            attn.register_hook(grad_hook)
        
        # Apply dropout if module has it
        if hasattr(attn_module, 'dropout'):
            dropout_val = attn_module.dropout
            if callable(dropout_val):
                attn_dropped = dropout_val(attn)
            elif isinstance(dropout_val, (float, int)) and dropout_val > 0:
                attn_dropped = F.dropout(attn, p=dropout_val, training=attn_module.training)
            else:
                attn_dropped = attn
        else:
            attn_dropped = attn
        
        # Compute output
        out = (attn_dropped @ value).transpose(1, 2).reshape(B, N, C)
        out = attn_module.out_proj(out)
        
        # Return 2 values for SiglipEncoderLayer compatibility
        if output_attentions:
            return out, attn
        return out, None
    
    # Replace forward method
    attn_module.forward = wrapped_forward
    
    # Add methods to retrieve attention weights and gradients
    def get_attn():
        return attn_storage.get('attn')
    
    def get_attn_gradients():
        return grad_storage.get('grad')
    
    attn_module.get_attn = get_attn
    attn_module.get_attn_gradients = get_attn_gradients
    attn_module._chefer_num_heads = num_heads
    attn_module._chefer_head_dim = head_dim
    attn_module._chefer_wrapped = True
    
    return attn_module


def wrap_pooling_head_attention(mha_module):
    """
    Wrap SigLIP's pooling head nn.MultiheadAttention for cross-attention capture.
    
    The pooling head does: output = attention(probe, patches, patches)
    where probe [B, 1, 768] is a learnable CLS-like token and patches [B, 196, 768]
    are the spatial features.  This wrapper manually computes the attention so we
    can register gradient hooks — needed for proper Chefer R[probe, patches] extraction.
    
    Args:
        mha_module: nn.MultiheadAttention (batch_first=True) from SiglipMultiheadAttentionPoolingHead
        
    Returns:
        The same module, now with get_attn() and get_attn_gradients() methods
    """
    # Idempotency guard: skip if already wrapped
    if hasattr(mha_module, '_chefer_wrapped'):
        return mha_module

    embed_dim = mha_module.embed_dim   # 768
    num_heads = mha_module.num_heads   # 12
    head_dim = embed_dim // num_heads  # 64
    scale = 1.0 / math.sqrt(head_dim)
    
    attn_storage = {}
    grad_storage = {}
    
    def wrapped_forward(query, key, value, **kwargs):
        """Manual cross-attention with gradient hooks on attention weights."""
        B, Nq, C = query.shape   # [B*T, 1, 768]  (probe)
        _, Nk, _ = key.shape     # [B*T, 196, 768] (patches)
        
        # Split in_proj_weight/bias into Q, K, V projections
        w = mha_module.in_proj_weight  # [3*768, 768]
        b = mha_module.in_proj_bias    # [3*768]
        
        Q = F.linear(query, w[:embed_dim], b[:embed_dim])
        K = F.linear(key, w[embed_dim:2*embed_dim], b[embed_dim:2*embed_dim])
        V = F.linear(value, w[2*embed_dim:], b[2*embed_dim:])
        
        # Reshape for multi-head: [B, N, H, D] -> [B, H, N, D]
        Q = Q.view(B, Nq, num_heads, head_dim).transpose(1, 2)  # [B, H, 1, 64]
        K = K.view(B, Nk, num_heads, head_dim).transpose(1, 2)  # [B, H, 196, 64]
        V = V.view(B, Nk, num_heads, head_dim).transpose(1, 2)  # [B, H, 196, 64]
        
        attn = (Q @ K.transpose(-2, -1)) * scale  # [B, H, 1, 196]
        attn = attn.softmax(dim=-1)
        
        # Store attention weights and register gradient hook
        attn_storage['attn'] = attn.detach()
        if DEBUG:
            print(f"  [DEBUG wrap_pooling_head] Stored attn: shape={list(attn.shape)}, "
                  f"requires_grad={attn.requires_grad}, "
                  f"row_sum[0,0,0]={attn[0,0,0].sum().item():.4f} (expect ~1.0)")
        if attn.requires_grad:
            def grad_hook(grad):
                grad_storage['grad'] = grad.detach()
                if DEBUG:
                    print(f"  [DEBUG wrap_pooling_head] Grad hook fired: shape={list(grad.shape)}, "
                          f"norm={grad.norm().item():.6e}")
            attn.register_hook(grad_hook)
        
        out = (attn @ V).transpose(1, 2).reshape(B, Nq, C)  # [B, 1, 768]
        out = mha_module.out_proj(out)
        
        return out, attn
    
    mha_module.forward = wrapped_forward
    
    mha_module.get_attn = lambda: attn_storage.get('attn')
    mha_module.get_attn_gradients = lambda: grad_storage.get('grad')
    mha_module._chefer_num_heads = num_heads
    mha_module._chefer_head_dim = head_dim
    mha_module._chefer_wrapped = True
    
    return mha_module


# ============================================================================
# Model Wrapping Functions
# ============================================================================

def wrap_matchvision_model(model):
    """
    Wrap a MatchVision model's attention modules for Chefer explainability.
    
    Per-frame spatial-only version: wraps only spatial attention and pooling head.
    Temporal attention (Timesformer) is NOT wrapped — it participates in the
    forward pass but is excluded from the relevance propagation.
    
    This wraps:
    1. SigLIP spatial attention in VisionTimesformer encoder blocks
    2. SigLIP pooling head cross-attention (probe -> patches, the CLS equivalent)
    
    Args:
        model: MatchVision model (MatchVision_Classifier or VisionTimesformer)
        
    Returns:
        Dictionary with wrapped attention modules organized by type
    """
    wrapped = {
        'spatial_attn': [],      # SigLIP spatial attention (per frame)
        'pooling_head_attn': None, # SigLIP pooling head cross-attention (probe -> patches)
    }
    
    # Get visual encoder - handle MatchVision_Classifier, full model, or standalone VisionTimesformer
    if hasattr(model, 'siglip_model'):
        visual_encoder = model.siglip_model
    elif hasattr(model, 'visual_encoder'):
        visual_encoder = model.visual_encoder
    elif hasattr(model, 'timesformer'):
        # Model IS the visual encoder (VisionTimesformer)
        visual_encoder = model
    else:
        raise ValueError("Model does not have 'siglip_model', 'visual_encoder' or 'timesformer' attribute")
    
    # Wrap Timesformer spatial attention only (NO temporal wrapping)
    if hasattr(visual_encoder, 'timesformer'):
        timesformer = visual_encoder.timesformer
        for i, block in enumerate(timesformer.resblocks):
            # Wrap spatial attention (SigLIP encoder layer)
            if hasattr(block, 'encoder'):
                encoder_layer = block.encoder
                if hasattr(encoder_layer, 'self_attn'):
                    try:
                        wrapped_spatial = wrap_siglip_attention_module(encoder_layer.self_attn)
                        wrapped['spatial_attn'].append((i, wrapped_spatial))
                    except Exception as e:
                        print(f"Warning: Could not wrap spatial attention in block {i}: {e}")
    
    # Wrap SigLIP pooling head attention (probe -> patches cross-attention)
    if hasattr(visual_encoder, 'head') and hasattr(visual_encoder.head, 'attention'):
        try:
            wrapped_head = wrap_pooling_head_attention(visual_encoder.head.attention)
            wrapped['pooling_head_attn'] = wrapped_head
        except Exception as e:
            print(f"Warning: Could not wrap pooling head attention: {e}")
    
    print(f"Wrapped attention modules (per-frame spatial-only):")
    print(f"  - Spatial (SigLIP): {len(wrapped['spatial_attn'])} layers")
    print(f"  - Pooling head (probe->patches): {'yes' if wrapped['pooling_head_attn'] is not None else 'no'}")
    
    return wrapped


# ============================================================================
# Heatmap Generation
# ============================================================================

def generate_per_frame_heatmaps(
    model,
    video_frames: torch.Tensor,
    target_label: Optional[int] = None,
    target_label_name: Optional[str] = None,
    device: str = 'cuda',
    num_frames: int = 30,
    patch_size: int = 14,  # 14x14 = 196 patches for 224x224 input
) -> np.ndarray:
    """
    Generate per-frame spatial heatmaps using the Chefer method (ICCV 2021).
    
    Per-frame spatial-only version: each frame has an independent R_pp [N, N]
    relevance matrix. Only spatial attention layers propagate relevance.
    Temporal attention is present in the forward pass but excluded from R.
    
    Algorithm (matches example.py exactly, applied per-frame):
        1. Forward pass -> class logits
        2. Backward from target class logit
        3. For each spatial attention layer:
               cam = (grad * attn).clamp(min=0).mean(heads)
               R_pp[t] += cam @ R_pp[t]
        4. Extract via pooling head probe: heatmap = cam_cross @ R_pp[t]
           (analog of R[cls, 1:] for CLS-less SigLIP)
        5. Min-max normalize per frame
    
    Args:
        model: MatchVision model with model.classifier for class logits
        video_frames: Video tensor [B, C, T, H, W]
        target_label: Class index for backprop (use with target_label_name)
        target_label_name: Class name string (resolved via LABEL_NAMES)
        device: Device to run on
        num_frames: Number of frames (T dimension)
        patch_size: Spatial grid size (14 for SigLIP-base = 196 patches)
    
    Returns:
        Heatmaps array [T, patch_size, patch_size] with values in [0, 1]
    """
    # Set model to eval mode but enable gradients for Chefer
    model.eval()
    
    # Wrap the model to capture attention weights AND gradients
    wrapped = wrap_matchvision_model(model)
    
    # Move inputs to device and enable gradients
    video_frames = video_frames.to(device)
    video_frames.requires_grad_(True)
    
    B = video_frames.shape[0]
    T = num_frames
    num_patches = patch_size * patch_size  # 196
    
    if DEBUG:
        print(f"[DEBUG] video_frames: {list(video_frames.shape)} (expect [B, C, T, H, W])")
        print(f"[DEBUG] B={B}, T={T}, num_patches={num_patches}")
        print(f"[DEBUG] Wrapped layers: spatial={len(wrapped['spatial_attn'])}, "
              f"pooling={'yes' if wrapped['pooling_head_attn'] else 'no'}")
    
    # Initialize per-frame relevance matrices (analog of R = eye(num_tokens) in example.py)
    # SigLIP has no CLS token, so R is [196, 196] per frame (patches only).
    # The probe (CLS equivalent) is handled at extraction via the pooling head.
    R_pp_per_frame = [torch.eye(num_patches, device=device) for _ in range(T)]
    if DEBUG:
        print(f"[DEBUG] R_pp_per_frame: {T} matrices of shape [{num_patches}, {num_patches}]")
    
    # Forward pass WITH gradients enabled (critical for Chefer!)
    model.zero_grad()
    
    # Get visual features
    if hasattr(model, 'siglip_model'):
        visual_encoder = model.siglip_model
    elif hasattr(model, 'visual_encoder'):
        visual_encoder = model.visual_encoder
    elif hasattr(model, 'timesformer'):
        visual_encoder = model
    else:
        raise ValueError("Model does not have 'siglip_model', 'visual_encoder' or 'timesformer' attribute")
    
    # Process video frames
    x = video_frames
    B_actual, _, T_actual, _, _ = x.shape
    
    x = rearrange(x, "b c t h w -> (b t) c h w")
    
    # Get patch embeddings
    x = visual_encoder.vision_model_embedding(x)  # [B*T, 196, 768]
    if DEBUG:
        print(f"[DEBUG] After vision_model_embedding: {list(x.shape)} (expect [B*T, 196, 768])")
    x = rearrange(x, "(b t) n m -> b n t m", b=B_actual, t=T_actual)
    x = x + visual_encoder.temporal_positional_embedding
    if DEBUG:
        tpe = visual_encoder.temporal_positional_embedding
        print(f"[DEBUG] After temporal_pos_embed addition: x={list(x.shape)}, "
              f"tpe_norm={tpe.norm().item():.4f}, tpe_max_abs={tpe.abs().max().item():.6f}")
    x = rearrange(x, "b n t m -> (b t) n m")
    
    # Forward through Timesformer
    x = visual_encoder.timesformer(x, B_actual, T_actual)
    if DEBUG:
        print(f"[DEBUG] After timesformer: {list(x.shape)} (expect [B*T, 196, 768])")
    
    # Continue forward through pooling to get final features
    x = visual_encoder.post_layernorm(x)
    x = visual_encoder.head(x)  # [B*T, 768] - spatial pooled
    if DEBUG:
        print(f"[DEBUG] After pooling head: {list(x.shape)} (expect [B*T, 768])")
    video_features = rearrange(x, "(b t) m -> b t m", b=B_actual, t=T_actual)
    if DEBUG:
        print(f"[DEBUG] video_features: {list(video_features.shape)} (expect [B, T, 768])")
    
    # Convert label name to index if provided
    if target_label_name is not None:
        if target_label_name.lower() not in LABEL_TO_IDX:
            raise ValueError(f"Unknown label '{target_label_name}'. Available: {LABEL_NAMES}")
        target_label = LABEL_TO_IDX[target_label_name.lower()]
        print(f"Resolved label name '{target_label_name}' to index {target_label}")
    
    # Backprop from classification logit
    # video_features is [B, T, 768] from visual encoder
    
    # Route through the full MatchVision_Classifier pipeline:
    #   classifier_ln1 -> transformer_encoder -> classifier_ln2 -> classifier
    # This matches MatchVision_Classifier.get_logits() exactly.
    if hasattr(model, 'classifier') and hasattr(model, 'classifier_ln1'):
        x_cls = model.classifier_ln1(video_features)  # [B, T, 768]
        if hasattr(model, 'use_transformer') and model.use_transformer and hasattr(model, 'transformer_encoder'):
            x_cls = x_cls.permute(1, 0, 2)  # [T, B, 768]
            x_cls = model.transformer_encoder(x_cls)
            if hasattr(model, 'classifier_transformer_type') and model.classifier_transformer_type == 'cls_token':
                x_cls = x_cls[0, :, :]  # [B, 768]
            else:
                x_cls = x_cls.mean(dim=0)  # [B, 768] — avg_pool (default)
        else:
            x_cls = x_cls.mean(dim=1)  # [B, 768]
        x_cls = model.classifier_ln2(x_cls)
        cls_logits = model.classifier(x_cls)  # [B, num_classes]
    elif hasattr(model, 'classifier'):
        # Simpler model with just a classifier head
        cls_features = video_features.mean(dim=1)
        cls_logits = model.classifier(cls_features)
    else:
        raise RuntimeError("Model does not have a classification head (model.classifier).")
    
    if DEBUG:
        print(f"[DEBUG] cls_logits: shape={list(cls_logits.shape)}, "
              f"min={cls_logits.min().item():.4f}, max={cls_logits.max().item():.4f}, "
              f"predicted_class={cls_logits.argmax(dim=-1).item()} "
              f"({LABEL_NAMES[cls_logits.argmax(dim=-1).item()]})")
    
    # Backprop from the target class logit (one-hot selection)
    target = cls_logits[0, target_label]
    print(f"Chefer class-logit backprop: class_idx={target_label}, logit={target.item():.4f}")
    
    # CRITICAL: Backward pass to compute gradients through attention
    target.backward(retain_graph=True)
    
    print(f"Computed backward pass. Target value: {target.item():.4f}")
    
    if DEBUG:
        # Verify gradients exist on video_frames (confirms backprop reached input)
        print(f"[DEBUG] video_frames.grad is None: {video_frames.grad is None}")
        if video_frames.grad is not None:
            print(f"[DEBUG] video_frames.grad norm: {video_frames.grad.norm().item():.6e}")
    
    # =========================================================================
    # Relevance propagation — mirrors example.py lines 23-30 exactly.
    # Only addition: per-frame loop (temporal axis from Timesformer).
    #
    # Original (example.py):
    #   for blk in image_attn_blocks:
    #       grad = blk.attn_grad
    #       cam = blk.attn_probs
    #       cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
    #       grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
    #       cam = grad * cam
    #       cam = cam.clamp(min=0).mean(dim=0)
    #       R += torch.matmul(cam, R)
    # =========================================================================
    for layer_idx, attn_module in wrapped['spatial_attn']:
        attn_probs = attn_module.get_attn()       # [B*T, num_heads, N, N]
        attn_grad = attn_module.get_attn_gradients()  # [B*T, num_heads, N, N]
        
        if DEBUG:
            print(f"[DEBUG] Layer {layer_idx} SPATIAL: "
                  f"attn={'None' if attn_probs is None else list(attn_probs.shape)} "
                  f"(expect [B*T={T}, heads, N={num_patches}, N={num_patches}]), "
                  f"grad={'None' if attn_grad is None else list(attn_grad.shape)}")
        
        if attn_probs is not None:
            for t in range(T):
                # --- identical to example.py lines 24-30, applied per frame ---
                cam = attn_probs[t]   # [num_heads, N, N]
                grad = attn_grad[t]   # [num_heads, N, N]
                cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                R_pp_per_frame[t] += torch.matmul(cam, R_pp_per_frame[t])
                
                if DEBUG and t == 0 and layer_idx == 0:
                    print(f"[DEBUG]   cam (frame 0, layer 0): shape={list(cam.shape)} "
                          f"(expect [{num_patches},{num_patches}]), "
                          f"norm={cam.norm().item():.6e}, "
                          f"max={cam.max().item():.6e}")
    
    if DEBUG:
        for t in range(min(3, T)):
            print(f"[DEBUG] R_pp_per_frame[{t}]: norm={R_pp_per_frame[t].norm().item():.4f}, "
                  f"diag_mean={R_pp_per_frame[t].diag().mean().item():.4f}")
    
    # =========================================================================
    # Extract heatmaps via pooling head cross-attention (CLS-token analog).
    #
    # SigLIP has no CLS token. Its pooling head has a learnable probe that
    # cross-attends to all 196 patches — functionally identical to CLS.
    #
    # Original Chefer (ViT with CLS):   image_relevance = R[0, 1:]
    # Our equivalent (SigLIP, no CLS):  image_relevance = cam_cross @ R
    #
    # cam_cross is the gradient-weighted probe→patches attention from the
    # pooling head, so cam_cross @ R reads the probe's relevance row.
    # =========================================================================
    
    head_attn_module = wrapped['pooling_head_attn']
    head_attn = head_attn_module.get_attn()           # [B*T, num_heads, 1, N]
    head_grad = head_attn_module.get_attn_gradients()  # [B*T, num_heads, 1, N]
    print(f"  Pooling head: cross-attention {head_attn.shape} + gradients {head_grad.shape}")
    
    heatmaps = []
    for t in range(T):
        # Gradient-weighted cross-attention (same rule as spatial layers)
        cam = head_attn[t]   # [num_heads, 1, N]
        grad = head_grad[t]  # [num_heads, 1, N]
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)          # [1, N]
        
        # Since SigLIP has no CLS token, cam_cross @ R is the equivalent of R[cls, 1:] in original Chefer
        image_relevance = torch.matmul(cam, R_pp_per_frame[t]).squeeze(0)  # [N]
        
        if DEBUG and t == 0:
            print(f"[DEBUG] image_relevance (frame 0): shape={list(image_relevance.shape)} "
                  f"(expect [{num_patches}]), "
                  f"min={image_relevance.min().item():.6f}, "
                  f"max={image_relevance.max().item():.6f}, "
                  f"mean={image_relevance.mean().item():.6f}")
        
        heatmap = image_relevance.detach().cpu().numpy().reshape(patch_size, patch_size)
        heatmaps.append(heatmap)
    
    heatmaps = np.stack(heatmaps, axis=0)  # [T, patch_size, patch_size]
    if DEBUG:
        print(f"[DEBUG] Raw heatmaps: shape={list(heatmaps.shape)} (expect [{T}, {patch_size}, {patch_size}]), "
              f"global_min={heatmaps.min():.6f}, global_max={heatmaps.max():.6f}")
    
    # Min-max normalization per frame
    for t in range(T):
        h = heatmaps[t]
        if h.max() > h.min():
            heatmaps[t] = (h - h.min()) / (h.max() - h.min())
        else:
            heatmaps[t] = np.zeros_like(h)
    
    if DEBUG:
        print(f"[DEBUG] Normalized heatmaps: shape={list(heatmaps.shape)}, "
              f"all in [0,1]: min={heatmaps.min():.4f}, max={heatmaps.max():.4f}")
        # Per-frame stats for first 3 frames
        for t in range(min(3, T)):
            h = heatmaps[t]
            print(f"[DEBUG]   frame {t}: mean={h.mean():.4f}, std={h.std():.4f}, "
                  f"max_pos=({np.unravel_index(h.argmax(), h.shape)})")
    
    return heatmaps


# ============================================================================
# Attribution Renderer
# ============================================================================

def chefer_attribution_renderer(frame_float, attribution_map):
    """
    Render a Chefer heatmap overlay on a frame.

    Replaces pytorch_grad_cam's show_cam_on_image so we have no dependency
    on that library.

    Args:
        frame_float: [H, W, 3] float32 image in [0, 1]
        attribution_map: [h, w] float32 heatmap in [0, 1] (may differ in size)

    Returns:
        [H, W, 3] uint8 RGB overlay image
    """
    H, W = frame_float.shape[:2]
    heatmap = cv2.resize(attribution_map, (W, H), interpolation=cv2.INTER_LINEAR)

    heatmap = np.clip(heatmap, 0, 1)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    alpha = 0.5
    overlay = alpha * heatmap_colored + (1 - alpha) * frame_float
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


# ============================================================================
# Main Inference
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Chefer explainability inference (per-frame spatial-only) for MatchVision classification.')
    parser.add_argument('--config_path', type=str,
                        default='config/pretrain_classification_ibex.py',
                        help='Path to the Python config file')
    parser.add_argument('--checkpoint_path', type=str,
                        default='/path/to/pretrained_classification.pth',
                        help='Path to the checkpoint file')
    parser.add_argument('--coco_json', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'annotations-coco.json'),
                        help='Path to annotations-coco.json for attribution evaluation')
    parser.add_argument('--cam_threshold', type=float, default=0.5,
                        help='Fraction of max to binarise heatmap for IoU (default: 0.5)')
    parser.add_argument('--eval_output_json', type=str, default=None,
                        help='Optional path to save per-video evaluation results as JSON')
    parser.add_argument('--output_dir', type=str,
                        default='../output_chefer_soccer/',
                        help='Directory to save attribution visualization outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Load config and setup
    # ----------------------------------------------------------------
    config = load_config(args.config_path)
    checkpoint_path = args.checkpoint_path

    config_dataset = config['dataset']
    config_test_dataset = config_dataset['test']

    config_training_settings = config['training_settings']
    device_ids = config_training_settings['device_ids']
    classifier_transformer_type = config_training_settings['classifier_transformer_type']
    encoder_type = config_training_settings['encoder_type']
    use_transformer = config_training_settings['use_transformer']

    devices = [torch.device(f'cuda:{i}') for i in device_ids]

    # ----------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------
    test_dataset, test_data_loader = create_test_dataloader(config_test_dataset)

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    classifier = load_classifier(
        config_test_dataset, classifier_transformer_type, encoder_type,
        use_transformer, checkpoint_path, devices, device_ids,
    )

    print(f"Model loaded. Keywords ({len(test_dataset.keywords)}): {test_dataset.keywords}")

    # ----------------------------------------------------------------
    # COCO evaluator (optional)
    # ----------------------------------------------------------------
    coco_json_path = os.path.abspath(args.coco_json)
    attribution_evaluator, all_eval_results = setup_attribution_evaluator(coco_json_path)

    # ----------------------------------------------------------------
    # Inference loop
    # ----------------------------------------------------------------
    all_predictions = []
    test_progress_bar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc='Chefer Per-Frame Inference')

    for _, (frames, caption, dummy_frames, video_path, caption_text) in test_progress_bar:
        video_name = video_path[0].split('/')[-1]
        print(f'\nProcessing video: {video_name}')

        matched_video_id = match_video_id(attribution_evaluator, video_path[0])

        frames = frames.to(devices[0])
        vp_parts = video_path[0].replace('\\', '/').split('/')
        match_name = vp_parts[-2] if len(vp_parts) >= 2 else 'unknown_match'
        video_timestamp = video_name.replace('.mp4', '')
        video_directory = os.path.join(args.output_dir, match_name)
        os.makedirs(video_directory, exist_ok=True)

        # ----------------------------------------------------------
        # Run Chefer explainability
        # ----------------------------------------------------------
        target_label_idx = caption[0].item()

        chefer_heatmaps = generate_per_frame_heatmaps(
            classifier.module,
            frames,
            target_label=target_label_idx,
            device=str(devices[0]),
            num_frames=frames.shape[2],
        )
        # chefer_heatmaps: [T, 14, 14] numpy float32 in [0, 1]
        print(f'Chefer per-frame heatmap shape: {chefer_heatmaps.shape}')

        # Compute per-frame attribution scores (mean heatmap intensity)
        chefer_scores = chefer_heatmaps.mean(axis=(1, 2))  # [T]
        print(f'Chefer scores shape: {chefer_scores.shape}')

        # ----------------------------------------------------------
        # Get model predictions (fresh forward pass without hooks)
        # ----------------------------------------------------------
        with torch.no_grad():
            logits = classifier.module.forward(frames)

        i = 0
        for sample_idx in range(frames.shape[0]):
            new_frames = dummy_frames[sample_idx]
            new_frames = new_frames.permute(0, 2, 3, 1)  # [T, H, W, C]

            # ----------------------------------------------------------
            # COCO evaluation
            # ----------------------------------------------------------
            evaluate_and_print_video(
                attribution_evaluator, chefer_heatmaps, matched_video_id,
                video_name, args.cam_threshold, all_eval_results,
            )

            # ----------------------------------------------------------
            # Predictions
            # ----------------------------------------------------------
            predictions = classifier.module.get_types(logits)
            prediction_text = test_dataset.keywords[predictions[sample_idx, 0].item()]
            ground_truth_text = caption_text[sample_idx]
            print(f'Prediction: {prediction_text}')
            print(f'Ground Truth: {ground_truth_text}')

            # ----------------------------------------------------------
            # Visualization
            # ----------------------------------------------------------
            save_lowres_visualization_video(
                video_directory=video_directory,
                video_name=video_timestamp,
                lowres_frames=new_frames,
                attribution_maps=chefer_heatmaps,
                attribution_scores=chefer_scores,
                prediction_text=prediction_text,
                ground_truth_text=ground_truth_text,
                attribution_evaluator=attribution_evaluator,
                matched_video_id=matched_video_id if attribution_evaluator else None,
                cam_threshold=args.cam_threshold,
                attribution_method_name='Chefer-PerFrame',
                attribution_renderer=chefer_attribution_renderer,
            )

            i += 1

        all_predictions.append(predictions.cpu())
        del frames

    all_predictions = torch.cat(all_predictions, dim=0)
    print(all_predictions)

    # ----------------------------------------------------------------
    # Global evaluation summary
    # ----------------------------------------------------------------
    if attribution_evaluator is not None and all_eval_results:
        eval_output = args.eval_output_json
        if eval_output is None:
            eval_output = os.path.join(args.output_dir, 'chefer_eval_results.json')
        print_and_save_eval_summary(
            all_eval_results,
            eval_output_path=eval_output,
            summary_title='Chefer Per-Frame Attribution Label-Group Evaluation Summary',
        )


if __name__ == '__main__':
    main()
