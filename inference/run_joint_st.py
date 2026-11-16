"""Save ground-truth-target Joint-ST heatmaps, without benchmark scoring.

Use the existing test manifest/config; see docs/joint_st.md for usage.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.video_utils_siglip import read_frames_decord
from inference_utils import load_config

MODELS = ("matchvision", "soccermaster", "clip", "siglip", "peav", "videoprism")


def load_model(args, keywords, config, device):
    """Load the same classifiers used by the attribution experiments."""
    if args.model == "matchvision":
        from inference_utils import load_classifier

        if not args.checkpoint:
            raise ValueError("MatchVision requires --checkpoint PATH")
        settings = config["training_settings"]
        index = device.index if device.index is not None else 0
        return load_classifier(
            config["dataset"]["test"],
            settings["classifier_transformer_type"],
            settings["encoder_type"],
            settings["use_transformer"],
            args.checkpoint,
            [device],
            [index],
        ).module.eval()
    if args.model == "soccermaster":
        from chefer_soccermaster_temporal import load_soccermaster_model

        if not args.checkpoint:
            raise ValueError("SoccerMaster requires --checkpoint DIRECTORY")
        return load_soccermaster_model(
            args.checkpoint,
            device,
            num_frames=args.num_frames,
            siglip2_path=args.model_name or "google/siglip2-large-patch16-512",
        )
    if args.model == "siglip":
        from model.SigLIP_classifier import SigLIP_Classifier

        return (
            SigLIP_Classifier(
                keywords=keywords,
                model_name=args.model_name or "google/siglip-base-patch16-224",
                temporal_aggregation="mean",
            )
            .eval()
            .to(device)
        )
    if args.model == "clip":
        from model.CLIP_classifier import CLIP_Classifier

        return (
            CLIP_Classifier(
                keywords=keywords,
                model_name=args.model_name or "openai/clip-vit-base-patch16",
                temporal_aggregation="mean",
            )
            .eval()
            .to(device)
        )
    if args.model == "peav":
        from model.PEAV_classifier import PEAVVideoClassifier as Classifier
    else:
        from model.VideoPrism_classifier import VideoPrismClassifier as Classifier
    options = {"model_name": args.model_name} if args.model_name else {}
    if args.revision:
        options["revision"] = args.revision
    return (
        Classifier(keywords=keywords, **options)
        .eval()
        .to(device=device, dtype=getattr(torch, args.dtype))
    )


def attribute(model, frames, target, model_key, device):
    """Return an unnormalized [T,H,W] relevance volume."""
    if model_key in {"peav", "videoprism"}:
        from chefer_video_foundation import (
            generate_peav_heatmaps,
            generate_videoprism_heatmaps,
        )

        generator = (
            generate_peav_heatmaps
            if model_key == "peav"
            else generate_videoprism_heatmaps
        )
        return generator(model, frames, target_label=target)[0]
    if model_key == "clip":
        from chefer_off_the_shelf import generate_clip_heatmaps

        return generate_clip_heatmaps(model, frames, target_label=target)[0]
    if model_key == "matchvision":
        from chefer_matchvision_temporal import generate_per_frame_heatmaps as generator
    elif model_key == "soccermaster":
        from chefer_soccermaster_temporal import (
            generate_per_frame_heatmaps_soccermaster as generator,
        )
    else:
        from chefer_siglip import generate_per_frame_heatmaps_siglip as generator
    options = {}
    if model_key == "siglip":
        vision = model.siglip_model.config
        options["patch_size"] = vision.image_size // vision.patch_size
    return generator(
        model,
        frames,
        target_label=target,
        device=str(device),
        num_frames=frames.shape[2],
        propagation_mode="joint_spatiotemporal",
        **options,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument(
        "--test_json", help="Optional replacement for the configured test manifest"
    )
    parser.add_argument("--video_base", help="Required with --test_json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--checkpoint",
        help="MatchVision checkpoint file or SoccerMaster checkpoint directory",
    )
    parser.add_argument("--model_name", help="Optional Hugging Face model ID/path")
    parser.add_argument(
        "--revision", help="PE-AV/VideoPrism checkpoint revision override"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="PE-AV/VideoPrism runtime precision",
    )
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--max_clips", type=int)
    args = parser.parse_args()
    if args.num_frames <= 0 or (args.max_clips is not None and args.max_clips <= 0):
        parser.error("Frame and clip counts must be positive")
    if bool(args.test_json) != bool(args.video_base):
        parser.error("Pass --test_json and --video_base together")
    config = load_config(args.config_path)
    test = config["dataset"]["test"]
    keywords = list(test["keywords"])
    manifests = [args.test_json] if args.test_json else test["json"]
    roots = [args.video_base] if args.video_base else test["video_base"]
    if len(manifests) != len(roots):
        raise ValueError("Test manifest and video-root counts differ")
    device = torch.device(args.device)
    model = load_model(args, keywords, config, device)
    processor = None
    if args.model not in {"peav", "videoprism"}:
        processor_name = args.model_name if args.model in {"clip", "siglip"} else None
        default_processor = (
            "openai/clip-vit-base-patch16"
            if args.model == "clip"
            else "google/siglip-base-patch16-224"
        )
        processor = AutoProcessor.from_pretrained(processor_name or default_processor)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "joint_st_run.json").write_text(
        json.dumps(
            {
                **vars(args),
                "method": "Joint-ST",
                "target": "ground_truth",
                "normalization": "none",
                "keywords": keywords,
                "checkpoint_revision": getattr(model, "checkpoint_revision", None),
            },
            indent=2,
        )
        + "\n"
    )
    count = 0
    seen = set()
    for manifest, video_root in zip(manifests, roots):
        for item in json.loads(Path(manifest).read_text()):
            if args.max_clips is not None and count >= args.max_clips:
                return
            clip_id = item["video"] + (
                "#" + item["variant"] if item.get("variant") else ""
            )
            if clip_id in seen:
                raise ValueError(f"Duplicate example: {clip_id}")
            seen.add(clip_id)
            target = keywords.index(item["caption"])
            raw, indices, duration = read_frames_decord(
                str(Path(video_root) / item["video"]),
                args.num_frames,
                test.get("sample", "middle"),
            )
            if processor is None:
                frames = model.preprocess(raw.unsqueeze(0).to(device))
            else:
                processed = torch.cat(
                    [
                        processor(images=frame, return_tensors="pt")["pixel_values"]
                        for frame in raw
                    ]
                )
                frames = processed.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            maps = attribute(model, frames, target, args.model, device)
            if (
                maps.ndim != 3
                or len(maps) != args.num_frames
                or not np.isfinite(maps).all()
                or (maps < 0).any()
            ):
                raise ValueError(f"Invalid Joint-ST map for {clip_id}")
            # Mirror the repository's saved-map filename convention.
            stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in clip_id)
            np.savez_compressed(
                output / (stem + ".npz"),
                saliency_maps=maps,
                frame_scores=maps.mean(axis=(1, 2)),
                frame_indices=indices,
            )
            (output / (stem + ".json")).write_text(
                json.dumps(
                    {
                        "clip_id": clip_id,
                        "method": "Joint-ST",
                        "target_index": target,
                        "ground_truth_text": item["caption"],
                        "duration": duration,
                        "normalization": "none",
                    },
                    indent=2,
                )
                + "\n"
            )
            count += 1
            print(f"[{count}] {clip_id}: {maps.shape}", flush=True)


if __name__ == "__main__":
    main()
