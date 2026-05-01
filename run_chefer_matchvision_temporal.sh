#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256GB
#SBATCH --time=03:59:59
#SBATCH --partition=batch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ahmed.sait@kaust.edu.sa
#SBATCH --account conf-neurips-2026.05.15-ghanembs
#SBATCH --output=/ibex/ai/home/saitaa0b/Projects/XMatchVision/SoccerExplainability/slurm_outputs/chefer_temporal-%j.out
# Run chefer_matchvision_temporal.py — Chefer spatial+temporal explainability pipeline
#   (adds per-spatial-position R_tt over the 12 backbone temporal_attn modules,
#    accumulated together with the 2 classifier head TransformerEncoder layers).

eval "$(conda shell.bash hook)"
conda activate UniSoccer

set -e

export HF_HOME="/ibex/ai/home/saitaa0b/.cache/huggingface"

SOCCER_DIR="/ibex/ai/home/saitaa0b/Projects/XMatchVision/SoccerExplainability"
OUTPUT_DIR="${SOCCER_DIR}/output_chefer_matchvision_temporal"

mkdir -p "${SOCCER_DIR}/slurm_outputs"
mkdir -p "${OUTPUT_DIR}"

cd "${SOCCER_DIR}/inference"

python chefer_matchvision_temporal.py \
    --config_path "${SOCCER_DIR}/config/pretrain_classification_ibex.py" \
    --checkpoint_path "/ibex/ai/home/saitaa0b/.cache/huggingface/hub/models--Homie0609--UniSoccer/snapshots/c8b4f77a2951f067ad2afd1ffb6cd479ff39601f/pretrained_classification.pth" \
    --coco_json "${SOCCER_DIR}/annotations-coco.json" \
    --cam_threshold 0.5 \
    --eval_output_json "${OUTPUT_DIR}/chefer_matchvision_temporal_eval_results.json" \
    --output_dir "${OUTPUT_DIR}"