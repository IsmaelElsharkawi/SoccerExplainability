#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256GB
#SBATCH --time=11:59:59
#SBATCH --partition=batch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ahmed.sait@kaust.edu.sa
#SBATCH --account conf-neurips-2026.05.15-ghanembs
#SBATCH --output=/ibex/ai/home/saitaa0b/Projects/XMatchVision/SoccerExplainability/slurm_outputs/chefer_soccermaster_fourmatrix-%j.out
# Run chefer_soccermaster_fourmatrix.py — Chefer four-matrix relevance for factored space-time attention.
#   Layers 0-15 spatial-only, layers 16-23 factored space-time. R_X dense [30, 1024, 30, 1024] fp16 ~1.9 GB.

eval "$(conda shell.bash hook)"
conda activate UniSoccer

set -e

export HF_HOME="/ibex/ai/home/saitaa0b/.cache/huggingface"

SOCCER_DIR="/ibex/ai/home/saitaa0b/Projects/XMatchVision/SoccerExplainability"
OUTPUT_DIR="${SOCCER_DIR}/output_chefer_soccermaster_fourmatrix"
CHECKPOINT_DIR="${SOCCER_DIR}/model/SoccerMaster/pretrained_models/SoccerMaster"

mkdir -p "${SOCCER_DIR}/slurm_outputs"
mkdir -p "${OUTPUT_DIR}"

cd "${SOCCER_DIR}/inference"

python chefer_soccermaster_fourmatrix.py \
    --config_path "${SOCCER_DIR}/config/pretrain_classification_ibex.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --siglip2_path "google/siglip2-large-patch16-512" \
    --coco_json "${SOCCER_DIR}/annotations-coco.json" \
    --cam_threshold 0.5 \
    --eval_output_json "${OUTPUT_DIR}/chefer_soccermaster_fourmatrix_eval_results.json" \
    --output_dir "${OUTPUT_DIR}" \
    --input_size 512
