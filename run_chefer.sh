#!/bin/bash
# Run chefer_inference.py — Chefer explainability pipeline

eval "$(conda shell.bash hook)"
conda activate your_env_name

set -e

export HF_HOME="/path/to/.cache/huggingface"

SOCCER_DIR="/path/to/SoccerExplainability"
OUTPUT_DIR="${SOCCER_DIR}/output_chefer_soccer"

mkdir -p "${OUTPUT_DIR}"

cd "${SOCCER_DIR}/inference"

python chefer_inference.py \
    --config_path "${SOCCER_DIR}/config/pretrain_classification_ibex.py" \
    --checkpoint_path "/path/to/pretrained_classification.pth" \
    --coco_json "${SOCCER_DIR}/annotations-coco.json" \
    --cam_threshold 0.5 \
    --eval_output_json "${OUTPUT_DIR}/chefer_eval_results.json" \
    --output_dir "${OUTPUT_DIR}"
