#!/bin/bash
# Run chefer_siglip.py - Chefer per-frame spatial-only explainability for plain SigLIP

set -e

SOCCER_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SOCCER_DIR}/output_chefer_siglip"

mkdir -p "${OUTPUT_DIR}"

cd "${SOCCER_DIR}/inference"

python chefer_siglip.py \
    --config_path "${SOCCER_DIR}/config/pretrain_classification_ibex.py" \
    --model_name "google/siglip-base-patch16-224" \
    --coco_json "${SOCCER_DIR}/annotations-coco.json" \
    --cam_threshold 0.5 \
    --eval_output_json "${OUTPUT_DIR}/chefer_siglip_eval_results.json" \
    --output_dir "${OUTPUT_DIR}"