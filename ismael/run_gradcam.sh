#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256GB
#SBATCH --time=03:59:59
#SBATCH --partition=batch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anon@example.com
#SBATCH --output=/content/SoccerExplainability/slurm_outputs/gradcam-%j.out
# Run gradcam_inference.py — GradCAM explainability pipeline

set -e

pip install -r /content/SoccerExplainability/environment.txt

export HF_HOME="/content/huggingface_cache"

SOCCER_DIR="/content/SoccerExplainability"
GOOGLE_DRIVE_DIR="/content/drive/MyDrive/SoccerExplainability-output-$(date +%Y-%m-%d)"
OUTPUT_DIR="${GOOGLE_DRIVE_DIR}/output_gradcam"

mkdir -p "${SOCCER_DIR}/slurm_outputs"
mkdir -p "${OUTPUT_DIR}"

cd "${SOCCER_DIR}/inference"

python gradcam_inference.py \
    --config_path "${SOCCER_DIR}/config/pretrain_classification.py" \
    --checkpoint_path "/path/to/UniSoccer/pretrained_classification.pth" \
    --coco_json "${SOCCER_DIR}/annotations-coco.json" \
    --cam_threshold 0.5 \
    --eval_output_json "${OUTPUT_DIR}/gradcam_eval_results.json" \
    --output_dir "${OUTPUT_DIR}"
