#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256GB
#SBATCH --time=03:59:59
#SBATCH --partition=batch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anon@example.com
#SBATCH --account <your-slurm-account>
#SBATCH --output=/content/SoccerExplainability/slurm_outputs/chefer_temporal-%j.out
# Run chefer_matchvision_temporal.py — Chefer spatial+temporal explainability pipeline
#   (adds per-spatial-position R_tt over the 12 backbone temporal_attn modules,
#    accumulated together with the 2 classifier head TransformerEncoder layers).

set -e

cd /content/SoccerExplainability
git checkout Rebuttal
cd /content/

pip install -r /content/SoccerExplainability/environment.txt

wget https://huggingface.co/Homie0609/UniSoccer/resolve/main/pretrained_classification.pth

export HF_HOME="/content/huggingface_cache"

SOCCER_DIR="/content/SoccerExplainability"
GOOGLE_DRIVE_DIR="/content/drive/MyDrive/SoccerExplainability-output-$(date +%Y-%m-%d)"
OUTPUT_DIR="${GOOGLE_DRIVE_DIR}/output_chefer_matchvision_temporal"
SALIENCY_DIR="${OUTPUT_DIR}/saliency"

mkdir -p "${SOCCER_DIR}/slurm_outputs"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SALIENCY_DIR}"

cd "${SOCCER_DIR}/inference"

python chefer_matchvision_temporal.py \
    --config_path "${SOCCER_DIR}/config/pretrain_classification.py" \
    --checkpoint_path "/content/pretrained_classification.pth" \
    --coco_json "${SOCCER_DIR}/annotations-coco.json" \
    --cam_threshold 0.5 \
    --eval_output_json "${OUTPUT_DIR}/chefer_matchvision_temporal_eval_results.json" \
    --output_dir "${OUTPUT_DIR}" \
    --saliency_save_dir "${SALIENCY_DIR}"

python convergence_analysis.py \
    --saliency_dir "${SALIENCY_DIR}" \
    --selected_videos_json "${SOCCER_DIR}/train_data/json/selected_videos_for_annotations.json"
