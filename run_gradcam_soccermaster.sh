#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256GB
#SBATCH --time=11:59:59
#SBATCH --partition=batch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anon@example.com
#SBATCH --account <your-slurm-account>
#SBATCH --output=/path/to/SoccerExplainability/slurm_outputs/gradcam_soccermaster-%j.out
# Run gradcam_inference.py for SoccerMaster

eval "$(conda shell.bash hook)"
conda activate UniSoccer

set -e

export HF_HOME="/path/to/huggingface_cache"

SOCCER_DIR="/path/to/SoccerExplainability"
OUTPUT_DIR="${SOCCER_DIR}/output_gradcam_soccermaster"
CHECKPOINT_DIR="${SOCCER_DIR}/model/SoccerMaster/pretrained_models/SoccerMaster"

mkdir -p "${SOCCER_DIR}/slurm_outputs"
mkdir -p "${OUTPUT_DIR}"

cd "${SOCCER_DIR}/inference"

python gradcam_inference.py \
    --model_type soccermaster \
    --config_path "${SOCCER_DIR}/config/pretrain_classification_cluster.py" \
    --soccermaster_checkpoint_dir "${CHECKPOINT_DIR}" \
    --coco_json "${SOCCER_DIR}/annotations-coco.json" \
    --cam_threshold 0.5 \
    --eval_output_json "${OUTPUT_DIR}/gradcam_soccermaster_eval_results.json" \
    --output_dir "${OUTPUT_DIR}"
