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
#SBATCH --output=/content/SoccerExplainability/slurm_outputs/chefer_soccermaster_temporal-%j.out
# Run chefer_soccermaster_temporal.py — Chefer spatial+temporal explainability for SoccerMaster
#   (adds per-spatial-position R_tt over the 11 backbone temporal_attn modules in layers 16-26,
#    accumulated together with the 2 CaptionClassificationHead TransformerEncoder layers).

set -e

cd /content/SoccerExplainability
git checkout Rebuttal
cd /content/

pip install -r /content/SoccerExplainability/environment.txt

mkdir models && cd models && git clone https://huggingface.co/xleprime/SoccerMaster

# Needed for G4 instances on Colab
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install latest nightly with CUDA 12.8 (Blackwell requires 12.4+)
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

export HF_HOME="/content/huggingface_cache"

SOCCER_DIR="/content/SoccerExplainability"
GOOGLE_DRIVE_DIR="/content/drive/MyDrive/SoccerExplainability-output-$(date +%Y-%m-%d)"
OUTPUT_DIR="${GOOGLE_DRIVE_DIR}/output_chefer_soccermaster_temporal"
SALIENCY_DIR="${OUTPUT_DIR}/saliency"
CHECKPOINT_DIR="/content/models/SoccerMaster/pretrained_models/SoccerMaster"

mkdir -p "${SOCCER_DIR}/slurm_outputs"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SALIENCY_DIR}"

cd "${SOCCER_DIR}/inference"

python chefer_soccermaster_temporal.py \
    --config_path "${SOCCER_DIR}/config/pretrain_classification.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --siglip2_path "google/siglip2-large-patch16-512" \
    --coco_json "${SOCCER_DIR}/annotations-coco.json" \
    --cam_threshold 0.5 \
    --eval_output_json "${OUTPUT_DIR}/chefer_soccermaster_temporal_eval_results.json" \
    --output_dir "${OUTPUT_DIR}" \
    --input_size 512 \
    --saliency_save_dir "${SALIENCY_DIR}"

python convergence_analysis.py --saliency_dir "${SALIENCY_DIR}"
