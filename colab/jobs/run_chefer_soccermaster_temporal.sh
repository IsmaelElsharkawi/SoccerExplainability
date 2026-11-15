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

# Install stable PyTorch with CUDA 12.8 (Blackwell requires 12.4+)
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

export HF_HOME="/content/huggingface_cache"

SOCCER_DIR="/content/SoccerExplainability"
DRIVE_ROOT="/content/drive/MyDrive/SoccerExplainability-output"
EXPERIMENT_NAME="chefer_soccermaster_temporal"
EXPERIMENT_DIR="${DRIVE_ROOT}/${EXPERIMENT_NAME}"

# Claim a fresh run directory. `mkdir` without -p fails when the directory
# already exists, so the claim is atomic: two jobs starting at the same instant
# can never be handed the same run id, and an id is never reused.
mkdir -p "${EXPERIMENT_DIR}"
RUN_DIR=""
for _attempt in $(seq 1 50); do
    RUN_ID=$(python3 -c 'import random; print("%04d" % random.randrange(10000))')
    if mkdir "${EXPERIMENT_DIR}/run_${RUN_ID}" 2>/dev/null; then
        RUN_DIR="${EXPERIMENT_DIR}/run_${RUN_ID}"
        break
    fi
done
if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: no unused run id found under ${EXPERIMENT_DIR} after 50 tries." >&2
    exit 1
fi
echo "Run directory: ${RUN_DIR}"

# Minimal provenance so a downloaded run can always be identified later.
{
    echo "experiment:  ${EXPERIMENT_NAME}"
    echo "run_id:      ${RUN_ID}"
    echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit:  $(git -C "${SOCCER_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} > "${RUN_DIR}/run_info.txt"

OUTPUT_DIR="${RUN_DIR}"
SALIENCY_DIR="${OUTPUT_DIR}/saliency"
CHECKPOINT_DIR="/content/models/SoccerMaster"

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

python convergence_analysis.py \
    --saliency_dir "${SALIENCY_DIR}" \
    --selected_videos_json "${SOCCER_DIR}/train_data/json/selected_videos_for_annotations.json"
