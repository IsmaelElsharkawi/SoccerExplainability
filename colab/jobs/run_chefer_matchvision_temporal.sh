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
DRIVE_ROOT="/content/drive/MyDrive/SoccerExplainability-output"
EXPERIMENT_NAME="chefer_matchvision_temporal"
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
