#!/bin/bash
# End-to-end reproduction of Table 1 (grounding results) on Google Colab.
#
#   Stage 1  SigLIP, MatchVision, SoccerMaster  (repo env: transformers 4.43.2)
#   Stage 2  CLIP, PE-AV, VideoPrism            (transformers 5.22.2, timm 1.0.31)
#   Stage 3  Energy / Pointing / S-AUROC / S-AUPRC / T-IoU for all six models,
#            scored from the saved maps by inference/evaluate_grounding.py at
#            frame resolution (headline) and on the native token grid
#            (validation against the paper's current numbers).
#
# Colab cells (G4 / RTX Pro 6000 runtime; PE-AV with 30 frames needs ~40 GB):
#
#   from google.colab import drive
#   drive.mount('/content/drive')
#
#   !git clone -b table1-eval https://github.com/IsmaelElsharkawi/SoccerExplainability.git /content/SoccerExplainability
#   !bash /content/SoccerExplainability/colab/run_table1.sh
#
# Clips are read from /content/drive/MyDrive/SoccerLens/ (config/pretrain_classification.py).
# Everything is written to ${DRIVE_ROOT}/table1/run_XXXX/:
#   <model>/saliency/*.npz|json   per-clip maps (the only input stage 3 needs)
#   table1/table1_frame.{md,tex}  headline table
#   table1/table1_native.md       token-grid table
#   table1/validation_*.md        cell-by-cell diff against the paper
#
# Optional environment variables:
#   RUN_DIR=/content/drive/.../run_1234   resume an earlier run (finished models and
#                                         clips are skipped)
#   MODELS="clip peav"                    subset of: siglip matchvision soccermaster clip peav videoprism
#   SKIP_EVAL=1                           stop after inference
#   EXTRA_RUNS="Label=/path/to/saliency"  more rows for the table, e.g. an older bundle
#   CANVAS=720x1280                       frame-resolution canvas (HEIGHTxWIDTH)

set -euo pipefail

SOCCER_DIR="/content/SoccerExplainability"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/SoccerExplainability-output}"
MODELS="${MODELS:-siglip matchvision soccermaster clip peav videoprism}"
CANVAS="${CANVAS:-720x1280}"
CONFIG="${SOCCER_DIR}/config/pretrain_classification.py"
COCO_JSON="${SOCCER_DIR}/annotations-coco.json"
EXPECTED_CLIPS=200
export HF_HOME="/content/huggingface_cache"
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
if [ ! -d "${SOCCER_DIR}/inference" ]; then
    echo "ERROR: repo not found at ${SOCCER_DIR} (the config hard-codes this path)." >&2
    exit 1
fi
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "ERROR: Google Drive is not mounted. Run drive.mount('/content/drive') in a cell first." >&2
    exit 1
fi
if [ ! -d "/content/drive/MyDrive/SoccerLens" ]; then
    echo "ERROR: clip folder /content/drive/MyDrive/SoccerLens not found." >&2
    exit 1
fi
nvidia-smi -L || { echo "ERROR: no GPU visible." >&2; exit 1; }

# ---------------------------------------------------------------------------
# Run directory (atomic claim, same scheme as the per-model scripts)
# ---------------------------------------------------------------------------
if [ -z "${RUN_DIR:-}" ]; then
    EXPERIMENT_DIR="${DRIVE_ROOT}/table1"
    mkdir -p "${EXPERIMENT_DIR}"
    for _attempt in $(seq 1 50); do
        RUN_ID=$(python3 -c 'import random; print("%04d" % random.randrange(10000))')
        if mkdir "${EXPERIMENT_DIR}/run_${RUN_ID}" 2>/dev/null; then
            RUN_DIR="${EXPERIMENT_DIR}/run_${RUN_ID}"
            break
        fi
    done
    if [ -z "${RUN_DIR:-}" ]; then
        echo "ERROR: no unused run id found under ${EXPERIMENT_DIR} after 50 tries." >&2
        exit 1
    fi
fi
mkdir -p "${RUN_DIR}"
echo "Run directory: ${RUN_DIR}"
{
    echo "experiment:  table1"
    echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_branch:  $(git -C "${SOCCER_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "git_commit:  $(git -C "${SOCCER_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty:   $(git -C "${SOCCER_DIR}" status --porcelain 2>/dev/null | grep -qv '^??' && echo yes || echo no)"
    echo "gpu:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "models:      ${MODELS}"
} >> "${RUN_DIR}/run_info.txt"

wants() { [[ " ${MODELS} " == *" $1 "* ]]; }

n_done() {
    local dir="$1"
    [ -d "${dir}" ] && find "${dir}" -maxdepth 1 -name '*.npz' | wc -l || echo 0
}

# Run a model unless its saliency dir is already complete.
needs_run() {
    local model="$1"
    local done
    done=$(n_done "${RUN_DIR}/${model}/saliency")
    if [ "${done}" -ge "${EXPECTED_CLIPS}" ]; then
        echo "[${model}] ${done} clips already present, skipping."
        return 1
    fi
    mkdir -p "${RUN_DIR}/${model}/saliency"
    return 0
}

install_repo_env() {
    # environment.txt pins torch==2.3.1, which current Colab Pythons cannot
    # install and which has no Blackwell (G4) kernels anyway. Install every
    # other pin, then a CUDA 12.8 torch build (harmless on older GPUs).
    grep -viE '^(torch|torchvision|torchaudio)([=<>~ ]|$)' "${SOCCER_DIR}/environment.txt" > /tmp/env_no_torch.txt
    pip install -q -r /tmp/env_no_torch.txt
    pip install -q --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
}

# ---------------------------------------------------------------------------
# Stage 1: soccer models + SigLIP with the repo's own scripts
# ---------------------------------------------------------------------------
if wants siglip || wants matchvision || wants soccermaster; then
    echo "=== Stage 1: repo environment ==="
    install_repo_env
    cd "${SOCCER_DIR}/inference"

    if wants siglip && needs_run siglip; then
        python chefer_siglip.py \
            --config_path "${CONFIG}" \
            --model_name "google/siglip-base-patch16-224" \
            --coco_json "${COCO_JSON}" \
            --cam_threshold 0.5 \
            --eval_output_json "${RUN_DIR}/siglip/chefer_siglip_eval_results.json" \
            --output_dir "${RUN_DIR}/siglip" \
            --saliency_save_dir "${RUN_DIR}/siglip/saliency"
    fi

    if wants matchvision && needs_run matchvision; then
        MV_CKPT="/content/pretrained_classification.pth"
        [ -f "${MV_CKPT}" ] || wget -q -O "${MV_CKPT}" \
            https://huggingface.co/Homie0609/UniSoccer/resolve/main/pretrained_classification.pth
        python chefer_matchvision_temporal.py \
            --config_path "${CONFIG}" \
            --checkpoint_path "${MV_CKPT}" \
            --coco_json "${COCO_JSON}" \
            --cam_threshold 0.5 \
            --eval_output_json "${RUN_DIR}/matchvision/chefer_matchvision_temporal_eval_results.json" \
            --output_dir "${RUN_DIR}/matchvision" \
            --saliency_save_dir "${RUN_DIR}/matchvision/saliency"
    fi

    if wants soccermaster && needs_run soccermaster; then
        SM_CKPT="/content/models/SoccerMaster"
        if [ ! -d "${SM_CKPT}" ]; then
            git lfs install --skip-repo >/dev/null 2>&1 || true
            mkdir -p /content/models
            git clone -q https://huggingface.co/xleprime/SoccerMaster "${SM_CKPT}"
        fi
        python chefer_soccermaster_temporal.py \
            --config_path "${CONFIG}" \
            --checkpoint_dir "${SM_CKPT}" \
            --siglip2_path "google/siglip2-large-patch16-512" \
            --coco_json "${COCO_JSON}" \
            --cam_threshold 0.5 \
            --eval_output_json "${RUN_DIR}/soccermaster/chefer_soccermaster_temporal_eval_results.json" \
            --output_dir "${RUN_DIR}/soccermaster" \
            --input_size 512 \
            --saliency_save_dir "${RUN_DIR}/soccermaster/saliency"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 2: off-the-shelf baselines on the newer stack
# ---------------------------------------------------------------------------
if wants clip || wants peav || wants videoprism; then
    echo "=== Stage 2: transformers 5.22.2 / timm 1.0.31 ==="
    # decord, einops, imageio etc. come from the repo environment.
    install_repo_env
    pip install -q "transformers==5.22.2" "timm==1.0.31" sentencepiece protobuf
    cd "${SOCCER_DIR}/inference"

    for model in clip peav videoprism; do
        if wants "${model}" && needs_run "${model}"; then
            python chefer_baselines.py \
                --model "${model}" \
                --config_path "${CONFIG}" \
                --saliency_save_dir "${RUN_DIR}/${model}/saliency" \
                2>&1 | tee -a "${RUN_DIR}/${model}/log.txt"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Stage 3: Table 1
# ---------------------------------------------------------------------------
if [ "${SKIP_EVAL:-0}" = "1" ]; then
    echo "SKIP_EVAL=1: inference done, outputs in ${RUN_DIR}"
    exit 0
fi

echo "=== Stage 3: scoring ==="
declare -A LABELS=(
    [siglip]=SigLIP [clip]=CLIP [peav]=PE-AV [videoprism]=VideoPrism
    [matchvision]=MatchVision [soccermaster]=SoccerMaster
)
RUN_ARGS=()
for model in siglip clip peav videoprism matchvision soccermaster; do
    dir="${RUN_DIR}/${model}/saliency"
    done=$(n_done "${dir}")
    if [ "${done}" -gt 0 ]; then
        [ "${done}" -lt "${EXPECTED_CLIPS}" ] && echo "WARNING: ${model} has only ${done}/${EXPECTED_CLIPS} clips."
        RUN_ARGS+=(--run "${LABELS[$model]}=${dir}")
    fi
done
for extra in ${EXTRA_RUNS:-}; do
    RUN_ARGS+=(--run "${extra}")
done
if [ "${#RUN_ARGS[@]}" -eq 0 ]; then
    echo "ERROR: no saliency maps found under ${RUN_DIR}." >&2
    exit 1
fi

cd "${SOCCER_DIR}/inference"
python evaluate_grounding.py \
    --coco_json "${COCO_JSON}" \
    "${RUN_ARGS[@]}" \
    --resolution frame native \
    --canvas "${CANVAS}" \
    --out_dir "${RUN_DIR}/table1" \
    2>&1 | tee "${RUN_DIR}/table1_log.txt"

echo "Done. Table 1: ${RUN_DIR}/table1/table1_frame.md (LaTeX: table1_frame.tex)"
echo "Validation against the paper: ${RUN_DIR}/table1/validation_native.md"
