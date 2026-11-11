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
# Colab exports its notebook backend (matplotlib_inline), which the venvs do not have.
export MPLBACKEND=Agg

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

# Each stage gets its own Python 3.11 venv (built with uv): the repo's 2024
# pins (transformers 4.43.2 -> tokenizers 0.19) have no wheels for Colab's
# current Python, and the two stages need different transformers versions.
# torch is not taken from environment.txt (2.3.1 has no Blackwell kernels);
# both venvs get a CUDA 12.8 build instead, which also works on older GPUs.
PY_VERSION="3.11"
VENV_REPO="/content/venv_repo"
VENV_NEW="/content/venv_baselines"

# make_venv VENV REQUIREMENTS_FILE [extra packages...]
make_venv() {
    local venv="$1" reqs="$2"; shift 2
    command -v uv >/dev/null 2>&1 || pip install -q uv
    if [ ! -x "${venv}/bin/python" ]; then
        uv venv --python "${PY_VERSION}" "${venv}"
    fi
    echo "Building ${venv} (Python ${PY_VERSION}; torch with CUDA 12.8 is a multi-GB download, expect 3-10 min)..."
    uv pip install --python "${venv}/bin/python" \
        --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        torch torchvision -r "${reqs}" "$@"
    "${venv}/bin/python" -c "import sys, torch, transformers; print('python', sys.version.split()[0], '| torch', torch.__version__, 'cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0), '| transformers', transformers.__version__)"
}

# Stage 1: every repo pin except torch/torchvision.
REQS_REPO="/tmp/reqs_repo.txt"
grep -viE '^(torch|torchvision|torchaudio)([=<>~ ]|$)' "${SOCCER_DIR}/environment.txt" > "${REQS_REPO}"
# grad-cam pulls opencv-python-headless, which shares the cv2 module; keep it on the same version.
echo "opencv-python-headless==4.10.0.84" >> "${REQS_REPO}"
# The SigLIP / SigLIP2 tokenizers need these; Colab's system Python ships them, the venv does not.
printf '%s\n' sentencepiece protobuf >> "${REQS_REPO}"
# Stage 2: only what chefer_baselines.py and the frame loader import.
REQS_NEW="/tmp/reqs_baselines.txt"
printf '%s\n' decord==0.6.0 einops imageio numpy opencv-python Pillow tqdm \
    transformers==5.22.2 timm==1.0.31 sentencepiece protobuf > "${REQS_NEW}"

# ---------------------------------------------------------------------------
# Stage 1: soccer models + SigLIP with the repo's own scripts
# ---------------------------------------------------------------------------
if wants siglip || wants matchvision || wants soccermaster; then
    echo "=== Stage 1: repo environment ==="
    make_venv "${VENV_REPO}" "${REQS_REPO}"
    PY="${VENV_REPO}/bin/python"
    cd "${SOCCER_DIR}/inference"

    if wants siglip && needs_run siglip; then
        "${PY}" chefer_siglip.py \
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
        "${PY}" chefer_matchvision_temporal.py \
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
        "${PY}" chefer_soccermaster_temporal.py \
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
    make_venv "${VENV_NEW}" "${REQS_NEW}"
    PY="${VENV_NEW}/bin/python"
    cd "${SOCCER_DIR}/inference"

    for model in clip peav videoprism; do
        if wants "${model}" && needs_run "${model}"; then
            "${PY}" chefer_baselines.py \
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
# Scoring needs only numpy + opencv; use whichever venv exists.
PY="${VENV_REPO}/bin/python"
[ -x "${PY}" ] || PY="${VENV_NEW}/bin/python"
[ -x "${PY}" ] || PY=python3
"${PY}" evaluate_grounding.py \
    --coco_json "${COCO_JSON}" \
    "${RUN_ARGS[@]}" \
    --resolution frame native \
    --canvas "${CANVAS}" \
    --out_dir "${RUN_DIR}/table1" \
    2>&1 | tee "${RUN_DIR}/table1_log.txt"

echo "Done. Table 1: ${RUN_DIR}/table1/table1_frame.md (LaTeX: table1_frame.tex)"
echo "Validation against the paper: ${RUN_DIR}/table1/validation_native.md"
