#!/bin/bash
# Relevance-guided frame selection for the saliency maps of a finished Table 1
# run, on Google Colab: keep only the k most relevant frames (by the saved map)
# and check whether the full-clip prediction survives, against uniform, random
# and least-relevant k-frame subsets.
#
# Reads ${RUN_DIR}/<model>/saliency/*.npz written by colab/run_table1.sh and
# writes ${RUN_DIR}/<model>/frame_selection[_drop]/{clips/,summary.md,summary.json,curves.csv,curves.png}
# plus a combined ${RUN_DIR}/frame_selection[_drop]/summary_all.md and curves_all.png.
#
# Colab cells:
#
#   from google.colab import drive
#   drive.mount('/content/drive')
#   !RUN_DIR=/content/drive/MyDrive/SoccerExplainability-output/table1/run_XXXX \
#       bash /content/SoccerExplainability/colab/run_frame_selection.sh
#
# Environment variables:
#   RUN_DIR      (required) the Table 1 run directory
#   MODELS       subset of: siglip matchvision soccermaster clip peav videoprism
#   BUDGETS      frame budgets (default "1 3 5 10")
#   MODE         mask (default; unselected frames mean-filled, every model) |
#                drop (model sees only k frames; SigLIP and CLIP only)
#   FILL         mean | zero (default mean)
#   RANDOM_SEEDS random k-frame draws per clip and budget (default 1)
#   MAP_KEY      saliency_maps (default) | chefer_spatial_maps
#   RANK_FROM    optional model whose maps also rank the frames for every other
#                model ("transfer" condition), e.g. RANK_FROM=matchvision
#   MAX_CLIPS    debug: stop after this many clips
#
# Cost: 1 + |BUDGETS| * (3 + seeds [+1 with RANK_FROM]) forward passes per clip
# (17 by default, 21 with RANK_FROM), no backward pass.

set -euo pipefail

SOCCER_DIR="/content/SoccerExplainability"
MODELS="${MODELS:-siglip matchvision soccermaster clip peav videoprism}"
BUDGETS="${BUDGETS:-1 3 5 10}"
MODE="${MODE:-mask}"
FILL="${FILL:-mean}"
RANDOM_SEEDS="${RANDOM_SEEDS:-1}"
MAP_KEY="${MAP_KEY:-saliency_maps}"
RANK_FROM="${RANK_FROM:-}"
CONFIG="${SOCCER_DIR}/config/pretrain_classification.py"
export HF_HOME="/content/huggingface_cache"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -z "${RUN_DIR:-}" ] || [ ! -d "${RUN_DIR}" ]; then
    echo "ERROR: set RUN_DIR to a finished Table 1 run directory (has <model>/saliency/)." >&2
    exit 1
fi
if [ ! -d "${SOCCER_DIR}/inference" ]; then
    echo "ERROR: repo not found at ${SOCCER_DIR}." >&2
    exit 1
fi
nvidia-smi -L || { echo "ERROR: no GPU visible." >&2; exit 1; }

OUT_NAME="frame_selection"
[ "${MODE}" = "drop" ] && OUT_NAME="frame_selection_drop"
if [ "${MODE}" = "drop" ]; then
    # Only the mean-pooled frame models accept a variable number of frames.
    MODELS="$(for m in ${MODELS}; do case "$m" in siglip|clip) printf '%s ' "$m";; esac; done)"
    echo "MODE=drop: restricting to frame-independent models: ${MODELS}"
fi

wants() { [[ " ${MODELS} " == *" $1 "* ]]; }

declare -A ALL_LABELS=([siglip]=SigLIP [clip]=CLIP [peav]=PE-AV [videoprism]=VideoPrism
                       [matchvision]=MatchVision [soccermaster]=SoccerMaster)

# Same two venvs as run_table1.sh (built there; rebuilt here if missing).
PY_VERSION="3.11"
VENV_REPO="/content/venv_repo"
VENV_NEW="/content/venv_baselines"

make_venv() {
    local venv="$1" reqs="$2"; shift 2
    command -v uv >/dev/null 2>&1 || pip install -q uv
    if [ ! -x "${venv}/bin/python" ]; then
        uv venv --python "${PY_VERSION}" "${venv}"
    fi
    uv pip install --python "${venv}/bin/python" \
        --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        torch torchvision -r "${reqs}" "$@"
}

REQS_REPO="/tmp/reqs_repo.txt"
grep -viE '^(torch|torchvision|torchaudio)([=<>~ ]|$)' "${SOCCER_DIR}/environment.txt" > "${REQS_REPO}"
echo "opencv-python-headless==4.10.0.84" >> "${REQS_REPO}"
printf '%s\n' sentencepiece protobuf >> "${REQS_REPO}"
REQS_NEW="/tmp/reqs_baselines.txt"
printf '%s\n' decord==0.6.0 einops imageio numpy opencv-python Pillow tqdm matplotlib \
    transformers==5.22.2 timm==1.0.31 sentencepiece protobuf > "${REQS_NEW}"

# shellcheck disable=SC2206
COMMON=(--config_path "${CONFIG}" --budgets ${BUDGETS} --mode "${MODE}" --fill "${FILL}"
        --random_seeds "${RANDOM_SEEDS}" --map_key "${MAP_KEY}")
[ -n "${MAX_CLIPS:-}" ] && COMMON+=(--max_clips "${MAX_CLIPS}")

run_model() {
    local py="$1" model="$2"; shift 2
    local sal="${RUN_DIR}/${model}/saliency"
    if [ ! -d "${sal}" ] || [ -z "$(ls "${sal}"/*.npz 2>/dev/null)" ]; then
        echo "[${model}] no saliency maps under ${sal}, skipping."
        return
    fi
    local extra=()
    if [ -n "${RANK_FROM}" ] && [ "${RANK_FROM}" != "${model}" ]; then
        if [ -d "${RUN_DIR}/${RANK_FROM}/saliency" ]; then
            extra=(--rank_saliency_dir "${RUN_DIR}/${RANK_FROM}/saliency" --rank_label "${ALL_LABELS[$RANK_FROM]}")
        else
            echo "[${model}] RANK_FROM=${RANK_FROM} has no saliency dir, no transfer condition."
        fi
    fi
    "${py}" frame_selection.py --model "${model}" --saliency_dir "${sal}" \
        --out_dir "${RUN_DIR}/${model}/${OUT_NAME}" "${COMMON[@]}" "${extra[@]}" "$@" \
        2>&1 | tee -a "${RUN_DIR}/${model}/${OUT_NAME}_log.txt"
}

cd "${SOCCER_DIR}/inference"

if wants siglip || wants matchvision || wants soccermaster; then
    echo "=== Stage 1: repo environment ==="
    [ -x "${VENV_REPO}/bin/python" ] || make_venv "${VENV_REPO}" "${REQS_REPO}"
    PY="${VENV_REPO}/bin/python"
    if wants siglip; then
        run_model "${PY}" siglip --label SigLIP --model_name "google/siglip-base-patch16-224"
    fi
    if wants matchvision; then
        MV_CKPT="/content/pretrained_classification.pth"
        [ -f "${MV_CKPT}" ] || wget -q -O "${MV_CKPT}" \
            https://huggingface.co/Homie0609/UniSoccer/resolve/main/pretrained_classification.pth
        run_model "${PY}" matchvision --label MatchVision --checkpoint_path "${MV_CKPT}"
    fi
    if wants soccermaster; then
        SM_CKPT="/content/models/SoccerMaster"
        if [ -d "${SM_CKPT}" ] && grep -lqs "^version https://git-lfs" "${SM_CKPT}"/*.pt; then
            rm -rf "${SM_CKPT}"
        fi
        "${PY}" -c "from huggingface_hub import snapshot_download; snapshot_download('xleprime/SoccerMaster', local_dir='${SM_CKPT}')"
        run_model "${PY}" soccermaster --label SoccerMaster --checkpoint_dir "${SM_CKPT}" \
            --siglip2_path "google/siglip2-large-patch16-512" --input_size 512
    fi
fi

if wants clip || wants peav || wants videoprism; then
    echo "=== Stage 2: transformers 5.22.2 / timm 1.0.31 ==="
    [ -x "${VENV_NEW}/bin/python" ] || make_venv "${VENV_NEW}" "${REQS_NEW}"
    PY="${VENV_NEW}/bin/python"
    for model in clip peav videoprism; do
        wants "${model}" && run_model "${PY}" "${model}" --label "${ALL_LABELS[$model]}"
    done
fi

# Combined table and figure over every model that has a summary.
RUNS=()
for model in siglip clip peav videoprism matchvision soccermaster; do
    [ -f "${RUN_DIR}/${model}/${OUT_NAME}/summary.json" ] && RUNS+=("${ALL_LABELS[$model]}=${RUN_DIR}/${model}/${OUT_NAME}")
done
if [ "${#RUNS[@]}" -gt 0 ]; then
    PY="${VENV_REPO}/bin/python"; [ -x "${PY}" ] || PY="${VENV_NEW}/bin/python"; [ -x "${PY}" ] || PY=python3
    mkdir -p "${RUN_DIR}/${OUT_NAME}"
    "${PY}" frame_selection.py --out_dir "${RUN_DIR}/${OUT_NAME}" --summarize_runs "${RUNS[@]}"
    echo "Done. Combined table: ${RUN_DIR}/${OUT_NAME}/summary_all.md, figure: curves_all.png"
fi
