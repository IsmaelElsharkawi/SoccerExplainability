#!/bin/bash
# Perturbation faithfulness (SaCo / deletion curves / flip rates) for the
# saliency maps of a finished Table 1 run, on Google Colab.
#
# Reads ${RUN_DIR}/<model>/saliency/*.npz written by colab/run_table1.sh and
# writes ${RUN_DIR}/<model>/perturbation/{clips/,summary.md,summary.json,curves.csv}
# plus a combined ${RUN_DIR}/perturbation/summary_all.md.
#
# Colab cells (after run_table1.sh has finished, same runtime or a new one):
#
#   from google.colab import drive
#   drive.mount('/content/drive')
#   !RUN_DIR=/content/drive/MyDrive/SoccerExplainability-output/table1/run_XXXX \
#       bash /content/SoccerExplainability/colab/run_perturbation.sh
#
# Environment variables:
#   RUN_DIR      (required) the Table 1 run directory
#   MODELS       subset of: siglip matchvision soccermaster clip peav videoprism
#   NUM_GROUPS   number of salience groups (default 10 = 10% bins)
#   RANKING      clip | frame (default clip)
#   FILL         mean | zero (default mean)
#   RANDOM_SEEDS random-map controls per clip (default 1)
#   MAP_KEY      saliency_maps (default) | chefer_spatial_maps
#   MAX_CLIPS    debug: stop after this many clips
#
# Cost: 2 + (3K-2)(1+seeds) forward passes per clip (58 for K=10, 1 seed), no
# backward pass. SoccerMaster (30 x 512^2, fp16) is the slowest model.

set -euo pipefail

SOCCER_DIR="/content/SoccerExplainability"
MODELS="${MODELS:-siglip matchvision soccermaster clip peav videoprism}"
NUM_GROUPS="${NUM_GROUPS:-10}"
RANKING="${RANKING:-clip}"
FILL="${FILL:-mean}"
RANDOM_SEEDS="${RANDOM_SEEDS:-1}"
MAP_KEY="${MAP_KEY:-saliency_maps}"
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

wants() { [[ " ${MODELS} " == *" $1 "* ]]; }

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
printf '%s\n' decord==0.6.0 einops imageio numpy opencv-python Pillow tqdm \
    transformers==5.22.2 timm==1.0.31 sentencepiece protobuf > "${REQS_NEW}"

COMMON=(--config_path "${CONFIG}" --groups "${NUM_GROUPS}" --ranking "${RANKING}" --fill "${FILL}"
        --random_seeds "${RANDOM_SEEDS}" --map_key "${MAP_KEY}")
[ -n "${MAX_CLIPS:-}" ] && COMMON+=(--max_clips "${MAX_CLIPS}")

run_model() {
    local py="$1" model="$2"; shift 2
    local sal="${RUN_DIR}/${model}/saliency"
    if [ ! -d "${sal}" ] || [ -z "$(ls "${sal}"/*.npz 2>/dev/null)" ]; then
        echo "[${model}] no saliency maps under ${sal}, skipping."
        return
    fi
    "${py}" perturbation_faithfulness.py --model "${model}" --saliency_dir "${sal}" \
        --out_dir "${RUN_DIR}/${model}/perturbation" "${COMMON[@]}" "$@" \
        2>&1 | tee -a "${RUN_DIR}/${model}/perturbation_log.txt"
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
    declare -A LABELS=([clip]=CLIP [peav]=PE-AV [videoprism]=VideoPrism)
    for model in clip peav videoprism; do
        wants "${model}" && run_model "${PY}" "${model}" --label "${LABELS[$model]}"
    done
fi

# Combined table over every model that has a summary.
RUNS=()
declare -A ALL_LABELS=([siglip]=SigLIP [clip]=CLIP [peav]=PE-AV [videoprism]=VideoPrism
                       [matchvision]=MatchVision [soccermaster]=SoccerMaster)
for model in siglip clip peav videoprism matchvision soccermaster; do
    [ -f "${RUN_DIR}/${model}/perturbation/summary.json" ] && RUNS+=("${ALL_LABELS[$model]}=${RUN_DIR}/${model}/perturbation")
done
if [ "${#RUNS[@]}" -gt 0 ]; then
    PY="${VENV_REPO}/bin/python"; [ -x "${PY}" ] || PY="${VENV_NEW}/bin/python"; [ -x "${PY}" ] || PY=python3
    mkdir -p "${RUN_DIR}/perturbation"
    "${PY}" perturbation_faithfulness.py --out_dir "${RUN_DIR}/perturbation" --summarize_runs "${RUNS[@]}"
    echo "Done. Combined table: ${RUN_DIR}/perturbation/summary_all.md"
fi
