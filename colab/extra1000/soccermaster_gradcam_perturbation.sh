#!/bin/bash
# SoccerMaster Grad-CAM perturbation on the 1000 merged clips (Colab G4 / 96 GB GPU).
# For each chunk of 100 clips: Grad-CAM maps -> pool to the 32x32 token grid -> perturbation.
# Everything is resumable: finished chunks and finished clips are skipped on a re-run.
#
# Colab:
#   from google.colab import drive; drive.mount('/content/drive')
#   !REPO_URL=<repository URL> bash colab/extra1000/soccermaster_gradcam_perturbation.sh
# Optional: OUT_ROOT (results folder), CLIP_ROOT (folder with the 1000 clips).
set -uo pipefail
export HF_HOME=/content/huggingface_cache HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S=/content/SoccerExplainability
CLIPS=${CLIP_ROOT:-/content/drive/MyDrive/SoccerLens}
OUT=${OUT_ROOT:-/content/drive/MyDrive/SoccerExplainability-output/table1/run_1000_final}/perturbation_colab
PERT=${OUT}/soccermaster_gradcam          # per-clip results + summary
MAPS=${OUT}/soccermaster_gradcam_maps     # pooled 32x32 maps
LOCAL=/content/sm_gc; PY=/content/venv_repo/bin/python; CHUNK=100
SM_CKPT=/content/models/SoccerMaster
log(){ echo "[$(date +%T)] $*"; }

[ -d "${CLIPS}" ] || { echo "ERROR: ${CLIPS} not found. Mount Drive first."; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "ERROR: no GPU"; exit 1; }
mkdir -p "${PERT}" "${MAPS}" "${LOCAL}/done"

# ---------------------------------------------------------------- code + env
if [ ! -d ${S}/.git ]; then
    git clone -q "${REPO_URL:?set REPO_URL to the repository URL}" ${S} || exit 1
fi
cd ${S} && git log --oneline -1
if [ ! -x ${PY} ]; then
    log "building the Python environment (3 to 10 min)"
    command -v uv >/dev/null 2>&1 || pip install -q uv
    grep -viE '^(torch|torchvision|torchaudio)([=<>~ ]|$)' environment.txt > /tmp/reqs.txt
    printf '%s\n' opencv-python-headless==4.10.0.84 sentencepiece protobuf >> /tmp/reqs.txt
    uv venv -q --python 3.11 /content/venv_repo
    uv pip install -q --python ${PY} --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
        torch torchvision -r /tmp/reqs.txt || exit 1
fi
${PY} -c "import torch, transformers; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), '| transformers', transformers.__version__)" || exit 1
${PY} -c "from huggingface_hub import snapshot_download; snapshot_download('xleprime/SoccerMaster', local_dir='${SM_CKPT}')" || exit 1


# ---------------------------------------------------------------- chunks
N_CHUNKS=$(${PY} - <<EOF
import json
d = json.load(open('${S}/train_data/json/selected_videos_for_annotations_merged.json'))
n = (len(d) + ${CHUNK} - 1) // ${CHUNK}
for i in range(n):
    json.dump(d[i*${CHUNK}:(i+1)*${CHUNK}], open('${LOCAL}/chunk_%02d.json' % i, 'w'))
print(n)
EOF
)
log "${N_CHUNKS} chunks of ${CHUNK} clips"
SM_ARGS="--model soccermaster --label SoccerMaster-GradCAM --checkpoint_dir ${SM_CKPT} --siglip2_path google/siglip2-large-patch16-512 --input_size 512"
COMMON="--groups 10 --ranking clip --fill mean --random_seeds 1"

cd ${S}/inference
for i in $(seq 0 $((N_CHUNKS - 1))); do
    c=$(printf '%02d' $i); cfg=${S}/config/chunk_${c}.py
    sed "s#/content/SoccerExplainability/train_data/json/selected_videos_for_annotations_800.json#${LOCAL}/chunk_${c}.json#" \
        ${S}/config/pretrain_classification_800.py > ${cfg}
    if [ -f "${PERT}/chunk_${c}.done" ]; then log "chunk ${c}: already done, skipping"; continue; fi

    if [ ! -f "${MAPS}/chunk_${c}.maps_done" ]; then
        log "chunk ${c}: Grad-CAM maps"
        rm -rf ${LOCAL}/full_${c}
        ${PY} gradcam_inference.py --model_type soccermaster --soccermaster_checkpoint_dir ${SM_CKPT} \
            --config_path ${cfg} --coco_json ${S}/annotations-coco-merged.json \
            --saliency_save_dir ${LOCAL}/full_${c} --output_dir ${LOCAL}/viz > ${LOCAL}/gradcam_${c}.log 2>&1
        rc=$?
        if [ ${rc} -ne 0 ]; then log "chunk ${c}: Grad-CAM FAILED (rc=${rc})"; tail -n 15 ${LOCAL}/gradcam_${c}.log; exit 1; fi
        ${PY} ${S}/inference/pool_saliency_maps.py ${LOCAL}/full_${c} ${MAPS} 32 || exit 1
        rm -rf ${LOCAL}/full_${c} ${LOCAL}/viz
        touch "${MAPS}/chunk_${c}.maps_done"
    fi

    log "chunk ${c}: perturbation (about 9 s per clip)"
    ${PY} perturbation_faithfulness.py ${SM_ARGS} --config_path ${cfg} --saliency_dir ${MAPS} \
        --out_dir ${PERT} ${COMMON} 2>&1 | grep --line-buffered -E "SaCo=|scored|Error|error|Traceback" | cut -c1-200
    n=$(ls "${PERT}/clips" 2>/dev/null | wc -l)
    log "chunk ${c}: finished, ${n} clips scored in total"
    touch "${PERT}/chunk_${c}.done"
done

# ---------------------------------------------------------------- summary over all clips
sed "s#selected_videos_for_annotations_800.json#selected_videos_for_annotations_merged.json#" \
    ${S}/config/pretrain_classification_800.py > ${S}/config/pretrain_classification_1000.py
${PY} perturbation_faithfulness.py ${SM_ARGS} --config_path ${S}/config/pretrain_classification_1000.py \
    --saliency_dir ${MAPS} --out_dir ${PERT} ${COMMON} --summarize_only > ${LOCAL}/summary.log 2>&1
log "clips scored: $(ls "${PERT}/clips" | wc -l) of 1000"
cat "${PERT}/summary.md" 2>/dev/null | head -n 40
log "COLAB_SM_GRADCAM_DONE  ->  ${PERT}"
