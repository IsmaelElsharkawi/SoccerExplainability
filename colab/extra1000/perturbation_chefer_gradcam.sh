#!/bin/bash
# Perturbation faithfulness on the 1000 merged clips: Chefer-T for MatchVision and SoccerMaster,
# and Grad-CAM for MatchVision. (SoccerMaster Grad-CAM: soccermaster_gradcam_perturbation.sh.)
# Consolidated form of the commands that produced the reported numbers; every step is resumable.
#
# Required:
#   MAPS      folder with matchvision/saliency and soccermaster/saliency (Chefer-T maps of the
#             1000 clips, written by colab/run_table1.sh with --saliency_save_dir)
#   OUT       results folder
# Optional: SOCCER_DIR (repo, default /content/SoccerExplainability), PY (python of the repo
# environment), MV_CKPT, SM_CKPT, CLIP_ROOT is taken from the config (video_base).
set -uo pipefail
S=${SOCCER_DIR:-/content/SoccerExplainability}; PY=${PY:-/content/venv_repo/bin/python}
MAPS=${MAPS:?set MAPS}; OUT=${OUT:?set OUT}
MV_CKPT=${MV_CKPT:-/content/pretrained_classification.pth}; SM_CKPT=${SM_CKPT:-/content/models/SoccerMaster}
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=${S}/config/pretrain_classification_1000.py
COMMON="--config_path ${CFG} --groups 10 --ranking clip --fill mean --random_seeds 1"
MV="--model matchvision --checkpoint_path ${MV_CKPT}"
SM="--model soccermaster --checkpoint_dir ${SM_CKPT} --siglip2_path google/siglip2-large-patch16-512 --input_size 512"
mkdir -p "${OUT}/logs"; cd ${S}/inference
log(){ echo "[$(date +%T)] $*"; }
pert(){ # name label saliency_dir model-args...
    local name=$1 label=$2 dir=$3; shift 3
    ${PY} perturbation_faithfulness.py "$@" --label "${label}" --saliency_dir "${dir}" --out_dir "${OUT}/${name}" ${COMMON} \
        >> "${OUT}/logs/${name}.log" 2>&1
    log "${name}: rc=$? clips=$(ls "${OUT}/${name}/clips" 2>/dev/null | wc -l)"
}

# SoccerMaster Chefer-T runs alongside the MatchVision chain (one 96 GB GPU is enough).
pert soccermaster_chefer "SoccerMaster Chefer-T" "${MAPS}/soccermaster/saliency" ${SM} &
P1=$!
pert matchvision_chefer "MatchVision Chefer-T" "${MAPS}/matchvision/saliency" ${MV}

# MatchVision Grad-CAM: maps at 224x224, pooled to the 14x14 token grid, then the same test.
if [ ! -f "${OUT}/matchvision_gradcam_maps.done" ]; then
    ${PY} gradcam_inference.py --model_type matchvision --checkpoint_path ${MV_CKPT} --config_path ${CFG} \
        --coco_json ${S}/annotations-coco-merged.json --saliency_save_dir "${OUT}/matchvision_gradcam_maps_full" \
        --output_dir /tmp/gradcam_viz > "${OUT}/logs/matchvision_gradcam_maps.log" 2>&1 \
        && ${PY} pool_saliency_maps.py "${OUT}/matchvision_gradcam_maps_full" "${OUT}/matchvision_gradcam_maps" 14 \
        && touch "${OUT}/matchvision_gradcam_maps.done"
fi
pert matchvision_gradcam "MatchVision Grad-CAM" "${OUT}/matchvision_gradcam_maps" ${MV}
wait ${P1}

RUNS=("MatchVision Chefer-T=${OUT}/matchvision_chefer" "MatchVision Grad-CAM=${OUT}/matchvision_gradcam" "SoccerMaster Chefer-T=${OUT}/soccermaster_chefer")
[ -f "${OUT}/soccermaster_gradcam/summary.json" ] && RUNS+=("SoccerMaster Grad-CAM=${OUT}/soccermaster_gradcam")
${PY} perturbation_faithfulness.py --out_dir "${OUT}/combined" --summarize_runs "${RUNS[@]}"
${PY} perturbation_subsets.py ${S}/train_data/json/selected_videos_for_annotations.json "${RUNS[@]}" | tee "${OUT}/combined/subsets.txt"
log "done -> ${OUT}/combined/summary_all.md"
