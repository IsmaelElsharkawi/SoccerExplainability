#!/bin/bash
# Two extra analyses on the 1000 merged clips (Colab G4). Independent of the perturbation runs:
# it can run in a second Colab notebook at the same time, or in the same one afterwards.
#
#   A. Grad-CAM sanity check (CPU, ~2 min): how many SoccerMaster Grad-CAM maps are constant/empty.
#   B. Frame selection (temporal faithfulness) for MatchVision and SoccerMaster with their Chefer-T
#      maps: keep the k most relevant frames (k = 1, 3, 5, 10 of 30) and compare with uniform,
#      random and least-relevant frames. 17 forward passes per clip, resumable.
#
# Colab:
#   from google.colab import drive; drive.mount('/content/drive')
#   !REPO_URL=<repository URL> bash colab/extra1000/frame_selection_and_gradcam_check.sh
# Optional: OUT_ROOT (results folder), CLIP_ROOT (folder with the 1000 clips).
# Needs ${OUT_ROOT}/saliency_1000_all_models.tar: a tar of maps/<model>/saliency/ from the Table 1 run.
set -uo pipefail
export HF_HOME=/content/huggingface_cache HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S=/content/SoccerExplainability
FINAL=${OUT_ROOT:-/content/drive/MyDrive/SoccerExplainability-output/table1/run_1000_final}
CLIPS=${CLIP_ROOT:-/content/drive/MyDrive/SoccerLens}
RES=${FINAL}/frame_selection_colab
L=/content/framesel; PY=/content/venv_repo/bin/python
SM_CKPT=/content/models/SoccerMaster; MV_CKPT=/content/pretrained_classification.pth
log(){ echo "[$(date +%T)] $*"; }

[ -d "${CLIPS}" ] || { echo "ERROR: ${CLIPS} not found. Mount Drive first."; exit 1; }
[ -f "${FINAL}/saliency_1000_all_models.tar" ] || { echo "ERROR: missing ${FINAL}/saliency_1000_all_models.tar"; exit 1; }
mkdir -p ${L} "${RES}"

# ---------------------------------------------------------------- A. Grad-CAM sanity check
GC="${FINAL}/perturbation_colab/soccermaster_gradcam_maps"
if [ -d "${GC}" ]; then
    log "A. checking the SoccerMaster Grad-CAM maps in ${GC}"
    python3 - "${GC}" "${FINAL}/perturbation_colab/soccermaster_gradcam" "${RES}/gradcam_map_check.json" <<'EOF'
import sys, os, glob, json, numpy as np
maps_dir, pert_dir, out = sys.argv[1:4]
rows = []
for f in sorted(glob.glob(os.path.join(maps_dir, '*.npz'))):
    m = np.load(f)['saliency_maps'].astype(np.float64)
    stem = os.path.basename(f)[:-4]
    saco = None
    p = os.path.join(pert_dir, 'clips', stem + '.json')
    if os.path.exists(p):
        try:
            d = json.load(open(p)); saco = (d.get('saliency') or {}).get('saco')
        except Exception:
            pass
    rows.append(dict(stem=stem, constant=bool(np.ptp(m) < 1e-8), all_zero=bool(np.abs(m).max() < 1e-8),
                     zero_frac=float((np.abs(m) < 1e-8).mean()), n_unique=int(np.unique(np.round(m, 8)).size),
                     constant_frames=int(sum(np.ptp(fr) < 1e-8 for fr in m)), frames=int(m.shape[0]), saco=saco))
n = len(rows)
const = sum(r['constant'] for r in rows); zero = sum(r['all_zero'] for r in rows)
print(f"maps: {n}")
print(f"constant maps (no ranking at all): {const} ({100*const/max(n,1):.1f}%)   all-zero: {zero}")
print(f"mean share of exactly-zero tokens per map: {100*np.mean([r['zero_frac'] for r in rows]):.1f}%")
print(f"maps with more than half of their tokens exactly zero: {sum(r['zero_frac'] > 0.5 for r in rows)}")
print(f"mean number of constant frames per clip: {np.mean([r['constant_frames'] for r in rows]):.1f} of {rows[0]['frames'] if rows else 0}")
with_saco = [r for r in rows if isinstance(r['saco'], (int, float))]
if with_saco:
    z = [r for r in with_saco if abs(r['saco']) < 1e-9]
    print(f"clips with SaCo exactly 0: {len(z)}; of these, constant maps: {sum(r['constant'] for r in z)}, "
          f"mean zero-token share: {100*np.mean([r['zero_frac'] for r in z]) if z else 0:.1f}%")
else:
    print("(per-clip SaCo key not found in the result files; map statistics above are still valid)")
json.dump(rows, open(out, 'w'))
print("saved", out)
EOF
else
    log "A. skipped: ${GC} not found"
fi

# ---------------------------------------------------------------- B. frame selection
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "ERROR: no GPU, part B needs one"; exit 1; }
[ -d ${S}/.git ] || git clone -q "${REPO_URL:?set REPO_URL to the repository URL}" ${S} || exit 1
cd ${S}
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
${PY} -c "import torch; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))" || exit 1
[ -f ${MV_CKPT} ] || wget -q -O ${MV_CKPT} https://huggingface.co/Homie0609/UniSoccer/resolve/main/pretrained_classification.pth || exit 1
${PY} -c "from huggingface_hub import snapshot_download; snapshot_download('xleprime/SoccerMaster', local_dir='${SM_CKPT}')" || exit 1
sed "s#selected_videos_for_annotations_800.json#selected_videos_for_annotations_merged.json#" \
    config/pretrain_classification_800.py > config/pretrain_classification_1000.py
CFG=${S}/config/pretrain_classification_1000.py
if [ ! -f ${L}/maps.ready ]; then
    log "unpacking the Chefer-T maps"
    tar -xf "${FINAL}/saliency_1000_all_models.tar" -C ${L} maps/matchvision maps/soccermaster || exit 1
    touch ${L}/maps.ready
fi

COMMON="--config_path ${CFG} --budgets 1 3 5 10 --mode mask --fill mean --random_seeds 1"
fs(){ # name label model-args...
    local name=$1 label=$2; shift 2
    cd ${S}/inference
    ${PY} frame_selection.py "$@" --label "${label}" --saliency_dir ${L}/maps/${name}/saliency \
        --out_dir "${RES}/${name}" ${COMMON} > ${L}/${name}.log 2>&1
    log "${name}: finished rc=$? with $(ls "${RES}/${name}/clips" 2>/dev/null | wc -l) of 1000 clips"
}
( while true; do sleep 300; echo "[$(date +%T)] progress: matchvision=$(ls "${RES}/matchvision/clips" 2>/dev/null | wc -l) soccermaster=$(ls "${RES}/soccermaster/clips" 2>/dev/null | wc -l)"; done ) &
MON=$!
log "B. frame selection: MatchVision and SoccerMaster in parallel"
fs soccermaster SoccerMaster --model soccermaster --checkpoint_dir ${SM_CKPT} --siglip2_path google/siglip2-large-patch16-512 --input_size 512 &
P1=$!
fs matchvision MatchVision --model matchvision --checkpoint_path ${MV_CKPT}
wait ${P1}; kill ${MON} 2>/dev/null
grep -lE "Traceback" ${L}/*.log 2>/dev/null | while read f; do echo "  ERROR in ${f}:"; tail -n 6 "${f}"; done

cd ${S}/inference
${PY} frame_selection.py --out_dir "${RES}/combined" \
    --summarize_runs "MatchVision=${RES}/matchvision" "SoccerMaster=${RES}/soccermaster" > ${L}/combined.log 2>&1 \
    || { echo "combined summary failed:"; tail -n 5 ${L}/combined.log; }
cp ${L}/*.log "${RES}/" 2>/dev/null
cat "${RES}/combined/summary_all.md" 2>/dev/null | head -n 80
log "COLAB_FRAME_SELECTION_DONE  ->  ${RES}"
