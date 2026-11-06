#!/usr/bin/env bash
# Danh gia tren tap test:  bash server/eval_clip.sh <MODE> <GPU>      (MODE: single | detail | zoom)
# Danh gia CA HAI checkpoint: epoch cuoi (last) va epoch co val tot nhat (best). Chay duoc khi model van dang train.
# Bao cao so cua "last" lam so chinh (khong chon theo val), "best" la so phu.
set -e
MODE=${1:?Thieu MODE (single | detail | zoom)}; GPU=${2:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV_DIR=${ENV_DIR:-$HOME/mcs-venv}
DATA_DIR=${DATA_DIR:-$(cat "$HOME/.mcs_data_dir")}
MAG=${MAG:-2}
case $MODE in
  single) VIEWS_ENV="VIEWS=single" ;;
  detail) VIEWS_ENV="VIEWS=detail DETAIL_MAG=$MAG DETAIL_K_TEST=9" ;;
  zoom)   VIEWS_ENV="VIEWS=zoom DETAIL_MAG=$MAG DETAIL_K_TEST=9" ;;
  *) echo "!! MODE phai la single | detail | zoom"; exit 1 ;;
esac
SUB="$HOME/runs/wikiart_fullpaper_selfattn_abl_srv_clip_$MODE"
mkdir -p "$HOME/results"; cd "$REPO"
for WHICH in last best; do
  if [ "$WHICH" = last ]; then SRC="$SUB/last.ckpt"; else SRC="$SUB/model.ckpt"; fi
  [ -f "$SRC" ] || { echo "!! Khong thay $SRC"; continue; }
  EP=$("$ENV_DIR/bin/python" -c "import torch;print(torch.load('$SRC',weights_only=False,map_location='cpu')['logs']['epoch'])")
  COPY="$HOME/runs/eval_clip_${MODE}_${WHICH}_ep$EP.ckpt"; cp "$SRC" "$COPY"
  OUT="$HOME/results/clip_${MODE}_${WHICH}_ep$EP"
  echo ">> $MODE / $WHICH (epoch $EP)"
  env CUDA_VISIBLE_DEVICES=$GPU HIP_VISIBLE_DEVICES=$GPU DATA_DIR="$DATA_DIR" BATCH_SIZE=${EVAL_BS:-16} INFER_TTA=0 \
    NET=vit_base_patch16_clip_quickgelu_224.openai PLAIN_CLS=1 TEXT_INIT=1 LABEL_ATTN=${LABEL_ATTN:-1} \
    DATASETS=wikiart LABEL_SPACE=union $VIEWS_ENV WORKERS=${WORKERS:-8} \
    INFER_CKPT="$COPY" INFER_OUT="$OUT" "$ENV_DIR/bin/python" infer.py | tee "$OUT.log"
done
echo ">> Ket qua: ~/results/clip_${MODE}_last_ep*/test_results.json  (predictions.npz de chay mcnemar.py)"
echo ">> Dong 'center_crop' la du doan da gop (toan canh + 9 manh); 'global_view' la chi nhin toan canh, 'detail_views' chi nhin 9 manh"
