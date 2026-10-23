#!/usr/bin/env bash
# Danh gia tren tap test:  bash server/eval.sh <MODEL> <GPU>      (MODEL: base | mla64 | mha)
# Co the chay khi model van dang train (dung ban copy cua last.ckpt).
set -e
MODEL=${1:?Thieu MODEL}; GPU=${2:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV_DIR=${ENV_DIR:-$HOME/mcs-venv}
DATA_DIR=${DATA_DIR:-$(cat "$HOME/.mcs_data_dir")}
case $MODEL in
  base)  ARCH="PART_ATTN=0 MLA_RANK=0";  SUB=wikiart_fullpaper_baseline ;;
  mla64) ARCH="PART_ATTN=1 MLA_RANK=64"; SUB=wikiart_fullpaper_selfattn ;;
  mha)   ARCH="PART_ATTN=1 MLA_RANK=0";  SUB=wikiart_fullpaper_selfattn ;;
  *) echo "!! MODEL phai la base | mla64 | mha"; exit 1 ;;
esac
SRC="$HOME/runs/${SUB}_abl_srv_$MODEL/last.ckpt"
[ -f "$SRC" ] || { echo "!! Khong thay $SRC"; exit 1; }
EP=$("$ENV_DIR/bin/python" -c "import torch;print(torch.load('$SRC',weights_only=False,map_location='cpu')['logs']['epoch'])")
mkdir -p "$HOME/results"
COPY="$HOME/runs/eval_${MODEL}_ep$EP.ckpt"; cp "$SRC" "$COPY"
OUT="$HOME/results/${MODEL}_ep$EP"
cd "$REPO"
env CUDA_VISIBLE_DEVICES=$GPU HIP_VISIBLE_DEVICES=$GPU DATA_DIR="$DATA_DIR" BATCH_SIZE=${EVAL_BS:-16} INFER_TTA=0 GLOBAL_CLS=1 $ARCH \
  INFER_CKPT="$COPY" INFER_OUT="$OUT" "$ENV_DIR/bin/python" infer.py | tee "$OUT.log"
echo ">> Ket qua: $OUT/test_results.json   (predictions.npz de chay mcnemar.py)"
