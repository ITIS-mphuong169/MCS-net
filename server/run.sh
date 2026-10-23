#!/usr/bin/env bash
# Chay train.  Cach dung:  bash server/run.sh <MODEL> <GPU>
#   MODEL: smoke (chay thu ~10 phut) | base (MCS-Net) | mla64 (MCS-Net + MLA) | mha (MCS-Net + self-attention)
#   GPU  : so thu tu GPU (0, 1, ...)
# Chay lai dung lenh nay sau khi bi dung -> tu train tiep tu epoch cuoi (last.ckpt).
set -e
MODEL=${1:?Thieu MODEL}; GPU=${2:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV_DIR=${ENV_DIR:-$HOME/mcs-venv}
DATA_DIR=${DATA_DIR:-$(cat "$HOME/.mcs_data_dir" 2>/dev/null)}
[ -d "$DATA_DIR" ] || { echo "!! Chua co du lieu. Chay: bash server/get_data.sh"; exit 1; }

case $MODEL in
  base)  ARCH="PART_ATTN=0 MLA_RANK=0" ;;
  mla64) ARCH="PART_ATTN=1 MLA_RANK=64" ;;
  mha)   ARCH="PART_ATTN=1 MLA_RANK=0" ;;
  smoke) ARCH="PART_ATTN=1 MLA_RANK=64 TRAIN_FRACTION=0.02 WANDB_MODE=offline" ;;
  *) echo "!! MODEL phai la smoke | base | mla64 | mha"; exit 1 ;;
esac

# Batch hieu dung luon = 32 (nhu bai bao), tu chon theo bo nho GPU:
#   >= 32GB : batch 32
#   16-31GB : batch 16 x tich luy 2
#   < 16GB  : batch 16 x tich luy 2 + bfloat16 (AMP) de vua bo nho
# Ep tay bang BATCH_SIZE=, GRAD_ACCUM=, AMP= (vd het bo nho: BATCH_SIZE=8 GRAD_ACCUM=4 AMP=1)
MEM=$(CUDA_VISIBLE_DEVICES=$GPU HIP_VISIBLE_DEVICES=$GPU "$ENV_DIR/bin/python" -c \
  "import torch;print(int(round(torch.cuda.get_device_properties(0).total_memory/2**30)))")
if   [ "$MEM" -ge 32 ]; then D_BS=32; D_ACC=1; D_AMP=0
elif [ "$MEM" -ge 16 ]; then D_BS=16; D_ACC=2; D_AMP=0
else                         D_BS=16; D_ACC=2; D_AMP=1; fi
BS=${BATCH_SIZE:-$D_BS}; ACC=${GRAD_ACCUM:-$D_ACC}; AMPV=${AMP:-$D_AMP}
EPOCHS=$([ "$MODEL" = smoke ] && echo 1 || echo "${EPOCHS:-100}")

mkdir -p "$HOME/runs"
LOG="$HOME/runs/srv_$MODEL.log"
CMD="cd '$REPO' && env CUDA_VISIBLE_DEVICES=$GPU HIP_VISIBLE_DEVICES=$GPU DATA_DIR='$DATA_DIR' SAVE_ROOT='$HOME/runs' \
BATCH_SIZE=$BS GRAD_ACCUM=$ACC AMP=$AMPV WORKERS=${WORKERS:-6} EPOCHS=$EPOCHS SEED=${SEED:-1} AUG_ROTATE=15 FEAT_NOISE=0.1 GLOBAL_CLS=1 $ARCH \
ABLATION_TAG=srv_$MODEL WANDB_ENTITY=phuongmt-b22cn641-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng \
'$ENV_DIR/bin/python' train.py 2>&1 | tee -a '$LOG'"

echo ">> MODEL=$MODEL  GPU=$GPU (${MEM}GB)  batch=$BS x tich luy $ACC = $((BS*ACC))  AMP=$AMPV  epochs=$EPOCHS  data=$DATA_DIR"
if [ "$MODEL" = smoke ]; then
  bash -c "$CMD"; rm -rf "$HOME"/runs/*srv_smoke*; echo ">> CHAY THU OK"; exit 0
fi
if [ "$3" = fg ]; then bash -c "$CMD"; exit 0; fi   # chay truc tiep (dung boi run_both.sh)
if command -v tmux >/dev/null; then
  tmux new -d -s "srv_$MODEL" "$CMD"
  echo ">> Dang chay nen trong tmux 'srv_$MODEL'. Xem: tmux attach -t srv_$MODEL  (thoat: Ctrl+B roi D)"
elif command -v screen >/dev/null; then
  screen -dmS "srv_$MODEL" bash -c "$CMD"
  echo ">> Dang chay nen trong screen 'srv_$MODEL'. Xem: screen -r srv_$MODEL  (thoat: Ctrl+A roi D)"
else
  nohup bash -c "$CMD" >/dev/null 2>&1 &
  echo ">> Dang chay nen (nohup). Xem log: tail -f $LOG"
fi
echo ">> Log: $LOG"
