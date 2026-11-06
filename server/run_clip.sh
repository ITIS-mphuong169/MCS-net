#!/usr/bin/env bash
# Train CLIP ViT-B/16 (head neo chu + attention theo nhan) tren WikiArt, 100 epoch.
#   bash server/run_clip.sh <MODE> <GPU>
#   MODE: smoke  chay thu ~10 phut, phai in "CHAY THU OK"
#         single 1 view 224 (moc so sanh, cung cong thuc)
#         detail toan canh 224 + 2 manh chi tiet cat tu anh goc (kiem tra 9 manh), phong to MAG lan (MAG=2 mac dinh)
#         zoom   doi chung cua detail: manh cat tu chinh anh 224 phong to (cung vung nhin, KHONG co diem anh moi)
#   GPU : so thu tu GPU (0, 1, ...)
# Chay lai dung lenh nay sau khi bi dung -> tu train tiep tu epoch cuoi (last.ckpt).
set -e
MODE=${1:?Thieu MODE (smoke | single | detail | zoom)}; GPU=${2:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV_DIR=${ENV_DIR:-$HOME/mcs-venv}
DATA_DIR=${DATA_DIR:-$(cat "$HOME/.mcs_data_dir" 2>/dev/null)}
[ -d "$DATA_DIR" ] || { echo "!! Chua co du lieu. Chay: bash server/get_data.sh"; exit 1; }
"$ENV_DIR/bin/python" -c "import timm, transformers" 2>/dev/null || { echo "!! Thieu timm/transformers. Chay: $ENV_DIR/bin/pip install timm transformers"; exit 1; }

MAG=${MAG:-2}
case $MODE in
  single) VIEWS_ENV="VIEWS=single" ;;
  detail) VIEWS_ENV="VIEWS=detail DETAIL_MAG=$MAG DETAIL_K=2 DETAIL_K_EVAL=2 DETAIL_K_TEST=9" ;;
  zoom)   VIEWS_ENV="VIEWS=zoom DETAIL_MAG=$MAG DETAIL_K=2 DETAIL_K_EVAL=2 DETAIL_K_TEST=9" ;;
  smoke)  VIEWS_ENV="VIEWS=detail DETAIL_MAG=$MAG DETAIL_K=2 DETAIL_K_EVAL=2 DETAIL_K_TEST=9 TRAIN_FRACTION=0.02 WANDB_MODE=offline" ;;
  *) echo "!! MODE phai la smoke | single | detail | zoom"; exit 1 ;;
esac

# Batch hieu dung luon = 32 anh. Tu chon theo bo nho GPU (view nhieu thi moi buoc nang gap 3):
#   single: >= 12GB batch 32, nho hon batch 16 x tich luy 2
#   detail/zoom: >= 24GB batch 32, nho hon batch 16 x tich luy 2
# Ep tay bang BATCH_SIZE=, GRAD_ACCUM= (vd het bo nho: BATCH_SIZE=8 GRAD_ACCUM=4)
MEM=$(CUDA_VISIBLE_DEVICES=$GPU HIP_VISIBLE_DEVICES=$GPU "$ENV_DIR/bin/python" -c \
  "import torch;print(int(round(torch.cuda.get_device_properties(0).total_memory/2**30)))")
if [ "$MODE" = single ]; then LIM=12; else LIM=24; fi
if [ "$MEM" -ge "$LIM" ]; then D_BS=32; D_ACC=1; else D_BS=16; D_ACC=2; fi
BS=${BATCH_SIZE:-$D_BS}; ACC=${GRAD_ACCUM:-$D_ACC}
EPOCHS=$([ "$MODE" = smoke ] && echo 1 || echo "${EPOCHS:-100}")

mkdir -p "$HOME/runs"
LOG="$HOME/runs/srv_clip_$MODE.log"
# Cong thuc (giu nguyen giua cac MODE de so sanh duoc): AdamW, lr head 1e-3, lr backbone 1e-5, cosine, khoi dong 2 epoch,
# bfloat16, label smoothing 0.1, stochastic depth 0.1
RECIPE="NET=vit_base_patch16_clip_quickgelu_224.openai PLAIN_CLS=1 TEXT_INIT=1 LABEL_ATTN=${LABEL_ATTN:-1} \
DATASETS=wikiart LABEL_SPACE=union OPTIM=adamw BACKBONE_LR_MULT=0.01 WEIGHT_DECAY=0.05 SCHED=cosine WARMUP_EPOCHS=2 \
AMP=1 AMP_DTYPE=bf16 LABEL_SMOOTH=0.1 DROP_PATH=0.1"
CMD="cd '$REPO' && env CUDA_VISIBLE_DEVICES=$GPU HIP_VISIBLE_DEVICES=$GPU DATA_DIR='$DATA_DIR' SAVE_ROOT='$HOME/runs' \
BATCH_SIZE=$BS GRAD_ACCUM=$ACC WORKERS=${WORKERS:-8} EPOCHS=$EPOCHS SEED=${SEED:-1} $RECIPE $VIEWS_ENV \
ABLATION_TAG=srv_clip_$MODE WANDB_ENTITY=phuongmt-b22cn641-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng \
'$ENV_DIR/bin/python' train.py 2>&1 | tee -a '$LOG'"

echo ">> MODE=$MODE  GPU=$GPU (${MEM}GB)  batch=$BS x tich luy $ACC = $((BS*ACC)) anh  epochs=$EPOCHS  mag=x$MAG  data=$DATA_DIR"
if [ "$MODE" = smoke ]; then
  bash -c "$CMD"; rm -rf "$HOME"/runs/*srv_clip_smoke*; echo ">> CHAY THU OK"; exit 0
fi
if [ "$3" = fg ]; then bash -c "$CMD"; exit 0; fi   # chay truc tiep (dung boi run_clip_all.sh)
if command -v tmux >/dev/null; then
  tmux new -d -s "clip_$MODE" "$CMD"
  echo ">> Dang chay nen trong tmux 'clip_$MODE'. Xem: tmux attach -t clip_$MODE  (thoat: Ctrl+B roi D)"
elif command -v screen >/dev/null; then
  screen -dmS "clip_$MODE" bash -c "$CMD"
  echo ">> Dang chay nen trong screen 'clip_$MODE'. Xem: screen -r clip_$MODE  (thoat: Ctrl+A roi D)"
else
  nohup bash -c "$CMD" >/dev/null 2>&1 &
  echo ">> Dang chay nen (nohup). Xem log: tail -f $LOG"
fi
echo ">> Log: $LOG   W&B run ket thuc bang -abl-srv_clip_$MODE"
