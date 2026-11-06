#!/usr/bin/env bash
# Server 1 GPU: chay lan luot single (moc so sanh) roi detail, chay nen.
#   bash server/run_clip_all.sh [GPU]
# single chay truoc de co duong cong val lam moc so sanh tren W&B trong luc detail dang chay.
# Bi dung giua chung: chay lai dung lenh nay -> run nao xong roi se bao "Nothing to do" va bo qua,
# run dang do se train tiep tu epoch cuoi.
GPU=${1:-0}
DIR=$(cd "$(dirname "$0")" && pwd)
CMD="bash '$DIR/run_clip.sh' single $GPU fg; bash '$DIR/run_clip.sh' detail $GPU fg"
if command -v tmux >/dev/null; then
  tmux new -d -s clip_all "$CMD"
  echo ">> Dang chay nen trong tmux 'clip_all'. Xem: tmux attach -t clip_all  (thoat: Ctrl+B roi D)"
elif command -v screen >/dev/null; then
  screen -dmS clip_all bash -c "$CMD"
  echo ">> Dang chay nen trong screen 'clip_all'. Xem: screen -r clip_all  (thoat: Ctrl+A roi D)"
else
  nohup bash -c "$CMD" >/dev/null 2>&1 &
  echo ">> Dang chay nen (nohup)."
fi
echo ">> Log: ~/runs/srv_clip_single.log  roi  ~/runs/srv_clip_detail.log"
