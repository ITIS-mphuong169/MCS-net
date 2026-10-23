#!/usr/bin/env bash
# Server 1 GPU: chay lan luot MCS-Net (base) roi MCS-Net + MLA (mla64), chay nen.
#   bash server/run_both.sh [GPU]
# Bi dung giua chung: chay lai dung lenh nay -> model nao xong roi se bao "Nothing to do"
# va bo qua, model dang do se train tiep tu epoch cuoi.
GPU=${1:-0}
DIR=$(cd "$(dirname "$0")" && pwd)
CMD="bash '$DIR/run.sh' base $GPU fg; bash '$DIR/run.sh' mla64 $GPU fg"
if command -v tmux >/dev/null; then
  tmux new -d -s srv_both "$CMD"
  echo ">> Dang chay nen trong tmux 'srv_both'. Xem: tmux attach -t srv_both  (thoat: Ctrl+B roi D)"
elif command -v screen >/dev/null; then
  screen -dmS srv_both bash -c "$CMD"
  echo ">> Dang chay nen trong screen 'srv_both'. Xem: screen -r srv_both  (thoat: Ctrl+A roi D)"
else
  nohup bash -c "$CMD" >/dev/null 2>&1 &
  echo ">> Dang chay nen (nohup)."
fi
echo ">> Log: ~/runs/srv_base.log  roi  ~/runs/srv_mla64.log"
