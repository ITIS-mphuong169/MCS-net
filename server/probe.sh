#!/usr/bin/env bash
# Phep thu "manh cat tu anh goc do phan giai cao co them thong tin phong cach khong" (~25 phut GPU).
#   bash server/probe.sh [GPU]
# Dung CLIP dong bang + hoi quy logistic (khong train backbone). In ra cuoi cung 2 dong VERDICT:
#   (a) nhanh chi tiet:  GO / NO-GO
#   (b) chon vi tri bang hoc (RL):  dang xay / khong dang
# GO khi "toan canh + chi tiet" hon CA "toan canh" LAN "toan canh + zoom 224 (khong co diem anh moi)" it nhat 1 diem,
# khoang tin cay 95% > 0.
set -e
GPU=${1:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV_DIR=${ENV_DIR:-$HOME/mcs-venv}
DATA_DIR=${DATA_DIR:-$(cat "$HOME/.mcs_data_dir" 2>/dev/null)}
[ -d "$DATA_DIR" ] || { echo "!! Chua co du lieu. Chay: bash server/get_data.sh"; exit 1; }
"$ENV_DIR/bin/python" -c "import timm" 2>/dev/null || { echo "!! Thieu timm. Chay: $ENV_DIR/bin/pip install timm transformers"; exit 1; }
mkdir -p "$HOME/results"; cd "$REPO"
env CUDA_VISIBLE_DEVICES=$GPU HIP_VISIBLE_DEVICES=$GPU DATA_DIR="$DATA_DIR" PROBE_OUT="$HOME/results/probe_detail" \
  PROBE_MAGS=${PROBE_MAGS:-2,3} PROBE_TRAIN_LIMIT=${PROBE_TRAIN_LIMIT:-20000} WORKERS=${WORKERS:-8} PROBE_BS=${PROBE_BS:-16} \
  "$ENV_DIR/bin/python" probe_detail.py 2>&1 | tee "$HOME/results/probe_detail.log"
echo ">> Ket qua: ~/results/probe_detail/probe_detail_results.json   log: ~/results/probe_detail.log"
