#!/usr/bin/env bash
# Buoc 2: tao moi truong Python rieng (~/mcs-venv) va cai PyTorch + thu vien.
set -e
ENV_DIR=${ENV_DIR:-$HOME/mcs-venv}
PY="$ENV_DIR/bin/python"

if [ ! -x "$PY" ]; then
  if python3 -m venv "$ENV_DIR" 2>/dev/null; then echo ">> Tao venv: $ENV_DIR"
  elif command -v conda >/dev/null; then echo ">> Tao conda env: $ENV_DIR"; conda create -y -p "$ENV_DIR" python=3.11
  else echo "!! Khong tao duoc moi truong: thieu python3-venv va conda. Hoi quan ly may cai 1 trong 2."; exit 1; fi
fi
"$PY" -m pip install -q --upgrade pip

gpu_ok() { "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }

if ! gpu_ok; then
  if command -v nvidia-smi >/dev/null; then
    echo ">> GPU NVIDIA: cai PyTorch mac dinh (CUDA 12)"
    "$PY" -m pip install -q torch torchvision
    if ! gpu_ok; then
      echo ">> Driver cu, thu ban CUDA 11.8"
      "$PY" -m pip install -q --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118
    fi
  elif command -v rocm-smi >/dev/null; then
    IDX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}
    echo ">> GPU AMD: cai PyTorch ROCm tu $IDX (doi bang TORCH_INDEX=... neu sai phien ban)"
    "$PY" -m pip install -q torch torchvision --index-url "$IDX"
  else
    echo "!! Khong thay GPU. Van cai PyTorch nhung se KHONG train duoc."
    "$PY" -m pip install -q torch torchvision
  fi
fi

"$PY" -m pip install -q numpy tqdm scikit-learn wandb pillow kaggle
"$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "| GPU dung duoc:", torch.cuda.is_available(), "| so GPU:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory/2**30:.0f} GB")
PYEOF
echo ">> Xong. Tiep theo: dang nhap W&B:  $ENV_DIR/bin/wandb login"
