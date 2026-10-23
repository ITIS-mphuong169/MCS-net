# Chạy MCS-Net trên server AMD (2× Radeon AI PRO R9700, ROCm)

Làm trực tiếp trên máy (mở Terminal). Mỗi bước có lệnh kiểm tra — sai ở đâu dừng ở đó.

## 0. Kiểm tra máy trước
```bash
rocm-smi                 # 2 GPU; xem GPU nào đang bận (VRAM%, GPU%)
free -h                  # RAM trống — mỗi run cần ~10GB
df -h ~                  # ổ đĩa trống — cần ~65GB (zip + giải nén WikiArt)
```
Nếu GPU đang có job của người khác hoặc RAM < 20GB trống → hỏi người quản lý trước khi chạy.

## 1. Môi trường Python + PyTorch ROCm
Cách nhanh nhất: dùng lại PyTorch mà job khác trên máy đang dùng (chắc chắn hợp với GPU này):
```bash
ps aux | grep -i python | grep -v grep        # lấy PID của job python đang chạy
ls -l /proc/<PID>/exe                          # -> đường dẫn python của môi trường đó
/duong/dan/python -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.device_count())"
```
Rồi tạo môi trường riêng cài **đúng phiên bản torch đó**:
```bash
python3 -m venv ~/mcs-venv
source ~/mcs-venv/bin/activate
pip install --upgrade pip
# thay X.Y.Z và rocmA.B bằng đúng phiên bản in ra ở trên, ví dụ torch==2.8.0, rocm6.4
pip install torch==X.Y.Z torchvision --index-url https://download.pytorch.org/whl/rocmA.B
pip install numpy tqdm scikit-learn wandb pillow kaggle
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```
Phải in `True 2`. Không có job nào để tham khảo → hỏi người quản lý máy phiên bản ROCm,
chọn wheel tương ứng tại https://pytorch.org/get-started/locally/ (Linux, Pip, ROCm).

## 2. Lấy code
```bash
cd ~
git clone -b exp/regularize https://github.com/ITIS-mphuong169/MCS-net.git MCS-Net
cd MCS-Net && git log --oneline -1          # phải là a63bba6 hoặc mới hơn
```

## 3. Tải WikiArt (cùng bộ Kaggle steubk/wikiart → cùng tập test như các run trước)
1. Trên trình duyệt: kaggle.com → avatar → Settings → API → **Create New Token** → tải `kaggle.json`.
2.
```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
mkdir -p ~/data && cd ~/data
kaggle datasets download -d steubk/wikiart           # ~30GB, lâu
unzip -q wikiart.zip -d wikiart && rm wikiart.zip
find ~/data/wikiart -maxdepth 1 -type d | wc -l       # phải ra 28 (27 phong cách + thư mục gốc)
ls ~/data/wikiart | head                              # Abstract_Expressionism, Action_painting, ...
```
Nếu các thư mục phong cách nằm sâu thêm một tầng (vd `~/data/wikiart/wikiart/...`), dùng
đường dẫn tầng đó làm `DATA_DIR`.

## 4. Đăng nhập W&B
```bash
source ~/mcs-venv/bin/activate
wandb login                  # dán API key W&B
```

## 5. Chạy thử nhanh (~10 phút) trước khi chạy thật
```bash
cd ~/MCS-Net && source ~/mcs-venv/bin/activate
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 DATA_DIR=~/data/wikiart SAVE_ROOT=~/runs \
BATCH_SIZE=32 WORKERS=6 PART_ATTN=1 MLA_RANK=64 ABLATION_TAG=smoke \
TRAIN_FRACTION=0.02 EPOCHS=1 WANDB_MODE=offline python train.py
```
Phải thấy `Train images: 1303` (2%), thanh tiến trình chạy hết, không lỗi bộ nhớ.
Xong thì xoá: `rm -rf ~/runs/*smoke*`.

## 6. Chạy thật — 2 model, mỗi GPU một model, trong tmux
```bash
export WANDB_ENTITY=phuongmt-b22cn641-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng
COMMON="DATA_DIR=$HOME/data/wikiart SAVE_ROOT=$HOME/runs BATCH_SIZE=32 WORKERS=6 EPOCHS=100 SEED=1 AUG_ROTATE=15 FEAT_NOISE=0.1"
mkdir -p ~/runs
```
GPU 0 — MCS-Net (baseline):
```bash
tmux new -s base
cd ~/MCS-Net && source ~/mcs-venv/bin/activate
env CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 $COMMON PART_ATTN=0 MLA_RANK=0 ABLATION_TAG=srv_base \
  WANDB_ENTITY=$WANDB_ENTITY python train.py 2>&1 | tee -a ~/runs/srv_base.log
```
Thoát ra để nó chạy tiếp: **Ctrl+B rồi D**.

GPU 1 — MCS-Net + MLA:
```bash
tmux new -s mla
cd ~/MCS-Net && source ~/mcs-venv/bin/activate
env CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 $COMMON PART_ATTN=1 MLA_RANK=64 ABLATION_TAG=srv_mla64 \
  WANDB_ENTITY=$WANDB_ENTITY python train.py 2>&1 | tee -a ~/runs/srv_mla64.log
```
(Biến `COMMON`, `WANDB_ENTITY` phải export lại trong mỗi cửa sổ tmux mới.)

Theo dõi: `tmux attach -t base` / `tmux attach -t mla`, `rocm-smi`, hoặc W&B run
`...-abl-srv_base` và `...-abl-srv_mla64`.

**Nếu máy tắt / bị dừng**: chạy lại đúng lệnh trên — train.py tự đọc `last.ckpt` trong
`~/runs/...` và chạy tiếp từ epoch cuối.

## 7. Đánh giá trên tập test (ở epoch 50 hoặc 100)
```bash
cd ~/MCS-Net && source ~/mcs-venv/bin/activate
env CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 DATA_DIR=$HOME/data/wikiart BATCH_SIZE=32 PART_ATTN=0 MLA_RANK=0 INFER_TTA=0 \
  INFER_CKPT=$HOME/runs/wikiart_fullpaper_baseline_abl_srv_base/last.ckpt INFER_OUT=$HOME/results/base python infer.py
env CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 DATA_DIR=$HOME/data/wikiart BATCH_SIZE=32 PART_ATTN=1 MLA_RANK=64 INFER_TTA=0 \
  INFER_CKPT=$HOME/runs/wikiart_fullpaper_selfattn_abl_srv_mla64/last.ckpt INFER_OUT=$HOME/results/mla64 python infer.py
python mcnemar.py ~/results/base/predictions.npz ~/results/mla64/predictions.npz
```
Đánh giá giữa chừng (khi đang train) thì copy `last.ckpt` ra file khác trước rồi trỏ INFER_CKPT vào bản copy.
