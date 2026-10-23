# Train MCS-Net trên server (Linux, GPU NVIDIA hoặc AMD)

Làm trực tiếp trên máy (mở Terminal). Mỗi bước tự kiểm tra; bước nào báo `!!` hoặc lỗi đỏ thì dừng lại.

```bash
# 0. Lấy code
cd ~ && git clone -b exp/regularize https://github.com/ITIS-mphuong169/MCS-net.git MCS-Net
cd ~/MCS-Net

# 1. Xem máy có gì (GPU, RAM, ổ đĩa, internet)
bash server/check.sh

# 2. Cài môi trường (~/mcs-venv) + PyTorch — phải in "GPU dung duoc: True"
bash server/setup.sh
~/mcs-venv/bin/wandb login            # dán API key W&B

# 3. Tải WikiArt (~30GB). Trước đó: kaggle.com -> Settings -> API -> Create New Token,
#    rồi: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
bash server/get_data.sh               # phải in "27 thu muc phong cach"

# 4. Chạy thử ~10–15 phút — phải in "CHAY THU OK"
bash server/run.sh smoke 0

# 5. Chạy thật (chạy nền, tắt terminal vẫn chạy)
#    Server 1 GPU: chạy lần lượt MCS-Net rồi MCS-Net + MLA
bash server/run_both.sh 0
#    Server 2 GPU: chạy song song
#    bash server/run.sh base 0 ; bash server/run.sh mla64 1

# 6. Đánh giá trên tập test (lúc nào cũng được, kể cả khi đang train)
bash server/eval.sh base 0
bash server/eval.sh mla64 0
~/mcs-venv/bin/python mcnemar.py ~/results/base_ep100/predictions.npz ~/results/mla64_ep100/predictions.npz
```

## Ghi chú
- **Batch hiệu dụng luôn là 32 như bài báo**, tự chọn theo bộ nhớ GPU:
  ≥32GB → batch 32; 16–31GB → 16 × tích lũy 2; <16GB (vd RTX 4070 Super 12GB) → 16 × tích lũy 2 + bfloat16.
- **Hết bộ nhớ (out of memory) ở bước 4**: thử
  `BATCH_SIZE=8 GRAD_ACCUM=4 AMP=1 bash server/run.sh smoke 0`. Nếu OK thì dùng **đúng các biến đó** cho bước 5:
  `BATCH_SIZE=8 GRAD_ACCUM=4 AMP=1 bash server/run_both.sh 0` (cả 2 model phải cùng cấu hình).
- **Bị dừng giữa chừng** (mất điện, khởi động lại): chạy lại đúng lệnh bước 5 → tự train tiếp từ epoch cuối.
- **Theo dõi**: `tmux attach -t srv_both` (thoát Ctrl+B rồi D), `tail -f ~/runs/srv_base.log`,
  hoặc W&B run `...-abl-srv_base`, `...-abl-srv_mla64`.
- GPU AMD mà setup.sh báo `GPU dung duoc: False`: chạy lại với đúng phiên bản ROCm, ví dụ
  `TORCH_INDEX=https://download.pytorch.org/whl/rocm6.3 bash server/setup.sh`.
- Server không có internet: chép thư mục `MCS-Net` và `wikiart` bằng USB,
  rồi `echo /duong/dan/wikiart > ~/.mcs_data_dir` thay cho bước 3 (bước 2 vẫn cần internet để pip install).
