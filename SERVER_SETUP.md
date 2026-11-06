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

---

# CLIP ViT-B/16: hướng "toàn cảnh + chi tiết độ phân giải cao" (100 epoch)

Cần thêm gói: `bash server/setup.sh` (đã thêm `timm transformers`) hoặc `~/mcs-venv/bin/pip install timm transformers`.
Cần có `git pull` để lấy các file `server/*clip*.sh`, `server/probe.sh`.

```bash
cd ~/MCS-Net && git pull

# 1. Phép thử (~25 phút GPU). Cuối log có 2 dòng VERDICT.
bash server/probe.sh 0
#    VERDICT (a) GO, magnification x2 (hoặc x3)  -> làm tiếp bước 2.   NO-GO -> dừng, báo lại.

# 2. Chạy thử ~10 phút (phải in "CHAY THU OK"), rồi chạy thật: single (mốc so sánh) rồi detail, mỗi cái 100 epoch.
bash server/run_clip.sh smoke 0
bash server/run_clip_all.sh 0            # nếu VERDICT nói x3:  MAG=3 bash server/run_clip_all.sh 0

# 3. Đánh giá test (lúc nào cũng được, kể cả khi đang train). Mỗi lần chấm cả epoch cuối và epoch val tốt nhất.
bash server/eval_clip.sh single 0
bash server/eval_clip.sh detail 0        # nếu dùng MAG=3 lúc train thì thêm MAG=3 ở đây
~/mcs-venv/bin/python mcnemar.py ~/results/clip_single_last_ep100/predictions.npz ~/results/clip_detail_last_ep100/predictions.npz
```

- **Theo dõi**: `tmux attach -t clip_all`, `tail -f ~/runs/srv_clip_detail.log`, W&B run kết thúc bằng `-abl-srv_clip_single` và `-abl-srv_clip_detail`.
- **Hết bộ nhớ**: `BATCH_SIZE=8 GRAD_ACCUM=4 bash server/run_clip.sh smoke 0`, rồi dùng đúng biến đó cho `run_clip_all.sh`.
- **Bị dừng giữa chừng**: chạy lại đúng lệnh bước 2, tự train tiếp từ epoch cuối.
- **Đối chứng thêm (tùy chọn, ~cùng thời gian với detail)**: `bash server/run_clip.sh zoom 0` rồi `bash server/eval_clip.sh zoom 0`. Nó dùng cùng số view nhưng mảnh cắt từ ảnh 224 phóng to, nên chênh giữa detail và zoom là phần do điểm ảnh thật.
