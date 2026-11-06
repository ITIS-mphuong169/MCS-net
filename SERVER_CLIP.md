# Chạy CLIP đa góc nhìn trên server Windows (không cần file .sh)

Mọi thứ chạy bằng một file Python: `server\clip.py`. Gõ lệnh trong **PowerShell**, đứng trong thư mục `MCS-Net`.
(Cũng chạy y hệt trên Linux và macOS.)

Hai run sẽ được chạy, cùng một công thức, mỗi run 100 epoch:

| Run | Mô hình thấy gì | Vai trò |
|---|---|---|
| `single` | 1 ảnh 224 | mốc so sánh |
| `detail` | ảnh 224 toàn cảnh + 2 mảnh 224 cắt từ ảnh gốc độ phân giải cao (lúc chấm test dùng 9 mảnh) | hướng chính |

Cả hai đều: CLIP ViT-B/16, lớp phân loại neo theo chữ, attention theo nhãn, AdamW, cosine, bfloat16.

## Cách nhanh nhất: bấm đúp một file

1. Mở PowerShell trong thư mục `MCS-Net`, chạy một lần: `git pull origin exp/regularize`
2. Bấm đúp `server\chay_clip.bat`.

Lần đầu nó hỏi đường dẫn thư mục WikiArt (rồi nhớ lại) và API key W&B. Sau đó nó tự làm lần lượt: cài gói, kiểm tra máy, chạy thử 10 phút, train hai run 100 epoch, chấm test và McNemar. Cuối cùng nó ghi bảng tóm tắt vào `C:\Users\<ten>\results\summary_clip.txt`: gửi tôi file đó.

- **Bị dừng giữa chừng** (mất điện, tắt máy): bấm đúp lại, nó tự tiếp tục từ epoch cuối, bước chạy thử không lặp lại.
- **Báo lỗi**: cửa sổ giữ nguyên, chụp màn hình các dòng có `!!` gửi tôi.
- **Máy không ngủ** khi đang chạy: script tự giữ máy thức. Chỉ cần để cửa sổ mở.
- **Đổi đường dẫn WikiArt**: xoá file `C:\Users\<ten>\.mcs_data_dir` rồi bấm đúp lại.

Các mục dưới đây là cách làm từng bước bằng tay, dùng khi cần.

## 1. Chuẩn bị một lần

**1a. Lấy code mới nhất**
```powershell
cd C:\duong\dan\toi\MCS-Net
git fetch origin
git checkout exp/regularize
git pull origin exp/regularize
```
Chưa có thư mục thì: `git clone -b exp/regularize https://github.com/ITIS-mphuong169/MCS-net.git MCS-Net`

**1b. Cài gói** vào đúng môi trường Python đã dùng để train MCS-Net trước đây:
```powershell
python -m pip install --upgrade pip
python -m pip install timm transformers wandb scikit-learn tqdm pillow numpy
```
Nếu bước 1d báo GPU không dùng được (PyTorch bản CPU), cài bản GPU:
```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Cần driver NVIDIA đủ mới. Nếu báo lỗi CUDA, cập nhật driver từ trang NVIDIA rồi chạy lại.

**1c. Đăng nhập W&B** (một lần): `wandb login`, dán API key.

**1d. Kiểm tra** (`--data` là thư mục WikiArt có 27 thư mục phong cách như `Impressionism`, `Baroque`; nếu nằm lồng thêm một tầng, script tự tìm):
```powershell
python server\clip.py check --data D:\wikiart
```
Phải thấy: tên card và 12 GB, mọi gói có số phiên bản, dòng "Du lieu" trỏ đúng thư mục, W&B đã đăng nhập. Dòng nào có `!!` thì sửa theo gợi ý ngay trên dòng đó.

**1e. Không cho máy ngủ** (chạy một lần, mở PowerShell bằng Run as administrator):
```powershell
powercfg /change standby-timeout-ac 0
```

## 2. Chạy thử khoảng 10 phút
```powershell
python server\clip.py train smoke --data D:\wikiart
```
Phải kết thúc bằng `>> CHAY THU OK`. Lần đầu nó tải trọng số CLIP nên cần internet.
Hết bộ nhớ GPU thì thêm `--batch-size 8 --grad-accum 4`, rồi dùng đúng hai tham số đó ở bước 3.
Máy treo hoặc hết RAM (mỗi tiến trình đọc ảnh trên Windows tốn khoảng 1 GB RAM) thì thêm `--workers 4`.

## 3. Chạy thật: hai run, một lệnh
```powershell
python server\clip.py train both --data D:\wikiart
```
Nó chạy `single` rồi `detail`, mỗi run 100 epoch, không cần làm gì giữa chừng. Giữ cửa sổ PowerShell mở và không đăng xuất.
Bị dừng giữa chừng (mất điện, tắt máy): chạy lại **đúng lệnh cũ**, nó tự tiếp tục từ epoch cuối, run nào xong rồi thì bỏ qua.

Theo dõi trên W&B, dự án `mcs-net-wikiart`, run kết thúc bằng `-abl-srv_clip_single` và `-abl-srv_clip_detail`:
- `val/top1` phải tăng rồi đi ngang; `train/raw_top1` cao hơn nhiều so với `val/top1` nghĩa là quá khớp.
- `train/view_w_global` chỉ có ở run `detail`: trọng số của ảnh toàn cảnh, bắt đầu 0.5. Nếu nó chạy về gần 1 thì mô hình bỏ qua các mảnh chi tiết.
- `train/attn_gate`: cổng của attention theo nhãn, bắt đầu 0. Gần 0 suốt run nghĩa là nhánh đó không đóng góp.

Thời gian chưa đo trên card của bạn. Đọc thời gian epoch 1 của mỗi run rồi nhân với số epoch. Run `detail` nặng khoảng 3 lần run `single`.

## 4. Chấm test
```powershell
python server\clip.py eval both --data D:\wikiart
```
Chấm cả epoch cuối và epoch có val tốt nhất của mỗi run, rồi tự chạy McNemar giữa `detail` và `single`. Gửi tôi các khối in ra:
- `=== center_crop ===` của `single` và của `detail` (ở `detail` đây là dự đoán đã gộp toàn cảnh + 9 mảnh).
- `=== global_view ===` và `=== detail_views ===` của `detail`: chỉ nhìn toàn cảnh, và chỉ nhìn 9 mảnh.
- Phần McNemar.

Kết quả lưu ở `C:\Users\<ten>\results`, các run ở `C:\Users\<ten>\runs`. Đổi bằng `--results` và `--save-root`.

## 5. Tùy chọn
- **Phép thử 25 phút trước khi chạy dài:** `python server\clip.py probe --data D:\wikiart`. Nó đo bằng CLIP đóng băng xem mảnh chi tiết có thêm thông tin không, in hai dòng `VERDICT` ở cuối. Nếu nó nói `GO, magnification x3` thì thêm `--mag 3` vào các lệnh `train` và `eval`.
- **ViT-L/14 thay cho ViT-B/16:** thêm `--arch l14`. Nặng khoảng 4 lần, batch tự giảm. Nên giảm epoch: `--epochs 30`.
- **Đối chứng zoom:** `python server\clip.py train zoom --data D:\wikiart`. Dùng cùng số view nhưng mảnh cắt từ chính ảnh 224 phóng to, nên chênh giữa `detail` và `zoom` là phần do điểm ảnh thật.
- **Chỉ xem sẽ chạy gì:** thêm `--dry-run` vào bất kỳ lệnh nào.

## Lưu ý
- 100 epoch có thể dài hơn mức cần. Lần chạy CLIP 8 epoch trước đó đã đạt train 87.8 và test 74.3, tức đã quá khớp khá sớm. Vì vậy bước 4 chấm cả epoch cuối lẫn epoch val tốt nhất, và số chính của bài là epoch cuối. Muốn nhanh hơn thì dùng `--epochs 30` cho cả hai run; lịch cosine hạ lr theo số epoch nên đừng dừng giữa chừng một run 100 epoch rồi coi đó là kết quả.
- Mọi lệnh Python bên dưới đã được thử trên Linux với dữ liệu giả; phần PowerShell và cài đặt Windows chưa được thử trên máy Windows nào.
