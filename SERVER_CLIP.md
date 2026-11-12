# Chạy CLIP đa góc nhìn trên server Windows (không cần file .sh)

Mọi thứ chạy bằng một file Python: `server\clip.py`. Gõ lệnh trong **Command Prompt** (cmd), đứng trong thư mục `MCS-Net`. PowerShell cũng chạy, các lệnh giống hệt. Trên Linux và macOS cũng chạy y hệt.

**Mở Command Prompt đúng thư mục:** mở thư mục `MCS-Net` bằng File Explorer, bấm vào thanh địa chỉ, gõ `cmd`, nhấn Enter. Hoặc mở cmd rồi gõ `cd /d E:\MCS-Net` (chữ `/d` cần khi đổi sang ổ đĩa khác).

**Đường dẫn dữ liệu:** `chay_clip.bat` hỏi một lần rồi nhớ. Khi gõ tay, thêm `--data D:\wikiart` vào lệnh nếu máy chưa nhớ đường dẫn.

Hai run sẽ được chạy, cùng một công thức, mỗi run 100 epoch:

| Run | Mô hình thấy gì | Vai trò |
|---|---|---|
| `single` | 1 ảnh 224 | mốc so sánh |
| `detail` | ảnh 224 toàn cảnh + 2 mảnh 224 cắt từ ảnh gốc độ phân giải cao (lúc chấm test dùng 9 mảnh) | hướng chính |

Cả hai đều: CLIP ViT-B/16, lớp phân loại neo theo chữ, attention theo nhãn, AdamW, cosine, bfloat16.

## Cách nhanh nhất: bấm đúp một file

1. Mở Command Prompt trong thư mục `MCS-Net`, chạy một lần: `git pull origin exp/regularize`
2. Bấm đúp `server\chay_clip.bat`.

Lần đầu nó hỏi đường dẫn thư mục WikiArt (rồi nhớ lại) và API key W&B. Sau đó nó tự làm lần lượt: cài gói, kiểm tra máy, chạy thử 10 phút, train hai run 100 epoch, chấm test và McNemar. Cuối cùng nó ghi bảng tóm tắt vào `C:\Users\<ten>\results\summary_clip.txt`: gửi tôi file đó.

- **Bị dừng giữa chừng** (mất điện, tắt máy): bấm đúp lại, nó tự tiếp tục từ epoch cuối, bước chạy thử không lặp lại.
- **Báo lỗi**: cửa sổ giữ nguyên, chụp màn hình các dòng có `!!` gửi tôi.
- **Máy không ngủ** khi đang chạy: script tự giữ máy thức. Chỉ cần để cửa sổ mở.
- **Đổi đường dẫn WikiArt**: xoá file `C:\Users\<ten>\.mcs_data_dir` rồi bấm đúp lại.

Các mục dưới đây là cách làm từng bước bằng tay, dùng khi cần.

## 1. Chuẩn bị một lần

**1a. Lấy code mới nhất**
```bat
cd /d C:\duong\dan\toi\MCS-Net
git fetch origin
git checkout exp/regularize
git pull origin exp/regularize
```
Chưa có thư mục thì: `git clone -b exp/regularize https://github.com/ITIS-mphuong169/MCS-net.git MCS-Net`

**1b. Cài gói** vào đúng môi trường Python đã dùng để train MCS-Net trước đây:
```bat
python -m pip install --upgrade pip
python -m pip install timm transformers wandb scikit-learn tqdm pillow numpy
```
Nếu bước 1d báo GPU không dùng được (PyTorch bản CPU), cài bản GPU:
```bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Cần driver NVIDIA đủ mới. Nếu báo lỗi CUDA, cập nhật driver từ trang NVIDIA rồi chạy lại.

**1c. Đăng nhập W&B** (một lần): `wandb login`, dán API key.

**1d. Kiểm tra** (`--data` là thư mục WikiArt có 27 thư mục phong cách như `Impressionism`, `Baroque`; nếu nằm lồng thêm một tầng, script tự tìm):
```bat
python server\clip.py check --data D:\wikiart
```
Phải thấy: tên card và 12 GB, mọi gói có số phiên bản, dòng "Du lieu" trỏ đúng thư mục, W&B đã đăng nhập. Dòng nào có `!!` thì sửa theo gợi ý ngay trên dòng đó.

**1e. Không cho máy ngủ** (chạy một lần, mở Command Prompt bằng Run as administrator):
```bat
powercfg /change standby-timeout-ac 0
```

## 2. Chạy thử khoảng 10 phút
```bat
python server\clip.py train smoke --data D:\wikiart
```
Phải kết thúc bằng `>> CHAY THU OK`. Lần đầu nó tải trọng số CLIP nên cần internet.
Hết bộ nhớ GPU thì giảm batch nhỏ: run `detail` đã mặc định `--batch-size 8 --grad-accum 4` trên card 12 GB; nếu vẫn hết thì `--batch-size 4 --grad-accum 8`. Tích hai số luôn là 32. Dùng đúng hai tham số đó ở bước 3.
Máy treo hoặc hết RAM (mỗi tiến trình đọc ảnh trên Windows tốn khoảng 1 GB RAM) thì thêm `--workers 4`.

## 3. Chạy thật: hai run, một lệnh
```bat
python server\clip.py train both --data D:\wikiart
```
Nó chạy `single` rồi `detail`, mỗi run 100 epoch, không cần làm gì giữa chừng. Giữ cửa sổ mở và không đăng xuất.
Bị dừng giữa chừng (mất điện, tắt máy): chạy lại **đúng lệnh cũ**, nó tự tiếp tục từ epoch cuối, run nào xong rồi thì bỏ qua.

Theo dõi trên W&B, dự án `mcs-net-wikiart`, run kết thúc bằng `-abl-srv_clip_single` và `-abl-srv_clip_detail`:
- `val/top1` phải tăng rồi đi ngang; `train/raw_top1` cao hơn nhiều so với `val/top1` nghĩa là quá khớp.
- `train/view_w_global` chỉ có ở run `detail`: trọng số của ảnh toàn cảnh, bắt đầu 0.5. Nếu nó chạy về gần 1 thì mô hình bỏ qua các mảnh chi tiết.
- `train/attn_gate`: cổng của attention theo nhãn, bắt đầu 0. Gần 0 suốt run nghĩa là nhánh đó không đóng góp.

Thời gian chưa đo trên card của bạn. Đọc thời gian epoch 1 của mỗi run rồi nhân với số epoch. Run `detail` nặng khoảng 3 lần run `single`.

## 4. Chấm test
```bat
python server\clip.py eval both --data D:\wikiart
```
Chấm cả epoch cuối và epoch có val tốt nhất của mỗi run, rồi tự chạy McNemar giữa `detail` và `single`. Gửi tôi các khối in ra:
- `=== center_crop ===` của `single` và của `detail` (ở `detail` đây là dự đoán đã gộp toàn cảnh + 9 mảnh).
- `=== global_view ===` và `=== detail_views ===` của `detail`: chỉ nhìn toàn cảnh, và chỉ nhìn 9 mảnh.
- Phần McNemar.

Kết quả lưu ở `C:\Users\<ten>\results`, các run ở `C:\Users\<ten>\runs`. Đổi bằng `--results` và `--save-root`.

## 5. Hai việc thêm sau khi `single` đã xong (không cần `detail`)

**5a. WiSE-FT, không train, khoảng 35 phút.** Trộn trọng số CLIP gốc với trọng số của run `single` theo tỉ lệ α (0, 0.1, ..., 1), chọn α trên tập val, chấm test, chạy McNemar với `single`.
```bat
python server\clip.py eval wise
```
Kết quả: bảng `alpha | val Acc | test Acc | test F1 | top5` ở cuối, dòng `<- chosen` là α được chọn. α = 1 là chính run `single`, α = 0 là CLIP zero-shot. Thư mục `C:\Users\<ten>\results\clip_wise`.

**5b. Nhãn mềm theo họa sĩ, một run 100 epoch, nặng bằng run `single`.** Giống `single` hoàn toàn, chỉ khác: 10% label smoothing không chia đều cho 26 lớp còn lại mà chia theo các phong cách khác của cùng họa sĩ trong tập train (tên họa sĩ lấy từ tên file `hoa-si_ten-tranh.jpg`). Lúc test không dùng gì ngoài ảnh.
```bat
python server\clip.py train soft
python server\clip.py eval soft
```
Ngay đầu run phải thấy dòng `wikiart: SOFT_LABELS=artist: N artists (...), M files without an artist name, P% of training images get an artist-informed target`. Nếu `M` lớn hoặc `P%` nhỏ, chụp dòng đó gửi tôi: tên file trên máy bạn không theo dạng `hoa-si_ten-tranh.jpg`. Script tự dừng khi quá nửa số file không có tên họa sĩ.

Mỗi lệnh `eval` chấm xong sẽ ghi lại `summary_clip.txt` gồm **mọi** run đã chấm và McNemar của từng run so với `single`.

## 6. Tùy chọn
- **Phép thử 25 phút trước khi chạy dài:** `python server\clip.py probe --data D:\wikiart`. Nó đo bằng CLIP đóng băng xem mảnh chi tiết có thêm thông tin không, in hai dòng `VERDICT` ở cuối. Nếu nó nói `GO, magnification x3` thì thêm `--mag 3` vào các lệnh `train` và `eval`.
- **ViT-L/14 thay cho ViT-B/16:** thêm `--arch l14`. Nặng khoảng 4 lần, batch tự giảm. Nên giảm epoch: `--epochs 30`.
- **Đối chứng zoom:** `python server\clip.py train zoom --data D:\wikiart`. Dùng cùng số view nhưng mảnh cắt từ chính ảnh 224 phóng to, nên chênh giữa `detail` và `zoom` là phần do điểm ảnh thật.
- **Chạy hoặc chấm từng run riêng:** `python server\clip.py train detail`, `python server\clip.py eval single`, `python server\clip.py eval detail`, `python server\clip.py eval soft`. Run đang dở thì chạy lại đúng lệnh cũ, nó tự tiếp tục.
- **Chỉ xem sẽ chạy gì:** thêm `--dry-run` vào bất kỳ lệnh nào.

## Lưu ý
- 100 epoch có thể dài hơn mức cần. Lần chạy CLIP 8 epoch trước đó đã đạt train 87.8 và test 74.3, tức đã quá khớp khá sớm. Vì vậy bước 4 chấm cả epoch cuối lẫn epoch val tốt nhất, và số chính của bài là epoch cuối. Muốn nhanh hơn thì dùng `--epochs 30` cho cả hai run; lịch cosine hạ lr theo số epoch nên đừng dừng giữa chừng một run 100 epoch rồi coi đó là kết quả.
- Mọi lệnh Python bên dưới đã được thử trên Linux với dữ liệu giả; phần Windows (cmd và cài đặt) mới chỉ được thử trên máy server của bạn với lệnh eval single, còn lại chưa được thử trên máy Windows nào.
