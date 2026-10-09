# Báo cáo tiến độ — So sánh các cơ chế "chú ý" (attention) để học mối quan hệ giữa các bộ phận ảnh

## Mục tiêu đề tài

Tái tạo lại mô hình trong bài báo khoa học MCS-Net, dùng để phân loại phong cách tranh vẽ (27 loại phong cách khác nhau, ví dụ: Ấn tượng, Lập thể, Baroque...) trên bộ dữ liệu WikiArt. Sau khi tái tạo được mô hình gốc, nhóm đang mở rộng thêm một phần mới: mô hình gốc tách mỗi bức tranh thành 32 "vùng đặc trưng" (gọi tắt là 32 phần, mỗi phần đại diện cho một khía cạnh của bức tranh như nét cọ, màu sắc, bố cục...), nhưng 32 phần này hiện tại chỉ đang được **ghép nối đơn giản lại với nhau**, không có bất kỳ sự tương tác hay liên hệ nào giữa chúng. Nhóm đang thử nghiệm thêm một module để cho 32 phần này "giao tiếp" với nhau trước khi đưa ra quyết định phân loại cuối cùng, nhằm cải thiện độ chính xác.

## Phương pháp thử nghiệm

Tất cả các phiên bản thử nghiệm đều được tạo ra từ **cùng một phiên bản gốc đã chạy ổn định**. Mỗi lần thử, nhóm **chỉ thay đổi duy nhất một yếu tố** — là cách 32 phần giao tiếp với nhau — còn lại mọi thứ khác (mạng trích xuất đặc trưng ảnh, công thức tính lỗi, số lượng ảnh xử lý mỗi lần...) đều được giữ nguyên không đổi, để đảm bảo khi so sánh kết quả, sự khác biệt chỉ đến từ đúng một yếu tố đang thử nghiệm.

## Thông số huấn luyện cố định (giống nhau ở TẤT CẢ các run, đã xác nhận qua code)

| Thông số | Giá trị |
|---|---|
| Optimizer | SGD |
| Learning rate (lr) | 1e-3 (0.001) |
| Momentum | 0.9 |
| Weight decay | 1e-5 |
| Kích thước ảnh đầu vào (image_size) | 224 × 224 |
| Số attention map (M) | 32 |
| Số worker tải dữ liệu | 4 |
| Seed (chia tập + khởi tạo) | 1231 |

Các thông số này **không đổi giữa các run** — chỉ batch_size, các trọng số loss (λ1/λ2/λ_rel/τ), và module quan hệ-giữa-phần là khác nhau giữa các lần thử nghiệm (xem bảng chi tiết bên dưới).

## Chi tiết kiến trúc từng module (không được wandb tự log, đọc trực tiếp từ code)

Đây là các tham số cố định trong code (hardcode, không truyền qua config.py nên không hiện trong wandb Config panel):

| Thông số | Giá trị | Áp dụng cho |
|---|---|---|
| BAP pooling type | GAP (Global Average Pooling) | Tất cả các run |
| part_dim (số chiều sau khi nén mỗi part) | 256 | Mọi run có module quan hệ-giữa-phần (self-attention/ISAB/GAT/CLS-token) |
| nhead / num_heads (số đầu attention) | 8 | Self-attention, ISAB, GAT |
| dim_feedforward (lớp FFN trong block attention) | 512 (= part_dim × 2) | Self-attention, GAT |
| num_layers (số lớp self-attention xếp chồng) | 1 | Self-attention, CLS-token |
| num_inds (số inducing point của ISAB) | 16 | ISAB |
| CLS-token khởi tạo (trunc_normal std) | 0.02 | CLS-token |
| AGM: λ_c (trọng số tương quan kênh) | 0.5 | Các run có AGM (rcal-v2, full-paper-baseline, full-paper-self-attention) |
| AGM: λ_s (trọng số tương quan không gian) | 0.5 | Các run có AGM |

**Lưu ý quan trọng:** AGM có **2 trọng số λ_c/λ_s riêng (0.5/0.5)** — khác hoàn toàn với λ1/λ2 của công thức loss tổng (0.8/0.6). Đừng nhầm lẫn 2 cặp số này khi viết vào báo cáo/paper.

## Số liệu dữ liệu

- Bộ dữ liệu WikiArt (Kaggle, `steubk/wikiart`): ước lượng khoảng **81.440 ảnh** (27 lớp phong cách).
- Chia tập: **80% train (~65.150 ảnh) / 10% validation (~8.145 ảnh) / 10% test (~8.145 ảnh)**.
- Cách chia: xáo trộn toàn bộ ảnh với seed cố định (1231), không chia đều theo từng lớp (không stratified).

## Kết quả đã có đầy đủ, đáng tin cậy

**Self-attention (cơ chế "tự chú ý", chia thành 8 "góc nhìn" độc lập):**
Đã huấn luyện xong hoàn chỉnh (chạy liên tục 4 ngày). Kết quả: **val/top1 = 60.02%, val/top5 = 94.29%**. Đây là kết quả tốt nhất hiện có, và là cơ chế được chọn làm đóng góp chính thức.

**ISAB (biến thể attention lấy ý tưởng từ "Set Transformer"):**
Đã huấn luyện xong, kết quả **kém hơn rõ rệt** so với self-attention. Nguyên nhân: ISAB nén 32 phần xuống chỉ còn 16 điểm đại diện trước khi xử lý — việc nén này không cần thiết vì số lượng phần ban đầu (32) đã đủ nhỏ, chỉ làm mất thông tin.

## Kết quả sơ bộ, chưa đủ dữ liệu kết luận

**GAT (Graph Attention Network):** Mới huấn luyện được một phần, có xu hướng bám sát self-attention hơn ISAB, nhưng chưa đủ epoch để kết luận chắc chắn.

**CLS-token readout (kiểu BERT/ViT):** Đã viết xong code, kiểm tra chạy thử không lỗi, nhưng **chưa huấn luyện được epoch nào** vì hết hạn mức GPU.

## Khoảng trống phương pháp luận đã phát hiện và xử lý

Ban đầu, tất cả các thử nghiệm so sánh attention (self-attention/ISAB/GAT/CLS-token) đều dùng phiên bản **đơn giản hoá** của mô hình — chưa có đủ 3 module chính của bài báo gốc (AGM, SCLM, CCAM). Để kết quả so sánh với bài báo công bố có ý nghĩa, nhóm đã tạo thêm 2 nhánh mới:

1. **`exp/full-paper-baseline`** — đầy đủ AGM+SCLM+CCAM của bài báo, **không thêm** module quan hệ-giữa-phần nào (đúng nguyên bản bài báo).
2. **`exp/full-paper-self-attention`** — đầy đủ AGM+SCLM+CCAM **cộng thêm** self-attention cho 32 phần (đóng góp của nhóm).

Xác nhận: module self-attention cho quan hệ-giữa-phần **không trùng** với bất kỳ module nào trong bài báo (AGM tinh chỉnh feature map *trước khi* tạo ra 32 vùng; SCLM/CCAM không mô hình hoá quan hệ *giữa* các vùng đã tìm được) — đây là khoảng trống thật trong bài báo gốc mà nhóm đang bổ sung.

Cả 2 nhánh mới: đã giảm mục tiêu từ 100 xuống **30 epoch** (do giới hạn GPU miễn phí), **chưa chạy lần nào**.

## Bảng toàn bộ các lần chạy (12 run)

| Run (tên wandb) | Backbone | lr | Batch | AGM | SCLM | CCAM | Module quan hệ-giữa-phần | RCAL | λ1/λ2/λ_rel/τ | Epoch mục tiêu | Kết quả / trạng thái |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `wikiart-resnet101-v2` | resnet101 | 1e-3 | 32 | ❌ | ❌ | ❌ | Không có (ghép thẳng) | ❌ | — | 100 | val/top1 tăng vọt ~60% sớm rồi tụt còn ~40%, nghi overfit sớm (ước lượng từ biểu đồ) |
| `wikiart-swin_tiny_patch4_window7_224-v2` | Swin-T | 1e-3 | 32 | ❌ | ❌ | ❌ | Không có | ❌ | — | 100 | val/top5 tăng đều, chưa có số top1 chính xác |
| `wikiart-parttransformer-resnet101-v2` | resnet101 | 1e-3 | 32 | ❌ | ❌ | ❌ | **Self-attention** | ❌ | — | 100 | **val/top1=60.02%, val/top5=94.29%** — tốt nhất (số chính xác) |
| `wikiart-parttransformer-resnet101-v3` | resnet101 | 1e-3 | 32 | ❌ | ❌ | ❌ | **ISAB** | ❌ | — | 100 | Trùng code với isab-fair, hội tụ chậm |
| `wikiart-rcal-resnet101-v2` | resnet101 | 1e-3 | 32 | ❌ | ❌ | ❌ | ISAB | ✅ (λ_rel=0.5) | loss gốc WSDAN | 100 | Thấp (~13-16%, ước lượng) |
| `wikiart-rcal-v2-resnet101-v2` | resnet101 | 1e-3 | 16 | ✅ | ✅ | ✅ | ISAB | ✅ (λ_rel=0.5) | 0.8/0.6/0.5/0.07 | 100 | Thấp, tương tự trên |
| `wikiart-isab-rcal-resnet101-v2` | resnet101 | 1e-3 | 32 | ❌ | ❌ | ❌ | ISAB | ✅ (λ_rel=0.5) | λ2=0.6 | 100 | Thấp (~13-16%) |
| `wikiart-isab-fair-resnet101-v2` | resnet101 | 1e-3 | 32 | ❌ | ❌ | ❌ | **ISAB** | ❌ | — | 100 | Hội tụ chậm hơn self-attention rõ rệt |
| `wikiart-gat-fair-resnet101-v2` | resnet101 | 1e-3 | 32 | ❌ | ❌ | ❌ | **GAT** | ❌ | — | 100 | Bám sát self-attention hơn ISAB, chưa đủ epoch |
| `wikiart-cls-token-resnet101-v2` | resnet101 | 1e-3 | 32 | ❌ | ❌ | ❌ | Self-attention + CLS token | ❌ | — | 100 | **Chưa train epoch nào** |
| `wikiart-fullpaper-baseline-resnet101-v2` | resnet101 | 1e-3 | 16 | ✅ | ✅ | ✅ | **Không có** (đúng bài báo gốc) | ❌ | 0.8/0.6/—/0.07 | 30 | **Chưa chạy** |
| `wikiart-fullpaper-selfattn-resnet101-v2` | resnet101 | 1e-3 | 16 | ✅ | ✅ | ✅ | **Self-attention** | ❌ | 0.8/0.6/—/0.07 | 30 | **Chưa chạy** |

*Ghi chú: các con số "ước lượng từ biểu đồ" là đọc dáng đường trên wandb, không phải số chính xác. Chỉ số của `parttransformer-v2` là chính xác vì đã đọc trực tiếp từ trang Overview/Summary.*

## Vướng mắc hiện tại

Quota GPU Kaggle (30h/tuần) đã dùng hết do chạy nhiều thí nghiệm song song — reset sau vài ngày. Đã cân nhắc TPU (quota riêng) và CPU nhưng đều không khả thi: code hiện dùng thẳng CUDA nên TPU cần viết lại gần như toàn bộ `train.py` bằng `torch_xla`, còn CPU thì ResNet101 sẽ chậm gấp 20-50 lần.

## Kế hoạch tiếp theo

1. Chạy `exp/full-paper-baseline` và `exp/full-paper-self-attention` (30 epoch mỗi nhánh) khi quota GPU reset — đây là 2 kết quả quan trọng nhất còn thiếu, cho phép so sánh trực tiếp: bài báo gốc vs. bài báo + đóng góp mới.
2. Kết luận tạm thời: self-attention (Multi-Head Attention) là lựa chọn phù hợp nhất để mô hình hoá quan hệ giữa 32 phần trừu tượng — do dữ liệu không có cấu trúc đồ thị thưa thật sự (loại GAT khỏi lợi thế chính) và không cần nén (loại ISAB khỏi lợi thế chính).
3. Hướng dự phòng nếu cần thêm: Relation Network (MLP cặp-đôi), Capsule Network (routing-by-agreement, có cơ sở lý thuyết tốt cho mô hình hoá quan hệ bộ-phận/tổng-thể).
