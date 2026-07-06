# Rà soát bước tiền xử lý dữ liệu — SNN-Flood

> Tài liệu xin ý kiến thầy. Các phát hiện dưới đây **dựa trên chẩn đoán chạy thật** trên 446 chip hand-labeled (`dataset/diagnose_preprocessing.py`), ngày 03/07/2026.
> **Liên quan:** [Quy trình xử lý](SNN-Flood_QuyTrinh-XuLy-DuLieu.md) · [Pipeline](SNN-Flood_Pipeline.md)

## Tóm tắt nhanh

Pipeline hiện tại **về cơ bản đúng hướng**, nhưng còn **4 vấn đề cần xử lý** và **4 điểm thiết kế cần chốt** trước khi chạy thí nghiệm chính. Vấn đề lớn nhất là **mất cân bằng lớp nghiêm trọng**.

| # | Vấn đề | Mức độ |
|:--:|--------|:------:|
| 1 | Mất cân bằng lớp nghiêm trọng + nhiều chip không có nước/lũ | 🔴 Cao |
| 2 | Rò rỉ không gian: thấp nhưng chưa bằng 0 (~14 cặp chip chồng nhau) | 🟡 Vừa |
| 3 | Chuẩn hóa chung [-50,0] cho cả VV lẫn VH (VH lệch thấp hơn nhiều) | 🟡 Vừa |
| 4 | Điền 0 vào pixel NaN có thể lan sang lân cận | 🟢 Thấp |
| 5–8 | Các điểm thiết kế cần chốt (JRC, protocol test, Setup A, reproducibility) | 🟡 Vừa |

---

## Bằng chứng số liệu (chạy thật)

- **Tỷ lệ pixel nước / chip:** trung vị **2.5%**, trung bình 10.75%. → nước lũ (sau khi trừ nước thường trực) sẽ **còn thấp hơn**.
- **52/446 chip (12%) KHÔNG có pixel nước nào**; 101 chip có <1% nước.
- Phân bố "trắng nước" lệch theo vùng: **Ghana 26/53 chip trắng nước** (≈50%), Paraguay 10/67, Pakistan 7/28, USA 27/69 chip có <1% nước.
- **NaN cục bộ:** 45/446 chip (10.1%) có pixel NaN, trung vị 3.2% diện tích.
- **Dải dB:** VV median −9.8, VH median −16.3. Clip [-50,0] chỉ mất **0.18%** pixel (trên 0dB) + 0.002% (dưới −50) → range hợp lý.
- **Chồng lấn không gian:** chỉ **~14 cặp** chip chồng nhau trên toàn bộ (India 3, Mekong 3, Sri-Lanka 3, USA 2, còn lại ≤1). Phần lớn chip trong cùng vùng **không** chồng nhau.

---

## Chi tiết & khuyến nghị

### 🔴 Vấn đề 1 — Mất cân bằng lớp nghiêm trọng
Trung vị chỉ 2.5% pixel là nước; lớp flood (class 2) sẽ càng hiếm. 52 chip hoàn toàn không có nước (Ghana chiếm một nửa số này).

**Hệ quả & khuyến nghị:**
- **Chỉ số chính:** IoU/mIoU, F1 (hoặc Dice), Precision, Recall — tính theo lớp (nhất là lớp flood). **Vẫn báo pixel accuracy làm chỉ số phụ** (thầy dặn KHÔNG bỏ). Lý do accuracy không làm chính: **mất cân bằng lớp** khiến đoán "toàn nền" đã ~97%, gây hiểu nhầm — chứ không phải vì data ít.
- **Loss Dice + Focal là lựa chọn đúng** cho imbalance — giữ nguyên. Cần **báo cáo tỷ lệ 3 lớp thật** sau khi dựng nhãn bằng JRC.
- **Protocol đánh giá:** theo Banerjee & Daou 2026, chỉ tính flood-IoU trên **chip có chứa pixel flood** (loại chip không có flood khỏi phép tính). → Cần chốt áp dụng.
- **Split nên cân nhắc flood-presence:** hiện chỉ phân tầng theo vùng, chưa đảm bảo mỗi tập đủ chip chứa flood. Ví dụ test-Ghana 11 chip nhưng ~nửa trắng nước → tín hiệu mỏng. Đề xuất **phân tầng kép (vùng × có/không flood)** hoặc kiểm tra lại sau split.

### 🟡 Vấn đề 2 — Rò rỉ không gian (spatial leakage)
Tin tốt: chỉ ~14 cặp chip chồng nhau → xác nhận Sen1Floods11 lấy mẫu **phân tán** (khác ETCI vốn cắt tile dày đặc). Split *stratified-by-region* về cơ bản chấp nhận được.

**Khuyến nghị:** để chặt tuyệt đối, **gom các cặp chip chồng nhau vào cùng một tập** (hiện có thể bị chia đôi train/test). Việc nhỏ nhưng ghi điểm với reviewer. Vùng cần chú ý: India, Mekong, Sri-Lanka.

### 🟡 Vấn đề 3 — Chuẩn hóa chung cho VV và VH
VH lệch thấp hơn VV rõ rệt (median −16.3 vs −9.8). Chuẩn hóa chung [-50,0] làm VH bị nén vào dải hẹp của [0,1], mô hình khó tận dụng.

**Khuyến nghị:** thử **chuẩn hóa từng kênh (per-channel)** — mỗi kênh dùng min/max riêng — làm một ablation nhỏ. Không bắt buộc nhưng nên có.

### 🟢 Vấn đề 4 — Điền 0 vào pixel NaN
Hiện NaN → gán label −1 (loss bỏ qua, đúng), nhưng **input** tại đó bị điền 0 sau chuẩn hóa; conv/spiking vẫn lan giá trị 0 sang pixel lân cận.

**Khuyến nghị:** chấp nhận được, nhưng có thể điền bằng **giá trị trung bình kênh** thay vì 0 để sạch hơn ở vùng rìa.

### 🟡 Điểm 5 — JRC: no-data & căn lưới
Khi có lớp JRC, cần kiểm tra: (a) cùng kích thước/lưới 512×512 với S1Hand; (b) JRC có giá trị **no-data riêng** không — đừng nhầm "không quan sát" thành "không có nước". Script `build_3class_labels.py` đã tự dò encoding nhưng chưa xử lý no-data của JRC.

### 🟡 Điểm 6 — Protocol test 512 vs train 256
Train trên crop 256 nhưng test ảnh 512 → thống kê khác nhau. Cần chốt: test bằng **sliding-window 256 có overlap** rồi ghép, hay test full 512. (SNN 512×T bước có thể hết RAM.)

### 🟡 Điểm 7 — Setup A (weak) đo việc khác Setup B
Weak-label là **nước nhị phân** (chưa có JRC cho chip weak), còn Setup B là **3 lớp flood**. Hai cái đo bài toán khác nhau → khi báo cáo phải nói rõ, không đặt cạnh nhau như so sánh trực tiếp.

### 🟡 Điểm 8 — Reproducibility
`splits/*.csv` đã sinh ra — nên **commit vào git** để cố định split, và kiểm tra `.gitignore` không loại nó.

---

## Hướng xử lý & trạng thái (để thầy duyệt)

### A. Đã xử lý xong trong code — không cần thầy quyết
- ✅ **Rò rỉ không gian:** gom 14 nhóm chip chồng nhau vào cùng một tập.
- ✅ **Split cân bằng flood-presence:** phân tầng (vùng × có nước) → **271/87/87**, chip-có-nước chia **237/78/78**, mọi vùng đều có mặt cả 3 tập.

### B. Cần thầy duyệt / cho ý kiến

| # | Hạng mục | Đề xuất cụ thể | Lý do | Trạng thái |
|:--:|----------|----------------|-------|:----------:|
| 1 | Metric | **Chính:** IoU/mIoU, F1 (hoặc Dice), Precision, Recall (theo lớp flood). **Phụ:** vẫn báo **pixel accuracy** | accuracy cao ảo do **mất cân bằng lớp** (không phải do data ít) → không làm chính nhưng vẫn báo | ✅ Thầy chốt 04/07 |
| 2 | Loss | **Dice + Focal (γ=2)** + `ignore_index=−1` | chống mất cân bằng lớp | ⬜ Xin thầy duyệt |
| 3 | **Protocol đánh giá** ⭐ | Chỉ tính flood-IoU trên **chip CÓ flood** (test **78/87**), kèm báo pooled-IoU toàn tập | theo Banerjee & Daou 2026; chip trắng-flood làm nhiễu số liệu | 🔴 **Cần thầy quyết** |
| 4 | Chip trắng-nước | Giữ trong **train** (hard-negative), **loại** khỏi flood-IoU của val/test | giảm false-positive mà không thổi phồng metric | ⬜ Xin thầy duyệt |
| 5 | Chuẩn hóa VV/VH | Baseline [-50,0] dùng chung + **ablation per-channel** | VH median −16.3 lệch xa VV −9.8 | ⬜ Ablation, xin ý kiến |
| 6 | JRC no-data & lưới | ✅ Đã tải 446/446; JRC **nhị phân** (permanent = jrc>0), căn lưới khớp; nhãn 3 lớp đã dựng | — | ✅ Xong |
| 7 | Test 512 vs crop 256 | SNN: **sliding-window 256 overlap** rồi ghép; CNN: test full 512 | nhất quán train/test, tránh hết RAM | ⬜ Xin thầy duyệt |
| 8 | Setup A ≠ Setup B | Báo cáo **tách riêng** (A: nhị phân/weak; B: 3 lớp) | hai bài toán khác nhau | ⬜ Xin thầy duyệt |
| 9 | Reproducibility | **Commit `splits/*.csv`** vào git | cố định split để tái lập | ⬜ Sẽ làm |

> ⭐ = quyết định quan trọng nhất cần thầy chốt (nó kéo theo cách tính mọi con số trong paper).
>
> **✅ Cập nhật 03/07/2026 — đã có lớp JRC:** tải xong **446/446** `JRCWaterHand` từ Google Cloud (JRC dạng nhị phân), đã dựng nhãn 3 lớp `Label3Class/`. **Phân bố pixel thật:** nền 77.0% · nước thường trực 2.7% · **nước lũ 6.5%** · bỏ qua 13.8%. Theo split: **393/445 chip có flood** (train/val/test = 237/78/78); nếu chọn protocol "chỉ tính trên chip có flood" thì **test dùng 78/87 chip**. Các con số flood giờ là THẬT, không còn ước lượng gián tiếp.

---

## Những gì đã ĐÚNG (thầy có thể yên tâm)

- ✅ Nhãn sạch, chỉ gồm {−1, 0, 1} như kỳ vọng.
- ✅ Dải clip dB [-50,0] được **dữ liệu xác nhận** hợp lý (mất <0.2% pixel).
- ✅ Chia 60/20/20 phân tầng theo vùng, seed cố định, tái lập được — sau nâng cấp là 271/87/87, đã vá rò rỉ + cân bằng flood-presence.
- ✅ Đã lọc chip toàn NaN (Paraguay_34417).
- ✅ Chọn tách nước lũ khỏi nước thường trực bằng JRC — đúng chuẩn mực, khớp Banerjee & Daou 2026.

---

*Rà soát dựa trên dữ liệu thật — cập nhật sau khi có lớp JRC và ý kiến thầy.*
