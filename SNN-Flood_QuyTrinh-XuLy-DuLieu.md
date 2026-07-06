# Quy trình xử lý dữ liệu — SNN-Flood

> **Mục tiêu:** biến Sen1Floods11 thô → nhãn **3 lớp flood-on-land** + **split 60/20/20** → tensor sẵn sàng đưa vào Spiking U-Net.
> **Cập nhật:** 03/07/2026 · **Liên quan:** [Pipeline tổng thể](SNN-Flood_Pipeline.md) · [Đề cương](dataset/SNN-Flood_De-cuong-Paper.md)

---

## Tổng quan các bước

| Bước | Nội dung | Thực hiện ở | Trạng thái |
|:--:|----------|-------------|-----------|
| 1 | Ghép chip S1↔Label, lọc chip toàn NaN | `make_splits.py`, `build_3class_labels.py` | ✅ đã chạy (445 chip) |
| 2 | Đọc mảng S1 / LabelHand / JRC | `build_3class_labels.py` | ✅ đã chạy |
| 3 | Tạo mask nước thường trực từ JRC | `build_3class_labels.py` | ✅ đã chạy |
| 4 | Dựng nhãn 3 lớp + lưu `Label3Class/` | `build_3class_labels.py` | ✅ đã chạy |
| 5 | Chia 60/20/20 phân tầng theo vùng | `make_splits.py` | ✅ đã chạy (265/90/90) |
| 6 | Chuẩn hóa dB → [0,1], NaN→-1 | `sen1floods11_dataset.py` | ✅ có sẵn |
| 7 | (tùy chọn) thêm kênh `1−VH/VV` | — | ⬜ ablation |
| 8 | Crop 256×256 khi train | dataset (giai đoạn train) | ⬜ sau |
| 9 | Mã hóa SAR→spike (T bước) | dataset (giai đoạn train) | ⬜ sau |

> **Offline (chạy 1 lần):** bước 1–5. **Lúc load/train:** bước 6–9.

---

## Chi tiết từng bước

### Bước 1 — Ghép & lọc chip
Match `chip_id` giữa `S1Hand/` và `LabelHand/`; loại chip mà toàn bộ pixel SAR là NaN (đọc giảm mẫu 64×64 để nhanh).
→ **446 → 445 chip** (loại `Paraguay_34417` toàn NaN).

### Bước 2 — Đọc mảng
```
s1    = read(S1Hand)      → (2, 512, 512) float32   # VV, VH (dB)
water = read(LabelHand)   → (512, 512)   {-1, 0, 1}
jrc   = read(JRCWaterHand)→ (512, 512)               # ⏸ cần tải
```

### Bước 3 — Mask nước thường trực từ JRC
```
perm = perm_mask(jrc)   # tự nhận diện: JRC nhị phân → jrc>0; occurrence → jrc≥50
```
Script tự in dải giá trị JRC để xác nhận ngưỡng (xem "Điểm cần chốt").

### Bước 4 — Dựng nhãn 3 lớp
Thứ tự ưu tiên: **nước thường trực đè lên flood** (giống Banerjee & Daou 2026).
```
label3 = 0                              # nền
label3[perm] = 1                        # nước thường trực
label3[(water==1) & (~perm)] = 2        # nước lũ = water ∧ ¬permanent
label3[nan_mask] = -1                   # NaN trong S1
label3[water==-1] = -1                  # no-data gốc
```
→ Lưu `Label3Class/{id}_Label3Class.tif` (int16, nodata=-1).

### Bước 5 — Chia 60/20/20
Phân tầng theo **11 vùng** (mọi vùng có mặt cả 3 tập), seed=42, deterministic. Ghi `splits/{train,val,test}.csv`.

### Bước 6 — Chuẩn hóa SAR *(đã có trong `sen1floods11_dataset.py`)*
```
s1 = nan_to_num(s1, nan=-50); s1 = clip(s1, -50, 0); s1 = (s1 + 50) / 50.0   # → [0,1]
```

### Bước 7 — (Tùy chọn) kênh thứ 3
`1 − (VH/VV)` để tăng tương phản nước (đổi dB→tuyến tính trước: `lin = 10^(dB/10)`). Dùng làm ablation.

### Bước 8 — Crop
Train: random crop **256×256**; eval: 512×512 (hoặc sliding-window rồi ghép).

### Bước 9 — Mã hóa SAR→spike
```
direct: x_t = x  ∀ t∈[1..T]    → (T, C, H, W)   # lặp, lớp LIF đầu sinh spike
rate:   x_t = Bernoulli(x)                       # ablation
```

---

## Kết quả đã chạy (thật)

**Split 60/20/20 phân tầng theo vùng** (`python make_splits.py --drop_all_nan`):

| Vùng | train | val | test | tổng |
|------|:--:|:--:|:--:|:--:|
| Bolivia | 9 | 3 | 3 | 15 |
| Ghana | 31 | 11 | 11 | 53 |
| India | 40 | 14 | 14 | 68 |
| Mekong | 18 | 6 | 6 | 30 |
| Nigeria | 10 | 4 | 4 | 18 |
| Pakistan | 16 | 6 | 6 | 28 |
| Paraguay | 40 | 13 | 13 | 66 |
| Somalia | 16 | 5 | 5 | 26 |
| Spain | 18 | 6 | 6 | 30 |
| Sri-Lanka | 26 | 8 | 8 | 42 |
| USA | 41 | 14 | 14 | 69 |
| **TỔNG** | **265** | **90** | **90** | **445** |
| Tỷ lệ | 59.6% | 20.2% | 20.2% | |

→ Đã ghi `dataset/splits/train.csv`, `val.csv`, `test.csv` (cột: `chip_id, region, s1, label, label3`).

---

## ✅ Lớp JRC & nhãn 3 lớp — đã hoàn tất (03/07/2026)

Đã tải **446/446** `JRCWaterHand` từ Google Cloud (`dataset/download_jrc.py`, JRC dạng nhị phân) và chạy `build_3class_labels.py` → `dataset/Label3Class/`.

**Phân bố pixel thật (3 lớp):** nền 77.0% · nước thường trực 2.7% · nước lũ 6.5% · bỏ qua 13.8%.
**Per-chip:** 393/445 chip có flood (52 chip không flood); theo split flood-chips train/val/test = **237/78/78**.

---

## Cách chạy

```bash
cd dataset
python make_splits.py --drop_all_nan     # Bước 5 — tạo splits/*.csv   (đã chạy)
python build_3class_labels.py            # Bước 1–4 — tạo Label3Class/ (cần JRC)
```

## Cấu trúc thư mục

```
dataset/
├── S1Hand/           ✅ ảnh SAR (446)
├── LabelHand/        ✅ nhãn nước nhị phân (446)
├── JRCWaterHand/     ✅ nước thường trực (446, đã tải từ GCS)
├── Label3Class/      ✅ nhãn 3 lớp (446, đã dựng)
├── splits/           ✅ train.csv / val.csv / test.csv
├── make_splits.py            (Bước 5)
├── build_3class_labels.py    (Bước 1–4)
└── sen1floods11_dataset.py   (Bước 6, loader)
```

---

## Điểm cần chốt

1. **Ngưỡng JRC** — script tự nhận diện nhị phân (jrc>0) hay occurrence (jrc≥50); cần xác nhận lại sau khi có lớp JRC bằng dải giá trị nó in ra.
2. **Chiến lược split** — hiện dùng *stratified-by-region* (mọi vùng có mặt mọi split). Cân nhắc thêm *region-holdout* (giữ vài vùng chỉ để test) làm ablation đo tổng quát hóa.

---

*Tài liệu quy trình — cập nhật khi có lớp JRC và chạy xong nhãn 3 lớp.*
