# SNN-Flood — Pipeline Hoàn Chỉnh (End-to-End)

> **Đề tài:** SNN-Flood: Energy-Efficient Flood Extent Mapping from Sentinel-1 SAR Imagery Using Spiking Neural Networks
> **Cập nhật:** 03/07/2026 · **Tài liệu liên quan:** [đề cương paper](dataset/SNN-Flood_De-cuong-Paper.md), [data loader](dataset/sen1floods11_dataset.py)

Pipeline này gộp toàn bộ quyết định đã chốt: bài toán **3 lớp flood-on-land**, dữ liệu **Setup B (hand-labeled, split 60/20/20)**, baseline lấy từ **Banerjee & Daou 2026**, và contribution lõi là **phân tích năng lượng bằng SynOps**.

---

## 0. Sơ đồ pipeline tổng thể

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 1 — DỮ LIỆU & NHÃN                                            │
│  S1Hand (446, VV/VH) ┐                                                    │
│  LabelHand (nước nhị phân) ├──► Ghép với JRC ──► Nhãn 3 lớp:              │
│  JRCWaterHand (cần tải)  ┘        permanent-water │ 0 = nền              │
│                                                    │ 1 = nước thường trực │
│                                                    │ 2 = nước lũ (flood)  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 2 — CHIA DỮ LIỆU (Setup B, kết quả chính)                     │
│  446 chip ──► split 60/20/20 (≈268/89/89), phân tầng theo vùng, seed=42  │
│              chia theo NHÓM SCENE/VÙNG để tránh spatial leakage           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 3 — TIỀN XỬ LÝ & MÃ HÓA SPIKE                                 │
│  clip dB[-50,0] → norm[0,1] → NaN→label=-1 → crop 256 → encode T bước    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
        ┌────────────────────────┴────────────────────────┐
        ▼                                                  ▼
┌────────────────────────┐                    ┌────────────────────────────┐
│ GIAI ĐOẠN 4-5          │                    │  BASELINES (cùng split)    │
│ Spiking U-Net          │                    │  Otsu · U-Net · U-Net++    │
│ LIF + surrogate grad   │                    │  DeepLabV3 · SegFormer     │
│ Dice+Focal loss (BPTT) │                    │  MobileNet-UNet/INT8       │
│                        │                    │  ANN2SNN                   │
└───────────┬────────────┘                    └─────────────┬──────────────┘
            └──────────────────────┬───────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 7 — ĐÁNH GIÁ                                                   │
│  Accuracy: IoU/F1/P/R (riêng lớp flood=2)                                 │
│  Energy:   SynOps(SNN) vs FLOPs(ANN) → mJ  |  spike rate                  │
│  Thống kê: Wilcoxon signed-rank per-chip + bootstrap 95% CI               │
│  Hình:     scatter Accuracy–Energy (Pareto)  +  bản đồ dự đoán            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Định hình bài toán (ĐÃ CHỐT — hướng 1)

- **Bài toán:** semantic segmentation **3 lớp** — `0` nền · `1` nước thường trực · `2` nước lũ trên đất (flooded land) · `-1` bỏ qua.
- **Metric chính:** **Flood-on-land IoU / F1 (lớp 2)** — đây là con số trung tâm của paper.
- **Vì sao 3 lớp, không phải binary water/non-water:** nhãn nước nhị phân gộp cả sông/hồ/ao có sẵn — không phải "flood". Tách nước thường trực (bằng lớp JRC) mới đúng nghĩa "flood extent mapping" và mới so được với [Banerjee & Daou 2026].
- ⚠️ **Hệ quả về con số:** flood-on-land IoU chỉ ~0.38–0.42 (khó hơn nhiều binary water ~0.7+). Đừng so IoU của bài này với các paper làm binary water.

---

## 2. Giai đoạn 1 — Dữ liệu & xây nhãn 3 lớp

**Đã có trong `dataset/`:** `S1Hand/` (446 ảnh VV/VH), `LabelHand/` (nhãn nước nhị phân), `S1Weak/`, `S1OtsuLabelWeak/`.

**CẦN LÀM — tải lớp JRC permanent-water:**
- Lấy `JRCWaterHand/` (446 chip, căn lưới trùng S1Hand) từ dataset gốc Sen1Floods11.
- Đây là bản đồ nước thường trực từ JRC Global Surface Water.

**Công thức tạo nhãn 3 lớp** (per pixel):
```
if pixel invalid / NaN:          label3 = -1   # bỏ qua khi tính loss
elif JRC == permanent water:     label3 = 1    # nước thường trực
elif LabelHand == water:         label3 = 2    # nước lũ (water ∧ ¬permanent)
else:                            label3 = 0    # nền
```
→ Lưu ra thư mục mới `Label3Class/` để khỏi tính lại mỗi lần load.

---

## 3. Giai đoạn 2 — Chia dữ liệu

**Setup B (KẾT QUẢ CHÍNH):**
- Chỉ dùng `S1Hand` + nhãn 3 lớp vừa tạo.
- Split **60/20/20** (≈268/89/89), **phân tầng theo 11 vùng địa lý**, **seed cố định**.
- Chia theo **nhóm scene/vùng**, không random từng chip → tránh spatial leakage (chip kề nhau lọt cả train+test làm IoU cao ảo).
- Hàm cần viết: `get_handlabeled_splits(val_ratio=0.2, test_ratio=0.2, seed=42, stratify_by_region=True)`.

**Setup A (Ý PHỤ — weak-supervised):**
- Train trên `S1Weak` + `S1OtsuLabelWeak` (~3.937 chip), test trên hand-label.
- ⚠️ Nhãn weak là **nước nhị phân**, chưa có JRC cho chip weak → Setup A giữ ở dạng **binary water** (hoặc chỉ minh họa kịch bản weak-supervised). Ghi rõ khác biệt này khi báo cáo.

---

## 4. Giai đoạn 3 — Tiền xử lý & mã hóa SAR→spike

1. **Chuẩn hóa SAR:** clip dB về `[-50, 0]` → chuẩn hóa `[0, 1]`; pixel NaN → gán label `-1` (đã có trong `sen1floods11_dataset.py`).
2. **(Tùy chọn) kênh thứ 3:** `1 − (VH/VV)` như ETCI để tăng tương phản nước — cân nhắc làm ablation.
3. **Crop 256×256** khi train (512×512 × T bước rất nặng bộ nhớ); test có thể ghép tile lại.
4. **Mã hóa spike** (ablation kiểu encoding):
   - *Direct/analog coding:* lặp ảnh chuẩn hóa qua `T` bước, để lớp spiking đầu tiên tự sinh spike (phổ biến cho ảnh tĩnh).
   - *Rate coding:* xác suất phát spike tỉ lệ cường độ pixel.

---

## 5. Giai đoạn 4 — Kiến trúc Spiking U-Net

- **Xương sống:** U-Net encoder–decoder + skip connection, thay mọi ReLU bằng **nơ-ron LIF (Leaky Integrate-and-Fire)**.
- **Huấn luyện trực tiếp** bằng **surrogate gradient** (đạo hàm xấp xỉ hàm spike).
- **Đầu ra:** lấy **membrane potential** lớp cuối (tích lũy qua T bước) → logit 3 lớp → argmax ra mask.
- **Framework:** **SpikingJelly** (khuyến nghị, tối ưu tốc độ) hoặc snnTorch.
- **Đối chứng nội bộ:** thêm biến thể **ANN2SNN** (train U-Net thường rồi convert) để chứng minh direct-training tốt hơn.

---

## 6. Giai đoạn 5 — Huấn luyện

| Thành phần | Lựa chọn |
|-----------|----------|
| Loss | **Dice + Focal (γ=2)** + CrossEntropy `ignore_index=-1` — chống mất cân bằng (pixel flood rất ít) |
| Tối ưu | AdamW, weight decay 1e-4 |
| Early stopping | theo **val flood IoU** (lớp 2), patience ~8 |
| Timesteps | T ∈ {2,4,6,8} — làm ablation |
| Mixed precision | bật để tiết kiệm bộ nhớ |

---

## 7. Giai đoạn 6 — Baselines (train trên ĐÚNG cùng split)

| Nhóm | Mô hình | Vai trò | IoU flood tham chiếu¹ |
|------|---------|---------|:--:|
| Cổ điển | Otsu threshold | sàn accuracy, ~0 năng lượng | — |
| CNN mạnh | **U-Net (ResNet34)** | baseline accuracy chính | 0.408 |
| CNN mạnh | U-Net++ (ResNet34) | | 0.402 |
| CNN mạnh | **DeepLabV3 (ResNet34)** | | 0.382 |
| Transformer | SegFormer-b0/b1 | | 0.388 / 0.409 |
| Transformer | **SegFormer-b2** | mốc accuracy trần | 0.418 |
| CNN tiết kiệm | MobileNet-UNet / U-Net INT8 | *đối thủ năng lượng thật sự* | — |
| SNN | ANN2SNN (conversion) | so sánh nội bộ họ SNN | — |
| **Ours** | **SNN-Flood** | điểm Pareto accuracy–energy | **?** |

¹ Số Sen1Floods11 flood-on-land, không TTA, từ Banerjee & Daou 2026 — **chỉ đúng nếu ta dùng split 251/89/28 của họ**; với split 60/20/20 riêng thì phải tự chạy lại toàn bộ.

---

## 8. Giai đoạn 7 — Đánh giá

**Chỉ số accuracy** (tính riêng từng lớp, nhấn mạnh lớp flood=2):
- **Chính:** IoU/mIoU, F1 (hoặc Dice), Precision, Recall.
- **Phụ:** pixel accuracy — vẫn báo cáo, nhưng không làm chính (mất cân bằng lớp → ~97% ảo).

**Energy** (contribution lõi — chưa ai làm trên bài toán này):
- Đếm **SynOps** (SNN, phép cộng AC) vs **FLOPs** (ANN, phép nhân-cộng MAC).
- Quy đổi: `E = SynOps × 0.9 pJ` (AC) so với `FLOPs × 4.6 pJ` (MAC) ở công nghệ 45nm.
- Báo cả **spike rate** (độ thưa) và **× lần tiết kiệm**.

**Thống kê ý nghĩa** (bám theo Banerjee & Daou 2026):
- **Wilcoxon signed-rank** trên flood IoU **per-chip** + **bootstrap 95% CI** (10.000 lần).

**Hình điểm nhấn:**
- **Scatter Accuracy (Y) vs Energy (X, log)** — mỗi model 1 điểm; SNN-Flood ở góc "accuracy cao – energy thấp" (Pareto-optimal).

---

## 9. Giai đoạn 8 — Ablation & định tính

- **Ablation:** số timestep T · kiểu encoding (direct vs rate) · kênh thứ 3 · loss · ngưỡng LIF.
- **Setup A** (weak-supervised) như một thực nghiệm phụ.
- **Định tính:** ≥3 hình SAR → mask dự đoán → ground truth, chọn vùng có flood phân mảnh (India/Paraguay như paper).

---

## 10. Bảng kết quả cuối (đích đến — điền sau khi chạy)

| Model | Nhóm | Flood IoU | Flood F1 | Params (M) | FLOPs/SynOps | Energy (mJ) | × tiết kiệm |
|-------|------|:--:|:--:|:--:|:--:|:--:|:--:|
| Otsu | Cổ điển | — | — | 0 | ~0 | ~0 | — |
| U-Net | CNN | — | — | — | FLOPs | — | 1× (ref) |
| DeepLabV3 | CNN | — | — | — | FLOPs | — | — |
| MobileNet-UNet | CNN nhẹ | — | — | — | FLOPs | — | — |
| SegFormer-b2 | Transformer | — | — | — | FLOPs | — | — |
| ANN2SNN | SNN | — | — | — | SynOps | — | — |
| **SNN-Flood (ours)** | SNN | — | — | — | SynOps | — | **X×** |

---

## 11. Trạng thái code — cái gì có, cái gì cần viết

| Thành phần | File | Trạng thái |
|-----------|------|-----------|
| Data loader nền | `dataset/sen1floods11_dataset.py` | ✅ có (weak train/val + test toàn bộ hand) |
| Tải lớp JRC | — | ⬜ cần |
| Script tạo nhãn 3 lớp | `build_3class_labels.py` | ⬜ cần |
| Hàm split 60/20/20 stratified | `get_handlabeled_splits()` | ⬜ cần |
| Spiking U-Net | `models/spiking_unet.py` | ⬜ cần |
| Baselines (U-Net, SegFormer…) | `models/baselines.py` | ⬜ cần |
| Vòng train (BPTT + Dice/Focal) | `train.py` | ⬜ cần |
| Đo SynOps/energy | `energy.py` | ⬜ cần |
| Eval + thống kê + Pareto plot | `evaluate.py` | ⬜ cần |

---

## 12. Thứ tự thực thi đề xuất

1. **Tải JRC + tạo nhãn 3 lớp + viết `get_handlabeled_splits()`** → dữ liệu sẵn sàng.
2. **Dựng U-Net baseline** (ANN) trên split 60/20/20 → lấy mốc IoU/F1 flood.
3. **Module đo FLOPs/SynOps/energy** → khung đánh giá năng lượng.
4. **Spiking U-Net** (T=4) → pilot, kiểm tra hội tụ; so accuracy & energy với U-Net.
5. Thêm các baseline còn lại (SegFormer-b2, DeepLabV3, MobileNet-UNet, ANN2SNN).
6. **Ablation T + encoding**, chạy thống kê, vẽ Pareto.
7. Định tính + hoàn thiện bảng + viết bản thảo.

---

*Tài liệu thiết kế — cập nhật khi có kết quả thực nghiệm.*
