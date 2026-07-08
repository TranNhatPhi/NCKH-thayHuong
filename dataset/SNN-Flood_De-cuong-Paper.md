# SNN-Flood: Energy-Efficient Flood Extent Mapping from Sentinel-1 SAR Imagery Using Spiking Neural Networks
### Đề cương paper hội nghị — phiên bản xin ý kiến thầy

| | |
|--|--|
| **Tác giả** | TranNhatPhi (phivt1234@gmail.com) |
| **Ngày** | 25/06/2026 |
| **Trạng thái** | Draft — chưa nộp, đang xin ý kiến thầy hướng dẫn |

---

## 0.0 CẬP NHẬT ĐỊNH HƯỚNG (07/07/2026 — chốt với thầy sau khi có kết quả n=3)

> Thầy đã duyệt: *"Ok có thể bắt tay viết paper với kết quả hiện tại rồi đó Phi"*. Các mục dưới **thay thế** framing cũ (binary / "SNN thắng CNN"). Framing mới = **phân tích đánh đổi năng lượng–độ chính xác trung thực** + baseline SNN đầu tiên trên Sen1Floods11.

### Tiêu đề chốt
**"Energy-Accuracy Trade-offs in Spiking U-Net for SAR Flood Mapping: A Systematic Study on Sen1Floods11"**

### Bài toán (cập nhật): **3 lớp** flood-on-land
0 = background, 1 = permanent water (dùng JRC tách), 2 = flood. (Khó hơn binary water → giải thích rõ vì sao IoU tuyệt đối khiêm tốn.)

### 3 đóng góp chốt (theo thầy)
1. **First direct-trained SNN baseline** cho 3-class flood-on-land segmentation trên Sen1Floods11 (chỉ dùng S1 SAR, 446 chip hand-labeled).
2. **Empirical characterization of T-stability regime** của Spiking U-Net: **T=2 và T=8 ổn định, T=4 và T=6 bimodal/bất ổn** (std ~0.12–0.13) → *phát hiện mới*. Đang chạy thêm **T=1,3,5,7,10 (×3 seed)** để xác nhận "T ổn định là hiếm". Ghi nhận thêm: **LR của SNN ảnh hưởng tới ĐỘ ỔN ĐỊNH nhiều hơn tới accuracy**.
3. **Pareto analysis** SNN vs CNN vs Transformer vs ANN2SNN trên trục năng lượng–accuracy:
   - Dải **60–200 mJ**: **MobileNet-UNet thắng** không bàn cãi (0.52 IoU @ 63 mJ; tune LR 2e-4 quan trọng).
   - Dải **< 40 mJ**: **SNN-T2 là lựa chọn học được DUY NHẤT còn giữ accuracy > 0.30** (0.381 @ 33 mJ) — MobileNet không xuống dưới 63 mJ → **SNN-T2 hợp cho thiết bị biên thực sự**.
   - **ANN2SNN thất bại** kể cả T=128 (0.21 @ 440 mJ, đắt hơn cả ANN gốc) → **direct-training vượt trội conversion**.

### Điểm nhấn khi viết bản thảo (theo thầy)
- **SNN đã học được lớp permanent-water** (pwIoU khá ở T2/T8) — nêu rõ.
- Viết tập trung **cả SNN-T2 (năng lượng thấp nhất)** *và* **SNN-T8 (cân bằng — pwIoU cao nhất họ SNN, 0.606)**.
- **T4/T6 bất ổn → biến thành phát hiện mới** (thách thức train SNN sâu cho segmentation), không giấu.

### Thí nghiệm bổ sung (theo yêu cầu thầy — vòng 1)
- [ ] T-sweep mở rộng **T=1,3,5,7,10** (×3 seed) → xác nhận vùng bimodal. *(configs + `run_advisor.sh`)*
- [ ] **Retrain T2 & T8 với LR 2e-4** (recipe MobileNet đang thắng) → xem có lên **0.42–0.45**? Nếu T8+LR2e-4 đạt ~0.42 là ok.
- [ ] **MobileNet-UNet INT8** (PTQ) → thêm điểm Pareto "ANN nén" ở dải năng lượng cực thấp. *(`quantize_int8.py`)*
- [x] **ANN2SNN**: số liệu hiện tại (T32/64/128) **đã đủ** — nhưng chạy **3 seed** để có σ (fix trong `run_full.sh`).

### Thí nghiệm bổ sung (theo yêu cầu thầy — vòng 2)
- [ ] **T=6 chạy 5 seed** (điểm nghi vấn cao nhất) → verify lucky hay stable. *(`run_advisor.sh`)*
- [ ] **Verify INT8 recipe**: PTQ (đang dùng) vs QAT; calibration set ~200 ảnh train (nâng `--calib_batches=50`). *(`quantize_int8.py`)*
- [ ] **INT4 cho MobileNet-UNet**: thử `torchao int4_weight_only`. Dự kiến **UNSUPPORTED cho Conv2d** (chỉ nén Linear) → **nếu INT4 fail thì củng cố ngách năng lượng của SNN**. *(`quantize_int4.py`)*
- [ ] **Wilcoxon per-chip theo cặp**: `unet_smp vs mobilenet_int8` (kỳ vọng significant), `mobilenet_int8 vs SNN_T6` (chắc significant), `SNN_T2 vs SNN_T6` (T6 có thực sự tốt hơn?). *(`analysis.py --pairs`)*
- [ ] **Spike rate** cho SNN_T2/T5/T6/T8 → hoàn thiện SynOps (đã thêm cột `Spike%` vào `summarize.py`).

> **Thầy chốt (08/07/2026):** *"Kết quả không như dự định ban đầu về SNN nhưng có kho dữ liệu benchmark đầy đủ để viết được paper với câu chuyện khác như trên cũng ok á Phi."* → xác nhận đi theo framing benchmark/trade-off.

---

## 0. Thông tin định hướng

| Mục | Nội dung |
|-----|----------|
| **Loại paper** | Conference paper (6–8 trang) |
| **Hội nghị nhắm tới** | ⚠️ Xem mục 0.1 bên dưới — IGARSS 2026 đã đóng deadline |
| **Dataset** | Sen1Floods11 (Bonafilia et al., 2020) — Sentinel-1 SAR, VV+VH |
| **Bài toán** | Binary semantic segmentation: water (1) / non-water (0) |
| **Đóng góp lõi** | Theo hiểu biết của chúng tôi (to the best of our knowledge), một trong những công trình đầu tiên áp dụng SNN cho flood extent mapping trên SAR, tiết kiệm năng lượng mà giữ độ chính xác cạnh tranh |

---

## 0.1 Deadline hội nghị thực tế (cập nhật tháng 6/2026)

> ⚠️ **IGARSS 2026 đã đóng** (abstract deadline ~Jan 2026, hội nghị tháng 7/2026). Không kịp nộp.

| Hội nghị | Lĩnh vực | Deadline ước tính | Hội nghị |
|----------|----------|-------------------|----------|
| **IGARSS 2027** ★ Ưu tiên 1 | Remote sensing + flood | ~Jan 2027 | Jul/Aug 2027 |
| **AAAI 2027** | AI tổng quát | ~Aug 2026 ← **SẮP TỚI** | Feb 2027 |
| **IJCNN 2027** | Neural networks + SNN | ~Jan–Feb 2027 | Jun 2027 |
| **EarthVision @ CVPR 2027** | Remote sensing + DL | ~Mar 2027 | Jun 2027 |
| **ICONS 2027** | Neuromorphic | ~Apr 2027 | Jul 2027 |
| IEEE TGRS / Remote Sensing (MDPI) | Journal (rolling) | Bất kỳ lúc nào | 3–6 tháng review |

**Đề xuất:** Nhắm **IGARSS 2027** (phù hợp nhất chủ đề) + chuẩn bị song song bản journal cho **IEEE TGRS**. Hỏi thầy về deadline AAAI 2027 nếu muốn nộp sớm (deadline ~tháng 8/2026 — còn ~2 tháng).

> 🔴 **Hỏi thầy:** Thầy nhắm hội nghị nào? Deadline IGARSS 2027 cần xác nhận chính xác trên trang web IEEE IGARSS.

---

## 1. Câu "claim" trung tâm (one-sentence pitch)

> *"Chúng tôi đề xuất SNN-Flood — theo hiểu biết của chúng tôi, một trong những mô hình nơ-ron xung (spiking) đầu tiên cho lập bản đồ ngập lụt từ ảnh SAR Sentinel-1 — đạt IoU xấp xỉ U-Net (chênh < Y%) trong khi giảm khoảng X lần chi phí năng lượng tính toán (đo bằng SynOps), mở đường cho triển khai trên thiết bị biên/neuromorphic cho cảnh báo lũ thời gian thực."*

> ⚠️ X và Y là số bạn điền sau khi chạy thí nghiệm. Đừng hứa con số trước.

---

## 2. Ba đóng góp chính (Contributions) — viết ở cuối Introduction

1. **Đóng góp ứng dụng:** Theo hiểu biết của chúng tôi (to the best of our knowledge), lần đầu đưa SNN vào bài toán flood extent mapping trên SAR (Sen1Floods11), một lĩnh vực tới nay chỉ dùng CNN/Transformer.
2. **Đóng góp kỹ thuật:** Đề xuất kiến trúc *Spiking U-Net* + sơ đồ mã hóa SAR→spike phù hợp với đặc thù ảnh dB 2 kênh (VV/VH), train trực tiếp bằng surrogate gradient.
3. **Đóng góp định lượng:** Phân tích đánh đổi accuracy–energy có hệ thống (SynOps, số spike, ước lượng năng lượng) so với baseline ANN, chứng minh hiệu quả năng lượng.

---

## 3. Cấu trúc paper (section-by-section)

### Abstract (~200 từ)
Vấn đề (lũ lụt cần lập bản đồ nhanh, SAR xuyên mây/đêm) → khoảng trống (mô hình DL hiện tại ngốn năng lượng, khó triển khai biên) → giải pháp (SNN-Flood) → kết quả chính (1 con số accuracy + 1 con số energy) → ý nghĩa.

### 1. Introduction
- Bối cảnh: lũ lụt & nhu cầu bản đồ ngập nhanh; vì sao SAR (xuyên mây, ngày/đêm) tốt hơn quang học.
- Vấn đề: mô hình DL truyền thống tốn năng lượng → khó chạy trên vệ tinh/UAV/edge.
- Giải pháp đề xuất + 3 contributions (mục 2 ở trên).

### 2. Related Work
- 2.1 Flood mapping từ SAR bằng deep learning (U-Net, nested U-Net, semi-supervised).
- 2.2 Spiking Neural Networks cho computer vision & semantic segmentation.
- 2.3 SNN cho ảnh remote sensing / SAR.
- → Kết đoạn: chỉ ra **khoảng trống** = chưa ai ghép SNN + flood SAR.

### 3. Dataset & Preprocessing
- Mô tả Sen1Floods11: 446 hand-label (test), ~4.384 Otsu weak-label (train/val), 11 vùng địa lý, chip 512×512, 2 kênh dB.
- Tiền xử lý: mask NaN → label=-1, clip dB [-50,0], normalize [0,1].
- **✅ Đã chốt với thầy (03/07/2026) — 2 setup, phân vai rõ ràng:**
  - **Setup B (hand-labeled split) — KẾT QUẢ CHÍNH:** Chỉ dùng ảnh `S1Hand` + nhãn `LabelHand` (446 chip), chia train/val/test theo tỷ lệ **70/20/10** (**312/88/45 chip** — đã chốt với thầy), **phân tầng theo vùng địa lý** (11 vùng) và **cố định seed** để tái lập. Train + val + test đều trên nhãn thủ công → đây là bảng kết quả trung tâm của paper.
  - **Setup A (weak-supervised) — Ý PHỤ / ABLATION:** Train ~3.937 chip Otsu weak → test trên hand-label. Chỉ dùng làm thực nghiệm phụ minh họa kịch bản weak-supervised (khi không có nhãn tay). Không dùng làm số liệu chính.
  - **Lý do thầy:** nhãn Otsu tự động có noise, dùng làm kết quả chính thì kém thuyết phục; train + test trên hand-label mới cho so sánh công bằng.
  - **⚠️ Lưu ý về "công bằng" khi chia 70/20/10 tự chọn (KHÁC file CSV official split gốc):** vì split do ta tự tạo nên **không trích trực tiếp được số IoU từ các paper dùng split gốc**. Tính "công bằng" đạt được bằng cách **train MỌI baseline (Otsu, U-Net, DeepLabv3+, ANN2SNN, SNN-Flood…) trên ĐÚNG cùng một split 70/20/10 này**. Nếu vẫn muốn giữ khả năng trích dẫn trực tiếp paper khác, có thể bổ sung official CSV split làm bảng phụ.
  - **⚠️ Cần chuẩn bị:** viết hàm `get_handlabeled_splits(val_ratio=0.2, test_ratio=0.2, seed=42, stratify_by_region=True)` trong `sen1floods11_dataset.py` (hiện chỉ có train/val trên weak-label + test trên toàn bộ hand-label). Cân nhắc chia theo **nhóm vùng/scene** thay vì random từng chip để tránh rò rỉ không gian (spatial leakage) — các chip kề nhau cùng một trận lũ rơi vào cả train lẫn test sẽ làm IoU cao ảo.

### 4. Method (phần lõi)
- 4.1 Tổng quan kiến trúc SNN-Flood (Spiking U-Net: encoder–decoder, skip connection).
- 4.2 Spiking neuron model (LIF — Leaky Integrate-and-Fire), tham số ngưỡng/leak.
- 4.3 Mã hóa input SAR→spike (direct/rate coding qua T timesteps).
- 4.4 Giải mã output (membrane potential ở lớp cuối → logit → mask).
- 4.5 Loss: CrossEntropy(ignore_index=-1) + Dice, chống mất cân bằng lớp.
- 4.6 Huấn luyện: surrogate gradient (BPTT), optimizer, lịch học.

### 5. Experiments
- 5.1 Setup: phần cứng, framework (snnTorch/SpikingJelly), siêu tham số, T.
- 5.2 Metrics: **Chính** — IoU/mIoU, F1 (hoặc Dice), Precision, Recall (theo lớp, nhấn mạnh lớp flood); **Phụ** — pixel accuracy (vẫn báo cáo, không làm chính vì mất cân bằng lớp) + **Energy** (SynOps, spike rate, năng lượng ước lượng).
- 5.3 Baselines: U-Net (ANN) cùng cấu hình; nếu được thêm 1 SOTA gần đây.
- 5.4 Kết quả chính: bảng accuracy vs energy (điểm nhấn của paper).
- 5.5 **So sánh chuyên sâu SNN vs CNN** (xem mục 5.X bên dưới) — head-to-head nhiều kiến trúc CNN.
- 5.6 Ablation: số timestep T, kiểu encoding, loss, ngưỡng neuron.
- 5.7 Qualitative: hình SAR → mask dự đoán → ground truth (vài vùng khác nhau).

#### 5.X So sánh đa mô hình (bước bổ sung)
Mục tiêu: chứng minh SNN-Flood đạt cân bằng accuracy–energy tốt nhất **so với mọi nhóm đối thủ**, đặc biệt chặn lập luận "CNN nhẹ/nén cũng tiết kiệm năng lượng".

**5 nhóm đối chứng (xếp theo phổ accuracy–energy):**

1. **Sàn cổ điển — Otsu thresholding:** gần như 0 năng lượng, không học. Cho thấy DL đáng giá và đặt mốc accuracy thấp nhất.
2. **CNN mạnh — U-Net, DeepLabv3+:** accuracy cao, năng lượng cao. Baseline accuracy chính.
3. **CNN tiết kiệm — MobileNet-UNet / EfficientNet-UNet + bản INT8 (quantized/pruned):** *đối thủ năng lượng thật sự* — cùng "vũ khí" tiết kiệm như SNN. Bắt buộc có để claim đứng vững.
4. **SNN khác — ANN2SNN conversion:** so sánh nội bộ họ SNN, chứng minh direct-training (surrogate gradient) của ta tốt hơn.
5. **(Nên có) Transformer SOTA — SegFormer / Swin-UNet:** mốc accuracy trần để biết ta cách SOTA bao xa.

**So trên 3 trục:** (a) Accuracy — IoU/F1; (b) Chi phí tính toán — FLOPs vs SynOps; (c) Kích thước — số tham số, dung lượng model.

**Năng lượng quy đổi:** FLOPs×E_MAC (ANN) vs SynOps×E_AC (SNN), dùng hệ số chuẩn (vd 4.6 pJ/MAC, 0.9 pJ/AC ở 45nm) để ra bảng "× lần tiết kiệm".

**Biểu đồ điểm nhấn:** scatter Accuracy (trục Y) vs Energy (trục X, log) — mỗi model 1 điểm; SNN-Flood nằm ở góc "accuracy cao – energy thấp" (Pareto-optimal) → hình ảnh thuyết phục reviewer. *(Xem sơ đồ minh họa kèm theo.)*

### 6. Discussion
- Vì sao SNN giảm năng lượng (sparse spike); hạn chế (nhãn Otsu weak, accuracy gap); triển vọng neuromorphic (Loihi).

### 7. Conclusion
- Tóm tắt đóng góp + 1–2 hướng tương lai (multi-temporal SAR, train trên neuromorphic chip).

### References
- Tối thiểu 20–30 trích dẫn. Bắt đầu từ 7 paper đã tìm.

---

## 4. Bảng Related Work (điền dần)

| Paper | Hướng | Dataset | Đóng góp | Liên quan tới ta |
|-------|-------|---------|----------|------------------|
| Bonafilia 2020 (Sen1Floods11) | Flood SAR | Sen1Floods11 | Dataset benchmark | Nguồn dữ liệu |
| Flood Seg. Semi-Supervised (2021) | CNN flood | Sen1Floods11 | IoU baseline | Baseline so sánh |
| Residual Wave U-Net (2024) | U-Net flood | Sentinel-1 | SOTA accuracy | Baseline accuracy |
| Beyond Classification — Shi et al. (2022, arXiv) | SNN segmentation | **PASCAL VOC2012, DDD17** (event-based) | Surrogate gradient train SNN seg trực tiếp | Kiến trúc lõi — bám sát để xây model |
| Energy-Efficient Spiking Segmenter — Su et al. (2023, IEEE TNNLS) | SNN seg energy | Frame/Event | Khung đo SynOps, tiết kiệm energy | Cách đo SynOps/energy chuẩn |
| Spiking CGNet — (2023) | SNN seg nhẹ | Cityscapes | Lightweight SNN segmenter | Tham khảo kiến trúc gọn |
| Boundary-Aware MS-SNN (2025) | SNN seg | — | Multi-scale + biên | Cải tiến kiến trúc |
| SNN SAR Ship (2022) | SNN trên SAR | SAR ship | SNN chạy trên SAR | Encode SAR→spike |
| SNN SAR Phase Unwrap (2025) | SNN SAR energy | InSAR | Khung energy-efficient | Lý luận energy |

---

## 5. Kế hoạch thí nghiệm tối thiểu để paper được nhận

**Bắt buộc có:**
- [ ] **Setup B (chính):** SNN-Flood train trên hand-labeled train split (60%), báo IoU/F1 trên hand-labeled test split (20%, ~89 chip).
- [ ] **Setup A (phụ):** SNN-Flood train trên weak-label Otsu, test trên hand-label — mục weak-supervised ablation.
- [ ] U-Net baseline cùng điều kiện (cả 2 setup).
- [ ] **So sánh đa mô hình** (Otsu + CNN mạnh + CNN tiết kiệm/INT8 + ANN2SNN) trên accuracy/FLOPs/params/energy.
- [ ] Bảng so sánh **accuracy vs energy (SynOps)** + biểu đồ scatter accuracy–energy.
- [ ] ≥ 1 ablation (số timestep T).
- [ ] ≥ 3 hình qualitative.

**Nên có (tăng cơ hội nhận):**
- [ ] Thêm baseline Transformer SOTA (SegFormer/Swin-UNet).
- [ ] Ablation encoding & loss.
- [ ] Release code công khai.

**Bảng head-to-head đa mô hình (điền sau khi chạy):**

| Model | Nhóm | IoU | F1 | Params (M) | FLOPs/SynOps | Energy (mJ) | × tiết kiệm |
|-------|------|-----|----|-----------|--------------|-------------|-------------|
| Otsu threshold | Cổ điển | — | — | 0 | ~0 | ~0 | — |
| U-Net | CNN mạnh | — | — | — | FLOPs | — | 1× (ref) |
| DeepLabv3+ | CNN mạnh | — | — | — | FLOPs | — | — |
| MobileNet-UNet | CNN nhẹ | — | — | — | FLOPs | — | — |
| U-Net INT8 (quantized) | CNN nén | — | — | — | FLOPs | — | — |
| ANN2SNN (conversion) | SNN | — | — | — | SynOps | — | — |
| SegFormer (SOTA, nên có) | Transformer | — | — | — | FLOPs | — | — |
| **SNN-Flood (ours)** | SNN | — | — | — | SynOps | — | **X×** |

---

## 6. Lộ trình thời gian (~2–3 tháng)

| Tuần | Việc |
|------|------|
| 1 | Chốt hội nghị + deadline; viết claim & contributions |
| 2 | Pipeline data + official split; dựng U-Net baseline |
| 3–5 | Xây & train Spiking U-Net; debug surrogate gradient |
| 6 | Module đo energy (SynOps); chạy baseline đối chiếu |
| 7 | Ablation + qualitative + hoàn thiện bảng kết quả |
| 8–9 | Viết bản nháp đầy đủ |
| 10 | Sửa tiếng Anh, format template, chuẩn bị code release, nộp |

---

## 7. Rủi ro & cách giảm

| Rủi ro | Giảm thiểu |
|--------|-----------|
| SNN accuracy thua xa U-Net | Tăng T, dùng hybrid ANN-SNN, hoặc ANN2SNN conversion |
| Nhãn Otsu weak làm trần thấp | Báo cáo rõ; thử fine-tune trên hand-label (k-fold) |
| Energy claim bị reviewer nghi ngờ | Dùng công thức SynOps chuẩn, trích paper neuromorphic, nêu giả định rõ |
| Mất cân bằng lớp (ít pixel nước) | Dice/weighted loss, báo cáo cả Recall lớp nước |

---

*Tài liệu làm việc — cập nhật khi có kết quả thực nghiệm.*
