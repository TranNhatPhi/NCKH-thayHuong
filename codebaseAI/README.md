# SNN-Flood — Codebase AI

Mã nguồn huấn luyện & đánh giá cho đề tài **SNN-Flood** (segmentation 3 lớp flood-on-land trên Sentinel-1 SAR).

> **Nguyên tắc lõi (theo yêu cầu thầy):** MỌI model — CNN, Transformer, SNN — dùng **chung một** `dataset`, `losses`, `metrics`, `train.py`, `evaluate.py`. Chỉ phần **model** là khác. Nhờ vậy tất cả được đánh giá bằng **đúng một protocol**, không "mỗi thằng một kiểu".

## Bố cục thư mục

```
codebaseAI/
├── train.py                 # ① vòng train CHUNG cho mọi model học được
├── evaluate.py              # ② eval CHUNG (cùng 1 protocol) + đo năng lượng
├── requirements.txt
├── configs/                 # mỗi thí nghiệm = 1 file YAML (kế thừa base.yaml)
│   ├── base.yaml            #   cấu hình chung (data, loss, lr, epochs…)
│   ├── unet.yaml · segformer_b2.yaml · spiking_unet.yaml · otsu.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py       # đọc splits/*.csv + Label3Class, chuẩn hóa, crop  ✅
│   │   └── encoding.py      # SAR→spike (direct / rate) cho SNN                ✅
│   ├── models/              # ← MỖI MODEL MỘT FILE, wire vào get_model()
│   │   ├── __init__.py      #   factory get_model(name) — điểm vào duy nhất    ✅
│   │   ├── unet.py          #   U-Net (ANN) tự chứa, chỉ cần torch            ✅
│   │   ├── unetpp.py        #   U-Net++ (smp, ResNet34)                        ✅*
│   │   ├── deeplabv3.py     #   DeepLabV3 (smp, ResNet34)                      ✅*
│   │   ├── segformer.py     #   SegFormer b0/b1/b2 (smp, MiT)                  ✅*
│   │   ├── spiking_unet.py  #   ★ Spiking U-Net (SNN) — ĐÓNG GÓP LÕI          🟡
│   │   ├── ann2snn.py       #   baseline chuyển đổi ANN→SNN                    🟡
│   │   └── otsu.py          #   sàn cổ điển (không học)                        ✅*
│   ├── losses.py            # CE(ignore -1) + Dice + Focal                     ✅
│   ├── metrics.py           # IoU/mIoU, F1/Dice, P/R (+ pixel-acc phụ)         ✅
│   ├── energy.py            # FLOPs (ANN) + SynOps (SNN) → mJ                  ✅/🟡
│   └── utils.py             # seed, device, config, checkpoint, sliding-window ✅
└── runs/                    # output: checkpoint + test_metrics.json (gitignored)
```
✅ chạy được · ✅* cần cài thư viện (smp/skimage) · 🟡 skeleton, cần kiểm thử khi chạy thật.

## Danh mục model (rõ ràng model nào ở đâu)

| Nhóm | Model | File | Vai trò | Trạng thái |
|------|-------|------|---------|:--:|
| Cổ điển | Otsu | `models/otsu.py` | sàn accuracy, ~0 năng lượng | ✅* |
| CNN mạnh | U-Net | `models/unet.py` | baseline accuracy chính | ✅ |
| CNN mạnh | U-Net++ | `models/unetpp.py` | baseline | ✅* |
| CNN mạnh | DeepLabV3 | `models/deeplabv3.py` | baseline | ✅* |
| Transformer | SegFormer-b0/b1/b2 | `models/segformer.py` | mốc accuracy trần | ✅* |
| **SNN (ours)** | **Spiking U-Net** | `models/spiking_unet.py` | **điểm Pareto accuracy–energy** | 🟡 |
| SNN | ANN2SNN | `models/ann2snn.py` | so sánh nội bộ họ SNN | 🟡 |

Thêm model mới = thêm 1 file trong `models/` + 1 dòng trong `get_model()` + 1 config. Không đụng train/eval.

## Cách chạy

```bash
cd codebaseAI
pip install -r requirements.txt

# Huấn luyện (đổi config là đổi model, mọi thứ khác giữ nguyên)
python train.py --config configs/unet.yaml
python train.py --config configs/spiking_unet.yaml

# Đánh giá trên test (cùng protocol cho tất cả)
python evaluate.py --config configs/unet.yaml
python evaluate.py --config configs/otsu.yaml      # baseline không cần train
```

Kết quả lưu ở `runs/<name>/best.pt` và `runs/<name>/test_metrics.json`.

### Chạy trên Mac Apple Silicon (M-series, ví dụ M4 Air 16GB)
Code tự dùng GPU **MPS (Metal)**. Lưu ý:
```bash
# cho phép op chưa hỗ trợ trên MPS chạy tạm bằng CPU (quan trọng cho SNN)
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/mac_smoke.yaml
```
- **16GB unified memory** dùng chung CPU+GPU → giữ `batch_size` nhỏ (2–4), `crop_size` ≤ 256; đóng bớt app khác khi train.
- **SNN (spiking_unet) KHÔNG nên train thật trên máy này** (BPTT×T ngốn RAM, MPS+SpikingJelly rủi ro) → dùng **cloud GPU (Colab/Kaggle)** cho các run chính.
- Nếu DataLoader treo, đặt `num_workers: 0`.
- Máy này chỉ nên dùng để **dev + smoke-test** (config `mac_smoke.yaml`).

## Metric (thầy chốt 04/07)

- **Chính:** `flood_IoU` (lớp 2), IoU/mIoU, F1(=Dice), Precision, Recall — theo lớp.
- **Phụ:** `pixel_accuracy` (vẫn báo, không làm chính vì mất cân bằng lớp).
- **Protocol flood-chip:** `metrics.py` xuất cả `flood_IoU` (gộp pixel) lẫn `flood_IoU_perchip_mean` (trung bình trên 78/87 chip có flood) — **chờ thầy chốt dùng cái nào làm số chính**.

## Lộ trình dựng tiếp (theo thứ tự)

1. Cài môi trường + **chạy `train.py --config configs/unet.yaml`** → mốc U-Net baseline (chạy được ngay).
2. Kiểm thử **`spiking_unet.py`** (forward/backward, tinh chỉnh T & ngưỡng LIF).
3. Hoàn thiện **`energy.py`** phần SynOps (hook đếm spike) → bảng accuracy–energy.
4. Chạy nốt baseline (U-Net++, DeepLabV3, SegFormer-b2, ANN2SNN) → ablation T → biểu đồ Pareto.

> Dữ liệu đã sẵn sàng ở `../dataset/` (splits 271/87/87, Label3Class 3 lớp). Xem `../SNN-Flood_Pipeline.md`.
