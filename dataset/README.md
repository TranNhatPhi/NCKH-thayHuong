# Sen1Floods11 Dataset

Bộ dữ liệu phát hiện lũ lụt từ ảnh radar Sentinel-1 SAR, dựa trên Sen1Floods11 (Bonafilia et al., 2020).

## Cấu trúc thư mục

```
dataset/
├── S1Hand/                   # Ảnh Sentinel-1 SAR (test set, nhãn thủ công)
├── LabelHand/                # Nhãn phân đoạn thủ công (test set)
├── S1Weak/                   # Ảnh Sentinel-1 SAR (train/val set, nhãn yếu)
├── S1OtsuLabelWeak/          # Nhãn tự động bằng ngưỡng Otsu (train/val set)
└── sen1floods11_dataset.py   # PyTorch Dataset + DataLoader
```

## Thống kê

| Tập dữ liệu | Số chip (sau lọc) | Nguồn nhãn | Mục đích |
|-------------|-------------------|------------|----------|
| Train       | ~3 936            | Otsu (tự động) | Huấn luyện |
| Val         | ~438              | Otsu (tự động) | Kiểm định |
| Test        | 446               | Thủ công (hand-labeled) | Đánh giá cuối |

> **Lọc tự động:** 10 chip trong S1Weak bị loại vì 100% pixel là NaN (không có tín hiệu SAR hợp lệ).

Mỗi chip kích thước **512 × 512 px**.

## Khu vực địa lý

Bolivia, Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri Lanka, USA.

## Định dạng dữ liệu

- **Ảnh SAR (S1):** GeoTIFF, 2 kênh (VV, VH), đơn vị dB.
  - Raw range thực tế: khoảng [-77, +29] dB (bao gồm outlier ở rìa ảnh).
  - ~8.5% chip trong S1Weak và ~10.3% chip trong S1Hand có pixel NaN (no-data).
- **Nhãn:** GeoTIFF, 1 kênh, giá trị nguyên:
  - `0` — không phải nước
  - `1` — nước / lũ lụt
  - `-1` — không có dữ liệu (bỏ qua khi tính loss)

## Vấn đề dữ liệu đã xử lý

| Vấn đề | Mô tả | Xử lý |
|--------|-------|-------|
| Chip toàn NaN | 10 chip trong S1Weak: 100% pixel NaN | Lọc bỏ khi khởi tạo dataset |
| NaN-label mismatch | Pixel NaN trong S1 nhưng label = 0 hoặc 1 | Set label = -1 tại vị trí NaN → loss bỏ qua |
| Outlier dB | Giá trị ngoài [-50, 0] dB (thường là nhiễu rìa ảnh) | Clip về [-50, 0] trước normalize |

## Tiền xử lý (normalize)

```python
# 1. Mask NaN pixels ra khỏi loss
nan_mask = np.isnan(s1).any(axis=0)
label[nan_mask] = -1          # CrossEntropyLoss với ignore_index=-1 sẽ bỏ qua

# 2. Normalize S1 về [0, 1]
s1 = np.nan_to_num(s1, nan=-50.0, posinf=0.0, neginf=-50.0)
s1 = np.clip(s1, -50, 0)
s1 = (s1 + 50) / 50.0
```

## Sử dụng nhanh

```python
from sen1floods11_dataset import get_train_val_datasets, get_test_dataset
from torch.utils.data import DataLoader

train_set, val_set = get_train_val_datasets(".", val_ratio=0.1, seed=42)
test_set = get_test_dataset(".")

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)

for s1, label, chip_id in train_loader:
    # s1:    (B, 2, 512, 512), float32, range [0, 1]
    # label: (B, 512, 512),   int64,   values {-1, 0, 1}
    # Dùng ignore_index=-1 trong loss để bỏ qua pixel no-data
    pass
```

### Ví dụ loss function

```python
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
```

## Nguồn gốc

- Paper: Bonafilia et al., *"Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1"*, CVPR EarthVision 2020.
- Dataset gốc: [github.com/cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)
