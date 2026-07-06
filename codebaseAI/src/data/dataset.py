"""
Dataset DÙNG CHUNG cho MỌI model (CNN / Transformer / SNN).

Đọc split CSV (dataset/splits/{train,val,test}.csv) → ảnh S1Hand + nhãn 3 lớp
Label3Class, chuẩn hóa dB→[0,1], (tùy chọn) random-crop + augment khi train.

Nhãn: -1 bỏ qua · 0 nền · 1 nước thường trực · 2 nước lũ (flood).
Cùng một Dataset này cấp dữ liệu cho tất cả model ⇒ đảm bảo "cùng 1 protocol".
"""
import os
import csv
import random

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader

S1_CLIP_MIN, S1_CLIP_MAX = -50.0, 0.0


class Sen1FloodsDataset(Dataset):
    def __init__(self, data_root, split, crop_size=None,
                 per_channel_norm=False, augment=False):
        """
        data_root: đường dẫn tới thư mục `dataset/` (chứa splits/, S1Hand/, Label3Class/).
        split: 'train' | 'val' | 'test'.
        crop_size: cắt ngẫu nhiên (chỉ dùng khi train); None = giữ nguyên 512.
        per_channel_norm: chuẩn hóa min/max riêng từng kênh VV/VH (ablation).
        """
        self.data_root = data_root
        self.crop_size = crop_size
        self.per_channel_norm = per_channel_norm
        self.augment = augment

        csv_path = os.path.join(data_root, "splits", f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Không thấy {csv_path}. Chạy `python dataset/make_splits.py` trước.")
        self.items = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.items.append((row["chip_id"], row["s1"], row["label3"]))

    def __len__(self):
        return len(self.items)

    def _normalize(self, s1):
        s1 = np.nan_to_num(s1, nan=S1_CLIP_MIN, posinf=S1_CLIP_MAX, neginf=S1_CLIP_MIN)
        s1 = np.clip(s1, S1_CLIP_MIN, S1_CLIP_MAX)
        if self.per_channel_norm:
            out = np.empty_like(s1)
            for c in range(s1.shape[0]):
                lo, hi = s1[c].min(), s1[c].max()
                out[c] = (s1[c] - lo) / (hi - lo + 1e-6)
            return out
        return (s1 - S1_CLIP_MIN) / (S1_CLIP_MAX - S1_CLIP_MIN)

    def __getitem__(self, idx):
        cid, s1_rel, lab_rel = self.items[idx]
        with rasterio.open(os.path.join(self.data_root, s1_rel)) as src:
            s1 = src.read().astype(np.float32)          # (2, H, W)
        with rasterio.open(os.path.join(self.data_root, lab_rel)) as src:
            label = src.read(1).astype(np.int64)        # (H, W) ∈ {-1,0,1,2}

        # Pixel NaN trong ảnh SAR → nhãn -1 (loss bỏ qua)
        nan = np.isnan(s1).any(axis=0)
        if nan.any():
            label[nan] = -1

        s1 = self._normalize(s1)
        if self.crop_size:
            s1, label = self._random_crop(s1, label, self.crop_size)
        if self.augment:
            s1, label = self._augment(s1, label)

        s1 = torch.from_numpy(np.ascontiguousarray(s1))
        label = torch.from_numpy(np.ascontiguousarray(label))
        return s1, label, cid

    @staticmethod
    def _random_crop(s1, label, size):
        _, H, W = s1.shape
        if H <= size or W <= size:
            return s1, label
        top = random.randint(0, H - size)
        left = random.randint(0, W - size)
        return s1[:, top:top + size, left:left + size], label[top:top + size, left:left + size]

    @staticmethod
    def _augment(s1, label):
        if random.random() < 0.5:              # lật ngang
            s1 = s1[:, :, ::-1]; label = label[:, ::-1]
        if random.random() < 0.5:              # lật dọc
            s1 = s1[:, ::-1, :]; label = label[::-1, :]
        return np.ascontiguousarray(s1), np.ascontiguousarray(label)


def get_dataloader(data_root, split, batch_size=8, num_workers=4, **ds_kwargs):
    ds = Sen1FloodsDataset(data_root, split, **ds_kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=num_workers, pin_memory=True, drop_last=(split == "train"))
