import os
import glob
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# SAR backscatter clip range (dB). Raw data spans ~[-77, +29] but extreme
# outliers are edge artifacts; [-50, 0] covers the informative flood signal.
S1_CLIP_MIN = -50.0
S1_CLIP_MAX = 0.0


class Sen1Floods11Dataset(Dataset):
    def __init__(self, s1_dir, label_dir, s1_suffix, label_suffix, normalize=True):
        self.s1_dir = s1_dir
        self.label_dir = label_dir
        self.normalize = normalize
        self.s1_suffix = s1_suffix
        self.label_suffix = label_suffix

        s1_files = sorted(glob.glob(os.path.join(s1_dir, f"*{s1_suffix}.tif")))
        chip_ids = [
            os.path.basename(f).replace(f"{s1_suffix}.tif", "")
            for f in s1_files
        ]

        valid_ids = []
        skipped_no_label = 0
        skipped_all_nan = 0
        for cid in chip_ids:
            label_path = os.path.join(self.label_dir, f"{cid}{self.label_suffix}.tif")
            if not os.path.exists(label_path):
                skipped_no_label += 1
                continue
            # Drop chips where every S1 pixel is NaN (no usable signal)
            s1_path = os.path.join(self.s1_dir, f"{cid}{self.s1_suffix}.tif")
            with rasterio.open(s1_path) as src:
                s1 = src.read().astype(np.float32)
            if np.all(np.isnan(s1)):
                skipped_all_nan += 1
                continue
            valid_ids.append(cid)

        if skipped_no_label > 0:
            print(f"[Warning] Skipped {skipped_no_label} chips with no matching label.")
        if skipped_all_nan > 0:
            print(f"[Warning] Skipped {skipped_all_nan} fully-NaN chips (no valid SAR signal).")
        self.chip_ids = valid_ids

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        chip_id = self.chip_ids[idx]
        s1_path = os.path.join(self.s1_dir, f"{chip_id}{self.s1_suffix}.tif")
        label_path = os.path.join(self.label_dir, f"{chip_id}{self.label_suffix}.tif")

        with rasterio.open(s1_path) as src:
            s1 = src.read().astype(np.float32)

        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int64)

        # Mask label at NaN pixels so loss ignores them (ignore_index=-1)
        nan_mask = np.isnan(s1).any(axis=0)
        if nan_mask.any():
            label[nan_mask] = -1

        if self.normalize:
            s1 = np.nan_to_num(s1, nan=S1_CLIP_MIN, posinf=S1_CLIP_MAX, neginf=S1_CLIP_MIN)
            s1 = np.clip(s1, S1_CLIP_MIN, S1_CLIP_MAX)
            s1 = (s1 - S1_CLIP_MIN) / (S1_CLIP_MAX - S1_CLIP_MIN)

        s1_tensor = torch.from_numpy(s1)
        label_tensor = torch.from_numpy(label)

        return s1_tensor, label_tensor, chip_id


def get_train_val_datasets(data_root, val_ratio=0.1, seed=42):
    full_train = Sen1Floods11Dataset(
        s1_dir=os.path.join(data_root, "S1Weak"),
        label_dir=os.path.join(data_root, "S1OtsuLabelWeak"),
        s1_suffix="_S1Weak",
        label_suffix="_S1OtsuLabelWeak",
    )
    n_val = int(len(full_train) * val_ratio)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)
    return train_set, val_set


def get_test_dataset(data_root):
    return Sen1Floods11Dataset(
        s1_dir=os.path.join(data_root, "S1Hand"),
        label_dir=os.path.join(data_root, "LabelHand"),
        s1_suffix="_S1Hand",
        label_suffix="_LabelHand",
    )


if __name__ == "__main__":
    DATA_ROOT = "."

    train_set, val_set = get_train_val_datasets(DATA_ROOT, val_ratio=0.1)
    test_set = get_test_dataset(DATA_ROOT)

    print(f"Train: {len(train_set)} chip")
    print(f"Val:   {len(val_set)} chip")
    print(f"Test:  {len(test_set)} chip (HandLabeled)")

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    s1_batch, label_batch, chip_ids = next(iter(train_loader))
    print(f"\nBatch shape - S1: {s1_batch.shape}, Label: {label_batch.shape}")
    print(f"Chip IDs trong batch: {chip_ids}")
    print(f"S1 value range sau normalize: [{s1_batch.min():.3f}, {s1_batch.max():.3f}]")
    n_ignore = (label_batch == -1).sum().item()
    total = label_batch.numel()
    print(f"Masked pixels (label=-1) in batch: {n_ignore}/{total} ({n_ignore/total*100:.1f}%)")
