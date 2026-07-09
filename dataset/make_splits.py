"""
Bước 5 — Chia hand-labeled thành train/val/test 60/20/20.

Nâng cấp (03/07/2026) để xử lý 2 vấn đề trong rà soát tiền xử lý:
  * Phân tầng theo (VÙNG × có/không nước)  -> mọi tập đều có đủ chip chứa nước/lũ.
  * Gom các chip CHỒNG NHAU về không gian vào cùng một tập -> chống rò rỉ (spatial leakage).
  * Loại chip toàn NaN.

Chạy:  python make_splits.py
Kết quả: dataset/splits/{train,val,test}.csv  (cột: chip_id, region, s1, label, label3)
"""
import os
import glob
import csv
import random
import argparse
from collections import defaultdict, Counter

import numpy as np
import rasterio

S1_DIR, S1_SUF = "S1Hand", "_S1Hand"
LAB_DIR, LAB_SUF = "LabelHand", "_LabelHand"
LAB3_DIR, LAB3_SUF = "Label3Class", "_Label3Class"


def region_of(chip_id):
    return chip_id.split("_")[0]


def list_chips(root):
    files = sorted(glob.glob(os.path.join(root, S1_DIR, f"*{S1_SUF}.tif")))
    return [os.path.basename(f).replace(f"{S1_SUF}.tif", "")
            for f in files
            if os.path.exists(os.path.join(root, LAB_DIR,
                              f"{os.path.basename(f).replace(f'{S1_SUF}.tif','')}{LAB_SUF}.tif"))]


def load_meta(root, chips):
    """Đọc 1 lần: has_water, bbox, crs, toàn-NaN cho từng chip."""
    meta = {}
    dropped_nan = []
    for cid in chips:
        with rasterio.open(os.path.join(root, LAB_DIR, f"{cid}{LAB_SUF}.tif")) as src:
            lab = src.read(1)
        with rasterio.open(os.path.join(root, S1_DIR, f"{cid}{S1_SUF}.tif")) as src:
            small = src.read(1, out_shape=(64, 64)).astype(np.float32)
            b, crs = src.bounds, str(src.crs)
        if np.all(np.isnan(small)):
            dropped_nan.append(cid)
            continue
        meta[cid] = dict(region=region_of(cid), has_water=bool((lab == 1).any()),
                         bounds=b, crs=crs)
    return meta, dropped_nan


def group_overlaps(meta):
    """Union-find: gom chip cùng vùng có bbox chồng nhau thành 1 nhóm nguyên tử."""
    chips = list(meta)
    parent = {c: c for c in chips}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_region = defaultdict(list)
    for c in chips:
        by_region[meta[c]["region"]].append(c)
    for lst in by_region.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = meta[lst[i]], meta[lst[j]]
                if a["crs"] != b["crs"]:
                    continue
                b1, b2 = a["bounds"], b["bounds"]
                if not (b1.right <= b2.left or b2.right <= b1.left or
                        b1.top <= b2.bottom or b2.top <= b1.bottom):
                    parent[find(lst[i])] = find(lst[j])
    groups = defaultdict(list)
    for c in chips:
        groups[find(c)].append(c)
    return list(groups.values())


def stratified_group_split(groups, meta, val_ratio, test_ratio, seed):
    """Chia theo NHÓM (giữ nguyên vẹn), phân tầng theo (vùng × nhóm-có-nước)."""
    strata = defaultdict(list)
    for g in groups:
        region = meta[g[0]]["region"]
        has_water = any(meta[c]["has_water"] for c in g)
        strata[(region, has_water)].append(g)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(strata):
        gl = sorted(strata[key], key=lambda g: sorted(g)[0])
        rng.shuffle(gl)
        n = len(gl)
        n_test = round(n * test_ratio)
        n_val = round(n * val_ratio)
        n_train = n - n_val - n_test
        for g in gl[:n_train]:
            splits["train"] += g
        for g in gl[n_train:n_train + n_val]:
            splits["val"] += g
        for g in gl[n_train + n_val:]:
            splits["test"] += g
    return splits


def write_csv(root, split_name, chip_ids):
    out_dir = os.path.join(root, "splits")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split_name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chip_id", "region", "s1", "label", "label3"])
        for cid in sorted(chip_ids):
            w.writerow([cid, region_of(cid),
                        f"{S1_DIR}/{cid}{S1_SUF}.tif",
                        f"{LAB_DIR}/{cid}{LAB_SUF}.tif",
                        f"{LAB3_DIR}/{cid}{LAB3_SUF}.tif"])
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.2)   # 60/20/20 (thầy chốt 09/07: test 45 chip quá ít)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    chips = list_chips(args.root)
    print(f"Chip có đủ ảnh + nhãn: {len(chips)}")
    meta, dropped = load_meta(args.root, chips)
    if dropped:
        print(f"Loại {len(dropped)} chip toàn NaN: {dropped}")

    groups = group_overlaps(meta)
    n_multi = sum(1 for g in groups if len(g) > 1)
    print(f"Chip hợp lệ: {len(meta)} | nhóm không gian: {len(groups)} "
          f"(trong đó {n_multi} nhóm gộp ≥2 chip chồng nhau)")

    splits = stratified_group_split(groups, meta, args.val_ratio, args.test_ratio, args.seed)

    regions = sorted({meta[c]["region"] for c in meta})
    print(f"\n{'Vùng':12s} {'train':>6s} {'val':>5s} {'test':>5s} {'tổng':>6s}"
          f"  | {'chip có nước (tr/va/te)':>24s}")
    print("-" * 68)
    per = {s: Counter(meta[c]["region"] for c in splits[s]) for s in splits}
    perw = {s: Counter(meta[c]["region"] for c in splits[s] if meta[c]["has_water"]) for s in splits}
    for r in regions:
        tr, va, te = per["train"][r], per["val"][r], per["test"][r]
        wtr, wva, wte = perw["train"][r], perw["val"][r], perw["test"][r]
        print(f"{r:12s} {tr:6d} {va:5d} {te:5d} {tr+va+te:6d}  | {wtr:8d}/{wva:d}/{wte:d}")
    print("-" * 68)
    n_tr, n_va, n_te = (len(splits[s]) for s in ("train", "val", "test"))
    tot = n_tr + n_va + n_te
    wtot = {s: sum(1 for c in splits[s] if meta[c]["has_water"]) for s in splits}
    print(f"{'TỔNG':12s} {n_tr:6d} {n_va:5d} {n_te:5d} {tot:6d}"
          f"  | {wtot['train']:8d}/{wtot['val']:d}/{wtot['test']:d}")
    print(f"Tỷ lệ:       {n_tr/tot:6.1%} {n_va/tot:5.1%} {n_te/tot:5.1%}")

    for s in ("train", "val", "test"):
        p = write_csv(args.root, s, splits[s])
        print(f"Đã ghi {p} ({len(splits[s])} chip)")


if __name__ == "__main__":
    main()
