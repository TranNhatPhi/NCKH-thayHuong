"""
Bước 1-4 — Dựng nhãn 3 lớp flood-on-land.

Ghép LabelHand (nước nhị phân) với JRC permanent-water để tách:
    -1 = bỏ qua (NaN / no-data)
     0 = nền
     1 = nước thường trực (permanent, từ JRC)   -> ưu tiên cao
     2 = nước lũ (flood) = water ∧ ¬permanent

Cần thư mục dataset/JRCWaterHand/ (tải từ dataset gốc Sen1Floods11).
Chạy:  python build_3class_labels.py                # tự nhận diện encoding JRC
       python build_3class_labels.py --jrc_threshold 50
Kết quả: dataset/Label3Class/{id}_Label3Class.tif
"""
import os
import glob
import argparse
from collections import Counter

import numpy as np
import rasterio

S1_DIR, S1_SUF = "S1Hand", "_S1Hand"
LAB_DIR, LAB_SUF = "LabelHand", "_LabelHand"
JRC_DIR, JRC_SUF = "JRCWaterHand", "_JRCWaterHand"
OUT_DIR, OUT_SUF = "Label3Class", "_Label3Class"


def list_chips(root):
    files = sorted(glob.glob(os.path.join(root, S1_DIR, f"*{S1_SUF}.tif")))
    chips = []
    for f in files:
        cid = os.path.basename(f).replace(f"{S1_SUF}.tif", "")
        if os.path.exists(os.path.join(root, LAB_DIR, f"{cid}{LAB_SUF}.tif")):
            chips.append(cid)
    return chips


def detect_jrc_mode(root, chips, n_sample=20):
    """Xem JRC là nhị phân (0/1) hay occurrence (0-100) để chọn ngưỡng permanent."""
    gmax = 0.0
    vals = Counter()
    for cid in chips[:n_sample]:
        p = os.path.join(root, JRC_DIR, f"{cid}{JRC_SUF}.tif")
        with rasterio.open(p) as src:
            a = src.read(1, out_shape=(64, 64)).astype(np.float32)
        a = a[np.isfinite(a)]
        if a.size:
            gmax = max(gmax, float(a.max()))
            vals.update(np.unique(a).astype(int).tolist())
    return gmax, vals


def build_label3(s1, water, jrc, perm_mask_fn):
    perm = perm_mask_fn(jrc)                       # nước thường trực
    label3 = np.zeros(water.shape, dtype=np.int16)  # 0 nền
    label3[perm] = 1                                # 1 permanent (ưu tiên)
    label3[(water == 1) & (~perm)] = 2              # 2 flood = water ∧ ¬permanent
    nan_mask = np.isnan(s1).any(axis=0)             # -1 bỏ qua
    label3[nan_mask] = -1
    label3[water == -1] = -1
    return label3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--jrc_threshold", type=float, default=50.0,
                    help="Ngưỡng occurrence coi là permanent (chỉ dùng khi JRC dạng 0-100)")
    args = ap.parse_args()

    chips = list_chips(args.root)
    print(f"Chip có đủ ảnh S1 + LabelHand: {len(chips)}")

    jrc_dir = os.path.join(args.root, JRC_DIR)
    if not os.path.isdir(jrc_dir):
        print("\n[CHẶN] Chưa có thư mục JRCWaterHand/ — không dựng được nhãn 3 lớp.")
        print("  Cần tải lớp JRC permanent-water (446 chip, cùng lưới với S1Hand) từ")
        print("  dataset gốc Sen1Floods11, đặt vào: dataset/JRCWaterHand/")
        print(f"  Định dạng tên: {{chip_id}}{JRC_SUF}.tif  (ví dụ India_842775{JRC_SUF}.tif)")
        print(f"  Sau khi có, chạy lại: python build_3class_labels.py")
        return

    gmax, vals = detect_jrc_mode(args.root, chips)
    if gmax <= 1.0:
        print(f"JRC dạng NHỊ PHÂN (max={gmax:g}); permanent = jrc > 0")
        perm_fn = lambda j: j > 0
    else:
        print(f"JRC dạng OCCURRENCE (max={gmax:g}); permanent = jrc >= {args.jrc_threshold:g}")
        print(f"  (giá trị JRC quan sát trên mẫu: {sorted(vals)[:15]} ...)")
        perm_fn = lambda j: j >= args.jrc_threshold

    out_dir = os.path.join(args.root, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    total = Counter()
    for i, cid in enumerate(chips, 1):
        with rasterio.open(os.path.join(args.root, S1_DIR, f"{cid}{S1_SUF}.tif")) as src:
            s1 = src.read().astype(np.float32)
        with rasterio.open(os.path.join(args.root, LAB_DIR, f"{cid}{LAB_SUF}.tif")) as src:
            water = src.read(1).astype(np.int16)
            profile = src.profile
        jrc_path = os.path.join(args.root, JRC_DIR, f"{cid}{JRC_SUF}.tif")
        if not os.path.exists(jrc_path):
            print(f"  [bỏ qua] thiếu JRC cho {cid}")
            continue
        with rasterio.open(jrc_path) as src:
            jrc = src.read(1).astype(np.float32)

        label3 = build_label3(s1, water, jrc, perm_fn)
        total.update(label3.flatten().tolist())

        profile.update(dtype=rasterio.int16, count=1, nodata=-1)
        with rasterio.open(os.path.join(out_dir, f"{cid}{OUT_SUF}.tif"), "w", **profile) as dst:
            dst.write(label3.astype(np.int16), 1)
        if i % 50 == 0:
            print(f"  ...đã xử lý {i}/{len(chips)}")

    n = sum(total.values())
    print(f"\nHoàn tất -> {out_dir}/")
    print("Phân bố pixel toàn tập:")
    for k in (-1, 0, 1, 2):
        print(f"  lớp {k:2d}: {total[k]:>12,d}  ({total[k]/n:6.2%})")


if __name__ == "__main__":
    main()
