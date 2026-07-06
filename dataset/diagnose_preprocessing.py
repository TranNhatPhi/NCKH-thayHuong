"""Chẩn đoán chất lượng dữ liệu trước tiền xử lý — tìm vấn đề để review với thầy."""
import glob
import os
from collections import defaultdict, Counter

import numpy as np
import rasterio

S1 = sorted(glob.glob("S1Hand/*_S1Hand.tif"))
def cid_of(f): return os.path.basename(f).replace("_S1Hand.tif", "")
def region_of(c): return c.split("_")[0]

label_values = Counter()
water_frac = {}          # chip -> % pixel nước (trên pixel hợp lệ)
zero_water, low_water = [], []
nan_any, nan_cov = 0, []
vv_vals, vh_vals = [], []
clip_lo = clip_hi = tot_valid = 0

for f in S1:
    c = cid_of(f)
    lf = f"LabelHand/{c}_LabelHand.tif"
    if not os.path.exists(lf):
        continue
    with rasterio.open(lf) as src:
        lab = src.read(1)
    label_values.update(np.unique(lab).tolist())
    valid = lab >= 0
    nv = int(valid.sum())
    wf = float((lab == 1).sum()) / nv if nv else 0.0
    water_frac[c] = wf
    if (lab == 1).sum() == 0:
        zero_water.append(c)
    elif wf < 0.01:
        low_water.append(c)

    with rasterio.open(f) as src:
        arr = src.read(out_shape=(2, 128, 128)).astype(np.float32)  # decimated
    nmask = np.isnan(arr).any(axis=0)
    frac_nan = float(nmask.mean())
    if frac_nan > 0:
        nan_any += 1
        nan_cov.append(frac_nan)
    fin = np.isfinite(arr)
    vv = arr[0][np.isfinite(arr[0])]; vh = arr[1][np.isfinite(arr[1])]
    vv_vals.append(vv[::10]); vh_vals.append(vh[::10])
    allv = arr[fin]
    tot_valid += allv.size
    clip_lo += int((allv < -50).sum()); clip_hi += int((allv > 0).sum())

n = len(water_frac)
print(f"=== 1. NHÃN ===")
print(f"Giá trị nhãn xuất hiện: {sorted(label_values)}")
print(f"\n=== 2. MẤT CÂN BẰNG LỚP (nhãn nhị phân hiện có) ===")
wf = np.array(list(water_frac.values()))
print(f"% pixel nước / chip:  trung vị {np.median(wf):.2%} | trung bình {wf.mean():.2%} | max {wf.max():.2%}")
print(f"Chip KHÔNG có pixel nước: {len(zero_water)}  {zero_water[:6]}{'...' if len(zero_water)>6 else ''}")
print(f"Chip có <1% nước:        {len(low_water)}")
print(f"  -> Sau khi tách nước lũ (trừ nước thường trực), tỷ lệ lớp flood sẽ CÒN THẤP HƠN nữa.")
print(f"\n=== 3. NaN CỤC BỘ ===")
print(f"Chip có ít nhất 1 pixel NaN: {nan_any}/{n} ({nan_any/n:.1%})")
if nan_cov:
    nc = np.array(nan_cov)
    print(f"Trong các chip đó, % diện tích NaN: trung vị {np.median(nc):.1%} | max {nc.max():.1%}")
print(f"\n=== 4. DẢI GIÁ TRỊ dB & MẤT MÁT DO CLIP [-50,0] ===")
vv = np.concatenate(vv_vals); vh = np.concatenate(vh_vals)
for name, v in [("VV", vv), ("VH", vh)]:
    p = np.percentile(v, [0.5, 50, 99.5])
    print(f"  {name}: p0.5={p[0]:.1f}  median={p[1]:.1f}  p99.5={p[2]:.1f}  (min={v.min():.1f}, max={v.max():.1f})")
print(f"Pixel bị clip: dưới -50dB = {clip_lo/tot_valid:.3%} | trên 0dB = {clip_hi/tot_valid:.3%}")

print(f"\n=== 5. RÒ RỈ KHÔNG GIAN: chip trong cùng vùng có chồng/sát nhau? ===")
# đọc bbox + crs cho từng chip
boxes = defaultdict(list)  # region -> list (cid, crs, bounds, width_map)
for f in S1:
    c = cid_of(f)
    with rasterio.open(f) as src:
        b = src.bounds
        boxes[region_of(c)].append((c, str(src.crs), b, b.right - b.left))
for region in sorted(boxes):
    items = boxes[region]
    crss = {crs for _, crs, _, _ in items}
    # đếm cặp chồng nhau (bbox giao nhau), chỉ so trong cùng CRS
    overlap = 0
    npair = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            _, c1, b1, _ = items[i]
            _, c2, b2, _ = items[j]
            if c1 != c2:
                continue
            npair += 1
            if not (b1.right <= b2.left or b2.right <= b1.left or
                    b1.top <= b2.bottom or b2.top <= b1.bottom):
                overlap += 1
    print(f"  {region:12s} n={len(items):2d} | CRS={len(crss)} | cặp chồng nhau: {overlap}/{npair}")
