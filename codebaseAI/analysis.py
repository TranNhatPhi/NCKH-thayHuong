"""
Phân tích sâu (theo yêu cầu framing mới): Wilcoxon signed-rank + bảng per-region.
Đọc runs/*/perchip.csv (do evaluate.py bản mới sinh ra). Gộp seed tự động.

    python analysis.py [--ref spiking_unet_T4]

- Wilcoxon: kiểm định khác biệt flood-IoU giữa <ref> và từng model (ghép theo chip) —
  trả lời "chênh lệch có THẬT hay chỉ do noise" (quan trọng vì ΔIoU nhỏ).
- Per-region: flood-IoU trung bình theo từng vùng địa lý cho mỗi model.
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


def base_name(name):
    return re.sub(r"_s\d+$", "", name)


def load():
    """model → chip_id → flood_iou (trung bình qua các seed). + map chip→region."""
    raw = defaultdict(lambda: defaultdict(list))
    region = {}
    for f in glob.glob("runs/*/perchip.csv"):
        model = base_name(os.path.basename(os.path.dirname(f)))
        with open(f) as fh:
            for row in csv.DictReader(fh):
                raw[model][row["chip_id"]].append(float(row["flood_iou"]))
                region[row["chip_id"]] = row["region"]
    avg = {m: {c: float(np.mean(v)) for c, v in chips.items()} for m, chips in raw.items()}
    return avg, region


def wilcoxon_vs(avg, ref):
    print(f"\n== Wilcoxon signed-rank: '{ref}' vs từng model (per-chip flood-IoU) ==")
    if ref not in avg:
        print(f"  [!] không thấy '{ref}' — chọn --ref khác:", ", ".join(sorted(avg)))
        return
    print(f"{'Model':16s}{'mean_ref':>9s}{'mean_oth':>9s}{'p-value':>10s}  kết luận")
    print("-" * 62)
    for m in sorted(avg, key=lambda k: -np.mean(list(avg[k].values()))):
        if m == ref:
            continue
        common = sorted(set(avg[ref]) & set(avg[m]))
        if len(common) < 5:
            continue
        a = np.array([avg[ref][c] for c in common])
        b = np.array([avg[m][c] for c in common])
        try:
            _, p = wilcoxon(a, b)
        except ValueError:
            p = float("nan")
        sig = "khác biệt THẬT (p<0.05)" if p < 0.05 else "ns (không chắc khác)"
        print(f"{m:16s}{a.mean():9.3f}{b.mean():9.3f}{p:10.4f}  {sig}")


def per_region(avg, region):
    print("\n== Per-region flood-IoU (trung bình theo vùng) ==")
    models = sorted(avg, key=lambda k: -np.mean(list(avg[k].values())))
    regions = sorted(set(region.values()))
    print(f"{'Vùng':12s}" + "".join(f"{m[:10]:>11s}" for m in models))
    print("-" * (12 + 11 * len(models)))
    for r in regions:
        row = f"{r:12s}"
        for m in models:
            vals = [avg[m][c] for c in avg[m] if region.get(c) == r]
            row += f"{np.mean(vals):11.3f}" if vals else f"{'-':>11s}"
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="spiking_unet_T4", help="model tham chiếu cho Wilcoxon")
    args = ap.parse_args()
    avg, region = load()
    if not avg:
        raise SystemExit("Không thấy runs/*/perchip.csv — chạy lại evaluate.py (bản mới) trước.")
    wilcoxon_vs(avg, args.ref)
    per_region(avg, region)
