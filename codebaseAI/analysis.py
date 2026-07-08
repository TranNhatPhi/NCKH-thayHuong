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


def is_junk(name):
    return name.endswith(("_OLDn1bak", "_bak", "_backup"))   # dir cách ly, bỏ qua


def load():
    """model → chip_id → flood_iou (trung bình qua các seed). + map chip→region."""
    raw = defaultdict(lambda: defaultdict(list))
    region = {}
    for f in glob.glob("runs/*/perchip.csv"):
        dirname = os.path.basename(os.path.dirname(f))
        if is_junk(dirname):
            continue
        model = base_name(dirname)
        with open(f) as fh:
            for row in csv.DictReader(fh):
                raw[model][row["chip_id"]].append(float(row["flood_iou"]))
                region[row["chip_id"]] = row["region"]
    avg = {m: {c: float(np.mean(v)) for c, v in chips.items()} for m, chips in raw.items()}
    return avg, region


OUTDIR = "runs/analysis"


def wilcoxon_vs(avg, ref):
    print(f"\n== Wilcoxon signed-rank: '{ref}' vs từng model (per-chip flood-IoU) ==")
    if ref not in avg:
        print(f"  [!] không thấy '{ref}' — chọn --ref khác:", ", ".join(sorted(avg)))
        return
    print(f"{'Model':24s}{'ref':>8s}{'model':>8s}{'p-value':>10s}  kết luận")
    print("-" * 72)
    rows = []
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
        better = "cao hơn" if b.mean() > a.mean() else "thấp hơn"
        sig = f"{better}, THẬT (p<0.05)" if p < 0.05 else "ns (ngang ref)"
        print(f"{m:24s}{a.mean():8.3f}{b.mean():8.3f}{p:10.4f}  {sig}")
        rows.append({"model": m, "ref_mean": round(a.mean(), 4),
                     "model_mean": round(b.mean(), 4), "p_value": round(p, 4),
                     "significant": int(p < 0.05)})
    _write_csv(f"wilcoxon_vs_{ref}.csv", ["model", "ref_mean", "model_mean", "p_value", "significant"], rows)


def wilcoxon_pairs(avg, pairs):
    """Wilcoxon per-chip cho các CẶP model cụ thể (theo yêu cầu thầy)."""
    print("\n== Wilcoxon per-chip cho các cặp quan trọng ==")
    print(f"{'Cặp A vs B':44s}{'A':>8s}{'B':>8s}{'p-value':>10s}  kết luận")
    print("-" * 92)
    rows = []
    for a_name, b_name in pairs:
        if a_name not in avg or b_name not in avg:
            miss = a_name if a_name not in avg else b_name
            print(f"{a_name+' vs '+b_name:44s}  [!] thiếu '{miss}' — bỏ qua")
            continue
        common = sorted(set(avg[a_name]) & set(avg[b_name]))
        if len(common) < 5:
            print(f"{a_name+' vs '+b_name:44s}  [!] <5 chip chung — bỏ qua")
            continue
        a = np.array([avg[a_name][c] for c in common])
        b = np.array([avg[b_name][c] for c in common])
        try:
            _, p = wilcoxon(a, b)
        except ValueError:
            p = float("nan")
        sig = "significant (p<0.05)" if p < 0.05 else "ns (không chắc khác)"
        print(f"{a_name+' vs '+b_name:44s}{a.mean():8.3f}{b.mean():8.3f}{p:10.4f}  {sig}")
        rows.append({"pair": f"{a_name} vs {b_name}", "A_mean": round(a.mean(), 4),
                     "B_mean": round(b.mean(), 4), "p_value": round(p, 4),
                     "significant": int(p < 0.05)})
    if rows:
        _write_csv("wilcoxon_pairs.csv", ["pair", "A_mean", "B_mean", "p_value", "significant"], rows)


def per_region(avg, region):
    """In dạng model-theo-hàng, vùng-theo-cột (gọn hơn, không wrap) + xuất CSV."""
    print("\n== Per-region flood-IoU (hàng = model, cột = vùng) ==")
    models = sorted(avg, key=lambda k: -np.mean(list(avg[k].values())))
    regions = sorted(set(region.values()))

    def cell(m, r):
        vals = [avg[m][c] for c in avg[m] if region.get(c) == r]
        return float(np.mean(vals)) if vals else None

    print(f"{'Model':24s}" + "".join(f"{r[:8]:>9s}" for r in regions) + f"{'MEAN':>9s}")
    print("-" * (24 + 9 * (len(regions) + 1)))
    rows = []
    for m in models:
        cells = {r: cell(m, r) for r in regions}
        mean_all = np.mean([v for v in cells.values() if v is not None])
        line = f"{m:24s}" + "".join(f"{cells[r]:9.3f}" if cells[r] is not None else f"{'-':>9s}" for r in regions)
        print(line + f"{mean_all:9.3f}")
        row = {"model": m, "MEAN": round(mean_all, 4)}
        row.update({r: (round(cells[r], 4) if cells[r] is not None else "") for r in regions})
        rows.append(row)
    _write_csv("per_region.csv", ["model"] + regions + ["MEAN"], rows)


def _write_csv(fname, fields, rows):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, fname)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"   → đã lưu {path}")


if __name__ == "__main__":
    # Cặp mặc định theo yêu cầu thầy
    default_pairs = ("unet_smp:mobilenet_int8,"
                     "mobilenet_int8:spiking_unet_T6,"
                     "spiking_unet_T2:spiking_unet_T6")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="spiking_unet_T4", help="model tham chiếu cho Wilcoxon")
    ap.add_argument("--pairs", default=default_pairs,
                    help="các cặp 'A:B' phân tách bởi dấu phẩy cho Wilcoxon per-chip")
    args = ap.parse_args()
    avg, region = load()
    if not avg:
        raise SystemExit("Không thấy runs/*/perchip.csv — chạy lại evaluate.py (bản mới) trước.")
    wilcoxon_vs(avg, args.ref)
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if ":" in p]
    wilcoxon_pairs(avg, pairs)
    per_region(avg, region)
