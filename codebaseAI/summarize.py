"""
Gom kết quả runs/*/test_metrics.json → bảng (GỘP mean±std các seed) + biểu đồ Pareto.
Các run cùng model khác seed (vd unet_smp_s0/_s1/_s2) được gộp tự động.
    python summarize.py
"""
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def base_name(name):
    return re.sub(r"_s\d+$", "", name)          # bỏ hậu tố _s0/_s1... để gộp multi-seed


def is_junk(name):
    return name.endswith(("_OLDn1bak", "_bak", "_backup"))   # dir cách ly, bỏ qua


def load_rows():
    rows = []
    for f in sorted(glob.glob("runs/*/test_metrics.json")):
        name = os.path.basename(os.path.dirname(f))
        if is_junk(name):
            continue
        d = json.load(open(f))
        rows.append({
            "model": name, "base": base_name(name),
            "flood_IoU": d.get("flood_IoU", 0.0),
            "flood_IoU_chip": d.get("flood_IoU_perchip_mean", 0.0),
            "flood_F1": d.get("flood_F1", 0.0),
            "pw_IoU": d.get("IoU_per_class", {}).get("permanent_water", 0.0),
            "mIoU": d.get("mIoU", 0.0),
            "params_M": d.get("params_M", 0.0),
            "compute_G": d.get("SynOps_G", d.get("FLOPs_G", 0.0)),
            "energy_mJ": d.get("energy_mJ_SNN", d.get("energy_mJ_ANN", 0.0)),
            "is_snn": "SynOps_G" in d,
        })
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["base"]].append(r)
    agg = []
    for base, rs in groups.items():
        def mean(k):
            return sum(r[k] for r in rs) / len(rs)
        fious = [r["flood_IoU"] for r in rs]
        agg.append({
            "model": base, "n": len(rs),
            "flood_IoU": mean("flood_IoU"),
            "flood_IoU_std": statistics.pstdev(fious) if len(fious) > 1 else 0.0,
            "flood_IoU_chip": mean("flood_IoU_chip"), "flood_F1": mean("flood_F1"),
            "pw_IoU": mean("pw_IoU"), "mIoU": mean("mIoU"),
            "params_M": rs[0]["params_M"], "compute_G": rs[0]["compute_G"],
            "energy_mJ": rs[0]["energy_mJ"], "is_snn": rs[0]["is_snn"],
        })
    return sorted(agg, key=lambda x: -x["flood_IoU"])


def print_table(agg):
    hdr = (f"{'Model':22s}{'FloodIoU(±std,n)':>21s}{'IoU/chip':>9s}{'F1':>7s}"
           f"{'pwIoU':>7s}{'Params':>8s}{'Energy(mJ)':>11s}")
    print(hdr); print("-" * len(hdr))
    for r in agg:
        fi = f"{r['flood_IoU']:.3f}±{r['flood_IoU_std']:.3f}(n{r['n']})"
        tag = " *SNN" if r["is_snn"] else ""
        print(f"{r['model']:22s}{fi:>21s}{r['flood_IoU_chip']:9.3f}{r['flood_F1']:7.3f}"
              f"{r['pw_IoU']:7.3f}{r['params_M']:8.2f}{r['energy_mJ']:11.1f}{tag}")


def save_csv(agg, path="runs/summary.csv"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader(); w.writerows(agg)
    print(f"\nĐã lưu {path}")


def plot_pareto(agg, path="runs/pareto.png"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for r in agg:
        x = r["energy_mJ"] if r["energy_mJ"] > 0 else 0.3
        snn = r["is_snn"]
        ax.errorbar(x, r["flood_IoU"], yerr=r["flood_IoU_std"],
                    fmt="*" if snn else "o", ms=18 if snn else 9,
                    color="crimson" if snn else "steelblue",
                    ecolor="gray", capsize=3, mec="black", zorder=3)
        ax.annotate(r["model"], (x, r["flood_IoU"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Năng lượng (mJ, thang log) — càng TRÁI càng tiết kiệm →")
    ax.set_ylabel("Flood IoU (mean±std) ↑")
    ax.set_title("Pareto: Accuracy vs Energy  (★ đỏ = SNN-Flood)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"Đã lưu {path}")


if __name__ == "__main__":
    rows = load_rows()
    if not rows:
        raise SystemExit("Không thấy runs/*/test_metrics.json — chạy evaluate.py trước.")
    agg = aggregate(rows)
    print_table(agg)
    save_csv(agg)
    plot_pareto(agg)
