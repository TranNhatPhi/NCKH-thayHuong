"""
Gom kết quả mọi model từ runs/*/test_metrics.json → bảng so sánh + biểu đồ Pareto.
    python summarize.py
Kết quả: in bảng ra màn hình + lưu runs/summary.csv + runs/pareto.png
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows():
    rows = []
    for f in sorted(glob.glob("runs/*/test_metrics.json")):
        name = os.path.basename(os.path.dirname(f))
        d = json.load(open(f))
        is_snn = "SynOps_G" in d
        rows.append({
            "model": name,
            "flood_IoU": d.get("flood_IoU", 0.0),
            "flood_IoU_chip": d.get("flood_IoU_perchip_mean", 0.0),
            "flood_F1": d.get("flood_F1", 0.0),
            "mIoU": d.get("mIoU", 0.0),
            "params_M": d.get("params_M", 0.0),
            "compute_G": d.get("SynOps_G", d.get("FLOPs_G", 0.0)),
            "energy_mJ": d.get("energy_mJ_SNN", d.get("energy_mJ_ANN", 0.0)),
            "is_snn": is_snn,
        })
    return rows


def print_table(rows):
    hdr = f"{'Model':16s}{'FloodIoU':>9s}{'IoU/chip':>9s}{'FloodF1':>8s}{'mIoU':>7s}{'Params(M)':>10s}{'Compute(G)':>11s}{'Energy(mJ)':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["flood_IoU"]):
        tag = " *SNN" if r["is_snn"] else ""
        print(f"{r['model']:16s}{r['flood_IoU']:9.3f}{r['flood_IoU_chip']:9.3f}"
              f"{r['flood_F1']:8.3f}{r['mIoU']:7.3f}{r['params_M']:10.2f}"
              f"{r['compute_G']:11.2f}{r['energy_mJ']:11.1f}{tag}")


def save_csv(rows, path="runs/summary.csv"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nĐã lưu bảng: {path}")


def plot_pareto(rows, path="runs/pareto.png"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for r in rows:
        x = r["energy_mJ"] if r["energy_mJ"] > 0 else 0.3   # Otsu ~0 -> đặt nhỏ cho log
        snn = r["is_snn"]
        ax.scatter(x, r["flood_IoU"],
                   s=340 if snn else 130,
                   marker="*" if snn else "o",
                   color="crimson" if snn else "steelblue",
                   edgecolor="black", linewidth=1.0, zorder=3)
        ax.annotate(r["model"], (x, r["flood_IoU"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Năng lượng (mJ, thang log) — càng TRÁI càng tiết kiệm →")
    ax.set_ylabel("Flood IoU — càng CAO càng chính xác ↑")
    ax.set_title("Pareto: Accuracy vs Energy  (★ đỏ = SNN-Flood)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Đã lưu biểu đồ Pareto: {path}")


if __name__ == "__main__":
    rows = load_rows()
    if not rows:
        raise SystemExit("Không thấy runs/*/test_metrics.json — chạy evaluate.py trước.")
    print_table(rows)
    save_csv(rows)
    plot_pareto(rows)
