"""
Sinh figures + thống kê cho paper (P1) — CHẠY TRÊN MÁY THƯỜNG, không cần GPU.
Đọc runs/*/test_metrics.json + runs/*/perchip.csv → xuất ra paper/figures/.

    python make_figures.py

Sinh:
  1. pareto_pooled.png / pareto_perchip.png  — accuracy vs energy (2 metric)
  2. tsweep.png                              — IoU vs T (error bar) + energy vs T
  3. per_region_heatmap.png                  — model × vùng
  4. stats_ci_effectsize.csv                 — bootstrap 95% CI + Cohen's d cho cặp chính
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "../paper/figures"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)


def base(name):
    return re.sub(r"_s\d+$", "", name)


def is_junk(n):
    return n.endswith(("_OLDn1bak", "_bak", "_backup"))


def load_summary():
    """base model -> dict(pooled_mean,pooled_std,perchip_mean,energy,is_snn,params)."""
    g = defaultdict(list)
    for f in sorted(glob.glob("runs/*/test_metrics.json")):
        n = os.path.basename(os.path.dirname(f))
        if is_junk(n):
            continue
        g[base(n)].append(json.load(open(f)))
    out = {}
    for m, rs in g.items():
        fi = [r.get("flood_IoU", 0.0) for r in rs]
        out[m] = dict(
            pooled_mean=float(np.mean(fi)),
            pooled_std=float(np.std(fi)),
            perchip_mean=float(np.mean([r.get("flood_IoU_perchip_mean", 0.0) for r in rs])),
            energy=rs[0].get("energy_mJ_SNN", rs[0].get("energy_mJ_ANN", 0.0)),
            spike=float(np.mean([r.get("spike_rate", 0.0) for r in rs])),
            is_snn="SynOps_G" in rs[0],
            params=rs[0].get("params_M", 0.0),
        )
    return out


def load_perchip():
    """base model -> {chip: mean flood_iou over seeds}; + chip->region."""
    raw = defaultdict(lambda: defaultdict(list))
    region = {}
    for f in glob.glob("runs/*/perchip.csv"):
        n = os.path.basename(os.path.dirname(f))
        if is_junk(n):
            continue
        m = base(n)
        for row in csv.DictReader(open(f)):
            raw[m][row["chip_id"]].append(float(row["flood_iou"]))
            region[row["chip_id"]] = row["region"]
    avg = {m: {c: float(np.mean(v)) for c, v in ch.items()} for m, ch in raw.items()}
    return avg, region


# ---------- 1. Pareto ----------
def pareto(summary, key, fname, ylabel):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for m, d in summary.items():
        x = d["energy"] if d["energy"] > 0 else 0.3
        snn = d["is_snn"]
        ax.scatter(x, d[key], s=170 if snn else 90, marker="*" if snn else "o",
                   c="crimson" if snn else "steelblue", edgecolor="black", zorder=3)
        ax.annotate(m.replace("spiking_unet", "SNN").replace("mobilenet", "mbnet"),
                    (x, d[key]), textcoords="offset points", xytext=(6, 5), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Energy (mJ, log)  — lower = greener →")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Accuracy–Energy Pareto ({key})")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=300); plt.close(fig)
    print(f"→ {OUT}/{fname}")


# ---------- 2. T-sweep ----------
def tsweep(summary):
    Ts = [1, 2, 3, 4, 5, 6, 7, 8, 10]
    xs, ys, es, en = [], [], [], []
    for T in Ts:
        k = f"spiking_unet_T{T}"
        if k in summary:
            xs.append(T); ys.append(summary[k]["pooled_mean"])
            es.append(summary[k]["pooled_std"]); en.append(summary[k]["energy"])
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.errorbar(xs, ys, yerr=es, fmt="o-", color="crimson", capsize=4, label="Flood-IoU")
    ax1.set_xlabel("Timesteps T"); ax1.set_ylabel("Pooled Flood-IoU (mean±std)", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson"); ax1.grid(True, ls="--", alpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(xs, en, "s--", color="steelblue", label="Energy")
    ax2.set_ylabel("Energy (mJ)", color="steelblue"); ax2.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_title("SNN-Flood: T-sweep (accuracy flat across T; energy grows)")
    fig.tight_layout(); fig.savefig(f"{OUT}/tsweep.png", dpi=300); plt.close(fig)
    print(f"→ {OUT}/tsweep.png")


# ---------- 2b. Ablation 3-panel: IoU / Spike% / Energy vs T (Fig 7) ----------
def ablation(summary):
    Ts = [1, 2, 3, 4, 5, 6, 7, 8, 10]
    xs, iou, err, spk, en = [], [], [], [], []
    for T in Ts:
        k = f"spiking_unet_T{T}"
        if k in summary:
            xs.append(T); iou.append(summary[k]["pooled_mean"]); err.append(summary[k]["pooled_std"])
            spk.append(summary[k]["spike"] * 100); en.append(summary[k]["energy"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].errorbar(xs, iou, yerr=err, fmt="o-", color="#4477AA", capsize=4)
    ax[0].set_xlabel("Timesteps T"); ax[0].set_ylabel("Pooled Flood-IoU (mean±std)")
    ax[0].set_title("(a) Accuracy vs T"); ax[0].grid(True, ls="--", alpha=0.4)
    ax[1].plot(xs, spk, "s-", color="#EE6677")
    ax[1].set_xlabel("Timesteps T"); ax[1].set_ylabel("Mean spike rate (%)")
    ax[1].set_title("(b) Spike rate vs T"); ax[1].grid(True, ls="--", alpha=0.4)
    ax[2].plot(xs, en, "^-", color="#228833")
    ax[2].set_xlabel("Timesteps T"); ax[2].set_ylabel("Energy (mJ)")
    ax[2].set_title("(c) Energy vs T"); ax[2].grid(True, ls="--", alpha=0.4)
    fig.suptitle("Ablation over timesteps T (Spiking U-Net)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/ablation_T.png", dpi=300); plt.close(fig)
    print(f"→ {OUT}/ablation_T.png")


# ---------- 3. Per-region heatmap ----------
def heatmap(avg, region):
    models = [m for m in ["segformer_b2", "unet_smp", "unetpp", "mobilenet_unet",
                          "mobilenet_int8", "unet", "spiking_unet_T2", "spiking_unet_T8",
                          "ann2snn_T128", "deeplabv3"] if m in avg]
    regions = sorted(set(region.values()))
    M = np.full((len(models), len(regions)), np.nan)
    for i, m in enumerate(models):
        for j, r in enumerate(regions):
            vals = [avg[m][c] for c in avg[m] if region.get(c) == r]
            if vals:
                M[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace("spiking_unet", "SNN") for m in models])
    for i in range(len(models)):
        for j in range(len(regions)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.3 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="per-region flood-IoU")
    ax.set_title("Per-region flood-IoU (model × region)")
    fig.tight_layout(); fig.savefig(f"{OUT}/per_region_heatmap.png", dpi=300); plt.close(fig)
    print(f"→ {OUT}/per_region_heatmap.png")


# ---------- 4. Bootstrap CI + Cohen's d ----------
def stats(avg):
    def boot_ci(x, n=10000):
        x = np.asarray(x); m = np.array([RNG.choice(x, len(x)).mean() for _ in range(n)])
        return np.percentile(m, 2.5), np.percentile(m, 97.5)

    rows = []
    for m, ch in avg.items():
        vals = list(ch.values()); lo, hi = boot_ci(vals)
        rows.append({"model": m, "perchip_mean": round(np.mean(vals), 4),
                     "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)})
    with open(f"{OUT}/../results/bootstrap_ci.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "perchip_mean", "ci95_lo", "ci95_hi"])
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: -r["perchip_mean"]))
    print(f"→ paper/results/bootstrap_ci.csv ({len(rows)} model)")

    pairs = [("unet_smp", "mobilenet_int8"), ("mobilenet_int8", "spiking_unet_T6"),
             ("spiking_unet_T2", "spiking_unet_T6")]
    prows = []
    for a, b in pairs:
        if a not in avg or b not in avg:
            continue
        common = sorted(set(avg[a]) & set(avg[b]))
        da = np.array([avg[a][c] for c in common]); db = np.array([avg[b][c] for c in common])
        diff = da - db
        d = diff.mean() / (diff.std(ddof=1) + 1e-12)   # paired Cohen's d
        prows.append({"pair": f"{a} vs {b}", "n": len(common),
                      "mean_diff": round(diff.mean(), 4), "cohens_d": round(d, 3)})
        print(f"   {a} vs {b}: mean_diff={diff.mean():.4f}, Cohen's d={d:.3f} (n={len(common)})")
    with open(f"{OUT}/../results/effect_size_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "n", "mean_diff", "cohens_d"])
        w.writeheader(); w.writerows(prows)
    print(f"→ paper/results/effect_size_pairs.csv")


def sync_results():
    """Copy summary.csv + analysis CSV (do summarize.py/analysis.py ghi vào runs/) sang paper/results/
    để make_docx.py đọc đúng bản mới nhất."""
    import shutil
    dst = os.path.join(OUT, "..", "results")
    os.makedirs(dst, exist_ok=True)
    for src in ["runs/summary.csv"] + glob.glob("runs/analysis/*.csv"):
        if os.path.isfile(src):
            shutil.copy(src, dst)
            print(f"   sync {src} → paper/results/")


if __name__ == "__main__":
    summary = load_summary()
    avg, region = load_perchip()
    if not summary:
        raise SystemExit("Không thấy runs/*/test_metrics.json — giải nén runs_full.tar.gz trước.")
    pareto(summary, "pooled_mean", "pareto_pooled.png", "Pooled Flood-IoU ↑")
    pareto(summary, "perchip_mean", "pareto_perchip.png", "Per-chip Flood-IoU ↑")
    tsweep(summary)
    ablation(summary)
    heatmap(avg, region)
    stats(avg)
    sync_results()
    print("\nXONG. Figures ở paper/figures/, CSV ở paper/results/. (Nhớ chạy summarize.py + analysis.py trước.)")
