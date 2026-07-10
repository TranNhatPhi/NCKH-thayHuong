"""
Hình qualitative GỘP cho paper: cùng chip, các cột =
    SAR (VV) | Ground truth | SegFormer | MobileNet | SNN-T2
Chọn chip có nhiều nước thường trực + flood để lộ khác biệt (SNN hay bỏ sót nước xanh).
    python make_qualitative.py
Xuất: ../paper/figures/qual_comparison.png
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.utils import load_config, get_device, load_checkpoint
from src.data import Sen1FloodsDataset
from src.models import get_model

CMAP = ListedColormap(["#454545", "#1f77b4", "#d62728"])
OUT = "../paper/figures/qual_comparison.png"
N = 5  # số vùng (chip từ 5 vùng khác nhau)

MODELS = [  # (nhãn cột, config, run_dir) — theo yêu cầu thầy: UNet++ / MobileNet(-INT8) / Spiking-UNet
    ("U-Net++", "configs/unetpp.yaml", "unetpp_s0"),
    ("MobileNet", "configs/mobilenet_unet_lr2e4.yaml", "mobilenet_unet_lr2e4_s0"),
    ("Spiking-UNet", "configs/spiking_unet_T2.yaml", "spiking_unet_T2_s0"),
]


def load(cfg_path, run):
    cfg = load_config(cfg_path)
    m = get_model(cfg["model"], in_channels=cfg.get("in_channels", 2), num_classes=3,
                  **cfg.get("model_kwargs", {})).to(get_device())
    load_checkpoint(m, os.path.join("runs", run, "best.pt"), map_location=get_device())
    m.eval()
    return cfg, m


def rgb(mask):
    m = mask.copy(); m[m == -1] = 0; return m


def main():
    dev = get_device()
    ds = Sen1FloodsDataset("../dataset", "test")
    # Chọn 1 chip đại diện cho mỗi vùng (nhiều nước thường trực + flood nhất), lấy N vùng
    best_per_region = {}
    for i in range(len(ds)):
        _, lab, cid = ds[i]
        region = cid.split("_")[0]
        score = (lab == 1).sum().item() + 0.3 * (lab == 2).sum().item()
        if region not in best_per_region or score > best_per_region[region][0]:
            best_per_region[region] = (score, i)
    top_regions = sorted(best_per_region.values(), reverse=True)[:N]
    picks = [i for _, i in top_regions]

    models = [(name,) + load(cfg, run) for name, cfg, run in MODELS]
    ncol = 2 + len(models)
    fig, axes = plt.subplots(N, ncol, figsize=(2.4 * ncol, 2.6 * N))
    if N == 1:
        axes = axes[None, :]
    for r, idx in enumerate(picks):
        s1, lab, cid = ds[idx]
        axes[r, 0].imshow(s1[0].numpy(), cmap="gray")
        axes[r, 0].set_ylabel(cid, fontsize=8)
        axes[r, 0].set_title("SAR (VV)" if r == 0 else "", fontsize=9)
        axes[r, 1].imshow(rgb(lab.numpy()), cmap=CMAP, vmin=0, vmax=2)
        axes[r, 1].set_title("Ground truth" if r == 0 else "", fontsize=9)
        for c, (name, cfg, m) in enumerate(models, start=2):
            with torch.no_grad():
                pred = m(s1.unsqueeze(0).to(dev)).argmax(1)[0].cpu().numpy()
            axes[r, c].imshow(rgb(pred), cmap=CMAP, vmin=0, vmax=2)
            axes[r, c].set_title(name if r == 0 else "", fontsize=9)
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    handles = [Patch(color="#454545", label="background"),
               Patch(color="#1f77b4", label="permanent water"),
               Patch(color="#d62728", label="flood")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print(f"Đã lưu {OUT}")


if __name__ == "__main__":
    main()
