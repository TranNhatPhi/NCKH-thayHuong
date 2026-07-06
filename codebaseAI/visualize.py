"""
Tạo hình QUALITATIVE: mỗi hàng = 1 chip test, 3 cột = SAR (VV) | Ground truth | Prediction.
Màu: nền (xám) · nước thường trực (xanh) · nước lũ/flood (đỏ).

    python visualize.py --config configs/spiking_unet.yaml        # SNN
    python visualize.py --config configs/unet.yaml --n 6          # U-Net
Kết quả: runs/<name>/qualitative.png
"""
import argparse
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

CMAP = ListedColormap(["#454545", "#1f77b4", "#d62728"])   # 0 nền · 1 perm-water · 2 flood


def to_rgb(mask):
    m = mask.copy()
    m[m == -1] = 0                     # pixel ignore hiển thị như nền
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=6, help="số chip đưa lên hình")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    model = get_model(cfg["model"], in_channels=cfg.get("in_channels", 2),
                      num_classes=3, **cfg.get("model_kwargs", {})).to(device)
    load_checkpoint(model, os.path.join("runs", cfg["name"], "best.pt"), map_location=device)
    model.eval()

    ds = Sen1FloodsDataset(cfg["data_root"], "test")
    # chọn chip có NHIỀU flood nhất để hình dễ nhìn
    scored = []
    for i in range(len(ds)):
        _, lab, cid = ds[i]
        scored.append(((lab == 2).sum().item(), i))
    scored.sort(reverse=True)
    picks = [i for _, i in scored[:args.n]]

    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    for row, idx in enumerate(picks):
        s1, lab, cid = ds[idx]
        with torch.no_grad():
            pred = model(s1.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
        axes[row, 0].imshow(s1[0].numpy(), cmap="gray")
        axes[row, 0].set_title(f"{cid}\nSAR (VV)", fontsize=8)
        axes[row, 1].imshow(to_rgb(lab.numpy()), cmap=CMAP, vmin=0, vmax=2)
        axes[row, 1].set_title("Ground truth", fontsize=8)
        axes[row, 2].imshow(to_rgb(pred), cmap=CMAP, vmin=0, vmax=2)
        axes[row, 2].set_title("Prediction", fontsize=8)
        for c in range(3):
            axes[row, c].axis("off")

    handles = [Patch(color="#454545", label="nền"),
               Patch(color="#1f77b4", label="nước thường trực"),
               Patch(color="#d62728", label="nước lũ (flood)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"Qualitative — {cfg['name']}", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out = os.path.join("runs", cfg["name"], "qualitative.png")
    fig.savefig(out, dpi=140)
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()
