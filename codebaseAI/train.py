"""
Vòng train DÙNG CHUNG cho MỌI model học được (CNN / Transformer / SNN).
Chỉ đổi --config là đổi model — cùng dataset, cùng loss, cùng protocol.

    python train.py --config configs/unet.yaml
    python train.py --config configs/spiking_unet.yaml
"""
import argparse
import os

import torch
from torch.optim import AdamW

from src.utils import set_seed, get_device, load_config, save_checkpoint, count_params
from src.data import get_dataloader
from src.models import get_model
from src.losses import CombinedLoss
from src.metrics import SegMetrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="ghi đè seed để chạy multi-seed")
    args = ap.parse_args()
    cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    run_name = cfg["name"] + (f"_s{seed}" if args.seed is not None else "")
    device = get_device()
    print(f"[{run_name}] model={cfg['model']} seed={seed} device={device}")

    train_loader = get_dataloader(
        cfg["data_root"], "train", batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4), crop_size=cfg.get("crop_size"),
        augment=True, per_channel_norm=cfg.get("per_channel_norm", False))
    val_loader = get_dataloader(
        cfg["data_root"], "val", batch_size=1, num_workers=cfg.get("num_workers", 4),
        per_channel_norm=cfg.get("per_channel_norm", False))

    model = get_model(cfg["model"], in_channels=cfg.get("in_channels", 2),
                      num_classes=3, **cfg.get("model_kwargs", {})).to(device)
    print(f"Số tham số: {count_params(model):,}")

    criterion = CombinedLoss(num_classes=3, **cfg.get("loss_kwargs", {}))
    optimizer = AdamW(model.parameters(), lr=cfg["lr"],
                      weight_decay=cfg.get("weight_decay", 1e-4))

    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    best, bad, patience = -1.0, 0, cfg.get("patience", 8)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0
        for s1, label, _ in train_loader:
            s1, label = s1.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(s1), label)
            loss.backward()
            optimizer.step()
            running += loss.item()

        # Validation — early-stopping theo flood_IoU (chỉ số CHÍNH)
        model.eval()
        metric = SegMetrics()
        with torch.no_grad():
            for s1, label, _ in val_loader:
                s1, label = s1.to(device), label.to(device)
                pred = model(s1).argmax(1)
                metric.update(pred, label)
                metric.update_perchip(pred[0], label[0])
        res = metric.compute()
        fiou = res["flood_IoU"]
        print(f"epoch {epoch:3d} | loss {running/len(train_loader):.4f} "
              f"| flood_IoU {fiou:.4f} | mIoU {res['mIoU']:.4f}")

        if fiou > best:
            best, bad = fiou, 0
            save_checkpoint(model, os.path.join(out_dir, "best.pt"),
                            epoch=epoch, metrics=res)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop ở epoch {epoch}.")
                break

    print(f"Xong. Best flood_IoU = {best:.4f}. Checkpoint: {out_dir}/best.pt")


if __name__ == "__main__":
    main()
