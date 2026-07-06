"""
Đánh giá DÙNG CHUNG cho MỌI model trên test split — "cùng 1 protocol".
Tính chỉ số CHÍNH (IoU/mIoU, F1/Dice, P/R) + PHỤ (pixel accuracy) + năng lượng.

    python evaluate.py --config configs/unet.yaml
    python evaluate.py --config configs/otsu.yaml     # baseline không học
"""
import argparse
import json
import os

import torch

from src.utils import load_config, get_device, load_checkpoint
from src.data import get_dataloader, Sen1FloodsDataset
from src.models import get_model
from src.metrics import SegMetrics
from src import energy as E


def evaluate_learned(cfg, device):
    model = get_model(cfg["model"], in_channels=cfg.get("in_channels", 2),
                      num_classes=3, **cfg.get("model_kwargs", {})).to(device)
    load_checkpoint(model, os.path.join("runs", cfg["name"], "best.pt"), map_location=device)
    model.eval()

    loader = get_dataloader(cfg["data_root"], "test", batch_size=1, num_workers=2,
                            per_channel_norm=cfg.get("per_channel_norm", False))
    metric = SegMetrics()
    with torch.no_grad():
        for s1, label, _ in loader:
            s1, label = s1.to(device), label.to(device)
            pred = model(s1).argmax(1)
            metric.update(pred, label)
            metric.update_perchip(pred[0], label[0])
    res = metric.compute()

    # Năng lượng (ANN: FLOPs×E_MAC). SNN: TODO SynOps qua energy.SpikeCounter.
    try:
        flops, params = E.count_flops_params(
            model, input_shape=(1, cfg.get("in_channels", 2), 512, 512), device="cpu")
        res["params_M"] = round(params / 1e6, 3)
        res["FLOPs_G"] = round(flops / 1e9, 3)
        res["energy_mJ_ANN"] = round(E.ann_energy_joules(flops) * 1e3, 4)
    except Exception as e:
        res["energy_note"] = f"Chưa đo được năng lượng: {e}"
    return res


def evaluate_otsu(cfg):
    from src.models.otsu import predict as otsu_predict
    ds = Sen1FloodsDataset(cfg["data_root"], "test")
    metric = SegMetrics()
    for i in range(len(ds)):
        s1, label, _ = ds[i]
        pred = torch.from_numpy(otsu_predict(s1.numpy()))
        metric.update(pred, label)
        metric.update_perchip(pred, label)
    res = metric.compute()
    res["params_M"] = 0.0
    res["energy_mJ_ANN"] = 0.0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    device = get_device()

    res = evaluate_otsu(cfg) if cfg["model"] == "otsu" else evaluate_learned(cfg, device)
    print(json.dumps(res, indent=2, ensure_ascii=False))

    out = os.path.join("runs", cfg["name"], "test_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()
