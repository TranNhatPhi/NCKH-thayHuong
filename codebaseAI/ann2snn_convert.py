"""
ANN2SNN — chuyển một U-Net (ANN) đã train sang SNN bằng spikingjelly, rồi đánh giá
accuracy + SynOps. Đây là "đối thủ nội bộ" chứng minh direct-training (SpikingUNet)
tốt hơn conversion.

    # train ANN trước, ví dụ:  python train.py --config configs/unet_smp.yaml
    python ann2snn_convert.py --ann_config configs/unet_smp.yaml --T 32

Trạng thái: 🟡 cần kiểm thử trên GPU (conversion + mô phỏng T bước).
"""
import argparse
import json
import os

import torch
import torch.nn as nn

from src.utils import load_config, get_device, load_checkpoint, count_params
from src.data import get_dataloader
from src.models import get_model
from src.metrics import SegMetrics
from src import energy as E


class SNNWrapper(nn.Module):
    """Bọc SNN single-step (đã convert) thành forward T bước + reset → logits trung bình,
    giữ cùng interface như SpikingUNet (để dùng chung eval + count_synops)."""
    def __init__(self, snn, T):
        super().__init__()
        self.snn, self.T = snn, T

    def forward(self, x):
        from spikingjelly.activation_based import functional
        functional.reset_net(self.snn)
        out = 0.0
        for _ in range(self.T):
            out = out + self.snn(x)
        return out / self.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_config", required=True, help="config của ANN đã train")
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--name", default="ann2snn")
    args = ap.parse_args()
    cfg = load_config(args.ann_config)
    device = get_device()

    # 1. Nạp ANN đã train
    ann = get_model(cfg["model"], in_channels=cfg.get("in_channels", 2),
                    num_classes=3, **cfg.get("model_kwargs", {})).to(device)
    load_checkpoint(ann, os.path.join("runs", cfg["name"], "best.pt"), map_location=device)
    ann.eval()

    # 2. Convert ANN → SNN (dùng train split để hiệu chỉnh)
    from spikingjelly.activation_based import ann2snn
    calib = get_dataloader(cfg["data_root"], "train", batch_size=4, num_workers=0,
                           crop_size=cfg.get("crop_size"))
    snn = ann2snn.Converter(mode="max", dataloader=calib)(ann).to(device)
    model = SNNWrapper(snn, args.T).to(device).eval()

    # 3. Đánh giá trên test (T bước)
    te = get_dataloader(cfg["data_root"], "test", batch_size=1, num_workers=0)
    metric = SegMetrics()
    with torch.no_grad():
        for s1, lab, _ in te:
            s1, lab = s1.to(device), lab.to(device)
            pred = model(s1).argmax(1)
            metric.update(pred, lab)
            metric.update_perchip(pred[0], lab[0])
    res = metric.compute()

    # 4. Năng lượng (SynOps qua T bước)
    res["params_M"] = round(count_params(snn) / 1e6, 3)
    dummy = torch.randn(1, cfg.get("in_channels", 2), 512, 512, device=device)
    synops = E.count_synops(model, dummy, device)
    res["SynOps_G"] = round(synops / 1e9, 3)
    res["energy_mJ_SNN"] = round(E.snn_energy_joules(synops) * 1e3, 4)

    out_dir = os.path.join("runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"Đã lưu runs/{args.name}/test_metrics.json")


if __name__ == "__main__":
    main()
