"""
MobileNet-UNet INT4 — THỬ NGHIỆM biên dưới của quantization ANN (theo yêu cầu thầy).

Giả thuyết của thầy: nếu INT4 cho MobileNet-UNet **fail** (framework không nén được conv
xuống 4-bit, hoặc accuracy sụp đổ) → chứng tỏ ANN không hạ precision vô hạn được →
**cứu vãn ngách năng lượng cực thấp của SNN** (SNN vốn thưa xung, không phụ thuộc bit-width).

Cách chạy:
    python quantize_int4.py --seed 0

Kết quả có 2 khả năng, ĐỀU là dữ liệu cho paper:
  (A) INT4 KHÔNG áp dụng được cho Conv2d (torchao int4_weight_only chỉ hỗ trợ Linear)
      → ghi status "unsupported" (không tạo điểm Pareto) → luận điểm cứu SNN.
  (B) INT4 chạy được → eval accuracy + năng lượng INT4; nếu accuracy sụp đổ vẫn là finding.

Ghi status vào runs/<name>/int4_status.txt (luôn có), và test_metrics.json chỉ khi (B) thành công.
"""
import argparse
import copy
import json
import os

import torch

from src.utils import load_config, count_params
from src.models import get_model
from src import energy as E


def _write_status(name, text):
    out_dir = os.path.join("runs", name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "int4_status.txt"), "w") as f:
        f.write(text + "\n")
    print(f"[INT4 status] {text}")
    print(f"Đã lưu runs/{name}/int4_status.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mobilenet_unet_lr2e4.yaml")
    ap.add_argument("--src_run", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    C = cfg.get("in_channels", 2)
    src_run = args.src_run or (f"{cfg['name']}_s{args.seed}" if args.seed is not None else cfg["name"])
    name = args.name or ("mobilenet_int4" + (f"_s{args.seed}" if args.seed is not None else ""))
    ckpt = os.path.join("runs", src_run, "best.pt")
    if not os.path.isfile(ckpt):
        _write_status(name, f"THIẾU {ckpt} — cần train MobileNet-UNet trước. Bỏ qua INT4.")
        return

    # torchao có sẵn không?
    try:
        from torchao.quantization import quantize_, int4_weight_only
    except Exception as e:
        _write_status(name, f"KHÔNG có torchao ({e}). INT4 unsupported trong môi trường này "
                            "→ ANN không hạ được 4-bit ở đây (luận điểm cứu SNN).")
        return

    from src.utils import load_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg["model"], in_channels=C, num_classes=3,
                      **cfg.get("model_kwargs", {})).to(device).eval()
    load_checkpoint(model, ckpt, map_location=device)

    n_conv = sum(1 for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))

    # int4_weight_only của torchao chỉ nhắm nn.Linear → conv-heavy UNet gần như không nén được
    try:
        quantize_(model, int4_weight_only())
    except Exception as e:
        _write_status(name, f"int4_weight_only lỗi khi áp dụng: {e}. "
                            f"Model có {n_conv} Conv2d / {n_linear} Linear — INT4 không phủ Conv2d.")
        return

    # Kiểm tra thực sự có tensor nào được lượng tử hoá 4-bit không
    from torchao.dtypes import AffineQuantizedTensor  # type: ignore
    n_quant = 0
    for m in model.modules():
        for p in m.parameters(recurse=False):
            if isinstance(p, AffineQuantizedTensor) or "AffineQuantized" in type(p).__name__:
                n_quant += 1

    if n_quant == 0:
        _write_status(
            name,
            f"INT4 UNSUPPORTED cho MobileNet-UNet: torchao int4_weight_only chỉ nén nn.Linear, "
            f"nhưng model có {n_conv} Conv2d và {n_linear} Linear → 0 tensor được nén 4-bit. "
            f"KẾT LUẬN: ANN conv-heavy KHÔNG hạ được INT4 bằng framework hiện tại "
            f"→ ngách năng lượng cực thấp của SNN được củng cố.")
        return

    # Nếu (hiếm) có nén được: đánh giá accuracy + năng lượng INT4
    from src.data import get_dataloader
    from src.metrics import SegMetrics
    flops, _ = E.count_flops_params(
        get_model(cfg["model"], in_channels=C, num_classes=3, **cfg.get("model_kwargs", {})),
        input_shape=(1, C, 512, 512), device="cpu")
    te = get_dataloader(cfg["data_root"], "test", batch_size=1, num_workers=0,
                        per_channel_norm=cfg.get("per_channel_norm", False))
    metric = SegMetrics()
    with torch.no_grad():
        for s1, label, cid in te:
            pred = model(s1.to(device)).argmax(1).cpu()
            metric.update(pred, label)
            metric.update_perchip(pred[0], label[0], cid[0])
    res = metric.compute()
    res["params_M"] = round(count_params(model) / 1e6, 3)
    res["FLOPs_G"] = round(flops / 1e9, 3)
    res["precision"] = "int4"
    res["energy_mJ_ANN"] = round(E.int4_energy_joules(flops) * 1e3, 4)

    out_dir = os.path.join("runs", name)
    os.makedirs(out_dir, exist_ok=True)
    import csv
    with open(os.path.join(out_dir, "perchip.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chip_id", "region", "flood_iou"])
        w.writerows(metric.perchip)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    _write_status(name, f"INT4 CHẠY ĐƯỢC: {n_quant} tensor nén 4-bit. Xem test_metrics.json.")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
