"""
MobileNet-UNet INT8 (Post-Training Quantization) — thêm 1 điểm Pareto "ANN nén"
để so trực tiếp với SNN ở dải năng lượng cực thấp (theo yêu cầu thầy).

Ý tưởng: lấy MobileNet-UNet FP32 đã thắng (recipe lr2e4), lượng tử hoá INT8 tĩnh
(FX graph-mode PTQ, hiệu chỉnh trên train split), đánh giá accuracy trên test.
Năng lượng = số MAC (không đổi) × E_MAC_INT8 (~0.23pJ, rẻ ~20× so với FP32).

    python quantize_int8.py --seed 0
    python quantize_int8.py --src_run mobilenet_unet_lr2e4_s0 --name mobilenet_int8_s0

Ghi chú recipe (để verify với thầy):
  * Phương pháp: **PTQ (post-training static)** — không cần train lại, hiệu chỉnh (calibrate)
    thống kê activation trên train split. (QAT sẽ chính xác hơn nhưng phải train lại; nêu
    trong Discussion như hướng cải thiện nếu PTQ tụt accuracy nhiều.)
  * Calibration set: lấy `--calib_batches` batch từ TRAIN split (mặc định 50 → ~200 ảnh,
    đủ đại diện cho phân bố activation).
  * qconfig: per-channel weight + per-tensor activation (mặc định backend x86/fbgemm).
  * kernel INT8 chạy trên CPU → eval để trên CPU. Nếu FX không trace được model smp,
    script báo lỗi rõ để ta chuyển sang eager-mode PTQ.
"""
import argparse
import copy
import json
import os

import torch

from src.utils import load_config, count_params
from src.data import get_dataloader
from src.models import get_model
from src.metrics import SegMetrics
from src import energy as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mobilenet_unet_lr2e4.yaml",
                    help="config của MobileNet-UNet FP32 nguồn")
    ap.add_argument("--src_run", default=None,
                    help="run dir FP32 nguồn (mặc định = name; multi-seed dùng vd mobilenet_unet_lr2e4_s0)")
    ap.add_argument("--seed", type=int, default=None,
                    help="tiện lợi: tự suy src_run/name theo seed")
    ap.add_argument("--name", default=None, help="tên run đầu ra (mặc định mobilenet_int8[_sN])")
    ap.add_argument("--calib_batches", type=int, default=50, help="số batch train để hiệu chỉnh")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")   # INT8 kernel chạy CPU
    C = cfg.get("in_channels", 2)

    src_run = args.src_run or (f"{cfg['name']}_s{args.seed}" if args.seed is not None else cfg["name"])
    name = args.name or ("mobilenet_int8" + (f"_s{args.seed}" if args.seed is not None else ""))
    ckpt = os.path.join("runs", src_run, "best.pt")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"Không thấy {ckpt} — cần train MobileNet-UNet (recipe lr2e4) trước.")

    # 1. Nạp FP32
    from src.utils import load_checkpoint
    model_fp32 = get_model(cfg["model"], in_channels=C, num_classes=3,
                           **cfg.get("model_kwargs", {})).to(device).eval()
    load_checkpoint(model_fp32, ckpt, map_location=device)

    # 2. FLOPs (bất biến với quantization) → năng lượng INT8
    flops, _ = E.count_flops_params(copy.deepcopy(model_fp32), input_shape=(1, C, 512, 512), device="cpu")

    # 3. FX static PTQ
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    try:
        backend = "x86"
        torch.backends.quantized.engine = backend
    except Exception:
        backend = "fbgemm"
        torch.backends.quantized.engine = backend

    calib = get_dataloader(cfg["data_root"], "train", batch_size=4, num_workers=0,
                           crop_size=cfg.get("crop_size"),
                           per_channel_norm=cfg.get("per_channel_norm", False))
    example = next(iter(calib))[0].to(device)
    qmap = get_default_qconfig_mapping(backend)
    try:
        prepared = prepare_fx(copy.deepcopy(model_fp32), qmap, example_inputs=(example,))
    except Exception as e:
        raise SystemExit(f"[FX PTQ] không trace được model: {e}\n"
                         "→ Model smp có op FX không hỗ trợ. Báo lại để chuyển sang eager-mode PTQ.")

    # Hiệu chỉnh (calibration)
    with torch.no_grad():
        for i, (s1, _, _) in enumerate(calib):
            prepared(s1.to(device))
            if i + 1 >= args.calib_batches:
                break
    model_int8 = convert_fx(prepared).eval()

    # 4. Đánh giá test
    te = get_dataloader(cfg["data_root"], "test", batch_size=1, num_workers=0,
                        per_channel_norm=cfg.get("per_channel_norm", False))
    metric = SegMetrics()
    with torch.no_grad():
        for s1, label, cid in te:
            pred = model_int8(s1.to(device)).argmax(1)
            metric.update(pred, label)
            metric.update_perchip(pred[0], label[0], cid[0])
    res = metric.compute()

    # 5. Năng lượng INT8 (đánh dấu là ANN để summarize xếp đúng nhóm)
    res["params_M"] = round(count_params(model_fp32) / 1e6, 3)
    res["FLOPs_G"] = round(flops / 1e9, 3)
    res["precision"] = "int8"
    res["energy_mJ_ANN"] = round(E.int8_energy_joules(flops) * 1e3, 4)

    out_dir = os.path.join("runs", name)
    os.makedirs(out_dir, exist_ok=True)
    import csv
    with open(os.path.join(out_dir, "perchip.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chip_id", "region", "flood_iou"])
        w.writerows(metric.perchip)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"Đã lưu runs/{name}/test_metrics.json  (backend={backend})")


if __name__ == "__main__":
    main()
