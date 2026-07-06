"""Smoke-test mọi model + pipeline dữ liệu thật (forward/loss/backward). Chạy CPU.
    cd codebaseAI && python smoke_test.py
"""
import traceback
import torch

DEV = torch.device("cpu")
B, C, H, W = 2, 2, 128, 128
NC = 3

print("== Phiên bản thư viện ==")
print("torch", torch.__version__)
try:
    import segmentation_models_pytorch as smp
    print("smp", smp.__version__)
except Exception as e:
    print("smp LỖI:", e)
try:
    import spikingjelly
    print("spikingjelly", getattr(spikingjelly, "__version__", "?"))
except Exception as e:
    print("spikingjelly LỖI:", e)

from src.models import get_model
from src.losses import CombinedLoss


def test_model(name, **kw):
    x = torch.rand(B, C, H, W)
    y = torch.randint(-1, NC, (B, H, W))               # gồm cả nhãn -1 (ignore)
    model = get_model(name, in_channels=C, num_classes=NC, **kw).to(DEV)
    model.train()
    out = model(x.to(DEV))
    assert out.shape == (B, NC, H, W), f"shape sai {tuple(out.shape)} (mong đợi {(B,NC,H,W)})"
    loss = CombinedLoss(num_classes=NC)(out, y.to(DEV))
    loss.backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n = sum(p.numel() for p in model.parameters())
    return tuple(out.shape), float(loss.item()), n, grads


print("\n== Test từng model (input giả 2×2×128×128) ==")
MODELS = [
    ("unet", {}),
    ("unetpp", {}),
    ("deeplabv3", {}),
    ("segformer", {"variant": "b0"}),
    ("spiking_unet", {"T": 2, "base": 16}),
]
for name, kw in MODELS:
    try:
        shape, loss, n, grads = test_model(name, **kw)
        print(f"[OK]   {name:13s} out={shape} loss={loss:.4f} params={n:,} layers_with_grad={grads}")
    except Exception as e:
        print(f"[FAIL] {name:13s} -> {type(e).__name__}: {e}")
        traceback.print_exc()

# Otsu — không phải nn.Module
try:
    import numpy as np
    from src.models.otsu import predict
    m = predict(np.random.rand(C, H, W).astype("float32"))
    print(f"[OK]   otsu          mask={m.shape} unique={np.unique(m).tolist()}")
except Exception as e:
    print(f"[FAIL] otsu -> {e}")

print("\n== Test pipeline trên DỮ LIỆU THẬT ==")
try:
    from src.data import get_dataloader
    dl = get_dataloader("../dataset", "train", batch_size=2, num_workers=0, crop_size=128)
    s1, lab, cid = next(iter(dl))
    print(f"[OK]   load data     s1={tuple(s1.shape)} lab={tuple(lab.shape)} "
          f"nhãn={sorted(torch.unique(lab).tolist())} range_s1=[{s1.min():.2f},{s1.max():.2f}]")
    out = get_model("unet").to(DEV)(s1.to(DEV))
    CombinedLoss()(out, lab.to(DEV)).backward()
    print("[OK]   pipeline thật U-Net forward+loss+backward trên data thật OK")
except Exception as e:
    print(f"[FAIL] data thật -> {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n== Test metrics ==")
try:
    from src.metrics import SegMetrics
    mt = SegMetrics(num_classes=NC)
    pred = torch.randint(0, NC, (B, H, W))
    tgt = torch.randint(-1, NC, (B, H, W))
    mt.update(pred, tgt)
    mt.update_perchip(pred[0], tgt[0])
    r = mt.compute()
    print(f"[OK]   metrics       flood_IoU={r['flood_IoU']:.3f} mIoU={r['mIoU']:.3f} "
          f"keys={list(r.keys())[:4]}...")
except Exception as e:
    print(f"[FAIL] metrics -> {e}")
