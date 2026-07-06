"""
Factory model — điểm vào DUY NHẤT để lấy bất kỳ mô hình nào.

    from src.models import get_model
    model = get_model("unet", in_channels=2, num_classes=3)

Mọi model trả về nn.Module có CÙNG interface:  forward(x) -> logits (B, num_classes, H, W)
→ nhờ vậy train.py / evaluate.py dùng chung được cho CNN, Transformer lẫn SNN
  (SNN tự lo phần mã hóa spike + mô phỏng T bước bên trong forward).

Bảng model:
  Học được (nn.Module):  unet · unetpp · deeplabv3 · segformer · spiking_unet · ann2snn
  Không học (đặc biệt):  otsu  (xử lý riêng trong evaluate.py)
"""

LEARNABLE = ["unet", "unetpp", "deeplabv3", "segformer", "spiking_unet", "ann2snn"]
CLASSICAL = ["otsu"]


def get_model(name, in_channels=2, num_classes=3, **kwargs):
    name = name.lower()
    if name == "unet":
        from .unet import UNet
        return UNet(in_channels, num_classes, **kwargs)
    if name == "unetpp":
        from .unetpp import build_unetpp
        return build_unetpp(in_channels, num_classes, **kwargs)
    if name == "deeplabv3":
        from .deeplabv3 import build_deeplabv3
        return build_deeplabv3(in_channels, num_classes, **kwargs)
    if name == "segformer":
        from .segformer import build_segformer
        return build_segformer(in_channels, num_classes, **kwargs)
    if name == "spiking_unet":
        from .spiking_unet import SpikingUNet
        return SpikingUNet(in_channels, num_classes, **kwargs)
    if name == "ann2snn":
        from .ann2snn import build_ann2snn
        return build_ann2snn(in_channels, num_classes, **kwargs)
    raise ValueError(f"Model chưa biết: '{name}'. Chọn trong {LEARNABLE + CLASSICAL}")
