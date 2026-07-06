"""
ANN2SNN — baseline chuyển đổi: train U-Net (ANN) rồi convert sang SNN.
Dùng để so sánh NỘI BỘ họ SNN (chứng minh direct-training của SpikingUNet tốt hơn).

Trạng thái: 🟡 SKELETON. Quy trình:
  1. Train UNet (ANN) bình thường (train.py --config configs/unet.yaml).
  2. Dùng spikingjelly.activation_based.ann2snn.Converter để chuyển checkpoint đó.
Yêu cầu:  pip install spikingjelly
"""


def build_ann2snn(in_channels=2, num_classes=3, ann_ckpt=None, T=32, mode="max"):
    from .unet import UNet
    ann = UNet(in_channels, num_classes)
    if ann_ckpt:
        import torch
        ann.load_state_dict(torch.load(ann_ckpt, map_location="cpu")["model"])
    # TODO: chuyển đổi bằng spikingjelly Converter (cần dữ liệu hiệu chỉnh)
    #   from spikingjelly.activation_based.ann2snn import Converter
    #   snn = Converter(mode=mode, dataloader=calib_loader)(ann)
    #   return snn
    raise NotImplementedError(
        "ann2snn: hoàn thiện sau khi có checkpoint UNet — dùng spikingjelly Converter.")
