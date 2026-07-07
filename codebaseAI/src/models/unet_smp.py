"""
U-Net với encoder ResNet-34 pretrain ImageNet (smp) — baseline U-Net "mạnh"
khớp Banerjee & Daou 2026. Khác với unet.py (tự chứa, train from-scratch, nhẹ).
Trạng thái: ✅ cần smp + mạng để tải weight pretrain.
"""


def build_unet_smp(in_channels=2, num_classes=3, encoder="resnet34",
                   encoder_weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
