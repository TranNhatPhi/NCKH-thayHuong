"""
DeepLabV3 baseline — smp, backbone ResNet-34 (khớp Banerjee & Daou 2026).
Trạng thái: ✅ CHẠY ĐƯỢC khi đã `pip install segmentation-models-pytorch`.
"""


def build_deeplabv3(in_channels=2, num_classes=3, encoder="resnet34",
                    encoder_weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3(
        encoder_name=encoder,
        encoder_weights=encoder_weights,      # "imagenet": pretrain
        in_channels=in_channels,
        classes=num_classes,
    )
