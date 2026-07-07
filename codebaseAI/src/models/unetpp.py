"""
U-Net++ (nested U-Net) baseline — dùng segmentation_models_pytorch (smp),
backbone ResNet-34 giống paper Banerjee & Daou 2026 để đối chiếu công bằng.
Trạng thái: ✅ CHẠY ĐƯỢC khi đã `pip install segmentation-models-pytorch`.
"""


def build_unetpp(in_channels=2, num_classes=3, encoder="resnet34",
                 encoder_weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,      # "imagenet": pretrain (smp tự thích ứng first conv cho 2 kênh)
        in_channels=in_channels,
        classes=num_classes,
    )
