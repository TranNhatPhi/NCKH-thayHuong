"""
SegFormer (Transformer SOTA) — mốc accuracy trần. Biến thể b0/b1/b2 như paper 2026.
Dùng smp (MiT encoder). Trạng thái: ✅ CHẠY ĐƯỢC khi đã cài smp.

    build_segformer(variant="b2")   # b0 | b1 | b2
"""


def build_segformer(in_channels=2, num_classes=3, variant="b2",
                    encoder_weights="imagenet"):
    import segmentation_models_pytorch as smp
    encoder = {"b0": "mit_b0", "b1": "mit_b1", "b2": "mit_b2"}[variant]
    return smp.Segformer(
        encoder_name=encoder,
        encoder_weights=encoder_weights,      # "imagenet": pretrain MiT encoder
        in_channels=in_channels,
        classes=num_classes,
    )
