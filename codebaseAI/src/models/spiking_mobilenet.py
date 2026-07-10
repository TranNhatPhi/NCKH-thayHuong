"""
★ Spiking-MobileNet-UNet — theo yêu cầu thầy (review 10/07): so SÁNH CÔNG BẰNG với INT8 MobileNet.

Khác Spiking U-Net thường ở chỗ thay full 3×3 conv bằng DEPTHWISE-SEPARABLE conv (như MobileNetV2):
    depthwise 3×3 (groups=cin) → pointwise 1×1
mỗi conv đều Conv-BN-LIF multi-step (step_mode='m', tdBN over (T,B)).
=> ít MAC/params hơn nhiều → SynOps thấp → năng lượng thấp, để đối chứng đúng "vũ khí" với MobileNet.

Cùng khung 4-level encoder–decoder + skip như SpikingUNet để so được trực tiếp.
Yêu cầu: pip install spikingjelly
"""
import torch
import torch.nn as nn


def _ds_unit(cin, cout):
    """1 đơn vị depthwise-separable spiking: DW(3×3,groups=cin)-BN-LIF → PW(1×1)-BN-LIF."""
    from spikingjelly.activation_based import layer, neuron, surrogate

    def lif():
        return neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(),
                              detach_reset=True, step_mode="m")
    return nn.Sequential(
        layer.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False, step_mode="m"),
        layer.BatchNorm2d(cin, step_mode="m"),
        lif(),
        layer.Conv2d(cin, cout, 1, bias=False, step_mode="m"),
        layer.BatchNorm2d(cout, step_mode="m"),
        lif(),
    )


def _block(cin, cout):
    """2 đơn vị DW-separable (khớp cấu trúc 2-conv/block của U-Net)."""
    return nn.Sequential(_ds_unit(cin, cout), _ds_unit(cout, cout))


class SpikingMobileUNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=3, base=32, T=4, encoding="direct"):
        super().__init__()
        from spikingjelly.activation_based import layer
        self.T = T
        self.encoding = encoding
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.d1 = _block(in_channels, c[0])
        self.d2 = _block(c[0], c[1])
        self.d3 = _block(c[1], c[2])
        self.d4 = _block(c[2], c[3])
        self.bott = _block(c[3], c[4])
        self.pool = layer.MaxPool2d(2, step_mode="m")
        self.up4 = layer.ConvTranspose2d(c[4], c[3], 2, stride=2, step_mode="m")
        self.up3 = layer.ConvTranspose2d(c[3], c[2], 2, stride=2, step_mode="m")
        self.up2 = layer.ConvTranspose2d(c[2], c[1], 2, stride=2, step_mode="m")
        self.up1 = layer.ConvTranspose2d(c[1], c[0], 2, stride=2, step_mode="m")
        self.u4 = _block(c[4], c[3])
        self.u3 = _block(c[3], c[2])
        self.u2 = _block(c[2], c[1])
        self.u1 = _block(c[1], c[0])
        self.head = layer.Conv2d(c[0], num_classes, 1, step_mode="m")

    def forward(self, x):
        from spikingjelly.activation_based import functional
        functional.reset_net(self)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)      # [B,C,H,W] → [T,B,C,H,W]
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        s4 = self.d4(self.pool(s3))
        b = self.bott(self.pool(s4))
        x = self.u4(torch.cat([self.up4(b), s4], dim=2))   # kênh ở dim=2 (T,B,C,H,W)
        x = self.u3(torch.cat([self.up3(x), s3], dim=2))
        x = self.u2(torch.cat([self.up2(x), s2], dim=2))
        x = self.u1(torch.cat([self.up1(x), s1], dim=2))
        out = self.head(x)                                 # [T,B,num_classes,H,W]
        return out.mean(0)                                 # [B,num_classes,H,W]
