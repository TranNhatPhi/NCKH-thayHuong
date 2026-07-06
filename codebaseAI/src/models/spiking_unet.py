"""
★ ĐÓNG GÓP LÕI CỦA ĐỀ TÀI — Spiking U-Net (SNN).

U-Net nhưng thay mọi ReLU bằng nơ-ron LIF (Leaky Integrate-and-Fire), huấn luyện
trực tiếp bằng surrogate gradient (BPTT). Mã hóa input SAR theo kiểu "direct":
lặp ảnh qua T bước, để lớp LIF đầu tự sinh spike; đầu ra là logits tích lũy /
trung bình qua T bước → giữ CÙNG interface forward(x)->(B,num_classes,H,W) như CNN.

Yêu cầu:  pip install spikingjelly
Trạng thái: 🟡 SKELETON — cần kiểm thử forward/backward trên GPU + tinh chỉnh T, ngưỡng.

Ghi chú thiết kế cần chốt (xem SNN-Flood_Pipeline.md):
  * T (số bước): làm ablation {2,4,6,8}.
  * Decoder: dùng membrane potential ở head (không spiking) để ra logits mượt.
  * Đo năng lượng: gắn hook đếm spike ở energy.py (SynOps).
"""
import torch
import torch.nn as nn


def _lif():
    from spikingjelly.activation_based import neuron, surrogate
    return neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)


class SpikingDoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.sn1 = _lif()
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.sn2 = _lif()

    def forward(self, x):
        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        return x


class SpikingUNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=3, base=32, T=4, encoding="direct"):
        super().__init__()
        self.T = T
        self.encoding = encoding
        c = [base, base * 2, base * 4, base * 8]
        self.d1 = SpikingDoubleConv(in_channels, c[0])
        self.d2 = SpikingDoubleConv(c[0], c[1])
        self.d3 = SpikingDoubleConv(c[1], c[2])
        self.bott = SpikingDoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.u3 = SpikingDoubleConv(c[3], c[2])
        self.u2 = SpikingDoubleConv(c[2], c[1])
        self.u1 = SpikingDoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], num_classes, 1)      # đầu ra thực (không spiking)

    def _forward_once(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        b = self.bott(self.pool(s3))
        x = self.u3(torch.cat([self.up3(b), s3], 1))
        x = self.u2(torch.cat([self.up2(x), s2], 1))
        x = self.u1(torch.cat([self.up1(x), s1], 1))
        return self.head(x)

    def forward(self, x):
        from spikingjelly.activation_based import functional
        functional.reset_net(self)               # xóa trạng thái màng trước mỗi mẫu
        # direct encoding: cùng ảnh x đưa vào T bước; LIF tích lũy theo thời gian
        out = 0.0
        for _ in range(self.T):
            out = out + self._forward_once(x)
        return out / self.T                      # logits trung bình qua T bước
