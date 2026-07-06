"""
U-Net baseline (ANN) — tự chứa, chỉ cần PyTorch (không phụ thuộc thư viện ngoài).
Đây là baseline accuracy chính, và cũng là "khuôn" để so với Spiking U-Net.
Trạng thái: ✅ CHẠY ĐƯỢC.
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=3, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.d1 = DoubleConv(in_channels, c[0])
        self.d2 = DoubleConv(c[0], c[1])
        self.d3 = DoubleConv(c[1], c[2])
        self.d4 = DoubleConv(c[2], c[3])
        self.bott = DoubleConv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.u4 = DoubleConv(c[4], c[3])
        self.u3 = DoubleConv(c[3], c[2])
        self.u2 = DoubleConv(c[2], c[1])
        self.u1 = DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        s4 = self.d4(self.pool(s3))
        b = self.bott(self.pool(s4))
        x = self.u4(torch.cat([self.up4(b), s4], 1))
        x = self.u3(torch.cat([self.up3(x), s3], 1))
        x = self.u2(torch.cat([self.up2(x), s2], 1))
        x = self.u1(torch.cat([self.up1(x), s1], 1))
        return self.head(x)                       # (B, num_classes, H, W)
