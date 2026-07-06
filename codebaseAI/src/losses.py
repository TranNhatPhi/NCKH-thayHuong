"""
Loss DÙNG CHUNG cho mọi model: CrossEntropy(ignore -1) + Dice + Focal.
Chống mất cân bằng lớp (nền 77% · nước thường trực 2.7% · flood 6.5%).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, ignore_index=-1):
        super().__init__()
        self.n = num_classes
        self.ignore = ignore_index

    def forward(self, logits, target):
        prob = F.softmax(logits, dim=1)
        mask = (target != self.ignore)
        t = target.clone()
        t[~mask] = 0
        oh = F.one_hot(t, self.n).permute(0, 3, 1, 2).float()
        m = mask.unsqueeze(1).float()
        prob, oh = prob * m, oh * m
        dims = (0, 2, 3)
        inter = (prob * oh).sum(dims)
        card = prob.sum(dims) + oh.sum(dims)
        dice = (2 * inter + 1e-6) / (card + 1e-6)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.ignore = ignore_index

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class CombinedLoss(nn.Module):
    """w_ce·CE + w_dice·Dice + w_focal·Focal (mặc định = tổng đều)."""
    def __init__(self, num_classes=3, ignore_index=-1,
                 w_ce=1.0, w_dice=1.0, w_focal=1.0, gamma=2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)
        self.focal = FocalLoss(gamma, ignore_index)
        self.w_ce, self.w_dice, self.w_focal = w_ce, w_dice, w_focal

    def forward(self, logits, target):
        return (self.w_ce * self.ce(logits, target)
                + self.w_dice * self.dice(logits, target)
                + self.w_focal * self.focal(logits, target))
