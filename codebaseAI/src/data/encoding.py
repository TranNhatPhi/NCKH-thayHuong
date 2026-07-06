"""
Mã hóa ảnh SAR tĩnh → chuỗi spike cho SNN, qua T bước thời gian.
Dùng cho các model SNN (spiking_unet). CNN/Transformer KHÔNG cần bước này.
"""
import torch


def direct_encode(x, T):
    """Direct/analog coding: lặp ảnh qua T bước; lớp LIF đầu tự sinh spike.
    x: (B, C, H, W) ∈ [0,1] → (T, B, C, H, W)."""
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)


def rate_encode(x, T):
    """Rate coding: pixel càng sáng, xác suất bắn spike càng cao (Bernoulli).
    x: (B, C, H, W) ∈ [0,1] → (T, B, C, H, W) nhị phân {0,1}."""
    x = x.clamp(0, 1).unsqueeze(0)
    return (torch.rand(T, *x.shape[1:], device=x.device) < x).float()
