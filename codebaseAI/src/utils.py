"""Tiện ích dùng chung: seed, device, config, checkpoint, sliding-window inference."""
import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():        # Mac Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # cho phép kế thừa từ base.yaml qua khóa `_base_`
    base = cfg.pop("_base_", None)
    if base:
        base_path = os.path.join(os.path.dirname(path), base)
        merged = load_config(base_path)
        merged.update(cfg)
        return merged
    return cfg


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, path, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), **extra}, path)


def load_checkpoint(model, path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    return ckpt


@torch.no_grad()
def sliding_window_inference(model, image, window=256, overlap=64, num_classes=3):
    """Suy luận ảnh lớn (512) bằng cửa sổ trượt — dùng khi train ở crop 256.
    image: (1, C, H, W) → logits (1, num_classes, H, W)."""
    _, _, H, W = image.shape
    stride = window - overlap
    logits = torch.zeros(1, num_classes, H, W, device=image.device)
    count = torch.zeros(1, 1, H, W, device=image.device)
    ys = list(range(0, max(H - window, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - window, 0) + 1, stride)) or [0]
    if ys[-1] != H - window and H > window:
        ys.append(H - window)
    if xs[-1] != W - window and W > window:
        xs.append(W - window)
    for y in ys:
        for x in xs:
            patch = image[:, :, y:y + window, x:x + window]
            out = model(patch)
            logits[:, :, y:y + window, x:x + window] += out
            count[:, :, y:y + window, x:x + window] += 1
    return logits / count.clamp(min=1)
