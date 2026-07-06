"""
Otsu thresholding — sàn cổ điển (không học), ~0 năng lượng.
Đặt mốc accuracy thấp nhất để cho thấy DL đáng giá.

Không phải nn.Module → evaluate.py gọi predict() trực tiếp (nhánh riêng).
Ngưỡng Otsu trên kênh VV; nước = pixel TỐI (backscatter thấp).
Lưu ý: Otsu chỉ tách nước/không-nước (nhị phân). Để ra 3 lớp, phần "nước" được
gộp thành lớp flood(2) — đây là baseline thô, không dùng JRC.
Trạng thái: ✅ CHẠY ĐƯỢC (cần scikit-image).
"""
import numpy as np


def predict(s1):
    """s1: (C, H, W) đã chuẩn hóa [0,1] → mask (H, W) ∈ {0 nền, 2 flood}."""
    from skimage.filters import threshold_otsu
    vv = s1[0]
    try:
        thr = threshold_otsu(vv)
    except ValueError:
        return np.zeros(vv.shape, dtype=np.int64)
    water = vv < thr                    # nước = tối
    out = np.zeros(vv.shape, dtype=np.int64)
    out[water] = 2
    return out
