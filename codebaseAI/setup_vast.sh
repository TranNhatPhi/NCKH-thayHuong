#!/usr/bin/env bash
# ============================================================================
# Thiết lập máy thuê Vast.ai để train SNN-Flood.
# Chọn image có sẵn CUDA + PyTorch (ví dụ template "PyTorch 2.x cuda12").
#
# Cách dùng trên máy Vast:
#   git clone <repo-của-bạn> && cd <repo>/codebaseAI
#   bash setup_vast.sh
# Giả định: thư mục ../dataset/ đã có S1Hand/ + LabelHand/ (đưa lên trước).
# ============================================================================
set -e

echo "==> 1. Cài phụ thuộc Python"
pip install -r requirements.txt
# SpikingJelly chạy nhanh hơn với CUDA kernel (tùy chọn; chọn đúng bản CUDA):
pip install cupy-cuda12x 2>/dev/null || echo "   (bỏ qua cupy — SNN vẫn chạy bằng backend torch)"

echo "==> 2. Kiểm tra GPU"
python -c "import torch; ok=torch.cuda.is_available(); print('CUDA:', ok, '|', torch.cuda.get_device_name(0) if ok else 'KHÔNG THẤY GPU')"

echo "==> 3. Tải dữ liệu từ Google Cloud + dựng nhãn 3 lớp"
cd ../dataset
[ -d S1Hand ]      || python download_data.py        # S1Hand+LabelHand+JRC (~300MB)
[ -d Label3Class ] || python build_3class_labels.py  # dựng nhãn 3 lớp
# splits/*.csv đã có sẵn trong repo (271/87/87) — không cần chạy make_splits
cd ../codebaseAI

echo "==> 4. Smoke-test nhanh (U-Net vài bước)"
python train.py --config configs/unet.yaml || echo "   (kiểm tra lỗi ở trên nếu có)"

echo ""
echo "XONG. Lưu ý: storage trên Vast là TẠM THỜI —"
echo "  tải thư mục runs/ (checkpoint + kết quả) về máy TRƯỚC KHI destroy instance."
