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
# torchao cho thử INT4 (không bắt buộc; thiếu thì quantize_int4.py tự ghi 'unsupported'):
pip install torchao 2>/dev/null || echo "   (bỏ qua torchao — INT4 sẽ báo unsupported, vẫn là finding)"

echo "==> 2. Kiểm tra GPU (quan trọng cho RTX 50-series / Blackwell)"
python -c "
import torch
print('torch', torch.__version__, '| CUDA build', torch.version.cuda)
assert torch.cuda.is_available(), 'KHONG thay GPU CUDA — chon template co CUDA!'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(2000, 2000, device='cuda'); float((x @ x).sum())   # test op that tren GPU
print('Test GPU op: OK — torch chay duoc tren GPU nay')
" || { echo '   [LOI] torch khong chay duoc tren GPU (5090 can PyTorch >=2.6, CUDA 12.8+).'; echo '         -> chon Template PyTorch moi hon.'; exit 1; }

echo "==> 3. Tải dữ liệu từ Google Cloud + dựng nhãn 3 lớp"
cd ../dataset
[ -d S1Hand ]      || python download_data.py        # S1Hand+LabelHand+JRC (~300MB)
[ -d Label3Class ] || python build_3class_labels.py  # dựng nhãn 3 lớp
# splits/*.csv đã có sẵn trong repo (271/87/87) — không cần chạy make_splits
cd ../codebaseAI

echo "==> 4. Smoke-test toàn bộ model + pipeline (forward/backward, KHÔNG train đầy đủ)"
python smoke_test.py

echo ""
echo "==> Sẵn sàng train. Ví dụ:"
echo "   python train.py    --config configs/unet.yaml"
echo "   python train.py    --config configs/spiking_unet.yaml"
echo "   python evaluate.py --config configs/unet.yaml"
echo ""
echo "LƯU Ý: storage Vast là TẠM THỜI — tải runs/ về máy TRƯỚC KHI destroy instance."
