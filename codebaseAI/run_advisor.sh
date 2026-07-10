#!/usr/bin/env bash
# ============================================================================
# VÒNG THÍ NGHIỆM THEO GÓP Ý CỦA THẦY:
#   1) T-sweep mở rộng: T=1,3,5,7,10 (x3 seed) -> xác nhận vùng T bimodal
#      ("T ổn định là hiếm" — T2/T8 ổn, T4/T6 bất ổn; cần biết T1/T10/T3/T5/T7 thế nào)
#   2) Retrain T2 & T8 với LR 2e-4 (recipe MobileNet đang thắng) -> có lên 0.42-0.45?
#   3) MobileNet-UNet INT8 (PTQ) x3 seed -> điểm Pareto "ANN nén" ở dải năng lượng thấp
#   4) summarize + analysis lại
# Chạy trong tmux:  bash run_advisor.sh 2>&1 | tee runs/run_advisor.log   (~1.5-2h)
# 1 run lỗi KHÔNG dừng cả script.
# ============================================================================
command -v python >/dev/null 2>&1 || { [ -f /venv/main/bin/activate ] && source /venv/main/bin/activate; }
SEEDS="0 1 2"
run(){ echo ">>> $*"; "$@" || echo "!!! FAILED: $*"; }

echo "===== 1) T-sweep mở rộng: T=1,3,5,7,10 (x3 seed) ====="
for cfg in spiking_unet_T1 spiking_unet_T3 spiking_unet_T5 spiking_unet_T7 spiking_unet_T10; do
  for s in $SEEDS; do
    run python train.py    --config configs/$cfg.yaml --seed $s
    run python evaluate.py --config configs/$cfg.yaml --seed $s
  done
done

echo "===== 2) T2 & T8 với LR 2e-4 (x3 seed) ====="
for cfg in spiking_unet_T2_lr2e4 spiking_unet_T8_lr2e4; do
  for s in $SEEDS; do
    run python train.py    --config configs/$cfg.yaml --seed $s
    run python evaluate.py --config configs/$cfg.yaml --seed $s
  done
done

echo "===== 3) T=6 thêm seed 3,4 (=> n=5, verify lucky hay stable) ====="
for s in 3 4; do
  run python train.py    --config configs/spiking_unet_T6.yaml --seed $s
  run python evaluate.py --config configs/spiking_unet_T6.yaml --seed $s
done

echo "===== 4) MobileNet-UNet INT8 PTQ (x3 seed, nguồn = mobilenet_unet_lr2e4_sN) ====="
for s in $SEEDS; do
  run python quantize_int8.py --seed $s
done

echo "===== 4b) Spiking-MobileNet (depthwise-separable + LIF) — so fair với INT8 (x3 seed) ====="
for cfg in spiking_mobilenet_T2 spiking_mobilenet_T4; do
  for s in $SEEDS; do
    run python train.py    --config configs/$cfg.yaml --seed $s
    run python evaluate.py --config configs/$cfg.yaml --seed $s
  done
done

# (INT4 đã bỏ theo ý thầy 10/07: tooling không hỗ trợ Conv2d → giảm giá trị benchmark)

echo "===== 5) Tổng hợp + phân tích ====="
run python summarize.py
run python analysis.py --ref spiking_unet_T4

echo ""
echo "XONG. Kiểm tra: T-sweep đầy đủ T1..T10 + T2/T8 lr2e4 + mobilenet_int8."
echo "Nếu ổn: tar -czf runs_full.tar.gz runs/  ->  tải về  ->  destroy máy."
