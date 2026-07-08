#!/usr/bin/env bash
# ============================================================================
# CHẠY TOÀN BỘ TỪ ĐẦU (bulletproof, không phụ thuộc gì đã train trước):
#   14 model × 3 seed (ANN baseline + LR sweep + SNN) + Otsu + ANN2SNN T-sweep
#   → summarize (mean±std, spike_rate) + analysis (Wilcoxon + per-region)
# Mọi thứ dùng code mới nhất (grad-clip, perchip, multi-step SNN).
# Chạy trong tmux:  bash run_full.sh 2>&1 | tee runs/run_full.log   (~2.5–3h)
# 1 run lỗi KHÔNG dừng cả script.
# ============================================================================
# Kích hoạt venv nếu shell chưa có python (vd tmux mới không auto-activate)
command -v python >/dev/null 2>&1 || { [ -f /venv/main/bin/activate ] && source /venv/main/bin/activate; }

SEEDS="0 1 2"
run(){ echo ">>> $*"; "$@" || echo "!!! FAILED: $*"; }

TRAINABLE="unet unet_smp unetpp deeplabv3 segformer_b2 mobilenet_unet \
mobilenet_unet_lr5e5 mobilenet_unet_lr2e4 \
spiking_unet spiking_unet_T2 spiking_unet_T6 spiking_unet_T8 \
spiking_unet_lr5e5 spiking_unet_lr2e4"

echo "===== TRAIN + EVAL 14 model × 3 seed ====="
for cfg in $TRAINABLE; do
  for s in $SEEDS; do
    run python train.py    --config configs/$cfg.yaml --seed $s
    run python evaluate.py --config configs/$cfg.yaml --seed $s
  done
done

echo "===== Otsu (không train) × 3 seed ====="
for s in $SEEDS; do run python evaluate.py --config configs/otsu.yaml --seed $s; done

echo "===== ANN2SNN quét T=32/64/128 (x3 seed nguồn: unet_smp_s0/s1/s2 -> có σ) ====="
for T in 32 64 128; do
  for s in $SEEDS; do
    run python ann2snn_convert.py --ann_config configs/unet_smp.yaml \
      --ann_run unet_smp_s$s --T $T --name ann2snn_T${T}_s${s}
  done
done

echo "===== Tổng hợp + phân tích ====="
run python summarize.py
run python analysis.py --ref spiking_unet_T4

echo ""
echo "XONG. Nhớ: tar -czf runs_full.tar.gz runs/  ->  tải về  ->  destroy máy."
