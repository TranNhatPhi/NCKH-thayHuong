#!/usr/bin/env bash
# ============================================================================
# VÒNG PHÂN TÍCH CUỐI (framing "honest energy-accuracy analysis"):
#   - LR sweep MobileNet + SNN (fairness đối xứng)
#   - Retrain toàn bộ SNN (T4 + T2/T6/T8) VỚI gradient-clipping mới (ổn định)
#   - Re-eval mọi ANN baseline để sinh perchip.csv + spike_rate
#   - ANN2SNN quét T=32/64/128
#   - summarize + analysis (Wilcoxon + per-region)
# Chạy trong tmux:  bash run_all2.sh 2>&1 | tee runs/run_all2.log   (~1.5–2h)
# ============================================================================
SEEDS="0 1 2"
run(){ echo ">>> $*"; "$@" || echo "!!! FAILED: $*"; }

echo "===== 1) LR sweep MobileNet + SNN (train+eval, 3 seed) ====="
for cfg in mobilenet_unet_lr5e5 mobilenet_unet_lr2e4 spiking_unet_lr5e5 spiking_unet_lr2e4; do
  for s in $SEEDS; do
    run python train.py    --config configs/$cfg.yaml --seed $s
    run python evaluate.py --config configs/$cfg.yaml --seed $s
  done
done

echo "===== 2) Retrain SNN (grad-clip) T4 + ablation T, 3 seed ====="
for cfg in spiking_unet spiking_unet_T2 spiking_unet_T6 spiking_unet_T8; do
  for s in $SEEDS; do
    run python train.py    --config configs/$cfg.yaml --seed $s
    run python evaluate.py --config configs/$cfg.yaml --seed $s
  done
done

echo "===== 3) Re-eval ANN baseline (CHỈ eval — lấy perchip + spike_rate) ====="
for cfg in unet unet_smp unetpp deeplabv3 segformer_b2 mobilenet_unet; do
  for s in $SEEDS; do run python evaluate.py --config configs/$cfg.yaml --seed $s; done
done
for s in $SEEDS; do run python evaluate.py --config configs/otsu.yaml --seed $s; done

echo "===== 4) ANN2SNN quét T=32/64/128 (có perchip) ====="
for T in 32 64 128; do
  run python ann2snn_convert.py --ann_config configs/unet_smp.yaml --ann_run unet_smp_s0 --T $T --name ann2snn_T$T
done

echo "===== 5) Tổng hợp + phân tích ====="
run python summarize.py
run python analysis.py --ref spiking_unet_T4

echo ""
echo "XONG. Nhớ: tar -czf runs_v3.tar.gz runs/  -> tải về -> destroy máy."
