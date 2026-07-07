#!/usr/bin/env bash
# ============================================================================
# Chạy TOÀN BỘ thí nghiệm vòng 2 (rigor): baselines pretrained + SNN, multi-seed.
# Chạy trong tmux:  bash run_all.sh 2>&1 | tee runs/run_all.log
# Một run lỗi sẽ KHÔNG dừng cả script (in FAILED rồi chạy tiếp).
# ============================================================================
SEEDS="0 1 2"
# unet = from-scratch (cặp so sánh CÔNG BẰNG với SNN); còn lại pretrained ImageNet
BASELINES="unet unet_smp unetpp deeplabv3 segformer_b2 mobilenet_unet"

run() { echo ">>> $*"; "$@" || echo "!!! FAILED: $*"; }

echo "===== BASELINES (ANN pretrained) x 3 seed ====="
for cfg in $BASELINES; do
  for s in $SEEDS; do
    run python train.py    --config configs/$cfg.yaml --seed $s
    run python evaluate.py --config configs/$cfg.yaml --seed $s
  done
done

echo "===== SNN Spiking U-Net (T4) x 3 seed ====="
for s in $SEEDS; do
  run python train.py    --config configs/spiking_unet.yaml --seed $s
  run python evaluate.py --config configs/spiking_unet.yaml --seed $s
done

echo "===== Ablation T (1 seed cho nhanh) ====="
for cfg in spiking_unet_T2 spiking_unet_T6 spiking_unet_T8; do
  run python train.py    --config configs/$cfg.yaml
  run python evaluate.py --config configs/$cfg.yaml
done

echo "===== Otsu (khong train) ====="
run python evaluate.py --config configs/otsu.yaml

echo "===== ANN2SNN (convert tu unet_smp seed 0) ====="
run python ann2snn_convert.py --ann_config configs/unet_smp.yaml --ann_run unet_smp_s0 --T 32 --name ann2snn

echo "===== Tong hop bang + Pareto ====="
run python summarize.py
echo ""
echo "XONG. Nho: tar -czf runs_backup.tar.gz runs/  -> tai ve -> destroy may."
