#!/usr/bin/env bash
# ============================================================================
# VÒNG BỔ SUNG THEO Ý THẦY (chat 10/07: "Được thì chạy luôn đi, cho nó chắc"):
#   Train Spiking-MobileNet (depthwise-separable + LIF, 1.52M params)
#   → so sánh CÔNG BẰNG về kiến trúc với MobileNet-UNet INT8.
#
# CHỈ train 6 run còn thiếu: spiking_mobilenet_T2 + T4, mỗi cái 3 seed.
# 80 run cũ (v2, split 60/20/20) ĐÃ CÓ ở máy local — không train lại.
#
# Cách dùng trên máy Vast H100 MỚI (sau khi setup_vast.sh xong):
#   tmux new -s mb
#   bash run_mobilenet_snn.sh 2>&1 | tee runs/run_mobilenet_snn.log
# Ước tính: ~6 run × 25-35 phút ≈ 2.5-3.5h trên H100.
# 1 run lỗi KHÔNG dừng cả script.
# ============================================================================
command -v python >/dev/null 2>&1 || { [ -f /venv/main/bin/activate ] && source /venv/main/bin/activate; }
mkdir -p runs
SEEDS="0 1 2"
run(){ echo ">>> $*"; "$@" || echo "!!! FAILED: $*"; }

echo "===== Spiking-MobileNet T2 & T4 (x3 seed) — split v2 60/20/20 ====="
for cfg in spiking_mobilenet_T2 spiking_mobilenet_T4; do
  for s in $SEEDS; do
    run python train.py    --config configs/$cfg.yaml --seed $s
    run python evaluate.py --config configs/$cfg.yaml --seed $s
  done
done

echo "===== Tổng hợp nhanh (chỉ trên runs/ của máy này) ====="
run python summarize.py

echo ""
echo "===== ĐÓNG GÓI để tải về (chỉ 6 run mới — nhẹ) ====="
tar -czf runs_spiking_mobilenet.tar.gz runs/spiking_mobilenet_T2_s* runs/spiking_mobilenet_T4_s* runs/run_mobilenet_snn.log 2>/dev/null \
  && ls -lh runs_spiking_mobilenet.tar.gz
echo ""
echo "XONG. Tải runs_spiking_mobilenet.tar.gz về máy local rồi mới destroy:"
echo "  scp -P <port> root@<ip>:/workspace/NCKH-thayHuong/codebaseAI/runs_spiking_mobilenet.tar.gz ."
echo "Về local: tar -xzf runs_spiking_mobilenet.tar.gz  (merge vào runs/ sẵn có)"
echo "         python summarize.py && python make_figures.py && python ../paper/make_docx.py"
