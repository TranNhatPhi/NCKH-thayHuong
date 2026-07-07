#!/usr/bin/env bash
# ============================================================================
# CHẠY TẤT CẢ TỪ ĐẦU (sau khi setup_vast.sh xong): baseline đầy đủ + ý thầy.
#   Bước 1: run_full.sh    — 14 model x 3 seed + Otsu + ANN2SNN T-sweep
#   Bước 2: run_advisor.sh — T-sweep mở rộng (T1/3/5/7/10) + T2/T8 lr2e4 + INT8
#   (mỗi bước tự summarize+analysis; bước 2 cần mobilenet_unet_lr2e4_sN từ bước 1)
# Chạy trong tmux:  bash run_everything.sh 2>&1 | tee runs/run_everything.log
# ⏱ ~4.5-5h tổng. 1 run lỗi KHÔNG dừng cả script.
# ============================================================================
command -v python >/dev/null 2>&1 || { [ -f /venv/main/bin/activate ] && source /venv/main/bin/activate; }
set +e

echo "############# BƯỚC 1/2: run_full.sh #############"
bash run_full.sh

echo "############# BƯỚC 2/2: run_advisor.sh #############"
bash run_advisor.sh

echo ""
echo "================= HOÀN TẤT TẤT CẢ ================="
echo ">>> QUAN TRỌNG — TẢI VỀ TRƯỚC KHI DESTROY:"
echo "    tar -czf runs_full.tar.gz runs/"
echo "    (rồi tải runs_full.tar.gz về máy, XONG mới destroy instance)"
