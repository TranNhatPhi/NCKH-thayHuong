#!/usr/bin/env bash
# ============================================================================
# CHẠY LẠI SNN ablation cho ĐỦ n=3 (T2/T6/T8), để bảng chính hợp lệ multi-seed.
# Vì sao: summary trước cho T2/T6/T8 = n1 (chỉ có dir cũ không hậu tố _sN),
#         trong khi T4 = n3 -> ablation T không so sánh công bằng được.
# Cách chạy trên H100 (trong tmux):
#     bash rerun_snn.sh 2>&1 | tee runs/rerun_snn.log        (~30-45 phút)
# 1 seed lỗi KHÔNG dừng cả script; mỗi run có log riêng trong runs/.
# ============================================================================
command -v python >/dev/null 2>&1 || { [ -f /venv/main/bin/activate ] && source /venv/main/bin/activate; }
mkdir -p runs
SEEDS="0 1 2"

for cfg in spiking_unet_T2 spiking_unet_T6 spiking_unet_T8; do
  # Cách ly run cũ (không hậu tố _sN) -> khỏi làm summarize đếm nhầm n
  if [ -d "runs/$cfg" ]; then
    echo "### cách ly dir cũ: runs/$cfg -> runs/${cfg}_OLDn1bak"
    rm -rf "runs/${cfg}_OLDn1bak"
    mv "runs/$cfg" "runs/${cfg}_OLDn1bak"
  fi
  for s in $SEEDS; do
    echo ">>> TRAIN $cfg seed $s"
    python train.py    --config configs/$cfg.yaml --seed $s 2>&1 | tee "runs/log_train_${cfg}_s${s}.txt"
    echo ">>> EVAL  $cfg seed $s"
    python evaluate.py --config configs/$cfg.yaml --seed $s 2>&1 | tee "runs/log_eval_${cfg}_s${s}.txt"
  done
done

echo ""
echo "===== Các dir SNN hiện có (kiểm tra n) ====="
ls -d runs/spiking_unet* 2>/dev/null

echo ""
echo "===== Tổng hợp lại ====="
python summarize.py
python analysis.py --ref spiking_unet_T4

echo ""
echo "XONG. Kiểm tra bảng: cột (n) của T2/T6/T8 phải = 3."
echo "Nếu ổn: tar -czf runs_full.tar.gz runs/  ->  tải về  ->  destroy máy."
