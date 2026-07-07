#!/usr/bin/env bash
# ============================================================================
# Chạy ANN2SNN cho ĐỦ n=3 (để bảng đồng đều với các model khác).
# Cách làm: convert từ 3 ANN nguồn khác seed (unet_smp_s0/s1/s2) cho mỗi T.
#   -> ra ann2snn_T{T}_s{seed}; summarize gộp _sN thành 1 dòng có n=3 + std.
# Cách chạy trên H100 (trong tmux):
#     bash rerun_ann2snn.sh 2>&1 | tee runs/rerun_ann2snn.log     (~20-40 phút)
# ============================================================================
command -v python >/dev/null 2>&1 || { [ -f /venv/main/bin/activate ] && source /venv/main/bin/activate; }
mkdir -p runs

# Cách ly các dir ann2snn cũ (không hậu tố _sN) -> tránh summarize đếm nhầm n
for T in 32 64 128; do
  d="runs/ann2snn_T${T}"
  if [ -d "$d" ]; then
    echo "### cách ly dir cũ: $d -> ${d}_OLDn1bak"
    rm -rf "${d}_OLDn1bak"; mv "$d" "${d}_OLDn1bak"
  fi
done

for T in 32 64 128; do
  for s in 0 1 2; do
    src="unet_smp_s${s}"
    if [ ! -f "runs/${src}/best.pt" ]; then
      echo "!!! THIẾU runs/${src}/best.pt — bỏ qua T=$T seed=$s"; continue
    fi
    echo ">>> ANN2SNN T=$T  nguồn=$src"
    python ann2snn_convert.py --ann_config configs/unet_smp.yaml \
      --ann_run "$src" --T "$T" --name "ann2snn_T${T}_s${s}" \
      2>&1 | tee "runs/log_ann2snn_T${T}_s${s}.txt"
  done
done

echo ""
echo "===== Các dir ann2snn hiện có (kiểm tra n) ====="
ls -d runs/ann2snn* 2>/dev/null

echo ""
echo "===== Tổng hợp lại ====="
python summarize.py
python analysis.py --ref spiking_unet_T4

echo ""
echo "XONG. Cột (n) của ann2snn_T32/T64/T128 giờ phải = 3."
echo "Nếu ổn: tar -czf runs_full.tar.gz runs/  ->  tải về  ->  destroy máy."
