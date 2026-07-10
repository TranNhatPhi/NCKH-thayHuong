#!/usr/bin/env bash
# Tải full-paper các tài liệu tham khảo open-access/arXiv (theo yêu cầu thầy).
# Bài paywall (Nature, IEEE) không tải tự động được -> xem REFERENCES.md gửi thầy.
#   bash download_refs.sh
cd "$(dirname "$0")"
get(){ echo ">>> $2"; curl -sL --max-time 60 -o "$2" "$1"; file "$2" | grep -q PDF && echo "   OK" || echo "   !!! LỖI (không phải PDF) — kiểm tra link"; }

# arXiv
get "https://arxiv.org/pdf/1505.04597"  "11_Ronneberger2015_UNet.pdf"
get "https://arxiv.org/pdf/1807.10165"  "12_Zhou2018_UNetpp.pdf"
get "https://arxiv.org/pdf/1801.04381"  "13_Sandler2018_MobileNetV2.pdf"
get "https://arxiv.org/pdf/1802.02611"  "14_Chen2018_DeepLabV3plus.pdf"
get "https://arxiv.org/pdf/2105.15203"  "15_Xie2021_SegFormer.pdf"
get "https://arxiv.org/pdf/1712.05877"  "16_Jacob2018_QuantizationIntegerOnly.pdf"
get "https://arxiv.org/pdf/1806.08342"  "17_Krishnamoorthi2018_QuantWhitepaper.pdf"
get "https://arxiv.org/pdf/2106.08295"  "18_Nagel2021_QuantWhitepaper.pdf"
get "https://arxiv.org/pdf/2310.16620"  "04_Fang2023_SpikingJelly.pdf"
get "https://arxiv.org/pdf/1809.05793"  "05_Wu2019_DirectTrainingSNN.pdf"
get "https://arxiv.org/pdf/1901.09948"  "06_Neftci2019_SurrogateGradient.pdf"
get "https://arxiv.org/pdf/2110.07742"  "07_Kim2022_SNN_SemanticSegmentation.pdf"
get "https://arxiv.org/pdf/2011.05280"  "10_Zheng2021_GoingDeeperSNN.pdf"
get "https://arxiv.org/pdf/2103.00476"  "21_Deng2021_OptimalANN2SNN.pdf"
# Open-access (CVF / Frontiers)
get "https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.pdf" "01_Bonafilia2020_Sen1Floods11.pdf"
get "https://www.frontiersin.org/articles/10.3389/fnins.2017.00682/pdf" "20_Rueckauer2017_ANN2SNN.pdf"

echo ""
echo "==== PAYWALL — thầy tải giúp (không mở tự do): ===="
echo "  [02] Pekel 2016, Nature 540:418-422 — https://doi.org/10.1038/nature20584"
echo "  [03] Torres 2012, Remote Sens. Environ. 120:9-24 — https://doi.org/10.1016/j.rse.2011.05.028"
echo "  [08] Davies 2018 (Loihi), IEEE Micro — https://doi.org/10.1109/MM.2018.112130359"
echo "  [09] Roy 2019, Nature 575:607-617 — https://doi.org/10.1038/s41586-019-1677-2"
echo "  [19] Horowitz 2014, ISSCC — https://doi.org/10.1109/ISSCC.2014.6757323"
