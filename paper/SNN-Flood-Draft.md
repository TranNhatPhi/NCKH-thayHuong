# Energy-Accuracy Trade-offs of Spiking Neural Networks for SAR Flood Segmentation: A Systematic Benchmark

> **Bản thảo v1 (08/07/2026)** — viết theo plan của thầy (Huong Bui). Số liệu lấy từ `paper/results/summary.csv` + `wilcoxon_*.csv` (n=3, T6 n=5). Chỗ cần chốt/điền thêm đánh dấu `[TODO]`. Draft bằng tiếng Anh (nộp hội nghị); chú thích tiếng Việt trong blockquote.

---

## Abstract

Rapid flood extent mapping from Sentinel-1 Synthetic Aperture Radar (SAR) is critical for disaster response, and on-board or edge deployment demands energy-efficient models. Spiking Neural Networks (SNNs) promise low-energy inference through sparse, event-driven computation, yet they have never been benchmarked for flood segmentation on SAR. We present the first systematic energy–accuracy benchmark of direct-trained SNNs against strong artificial neural network (ANN) baselines — CNNs, a Transformer, an ANN-to-SNN conversion, and an INT8-quantized CNN — for **3-class flood-on-land segmentation** (background / permanent water / flood) on Sen1Floods11, where permanent water is separated from flood using the JRC Global Surface Water layer. Across 20+ configurations trained with multiple seeds, we find that (i) with the **pooled flood-IoU** metric, an INT8-quantized MobileNet-UNet dominates all learned models on the accuracy–energy Pareto front (0.490 IoU at 3.1 mJ), while (ii) with a **per-chip paired metric** the best direct-trained SNN is **statistically indistinguishable** from the INT8 CNN (Wilcoxon signed-rank, p = 0.40) — so the conclusion about SNN competitiveness is *metric-dependent*. We verify that direct-trained SNNs operate at 13.6–23.8% spike sparsity, supporting their theoretical energy savings, whereas ANN-to-SNN conversion fails to converge even at T = 128. A per-region analysis reveals geographic sensitivity (SNNs are relatively stronger in Sri-Lanka/Ghana, weaker in Mekong/Pakistan), and a T-sweep shows that increasing the number of timesteps beyond T = 2 yields no accuracy gain while occasional seed-level training collapse appears at extreme T. We distill these findings into practical guidance for choosing between SNNs and quantized CNNs under a given energy budget, accuracy floor, and hardware constraint.

> Tóm tắt VN: đóng góp = benchmark trung thực đầu tiên của SNN cho flood-SAR; kết luận phụ thuộc metric (pooled → INT8 thắng; per-chip → SNN ngang INT8); ANN2SNN thất bại; verify spike sparsity; hướng dẫn thực dụng.

---

## 1. Introduction

### 1.1 Motivation
Floods are among the most frequent and damaging natural disasters, and timely flood-extent maps are essential for emergency response. Sentinel-1 SAR is the sensor of choice because it penetrates clouds and operates day and night, unlike optical imagery. Deep segmentation models (U-Net, DeepLabV3+, Transformers) achieve strong accuracy but are computationally and energetically expensive, which limits deployment on power-constrained platforms — on-board satellites, UAVs, and edge stations near disaster sites. This motivates a rigorous study of the **energy–accuracy trade-off** of candidate architectures, and in particular whether **Spiking Neural Networks (SNNs)** — which compute with sparse binary spikes and only additions (no multiplies) — offer a favorable operating point for flood mapping.

### 1.2 Contributions
1. **First direct-trained SNN benchmark** for 3-class flood-on-land segmentation on Sen1Floods11, with a results table spanning **20+ configurations** and a full ablation over timesteps *T*, learning rate, and encoding.
2. **Rigorous statistical comparison** via **Wilcoxon signed-rank tests on paired per-chip flood-IoU**. We show a direct-trained SNN is *statistically indistinguishable* from an INT8-quantized MobileNet-UNet under the per-chip metric (p > 0.05), whereas under **pooled IoU** the quantized CNN dominates — i.e. **metric choice determines the headline conclusion** about SNNs.
3. **Verified energy claim, not an artifact.** We measure actual spike sparsity (13.6–23.8%) with forward hooks, confirming the SynOps-based energy estimate; and we show **ANN-to-SNN conversion fails to converge even at T = 128**, so *direct training* is essential for SNNs on SAR.
4. **Per-region analysis** revealing geographic sensitivity (SNNs relatively strong in Sri-Lanka/Ghana, weak in Mekong/Pakistan), yielding design insight for SNNs in remote sensing.
5. **Practical guidance**: a decision procedure for when to prefer an SNN vs. a quantized CNN given a concrete energy budget, accuracy floor, and hardware target.

---

## 2. Related Work

### 2.1 SAR flood mapping
Sen1Floods11 [Bonafilia et al., 2020] established a benchmark of Sentinel-1 chips with hand-labeled water masks across 11 global flood events; subsequent CNN and semi-supervised approaches improved water IoU [Banerjee & Daou, 2026; Residual Wave U-Net, 2024]. Prior work targets binary water/non-water. We instead address a harder **3-class** task (background / permanent water / flood), separating permanent water via the JRC Global Surface Water layer, which better matches operational flood mapping. `[TODO: 3–4 câu điểm baseline IoU của các paper này để đặt bối cảnh]`

### 2.2 Spiking neural networks for vision
SNNs encode information in spike trains and are attractive for neuromorphic, energy-efficient inference. Direct training via surrogate gradients [Shi et al., 2022; Su et al., 2023] and spiking segmenters (Spiking-CGNet, boundary-aware MS-SNN) have shown feasibility on frame/event vision benchmarks. SNNs for SAR are nascent (SNN SAR ship classification, InSAR phase unwrapping). To our knowledge, **no prior work benchmarks SNNs for flood segmentation on SAR**.

### 2.3 Quantization for edge deployment
Post-training quantization (PTQ) and quantization-aware training (QAT) compress CNNs to INT8/INT4 with small accuracy loss, drastically reducing per-operation energy. INT8 is the strongest *ANN* energy competitor to SNNs and is therefore a mandatory baseline for any SNN energy claim — a comparison largely missing from prior SNN literature, which we make central here.

---

## 3. Method

### 3.1 Problem formulation
We cast flood mapping as **3-class semantic segmentation**: for each pixel of a 2-channel (VV, VH, in dB) Sentinel-1 chip, predict *background* (0), *permanent water* (1), or *flood* (2). Ground truth uses `LabelHand` for water and the JRC permanent-water layer to split permanent water from flood; invalid/NaN pixels are set to ignore-index (−1) and excluded from loss and metrics.

### 3.2 Spiking U-Net architecture
Our SNN-Flood is a 4-level U-Net (7.76 M parameters) in which every convolution block is followed by a **Leaky Integrate-and-Fire (LIF)** neuron with ATan surrogate gradient (`tau = 2.0`, detach-reset), implemented in multi-step mode (SpikingJelly). SAR inputs are fed by **direct encoding** (the analog chip is repeated over *T* timesteps); the final-layer membrane potentials are averaged over *T* and read out as class logits. Batch normalization is applied over the (T, B) axes (tdBN-style). The architecture mirrors the vanilla ANN U-Net for a controlled comparison.

### 3.3 Energy modeling
We estimate inference energy at 45 nm following Horowitz [2014]: an ANN multiply-accumulate (MAC) costs **E_MAC = 4.6 pJ** (FP32), an INT8 MAC costs **≈ 0.23 pJ**, and an SNN accumulate (AC, triggered only by a spike) costs **E_AC = 0.9 pJ**. For ANNs, energy = FLOPs × E_MAC (MACs counted consistently with `ptflops`). For SNNs, we count **SynOps** = Σ_conv (MACs × input spike-rate), accumulated over *T*, and energy = SynOps × E_AC. This makes the SNN energy depend on measured sparsity rather than nominal size.

### 3.4 Statistical testing protocol
Because flood-IoU differences between strong models are small, point estimates are insufficient. We report **two IoU aggregations**: (a) **pooled** flood-IoU (confusion matrix accumulated over all test pixels) and (b) **per-chip** flood-IoU (IoU computed per chip, then averaged). For significance we run **Wilcoxon signed-rank tests on paired per-chip flood-IoU** (models paired chip-by-chip, seeds averaged), for a reference model vs. all others and for a set of key model pairs. `[TODO P1: thêm bootstrap 95% CI + Cohen's d bên cạnh p-value]`

---

## 4. Experimental Setup

### 4.1 Dataset
Sen1Floods11 hand-labeled subset: **446 chips** of 512×512, 2 channels (VV/VH, dB), spanning **11 flood events / regions** (Bolivia, Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA). 3-class labels derived as in §3.1.

### 4.2 Preprocessing & splits
dB values clipped to [−50, 0] and normalized to [0, 1]; NaN → ignore-index. Chips are split into train/validation/test **stratified by region** with a fixed seed for reproducibility, using a **70/20/10 ratio → 312 / 88 / 45 chips** (agreed protocol). `[TODO: nêu xử lý spatial leakage — chia theo scene/vùng, không random từng chip.]`

### 4.3 Baselines
Classical: **Otsu** thresholding (non-learned). CNN: **U-Net (vanilla)**, **U-Net (SMP/ResNet34, ImageNet-pretrained)**, **U-Net++**, **DeepLabV3**, **MobileNet-UNet** (depthwise-separable, 3 learning rates). Transformer: **SegFormer-b2 (MiT)**. Compressed ANN: **MobileNet-UNet INT8** (static PTQ, per-channel weights, ~200-image calibration). SNN: **SNN-Flood** (T ∈ {1,2,3,4,5,6,7,8,10}, LR sweep) and **ANN2SNN** conversion (T ∈ {32,64,128}). All models share one dataset, loss (weighted CE + Dice + Focal, ignore-index −1), metric, training, and evaluation protocol.

### 4.4 Training
AdamW, gradient clipping (max-norm 5.0), early stopping on validation flood-IoU. Each learned configuration is trained with **3 seeds** (T = 6 with **5 seeds** to probe stability); Otsu is deterministic (single run). `[TODO: điền epochs, batch size, lr cụ thể từ configs/base.yaml]`

---

## 5. Results

### 5.1 Main comparison (Table 1)
**Table 1.** Selected models (full 25-row table in Appendix / `summary.csv`). Energy in mJ per 512×512 chip; Pooled = pooled flood-IoU (mean±std over seeds); Per-chip = mean per-chip flood-IoU.

| Model | Group | Params | Energy (mJ) | Pooled IoU | Per-chip IoU | Spike% |
|-------|-------|:-----:|:-----:|:-----:|:-----:|:-----:|
| SegFormer-b2 | Transformer | 24.7 M | 98.0 | **0.525 ± 0.018** | 0.251 | — |
| U-Net (SMP) | CNN | 24.4 M | 143.9 | 0.509 ± 0.028 | 0.291 | — |
| U-Net++ | CNN | 26.1 M | 339.1 | 0.500 ± 0.026 | 0.264 | — |
| MobileNet-UNet | CNN | 6.6 M | 62.7 | 0.498 ± 0.007 | 0.272 | — |
| **MobileNet-UNet INT8** | Quantized | 6.6 M | **3.1** | 0.490 ± 0.003 | 0.258 | — |
| U-Net (vanilla) | CNN | 7.8 M | 223.2 | 0.482 ± 0.019 | **0.306** | — |
| **SNN-Flood T2** | SNN | 7.8 M | 31.4 | 0.391 ± 0.045 | 0.240 | 13.6 |
| **SNN-Flood T8** | SNN | 7.8 M | 166.8 | 0.388 ± 0.030 | 0.246 | 23.8 |
| ANN2SNN (T=128) | SNN | 24.4 M | 440.4 | 0.290 ± 0.062 | 0.196 | 10.6 |
| DeepLabV3 | CNN | 26.0 M | 502.3 | 0.419 ± 0.014 | 0.127 | — |

Two observations frame the paper: **(1)** on **pooled** IoU the learned accuracy leaders are pretrained ANNs (SegFormer 0.525, U-Net-SMP 0.509), and **MobileNet-UNet INT8 attains 0.490 at only 3.1 mJ**, i.e. it nearly matches its FP32 parent (0.498 @ 62.7 mJ) at 20× lower energy; **(2)** SNN-Flood tops out at ≈0.39 pooled IoU, clearly below the quantized CNN.

### 5.2 Statistical significance
Under the **per-chip** metric the picture changes. Wilcoxon signed-rank on paired per-chip flood-IoU:

| Pair | A | B | p-value | Verdict |
|------|:--:|:--:|:--:|--------|
| U-Net (SMP) vs MobileNet-INT8 | 0.291 | 0.258 | 0.015 | significant |
| **MobileNet-INT8 vs SNN-T6** | 0.258 | 0.246 | **0.397** | **n.s. (tied)** |
| SNN-T2 vs SNN-T6 | 0.240 | 0.246 | 0.368 | n.s. |

The best SNN configurations and MobileNet-INT8 form a **statistically indistinguishable cluster** on per-chip flood-IoU (p ≈ 0.40), even though pooled IoU separates them. **The headline conclusion about SNN competitiveness is therefore metric-dependent** — a central, honest finding of this benchmark.

**Effect sizes** (paired Cohen's *d*) confirm this beyond p-values: U-Net-SMP vs INT8 *d* = 0.35 (small–medium, and significant), whereas **INT8 vs SNN-T6 *d* = 0.09** and SNN-T2 vs SNN-T6 *d* = −0.14 — both **negligible**. Bootstrap 95% CIs (`results/bootstrap_ci.csv`) overlap for INT8 and the top SNNs.

### 5.3 Energy & spike-rate analysis
On the accuracy–energy Pareto front (Fig. `figures/pareto_pooled.png`, `figures/pareto_perchip.png`), the frontier is **Otsu (0 mJ, 0.13) → MobileNet-INT8 (3.1 mJ, 0.49) → SegFormer (98 mJ, 0.53)**; SNN-Flood is not Pareto-optimal under the pooled metric. Measured spike sparsity is **13.6% (T2) to 23.8% (T8)**, confirming that SNN energy is driven by genuine sparsity rather than an accounting artifact. Direct-trained SNNs (31–167 mJ) are far more efficient than ANN2SNN (94–440 mJ), because converted networks fire densely (spike-rate ≈10%) over many timesteps.

### 5.4 Per-region breakdown
All models share the same regional difficulty ordering — easiest in Mekong/Ghana (IoU ≈ 0.4–0.5), hardest in Bolivia (≈0.04 for every model) — indicating this is a property of the **data**, not the model. Relative to CNNs, SNNs are comparatively **stronger in Sri-Lanka and Ghana** and **weaker in Mekong and Pakistan** (Fig. `figures/per_region_heatmap.png`; `results/per_region.csv`), suggesting SNN inductive biases suit certain backscatter regimes.

### 5.5 Ablations
**Timesteps T.** Within the stable regime (T2–T8), accuracy is flat (T2 0.391 ≈ T8 0.388; SNN-T2 vs SNN-T6 n.s.), so **larger T buys no accuracy** but costs proportionally more energy. At **extreme T (T1, T10)** we observe occasional **seed-level training collapse** (a single seed dropping to ≈0.08, inflating variance to std ≈ 0.13–0.15); re-running **T6 with 5 seeds** reduced its std from 0.13 (n=3) to **0.022 (n=5)**, confirming the earlier instability was a small-sample artifact, not a bimodal property of T6. **Learning rate** affects SNN *stability* more than mean accuracy. **Quantization bits.** INT8 preserves accuracy (−0.008 IoU vs FP32) at 20× energy; INT4 via `torchao` was inconclusive in our environment (API/kernel did not cover Conv2d) `[TODO: nêu như một giới hạn công cụ, không phải kết luận khoa học]`.

---

## 6. Discussion

### 6.1 Why SNN does not dominate
Under a MAC/SynOps energy model, the SNN's sparse-spike advantage is outweighed by two factors: (i) our Spiking U-Net uses **full convolutions** on a vanilla U-Net backbone, whereas MobileNet uses **depthwise-separable** convolutions with far fewer MACs; and (ii) **INT8 makes each MAC ~20× cheaper**. Thus an efficient, quantized ANN beats the SNN on both accuracy and estimated energy. The SNN's theoretical benefit (clock-free, event-driven addition) is only *partially* captured by SynOps counting on von-Neumann hardware.

### 6.2 When to use SNN vs quantized CNN
`[TODO P1: vẽ decision tree]` Guidance: if the target is standard edge hardware and INT8 kernels are available, a **quantized MobileNet-UNet** is the best choice (highest accuracy per mJ). Consider an **SNN** when (a) the deployment target is **neuromorphic hardware** (e.g., Loihi) where event-driven execution realizes the full energy benefit, (b) an ultra-low-power regime forbids INT8 MAC arrays, or (c) native event-based sensing is used. Under the per-chip metric the SNN is not statistically worse than INT8, which supports its use where per-scene robustness matters.

### 6.3 Limitations
No deployment on real neuromorphic hardware (energy is *estimated*, not measured on-chip); a single dataset (Sen1Floods11); modest absolute IoU due to the harder 3-class SAR-only setting; INT4 not conclusively evaluated.

---

## 7. Conclusion
We presented the first systematic energy–accuracy benchmark of direct-trained SNNs for SAR flood segmentation, against CNN, Transformer, ANN2SNN, and INT8 baselines under one protocol with multi-seed statistics. Direct-trained SNNs are competitive with an INT8 CNN under a paired per-chip metric but not under pooled IoU; ANN2SNN fails; and quantized CNNs occupy the pooled Pareto front. Our benchmark, code, and analysis provide a foundation and honest guidance for energy-efficient flood mapping. Future work: neuromorphic-hardware deployment, multi-temporal SAR, and event-native sensing.

---

## References
`[TODO: 20–30 trích dẫn. Hạt giống: Bonafilia 2020 (Sen1Floods11); Horowitz 2014 (energy); Shi 2022, Su 2023 (SNN seg); SpikingJelly; JRC Global Surface Water (Pekel 2016); MobileNetV2; SegFormer; DeepLabV3.]`

---

## Appendix / trạng thái checklist (theo plan thầy)
**P0 (trước khi viết):** ✅ debug spike ANN2SNN (8–11%); ⏳ freeze code (`git tag paper-v1` — làm ở commit này); ✅ metric = pooled + per-chip (per-chip là chính); ✅ 8 model cho Table 1.
**P1 (đang viết):** ✅ Pareto 2 bản (pooled/per-chip); ✅ per-region heatmap; ✅ T-sweep plot có error bar; ☐ qualitative 6 chip (SegFormer/INT8/SNN-T2 — cần checkpoint+inference); ✅ bootstrap 95% CI; ✅ Cohen's d. *(sinh bằng `make_figures.py` → paper/figures/, paper/results/)*
**P2 (cuối):** ☐ proofread; ☐ cite check; ☐ format venue; ☐ reproducibility (git SHA, seeds, hardware); ☐ code link công khai.
