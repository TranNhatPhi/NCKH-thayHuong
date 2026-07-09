"""
Sinh file Word mẫu (SNN-Flood-Paper.docx) theo plan của thầy — format conference paper.
    python make_docx.py            (chạy từ thư mục paper/ hoặc repo root)
Nhúng sẵn Table 1 (bảng Word) + 3 figure (pareto_pooled, tsweep, per_region_heatmap).
"""
import csv
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "SNN-Flood-Paper.docx")

doc = Document()
# Font mặc định Times New Roman 11pt
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(11)


def h(text, size=13, space_before=10):
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold = True; r.font.size = Pt(size)
    p.paragraph_format.space_before = Pt(space_before); p.paragraph_format.space_after = Pt(4)
    return p


def body(text, italic=False, justify=True):
    p = doc.add_paragraph()
    r = p.add_run(text); r.italic = italic
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    return p


def set_col_widths(table, widths):
    """Ép độ rộng từng cột (inch) để tên model không vỡ dòng."""
    table.autofit = False
    table.allow_autofit = False
    for row in table.rows:
        for i, w in enumerate(widths):
            if i < len(row.cells):
                row.cells[i].width = Inches(w)


def add_full_table(table_no):
    """Bảng đầy đủ 25 configs đọc từ results/summary.csv, chèn tại chỗ gọi."""
    csv_path = os.path.join(RES, "summary.csv")
    if not os.path.isfile(csv_path):
        body("[TODO: chạy make_figures.py/summarize.py để có results/summary.csv rồi tạo lại docx.]")
        return
    rows = list(csv.DictReader(open(csv_path)))
    rows.sort(key=lambda r: -float(r["flood_IoU"]))
    hdr = ["Model", "Pooled IoU (±std, n)", "Per-chip", "F1", "pwIoU", "Params", "Energy(mJ)", "Spike%"]
    at = doc.add_table(rows=len(rows) + 1, cols=len(hdr))
    at.style = "Light Grid Accent 1"
    for j, x in enumerate(hdr):
        c = at.cell(0, j); c.text = x
        for p in c.paragraphs:
            p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
            for r in p.runs:
                r.bold = True; r.font.size = Pt(8)
    for i, row in enumerate(rows, start=1):
        snn = row["is_snn"].strip().lower() in ("true", "1")
        spk = f"{float(row['spike_rate'])*100:.1f}" if snn else "—"
        vals = [row["model"],
                f"{float(row['flood_IoU']):.3f}±{float(row['flood_IoU_std']):.3f} (n{row['n']})",
                f"{float(row['flood_IoU_chip']):.3f}", f"{float(row['flood_F1']):.3f}",
                f"{float(row['pw_IoU']):.3f}", f"{float(row['params_M']):.2f}",
                f"{float(row['energy_mJ']):.1f}", spk]
        for j, v in enumerate(vals):
            c = at.cell(i, j); c.text = v
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
                for r in p.runs:
                    r.font.size = Pt(8)
    set_col_widths(at, [1.6, 1.45, 0.6, 0.5, 0.55, 0.6, 0.72, 0.53])
    cap = doc.add_paragraph()
    cr = cap.add_run(f"Table {table_no}. Full benchmark — all {len(rows)} configurations "
                     "(mean over seeds; n in parentheses). F1 = flood Dice; pwIoU = permanent-water IoU. "
                     "Sorted by pooled flood-IoU.")
    cr.italic = True; cr.font.size = Pt(9); cap.paragraph_format.space_after = Pt(8)


def figure(fname, caption, width=6.2):
    path = os.path.join(FIG, fname)
    if os.path.isfile(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cr = cap.add_run(caption); cr.italic = True; cr.font.size = Pt(9)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER; cap.paragraph_format.space_after = Pt(10)


# ---------- Title block ----------
t = doc.add_paragraph()
tr = t.add_run("Energy-Accuracy Trade-offs of Spiking Neural Networks for SAR "
               "Flood Segmentation: A Systematic Benchmark")
tr.bold = True; tr.font.size = Pt(16)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER

a = doc.add_paragraph()
ar = a.add_run("Phi Tran¹, Huong Bui², Hoang Dinh²"); ar.font.size = Pt(12)
a.alignment = WD_ALIGN_PARAGRAPH.CENTER
aff = doc.add_paragraph()
afr = aff.add_run("¹ [Affiliation 1] — [email]    ² [Affiliation 2] — [emails]    [TODO điền cơ quan + email]")
afr.font.size = Pt(10); afr.italic = True
aff.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ---------- Banner trạng thái (xoá khi số liệu 60/20/20 đã cập nhật) ----------
bn = doc.add_paragraph()
bnr = bn.add_run("⚠️ DRAFT NOTE: numbers below are v1 (70/20/10 split). The split is now 60/20/20 "
                 "(agreed 09/07); models are being retrained and Table 1 / figures will be refreshed "
                 "from the new run. Remove this note after updating.")
bnr.italic = True; bnr.font.size = Pt(9); bnr.font.color.rgb = RGBColor(0xB0, 0x00, 0x00)

# ---------- Abstract ----------
h("Abstract", 12)
body("Rapid flood-extent mapping from Sentinel-1 SAR is critical for disaster response, and "
     "on-board or edge deployment demands energy-efficient models. Spiking Neural Networks (SNNs) "
     "promise low-energy inference through sparse, event-driven computation, yet they have never "
     "been benchmarked for flood segmentation on SAR. We present the first systematic energy–accuracy "
     "benchmark of direct-trained SNNs against strong ANN baselines — CNNs, a Transformer, an "
     "ANN-to-SNN conversion, and an INT8-quantized CNN — for 3-class flood-on-land segmentation "
     "(background / permanent water / flood) on Sen1Floods11, where permanent water is separated "
     "from flood using the JRC Global Surface Water layer. Across 20+ configurations trained with "
     "multiple seeds, we find that (i) with pooled flood-IoU an INT8 MobileNet-UNet dominates the "
     "accuracy–energy Pareto front (0.490 IoU at 3.1 mJ), while (ii) with a per-chip paired metric "
     "the best direct-trained SNN is statistically indistinguishable from the INT8 CNN (Wilcoxon, "
     "p = 0.40; Cohen's d = 0.09) — the conclusion about SNN competitiveness is metric-dependent. "
     "We verify 13.6–23.8% spike sparsity supporting the energy estimate, show ANN-to-SNN conversion "
     "fails even at T = 128, analyse per-region geographic sensitivity, and give practical guidance "
     "for choosing SNN vs. quantized CNN.")
kw = doc.add_paragraph()
kwr = kw.add_run("Keywords: "); kwr.bold = True
kw.add_run("Spiking Neural Networks; SAR flood mapping; Sen1Floods11; energy-efficient "
           "segmentation; quantization; neuromorphic computing.")

# ---------- 1 Introduction ----------
h("1. Introduction")
h("1.1 Motivation", 11, 6)
body("Floods are among the most frequent and damaging natural disasters, and timely flood-extent "
     "maps are essential for emergency response. Sentinel-1 SAR penetrates clouds and operates day "
     "and night, unlike optical imagery. Deep segmentation models achieve strong accuracy but are "
     "energetically expensive, limiting deployment on satellites, UAVs and edge stations. This "
     "motivates studying the energy–accuracy trade-off of candidate architectures, and whether SNNs "
     "— computing with sparse binary spikes and additions only — offer a favourable operating point.")
h("1.2 Contributions", 11, 6)
for c in [
    "First direct-trained SNN benchmark for 3-class flood-on-land segmentation on Sen1Floods11, "
    "with 20+ configurations and a full ablation over timesteps T, learning rate and encoding.",
    "Rigorous statistical comparison via Wilcoxon signed-rank on paired per-chip flood-IoU: a "
    "direct-trained SNN is statistically indistinguishable from an INT8 MobileNet-UNet (p > 0.05), "
    "whereas pooled IoU lets the quantized CNN dominate — metric choice determines the conclusion.",
    "Verified energy claim: measured spike sparsity 13.6–23.8% confirms the SynOps estimate; and "
    "ANN-to-SNN conversion fails to converge even at T = 128, so direct training is essential.",
    "Per-region analysis revealing geographic sensitivity (SNN relatively strong in Sri-Lanka/Ghana, "
    "weak in Mekong/Pakistan).",
    "Practical guidance: a decision procedure for SNN vs. quantized CNN given energy budget, accuracy "
    "floor and hardware target.",
]:
    doc.add_paragraph(c, style="List Number")

# ---------- 2 Related Work ----------
h("2. Related Work")
body("2.1 SAR flood mapping. Sen1Floods11 [1] established a Sentinel-1 benchmark; later CNN and "
     "semi-supervised methods improved water IoU. Prior work targets binary water; we address a "
     "harder 3-class task separating permanent water via JRC [TODO cite].")
body("2.2 Spiking neural networks for vision. Direct training via surrogate gradients and spiking "
     "segmenters have shown feasibility on frame/event benchmarks [TODO cite]; SNNs for SAR are "
     "nascent, and none benchmark flood segmentation.")
body("2.3 Quantization for edge deployment. PTQ/QAT compress CNNs to INT8/INT4 with small accuracy "
     "loss, drastically cutting per-operation energy; INT8 is the strongest ANN energy competitor to "
     "SNNs and is a mandatory baseline here.")

# ---------- 3 Method ----------
h("3. Method")
body("3.1 Problem formulation. Per-pixel 3-class segmentation of 2-channel (VV, VH, dB) chips into "
     "background (0), permanent water (1) and flood (2); invalid pixels use ignore-index (−1).")
body("3.2 Spiking U-Net. A 4-level U-Net (7.76 M params) with a LIF neuron (ATan surrogate, τ=2.0, "
     "detach-reset) after each conv block, multi-step mode (SpikingJelly); direct SAR encoding over "
     "T timesteps; final membrane potentials averaged to logits; tdBN-style BatchNorm over (T,B).")
body("3.3 Energy modeling. At 45 nm [Horowitz 2014]: FP32 MAC = 4.6 pJ, INT8 MAC ≈ 0.23 pJ, "
     "SNN AC = 0.9 pJ. ANN energy = FLOPs × E_MAC (ptflops); SNN energy = SynOps × E_AC where "
     "SynOps = Σ_conv(MACs × input spike-rate) accumulated over T.")
body("3.4 Statistical testing protocol. We report pooled and per-chip flood-IoU, and run Wilcoxon "
     "signed-rank on paired per-chip flood-IoU (seeds averaged), plus bootstrap 95% CIs and paired "
     "Cohen's d for key pairs.")

# ---------- 4 Experimental Setup ----------
h("4. Experimental Setup")
body("4.1 Dataset. Sen1Floods11 hand-labeled subset: 446 chips of 512×512, 2 channels (VV/VH dB), "
     "across 11 flood events/regions. 3-class labels via JRC permanent-water.")
body("4.2 Preprocessing & splits. dB clipped to [−50,0], normalized to [0,1]; NaN → ignore-index. "
     "One NaN-only chip is dropped (445 valid). Region-stratified split 60/20/20 → 271/87/87 chips "
     "(agreed protocol; larger test set than the earlier 70/20/10 for more reliable evaluation), with "
     "spatially-overlapping chips kept in the same split to avoid leakage, fixed seed 42.")
body("4.3 Baselines. Otsu; U-Net (vanilla / SMP-ResNet34 pretrained / U-Net++); DeepLabV3; "
     "MobileNet-UNet (3 LRs) and MobileNet-UNet INT8 (static PTQ); SegFormer-b2; SNN-Flood "
     "(T∈{1..8,10}, LR sweep); ANN2SNN (T∈{32,64,128}). Shared dataset/loss/metric/protocol.")
body("4.4 Training. AdamW, gradient clipping (max-norm 5.0), early stopping on val flood-IoU; "
     "3 seeds per config (T=6 with 5 seeds); Otsu deterministic. [TODO: epochs, batch size, lr từ base.yaml].")

# ---------- 5 Results ----------
h("5. Results")
h("5.1 Main comparison (Table 1)", 11, 6)
add_full_table(1)
body("On pooled IoU the accuracy leaders are pretrained ANNs (SegFormer 0.525); MobileNet-UNet INT8 "
     "reaches 0.490 at only 3.1 mJ (≈ its FP32 parent at 20× lower energy). SNN-Flood tops out at "
     "≈0.39 pooled IoU, below the quantized CNN. The statistical grouping (Cluster A/B/C) is analysed "
     "in §5.2: the INT8 CNN and the best SNNs form one indistinguishable cluster on per-chip flood-IoU.")

h("5.2 Statistical significance", 11, 6)
body("Under the per-chip metric: U-Net-SMP vs MobileNet-INT8 p = 0.015 (significant, Cohen's d = 0.35); "
     "MobileNet-INT8 vs SNN-T6 p = 0.397 (n.s., d = 0.09); SNN-T2 vs SNN-T6 p = 0.368 (n.s., d = −0.14). "
     "The best SNNs and INT8 form a statistically indistinguishable cluster — the headline conclusion is "
     "metric-dependent.")

h("5.3 Energy & spike-rate analysis", 11, 6)
body("The pooled Pareto frontier is Otsu (0 mJ, 0.13) → MobileNet-INT8 (3.1 mJ, 0.49) → SegFormer "
     "(98 mJ, 0.53); SNN-Flood is not Pareto-optimal under pooled IoU. Measured spike sparsity "
     "13.6% (T2)–23.8% (T8) confirms energy is driven by genuine sparsity. Direct-trained SNNs "
     "(31–167 mJ) are far more efficient than ANN2SNN (94–440 mJ).")
figure("pareto_pooled.png", "Figure 1. Accuracy–energy Pareto (pooled flood-IoU).")
figure("pareto_perchip.png", "Figure 2. Accuracy–energy Pareto (per-chip flood-IoU).")

h("5.4 Per-region breakdown", 11, 6)
body("All models share the same regional difficulty ordering — easiest in Mekong/Ghana (≈0.4–0.5), "
     "hardest in Bolivia (≈0.04 for every model) — a property of the data, not the model. SNNs are "
     "comparatively stronger in Sri-Lanka/Ghana and weaker in Mekong/Pakistan.")
figure("per_region_heatmap.png", "Figure 3. Per-region flood-IoU (model × region).")

h("5.5 Ablations", 11, 6)
body("Timesteps T. Within T2–T8 accuracy is flat (T2 0.391 ≈ T8 0.388; SNN-T2 vs SNN-T6 n.s.), so "
     "larger T buys no accuracy but costs more energy. At extreme T (T1, T10) occasional seed-level "
     "training collapse appears (a seed drops to ≈0.08); re-running T6 with 5 seeds reduced std from "
     "0.13 (n=3) to 0.022 (n=5), confirming a small-sample artifact, not a bimodal property.")
body("Learning rate. LR = 2e-4 improved the CNN (MobileNet-UNet 0.480 → 0.498) but degraded the SNN "
     "(SNN-T2 0.391 → 0.370; SNN-T8 0.388 → 0.355): the winning CNN recipe does not transfer to SNNs, "
     "and LR mainly affects SNN training stability rather than raising its accuracy ceiling. The "
     "symmetric LR sweep also establishes a fair comparison (both families tuned).")
body("Quantization bits. INT8 preserves accuracy (−0.008 IoU) at 20× lower energy; INT4 was "
     "inconclusive in our tooling (torchao int4 did not cover Conv2d) — reported as a tooling "
     "limitation, not a scientific conclusion.")
figure("tsweep.png", "Figure 4. T-sweep: accuracy flat, energy grows; collapse at T1/T10.")

h("5.6 Qualitative results", 11, 6)
body("Figure 5 compares predictions on four permanent-water-rich chips. SegFormer and MobileNet "
     "delineate the permanent-water channels (blue) cleanly, whereas SNN-T2 produces salt-and-pepper "
     "predictions and largely merges water into the flood class — consistent with its low permanent-water "
     "IoU (Table 1). All models still struggle on the hardest scenes (e.g., Bolivia).")
figure("qual_comparison.png",
       "Figure 5. Qualitative comparison on test chips (columns: SAR VV | Ground truth | SegFormer | "
       "MobileNet | SNN-T2). MobileNet-INT8 is visually identical to MobileNet FP32 (−0.008 IoU).")

# ---------- 6 Discussion ----------
h("6. Discussion")
body("6.1 Why SNN does not dominate. Under a MAC/SynOps energy model, the SNN's sparse-spike "
     "advantage is outweighed by (i) using full convolutions on a vanilla U-Net vs. MobileNet's "
     "depthwise-separable convs, and (ii) INT8 making each MAC ~20× cheaper. The SNN's event-driven "
     "benefit is only partially captured on von-Neumann hardware.")
body("6.2 When to use SNN vs quantized CNN. On standard edge hardware with INT8 kernels, a quantized "
     "MobileNet-UNet is best. Prefer an SNN when the target is neuromorphic hardware (e.g., Loihi), "
     "when INT8 MAC arrays are unavailable, or with native event-based sensing; under the per-chip "
     "metric the SNN is not statistically worse than INT8. [TODO: vẽ decision-tree figure]")
body("6.3 Limitations. No deployment on real neuromorphic hardware (energy estimated, not on-chip); "
     "single dataset; modest absolute IoU (harder 3-class SAR-only); INT4 not conclusively evaluated.")

# ---------- 7 Conclusion ----------
h("7. Conclusion")
body("We presented the first systematic energy–accuracy benchmark of direct-trained SNNs for SAR "
     "flood segmentation against CNN, Transformer, ANN2SNN and INT8 baselines under one protocol with "
     "multi-seed statistics. SNNs are competitive with an INT8 CNN under a paired per-chip metric but "
     "not under pooled IoU; ANN2SNN fails; quantized CNNs occupy the pooled Pareto front. Future work: "
     "neuromorphic deployment, multi-temporal SAR, event-native sensing.")

# ---------- References ----------
h("References")
for r in [
    "[1] Bonafilia et al., Sen1Floods11: a georeferenced dataset for Sentinel-1 flood mapping, CVPRW 2020.",
    "[2] Horowitz, Computing's energy problem (and what we can do about it), ISSCC 2014.",
    "[3] Pekel et al., High-resolution mapping of global surface water (JRC), Nature 2016.",
    "[TODO] SpikingJelly; SNN segmentation (Shi 2022, Su 2023); MobileNetV2; SegFormer; DeepLabV3; "
    "SAR-SNN works. Cần đủ 20–30 trích dẫn.",
]:
    p = doc.add_paragraph(r); p.paragraph_format.space_after = Pt(2)
    for run in p.runs:
        run.font.size = Pt(9)

# ---------- Appendix A: Reproducibility (bảng đầy đủ đã lên §5.1 Table 2) ----------
doc.add_page_break()
h("Appendix A. Reproducibility")
body("Code, configs, region-stratified splits and fixed seeds are released at [GitHub repo link]. "
     "We keep all experiment versions: v1 = 70/20/10 split (git tag paper-v1, archived in "
     "paper/results_v1_70-20-10/); v2 = 60/20/20 split (current). Hardware: 1× NVIDIA H100 (training); "
     "energy is estimated analytically at 45 nm (not measured on-chip). Seeds: {0,1,2} (T=6 also {3,4}). "
     "[TODO: điền git SHA + repo URL].", italic=True)

doc.save(OUT)
print(f"Đã tạo {OUT}")
