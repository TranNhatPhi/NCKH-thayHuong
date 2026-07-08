"""
Đo NĂNG LƯỢNG — đóng góp định lượng lõi (chưa ai làm trên bài toán này).

Ý tưởng: quy đổi số phép toán ra năng lượng ở công nghệ 45nm (Horowitz 2014):
    ANN:  FLOPs (phép nhân-cộng MAC)  × E_MAC ≈ 4.6 pJ
    SNN:  SynOps (phép cộng AC theo spike) × E_AC  ≈ 0.9 pJ
So sánh "× lần tiết kiệm" = năng_lượng_ANN / năng_lượng_SNN.

Trạng thái:
  * FLOPs/params (ANN): ✅ qua `fvcore` hoặc `thop` nếu đã cài.
  * SynOps (SNN): 🟡 skeleton — gắn forward-hook đếm spike ở mỗi lớp LIF.
"""
E_MAC_45NM = 4.6e-12      # J / phép MAC (FP32, Horowitz 2014)
E_AC_45NM = 0.9e-12       # J / phép AC  (cộng dồn spike)
E_MAC_INT8_45NM = 0.23e-12  # J / phép MAC INT8 (8b mult 0.2pJ + 8b add 0.03pJ) — cho quantization
E_MAC_INT4_45NM = 0.12e-12  # J / phép MAC INT4 (ước lượng ~0.5× INT8; ghi rõ là xấp xỉ trong paper)


def int8_energy_joules(macs):
    """Năng lượng nếu chạy INT8 (cùng số MAC, mỗi MAC rẻ hơn ~20× so với FP32)."""
    return macs * E_MAC_INT8_45NM


def int4_energy_joules(macs):
    """Năng lượng nếu chạy INT4 (xấp xỉ, cùng số MAC). Chỉ dùng nếu INT4 chạy được."""
    return macs * E_MAC_INT4_45NM


def count_flops_params(model, input_shape=(1, 2, 512, 512), device="cpu"):
    """Đếm MACs & params cho model ANN bằng ptflops — đếm NHẤT QUÁN cả CNN lẫn
    Transformer (fvcore bỏ sót op transformer). Trả về (macs, params)."""
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        model.to(device).eval(), tuple(input_shape[1:]),
        as_strings=False, print_per_layer_stat=False, verbose=False)
    return int(macs), int(params)


def ann_energy_joules(flops):
    return flops * E_MAC_45NM


def snn_energy_joules(synops):
    return synops * E_AC_45NM


def count_synops(model, sample_input, device="cpu"):
    """
    Đếm SynOps của model SNN: với mỗi lớp conv, chỉ những kết nối có SPIKE ở input mới
    tạo phép cộng. SynOps = Σ_conv (MACs_conv × tỉ_lệ_spike_ở_input), cộng dồn qua T bước
    (forward của SNN tự gọi mỗi conv T lần).

    Giả định (ghi rõ trong paper): coi input mỗi conv là spike nhị phân; lớp mã hóa đầu
    tính gộp luôn. Trả về tổng SynOps (số phép cộng AC).
    """
    import torch
    import torch.nn as nn

    total = [0.0]
    handles = []

    def hook(module, inp, out):
        x = inp[0]
        rate = (x != 0).float().mean().item()          # tỉ lệ spike ở input conv
        if isinstance(module, nn.Conv2d):
            out_elems = out.numel()
            macs = out_elems * (module.in_channels // module.groups) \
                * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.ConvTranspose2d):
            in_elems = x.numel()
            macs = in_elems * (module.out_channels // module.groups) \
                * module.kernel_size[0] * module.kernel_size[1]
        else:
            return
        total[0] += macs * rate

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(sample_input.to(device))                 # gọi conv T lần → cộng dồn qua T
    for h in handles:
        h.remove()
    return int(total[0])


def mean_spike_rate(model, sample_input, device="cpu"):
    """Tỉ lệ phát spike TRUNG BÌNH trên mọi nơ-ron LIF (0–1). Càng thấp = càng thưa = càng tiết kiệm."""
    import torch
    try:
        from spikingjelly.activation_based import neuron
    except ImportError:
        return 0.0
    total, count, handles = [0.0], [0], []

    def hook(_m, _inp, out):
        total[0] += float((out != 0).float().mean().item())
        count[0] += 1

    for m in model.modules():
        if isinstance(m, neuron.BaseNode):
            handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(sample_input.to(device))
    for h in handles:
        h.remove()
    return total[0] / count[0] if count[0] else 0.0


def is_spiking(model):
    """True nếu model chứa nơ-ron spiking (SpikingJelly)."""
    try:
        from spikingjelly.activation_based import neuron
        return any(isinstance(m, neuron.BaseNode) for m in model.modules())
    except ImportError:
        return False


class SpikeCounter:
    """Gắn hook lên các lớp LIF để đếm tổng số spike (→ SynOps) trong 1 lần forward.
    SynOps ≈ Σ (spike ở lớp trước × số kết nối synapse tới lớp sau).

    TODO khi chạy thật:
      * Với spikingjelly, đăng ký hook ở mỗi neuron.LIFNode để lấy output spike.
      * Nhân số spike với fan-out (số phép cộng mỗi spike kích hoạt) của lớp kế.
      * Cộng dồn qua T bước.
    """
    def __init__(self, model):
        self.model = model
        self.total_spikes = 0
        self.handles = []

    def __enter__(self):
        from spikingjelly.activation_based import neuron

        def hook(_m, _inp, out):
            self.total_spikes += out.sum().item()

        for m in self.model.modules():
            if isinstance(m, neuron.BaseNode):
                self.handles.append(m.register_forward_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
