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
E_MAC_45NM = 4.6e-12      # J / phép MAC
E_AC_45NM = 0.9e-12       # J / phép AC


def count_flops_params(model, input_shape=(1, 2, 512, 512), device="cpu"):
    """Đếm FLOPs & số tham số cho model ANN. Trả về (flops, params)."""
    import torch
    x = torch.randn(*input_shape, device=device)
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        flops = FlopCountAnalysis(model.to(device), x).total()
        params = sum(v for v in parameter_count(model).values())
        return int(flops), int(params)
    except ImportError:
        pass
    try:
        from thop import profile
        flops, params = profile(model.to(device), inputs=(x,), verbose=False)
        return int(flops), int(params)
    except ImportError:
        raise ImportError("Cần cài `fvcore` hoặc `thop` để đếm FLOPs.")


def ann_energy_joules(flops):
    return flops * E_MAC_45NM


def snn_energy_joules(synops):
    return synops * E_AC_45NM


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
