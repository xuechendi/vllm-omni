# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3_GEMM_RELAYOUT must leave quantized Linears alone.

A quantized quant_method owns its weight's layout and packing (fp8 blockwise
scales, int4 packing), so re-striding the raw storage under it corrupts the
weight. The relayout loop skips any Linear whose quant_method is not (an)
UnquantizedLinearMethod -- matched by MRO class name -- and accounts for the
skips in its self-evidence line (skipped_quant=N). The quant check precedes
the device check, so a quantized weight is never misreported as off-device.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class UnquantizedLinearMethod:  # matched by class name, like vllm's
    pass


class _SubclassedUnquantized(UnquantizedLinearMethod):
    pass


class Fp8LinearMethod:
    pass


class _FakeLinear:
    def __init__(self, quant_method=None):
        self.weight = torch.nn.Parameter(torch.randn(4, 4))
        if quant_method is not None:
            self.quant_method = quant_method

    def modules(self):
        return iter(())


class _Root:
    od_config = None

    def __init__(self, mods):
        self._mods = mods

    def modules(self):
        return iter(self._mods)


@pytest.fixture
def relayout(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    monkeypatch.setenv("H3_GEMM_RELAYOUT", "1")
    return h3._h3_relayout_linear_weights


def test_quantized_linear_is_skipped_and_accounted(relayout, capsys):
    root = _Root([_FakeLinear(Fp8LinearMethod()), _FakeLinear()])

    assert relayout(root) == 0

    err = capsys.readouterr().err
    assert "skipped_quant=1" in err, err


def test_unquantized_method_is_not_skipped(relayout, capsys):
    root = _Root([_FakeLinear(UnquantizedLinearMethod()), _FakeLinear(_SubclassedUnquantized())])

    relayout(root)

    err = capsys.readouterr().err
    assert "skipped_quant=0" in err, err


def test_all_quantized_still_leaves_self_evidence(relayout, capsys):
    root = _Root([_FakeLinear(Fp8LinearMethod())])

    assert relayout(root) == 0

    err = capsys.readouterr().err
    assert "quant" in err, f"no self-evidence for the quant-only skip: {err!r}"


class _DeviceProxy:
    """Wraps a CPU tensor but reports ``device.type == 'xpu'``: exactly the
    surface the relayout loop reads (device / dim / stride / t / contiguous).
    Plain CPU Parameters are skipped as off-device before the assignment is
    reached, so only a device-gate-passing weight can prove the quant skip."""

    _t: torch.Tensor

    def __init__(self, t):
        object.__setattr__(self, "_t", t)

    @property
    def device(self):
        import types

        return types.SimpleNamespace(type="xpu")

    def dim(self):
        return self._t.dim()

    def stride(self, i=None):
        return self._t.stride(i) if i is not None else self._t.stride()

    def t(self):
        return _DeviceProxy(self._t.t())

    def contiguous(self):
        return _DeviceProxy(self._t.contiguous())

    def data_ptr(self):
        return self._t.data_ptr()


class _XpuParam(torch.nn.Parameter):
    """A real nn.Parameter (the loop's isinstance check) whose .data proxy
    claims XPU, so the device gate passes on a CPU-only runner."""

    def __new__(cls, t):
        return super().__new__(cls, t, requires_grad=False)

    @property
    def data(self):
        return self._proxy

    @data.setter
    def data(self, value):
        self._proxy = value


def _xpu_linear(quant_method=None):
    lin = _FakeLinear(quant_method)
    lin.weight = _XpuParam(torch.randn(4, 4))
    lin.weight._proxy = _DeviceProxy(torch.randn(4, 4))
    return lin


def test_quantized_stride_is_untouched_while_bf16_is_relayouted(relayout, capsys):
    """The decisive contract, on weights that pass the device gate: the
    quantized weight keeps its stride and storage, the unquantized one is
    actually re-strided. Red when the quant skip is deleted."""
    quant = _xpu_linear(Fp8LinearMethod())
    plain = _xpu_linear(UnquantizedLinearMethod())
    quant_proxy = quant.weight.data
    quant_ptr = quant_proxy.data_ptr()

    assert relayout(_Root([quant, plain])) == 1

    assert quant.weight.data is quant_proxy, "quantized weight was reassigned"
    assert quant.weight.data.stride() == (4, 1), "quantized weight was re-strided"
    assert quant.weight.data.data_ptr() == quant_ptr, "quantized storage was replaced"
    assert plain.weight.data.stride() == (1, 4), "unquantized weight was not relayouted"
    err = capsys.readouterr().err
    assert "1 linear weights -> transpose-contiguous" in err and "skipped_quant=1" in err, err


def test_all_quantized_self_evidence_carries_the_unified_field(relayout, capsys):
    """The quant-only branch reports the same machine-greppable
    ``skipped_quant=<n>`` field as the other branches, not only prose."""
    root = _Root([_FakeLinear(Fp8LinearMethod())])

    assert relayout(root) == 0

    err = capsys.readouterr().err
    assert "skipped_quant=1" in err, f"unified field missing on the quant-only line: {err!r}"
