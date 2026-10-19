"""CPU tests for the exl3_moe binding-arity detection that keeps the fused launch working
across exllamav3 1.4.x (30 positional arguments) and 1.5.0 (35)."""

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3


def _pybind_doc(n_args: int) -> str:
    # Shape of the docstring pybind11 generates for an unnamed-argument binding.
    params = ", ".join(
        f"arg{i}: torch.Tensor" if i < 9 else f"arg{i}: typing.SupportsInt"
        for i in range(n_args)
    )
    return f"exl3_moe({params}) -> None\n\nexl3_moe"


class _Bound:
    def __init__(self, doc: str) -> None:
        self.__doc__ = doc
        self.calls: list[tuple] = []

    def __call__(self, *args):
        self.calls.append(args)


def test_arity_from_pybind_docstring():
    assert exl3._exl3_moe_arity(_Bound(_pybind_doc(30))) == 30
    assert exl3._exl3_moe_arity(_Bound(_pybind_doc(35))) == 35


def test_arity_from_python_signature_and_unknown():
    def thirty(*a):
        return None

    assert exl3._exl3_moe_arity(thirty) is None

    def named(a, b, c):
        return None

    assert exl3._exl3_moe_arity(named) == 3


def test_tail_is_empty_for_1_4_x():
    assert exl3._exl3_moe_tail(_Bound(_pybind_doc(exl3.EXL3_MOE_ARITY_147)), 2048, 16) == ()


def test_tail_reproduces_the_all_fused_launch_for_1_5_0():
    tail = exl3._exl3_moe_tail(_Bound(_pybind_doc(exl3.EXL3_MOE_ARITY_150)), 2048, 16)
    assert tail == (None, None, 1, 2048, 16)


def test_unknown_arity_gets_no_tail():
    def fn(*a):
        return None

    assert exl3._exl3_moe_tail(fn, 2048, 16) == ()


def _pybind_binding(n_args: int) -> _Bound:
    return _Bound(_pybind_doc(n_args))


def _fused_layer(*, k: int) -> SimpleNamespace:
    return SimpleNamespace(
        _exl3_ptrs={
            key: object()
            for key in (
                "gate_trellis",
                "gate_suh",
                "gate_svh",
                "up_trellis",
                "up_suh",
                "up_svh",
                "down_trellis",
                "down_suh",
                "down_svh",
            )
        },
        _exl3_fused_temps=(
            torch.zeros(1, exl3.TEMP_ROWS_FUSED, 16, dtype=torch.float16),
            None,
            None,
            None,
        ),
        _exl3_k=k,
    )


def _record_fused_launch(
    monkeypatch: pytest.MonkeyPatch, n_args: int, *, k: int
) -> list[tuple]:
    """Run the real fused entry point against a binding of ``n_args`` positional args."""
    fn = _pybind_binding(n_args)
    monkeypatch.setattr(exl3, "get_moe_kernel_backend", lambda: "exllamav3")
    monkeypatch.setitem(sys.modules, "exllamav3_ext", SimpleNamespace(exl3_moe=fn))
    exl3.apply_exl3_fused_moe(
        torch.zeros(2, 16, dtype=torch.float16),
        torch.zeros(2, 1, dtype=torch.long),
        torch.ones(2, 1),
        _fused_layer(k=k),
        [{}],
        None,
    )
    return fn.calls


def test_fused_launch_pads_exactly_the_binding_arity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The launch itself, not a copy of its call expression: args + num_active, plus the
    # 1.5.0 tail only when the bound arity has room for it.
    for n_args, expect in ((30, 30), (35, 35)):
        call = _record_fused_launch(monkeypatch, n_args, k=4)[0]
        assert len(call) == expect, (n_args, len(call))
        assert call[29] == -1
        assert call[10:13] == (4, 4, 4)
        assert call[30:] == (
            (None, None, 1, exl3.TEMP_ROWS_FUSED, 16) if n_args == 35 else ()
        )


def test_fused_launch_carries_each_layers_own_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mixed-K stacks: the launch takes K from the layer it belongs to, never a global.
    for k, n_args in ((2, 30), (3, 30), (5, 35)):
        call = _record_fused_launch(monkeypatch, n_args, k=k)[0]
        assert call[10:13] == (k, k, k), (k, n_args)


def test_temp_rows_from_buffers_or_default():
    temps = (torch.zeros(2, 512, 8), None, None, None)
    assert exl3._exl3_moe_temp_rows(temps) == 512
    assert exl3._exl3_moe_temp_rows((None, None, None, None)) == exl3.TEMP_ROWS_FUSED
    assert exl3._exl3_moe_temp_rows(None) == exl3.TEMP_ROWS_FUSED
