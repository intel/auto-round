from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

import auto_round.algorithms.transforms.svdquant.residual as residual_module
from auto_round.algorithms.transforms.svdquant.residual import ResidualQuantScheme, rtn_qdq_residual
from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.data_type.mxfp import quant_mx
from auto_round.data_type.utils import get_quant_func


def _deterministic_weight(dtype=torch.float32):
    return torch.linspace(-3.0, 3.0, steps=3 * 64, dtype=dtype).reshape(3, 64)


def test_rtn_qdq_residual_preserves_tensor_contract():
    weight = _deterministic_weight()
    scheme = ResidualQuantScheme(data_type="mx_fp4e2m1", bits=4, group_size=32, sym=True)

    qdq = rtn_qdq_residual(weight, scheme)

    assert qdq.shape == weight.shape
    assert qdq.dtype == weight.dtype
    assert qdq.device == weight.device
    assert torch.isfinite(qdq).all()


def test_rtn_qdq_residual_matches_registered_quant_function():
    weight = _deterministic_weight(dtype=torch.float16)
    scheme = ResidualQuantScheme(data_type="mx_fp4e2m1", bits=4, group_size=32, sym=True)
    quant_func, resolved_dtype = get_quant_func(
        dtype=scheme.data_type,
        bits=scheme.bits,
        sym=scheme.sym,
        disable_opt_rtn=True,
        group_size=scheme.group_size,
        iters=0,
    )

    expected, _, _ = quant_func(
        tensor=weight,
        bits=scheme.bits,
        group_size=scheme.group_size,
        data_type=resolved_dtype.removeprefix("rtn_"),
    )

    torch.testing.assert_close(rtn_qdq_residual(weight, scheme), expected)


def test_rtn_qdq_residual_passes_exact_mxfp4_dtype_to_quant_mx(monkeypatch):
    calls = []

    def quant_mx_spy(*args, **kwargs):
        calls.append(kwargs["data_type"])
        return quant_mx(*args, **kwargs)

    monkeypatch.setitem(QUANT_FUNC_WITH_DTYPE, "mx_fp4e2m1", quant_mx_spy)
    scheme = ResidualQuantScheme(data_type="mx_fp4e2m1", bits=4, group_size=32, sym=True)

    rtn_qdq_residual(_deterministic_weight(), scheme)

    assert calls == ["mx_fp4e2m1"]


@pytest.mark.parametrize("data_type", ["mx_fp4", "mx_fp4e2m1"])
def test_residual_quant_scheme_rejects_inconsistent_suffixed_mxfp_bits(data_type):
    with pytest.raises(ValueError, match="data_type.*bits|bits.*data_type"):
        ResidualQuantScheme(data_type=data_type, bits=8, group_size=32, sym=True)


@pytest.mark.parametrize("data_type", ["mx_fp", "mx_fp4", "mx_fp4e2m1"])
@pytest.mark.parametrize("group_size", [16, 64, (1, 32)])
def test_rtn_qdq_residual_rejects_non_deployable_mxfp4_group_size(data_type, group_size):
    scheme = ResidualQuantScheme(data_type=data_type, bits=4, group_size=group_size, sym=True)

    with pytest.raises(ValueError, match="group_size"):
        rtn_qdq_residual(_deterministic_weight(), scheme)


def test_rtn_qdq_residual_applies_group_size_to_resolved_mxfp4_dtype(monkeypatch):
    def resolve_as_mxfp4(**kwargs):
        return quant_mx, "mx_fp4"

    monkeypatch.setattr(residual_module, "get_quant_func", resolve_as_mxfp4)
    scheme = ResidualQuantScheme(data_type="registry_alias", bits=4, group_size=16, sym=True)

    with pytest.raises(ValueError, match="group_size"):
        rtn_qdq_residual(_deterministic_weight(), scheme)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"data_type": None, "bits": 4, "group_size": 32, "sym": True}, "data_type"),
        ({"data_type": "mx_fp4e2m1", "bits": 0, "group_size": 32, "sym": True}, "bits"),
        ({"data_type": "mx_fp4e2m1", "bits": 4, "group_size": None, "sym": True}, "group_size"),
        ({"data_type": "mx_fp4e2m1", "bits": 4, "group_size": 32, "sym": "true"}, "sym"),
    ],
)
def test_residual_quant_scheme_rejects_malformed_values(kwargs, field):
    with pytest.raises(ValueError, match=field):
        ResidualQuantScheme(**kwargs)


@pytest.mark.parametrize("group_size", [-2, (0, 32), (32, 0), (-1, 32), (32, -1)])
def test_residual_quant_scheme_rejects_invalid_group_size(group_size):
    with pytest.raises(ValueError, match="group_size"):
        ResidualQuantScheme(data_type="int", bits=4, group_size=group_size, sym=True)


@pytest.mark.parametrize("group_size", [0, -1])
def test_residual_quant_scheme_allows_non_mx_group_size_sentinels(group_size):
    scheme = ResidualQuantScheme(data_type="int", bits=4, group_size=group_size, sym=True)
    weight = _deterministic_weight()

    assert scheme.group_size == group_size
    assert rtn_qdq_residual(weight, scheme).shape == weight.shape


@pytest.mark.parametrize("omitted_field", ["data_type", "bits", "group_size", "sym"])
def test_residual_quant_scheme_rejects_omitted_required_value(omitted_field):
    kwargs = {"data_type": "mx_fp4e2m1", "bits": 4, "group_size": 32, "sym": True}
    kwargs.pop(omitted_field)

    with pytest.raises(ValueError, match=omitted_field):
        ResidualQuantScheme(**kwargs)


def test_rtn_qdq_residual_rejects_missing_scheme_attribute():
    scheme = SimpleNamespace(data_type="mx_fp4e2m1", bits=4, group_size=32)

    with pytest.raises(ValueError, match="sym"):
        rtn_qdq_residual(_deterministic_weight(), scheme)


def test_residual_quant_scheme_is_immutable():
    scheme = ResidualQuantScheme(data_type="mx_fp4e2m1", bits=4, group_size=32, sym=True)

    with pytest.raises(FrozenInstanceError):
        scheme.bits = 8


def test_rtn_qdq_residual_rejects_non_finite_quantizer_result(monkeypatch):
    def non_finite_quantizer(tensor, **kwargs):
        return torch.full_like(tensor, torch.inf), None, None

    monkeypatch.setitem(QUANT_FUNC_WITH_DTYPE, "mx_fp4e2m1", non_finite_quantizer)
    scheme = ResidualQuantScheme(data_type="mx_fp4e2m1", bits=4, group_size=32, sym=True)

    with pytest.raises(ValueError, match="non-finite"):
        rtn_qdq_residual(_deterministic_weight(), scheme)


def test_rtn_qdq_residual_rejects_quantizer_result_on_different_device(monkeypatch):
    def wrong_device_quantizer(tensor, **kwargs):
        return torch.empty(tensor.shape, dtype=tensor.dtype, device="meta"), None, None

    monkeypatch.setitem(QUANT_FUNC_WITH_DTYPE, "mx_fp4e2m1", wrong_device_quantizer)
    scheme = ResidualQuantScheme(data_type="mx_fp4e2m1", bits=4, group_size=32, sym=True)

    with pytest.raises(ValueError, match="device"):
        rtn_qdq_residual(_deterministic_weight(), scheme)
