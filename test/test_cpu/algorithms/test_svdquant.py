import types

import pytest
import torch

import auto_round.algorithms.transforms.svdquant.residual as residual_module
from auto_round.algorithms.pipeline import QuantizationPipeline
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.data_type import QUANT_FUNC_WITH_DTYPE


def test_svdquant_config_residual_iteration_defaults():
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    config = SVDQuantConfig()

    assert config.smooth_enabled is True
    assert config.residual_iters == 1
    assert config.residual_early_stop is False
    assert config.residual_quant_method == "rtn"
    assert "residual_iters=1" in repr(config)
    assert "residual_early_stop=False" in repr(config)
    assert "residual_quant_method='rtn'" in repr(config)
    assert "smooth_enabled=True" in repr(config)


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_svdquant_config_rejects_non_bool_smooth_enabled(value):
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    with pytest.raises(ValueError, match="smooth_enabled"):
        SVDQuantConfig(smooth_enabled=value)


def test_svdquant_config_rejects_invalid_residual_iteration_count():
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    with pytest.raises(ValueError, match="residual_iters"):
        SVDQuantConfig(residual_iters=0)


def test_svdquant_config_rejects_float_residual_iteration_count():
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    with pytest.raises(ValueError, match="residual_iters"):
        SVDQuantConfig(residual_iters=1.5)


def test_svdquant_config_rejects_bool_residual_iteration_count():
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    with pytest.raises(ValueError, match="residual_iters"):
        SVDQuantConfig(residual_iters=True)


def test_svdquant_config_rejects_non_rtn_multi_round_method():
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    with pytest.raises(ValueError, match="residual_quant_method"):
        SVDQuantConfig(residual_iters=2, residual_quant_method="signround")


def test_svdquant_config_accepts_one_round_signround_and_normalizes_method():
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    config = SVDQuantConfig(
        residual_iters=1,
        residual_early_stop=True,
        residual_quant_method="SignRound",
    )

    assert config.residual_iters == 1
    assert config.residual_early_stop is True
    assert config.residual_quant_method == "signround"


def test_svdquant_config_is_pipeline_preprocessor():
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    pipeline = QuantizationPipeline.from_configs([SVDQuantConfig(rank=8), RTNConfig()])

    assert len(pipeline.preprocessors) == 1
    assert isinstance(pipeline.preprocessors[0], SVDQuantTransform)
    assert pipeline.block_quantizer.__class__.__name__ in {"RTNQuantizer", "OptimizedRTNQuantizer"}


def test_svdquant_linear_matches_manual_reference():
    from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear

    residual = torch.nn.Linear(3, 2, bias=True)
    lora_down = torch.nn.Linear(3, 1, bias=False)
    lora_up = torch.nn.Linear(1, 2, bias=False)
    smooth = torch.tensor([2.0, 0.5, 1.0])

    with torch.no_grad():
        residual.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]))
        residual.bias.copy_(torch.tensor([0.25, -0.5]))
        lora_down.weight.copy_(torch.tensor([[0.5, -1.0, 2.0]]))
        lora_up.weight.copy_(torch.tensor([[3.0], [-2.0]]))

    layer = SVDQuantLinear(residual, lora_down, lora_up, smooth)
    x = torch.tensor([[1.0, 2.0, -1.0], [0.0, -2.0, 3.0]])

    x_hat = x * smooth
    expected = residual(x_hat) + lora_up(lora_down(x_hat))

    torch.testing.assert_close(layer(x), expected)


def test_svdquant_transform_replaces_linear_with_residual_branch():
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
    from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear

    block = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=True))
    ctx = types.SimpleNamespace(block=block, block_name="model.layers.0")
    transform = SVDQuantTransform(SVDQuantConfig(rank=2, smooth_alpha=0.5, low_rank_dtype="float32"))

    transform.pre_quantize_block(ctx)

    assert isinstance(block[0], SVDQuantLinear)
    assert isinstance(block[0].residual_linear, torch.nn.Linear)
    assert block[0].residual_linear.weight.shape == (3, 4)
    assert block[0].lora_down.out_features == 2
    assert block[0].lora_up.in_features == 2


def test_svdquant_disabled_smoothing_skips_activation_collection():
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    block = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False))
    ctx = types.SimpleNamespace(block=block, block_name="model.layers.0")
    transform = SVDQuantTransform(SVDQuantConfig(smooth_enabled=False))

    with transform.block_forward_hooks(ctx) as handles:
        assert handles == []
        block(torch.tensor([[1.0, -2.0, 3.0, -4.0]]))

    assert transform._act_max == {}


def test_svdquant_disabled_smoothing_ignores_stale_act_max_and_preserves_forward():
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    layer = torch.nn.Linear(3, 2, bias=True)
    weight = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 4.0, 1.5]])
    bias = torch.tensor([0.25, -0.75])
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    x = torch.tensor([[1.0, -2.0, 3.0], [-0.5, 4.0, 2.0]])
    expected = layer(x)
    block = torch.nn.Sequential(layer)
    transform = SVDQuantTransform(
        SVDQuantConfig(rank=1, smooth_enabled=False, low_rank_dtype="float32")
    )
    transform._act_max[id(layer)] = torch.tensor([100.0, 0.01, 7.0])

    transform.pre_quantize_block(types.SimpleNamespace(block=block, block_name="model.layers.0"))

    torch.testing.assert_close(block[0].smooth, torch.ones(3), rtol=0, atol=0)
    assert id(layer) not in transform._act_max
    effective_weight = block[0].residual_linear.weight + block[0].lora_up.weight @ block[0].lora_down.weight
    torch.testing.assert_close(effective_weight, weight)
    torch.testing.assert_close(block(x), expected)


def test_svdquant_default_smoothing_collects_act_max_and_uses_nonidentity_scale():
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    layer = torch.nn.Linear(2, 2, bias=False)
    weight = torch.tensor([[4.0, 1.0], [-2.0, 0.5]])
    with torch.no_grad():
        layer.weight.copy_(weight)
    x = torch.tensor([[1.0, -4.0], [-0.5, 2.0]])
    expected = layer(x)
    block = torch.nn.Sequential(layer)
    ctx = types.SimpleNamespace(block=block, block_name="model.layers.0")
    transform = SVDQuantTransform(SVDQuantConfig(rank=1, low_rank_dtype="float32"))

    with transform.block_forward_hooks(ctx) as handles:
        assert len(handles) == 1
        block(x)
    transform.pre_quantize_block(ctx)

    torch.testing.assert_close(block[0].smooth, torch.tensor([2.0, 0.5]), rtol=0, atol=0)
    assert id(layer) not in transform._act_max
    torch.testing.assert_close(block(x), expected)


def test_svdquant_consumes_act_max_when_decomposition_fails(monkeypatch):
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    layer = torch.nn.Linear(3, 2, bias=False)
    block = torch.nn.Sequential(layer)
    transform = SVDQuantTransform(SVDQuantConfig(rank=1))
    transform._act_max[id(layer)] = torch.tensor([1.0, 2.0, 3.0])

    def fail_svd(*args, **kwargs):
        raise RuntimeError("forced SVD failure")

    monkeypatch.setattr(torch.linalg, "svd", fail_svd)

    with pytest.raises(ValueError, match="forced SVD failure"):
        transform.pre_quantize_block(types.SimpleNamespace(block=block, block_name="model.layers.0"))

    assert id(layer) not in transform._act_max


def test_svdquant_one_round_matches_one_shot_svd_without_residual_qdq(monkeypatch):
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    weight = torch.tensor(
        [
            [3.0, -1.0, 0.5, 2.0],
            [-2.0, 4.0, 1.5, -0.5],
            [0.25, -3.0, 2.5, 1.0],
        ]
    )
    layer = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)

    qdq_calls = []

    def qdq_spy(residual, scheme):
        qdq_calls.append((residual, scheme))
        return residual

    monkeypatch.setattr(residual_module, "rtn_qdq_residual", qdq_spy)
    block = torch.nn.Sequential(layer)
    transform = SVDQuantTransform(SVDQuantConfig(rank=2, residual_iters=1, low_rank_dtype="float32"))

    transform.pre_quantize_block(types.SimpleNamespace(block=block, block_name="model.layers.0"))

    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    expected_low_rank = (u[:, :2] * s[:2].reshape(1, -1)) @ vh[:2]
    actual_low_rank = block[0].lora_up.weight @ block[0].lora_down.weight
    torch.testing.assert_close(actual_low_rank, expected_low_rank)
    torch.testing.assert_close(block[0].residual_linear.weight, weight - expected_low_rank)
    assert qdq_calls == []


def test_svdquant_multi_round_uses_registered_rtn_and_retains_no_worse_candidate(monkeypatch):
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
    from auto_round.algorithms.transforms.svdquant.residual import ResidualQuantScheme, rtn_qdq_residual

    weight = torch.tensor(
        [
            [3.0, -1.0, 0.5, 2.0, -4.0, 1.25, 0.75, -2.5],
            [-2.0, 4.0, 1.5, -0.5, 2.25, -3.5, 1.0, 0.25],
            [0.25, -3.0, 2.5, 1.0, -1.5, 3.25, -2.0, 4.5],
            [1.5, 0.5, -2.25, 3.5, 0.75, -1.0, 4.0, -3.0],
        ]
    )
    layer = torch.nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.data_type = "int"
    layer.bits = 2
    layer.group_size = 4
    layer.sym = True
    layer.global_name = "model.layers.0.proj"

    scheme = ResidualQuantScheme(data_type="int", bits=2, group_size=4, sym=True)
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    first_low_rank = (u[:, :2] * s[:2].reshape(1, -1)) @ vh[:2]
    first_residual = weight - first_low_rank
    first_qdq = rtn_qdq_residual(first_residual, scheme)
    first_error = torch.sum((weight - (first_qdq + first_low_rank)).square())

    registered_rtn = QUANT_FUNC_WITH_DTYPE["rtn_int_sym"]
    registered_calls = []

    def registered_rtn_spy(*args, **kwargs):
        registered_calls.append(kwargs.copy())
        return registered_rtn(*args, **kwargs)

    monkeypatch.setitem(QUANT_FUNC_WITH_DTYPE, "rtn_int_sym", registered_rtn_spy)
    block = torch.nn.Sequential(layer)
    transform = SVDQuantTransform(SVDQuantConfig(rank=2, residual_iters=3, low_rank_dtype="float32"))

    transform.pre_quantize_block(types.SimpleNamespace(block=block, block_name="model.layers.0"))

    assert len(registered_calls) == 3
    assert all(call["bits"] == 2 and call["group_size"] == 4 for call in registered_calls)
    selected_low_rank = block[0].lora_up.weight @ block[0].lora_down.weight
    selected_qdq = rtn_qdq_residual(block[0].residual_linear.weight, scheme)
    selected_error = torch.sum((weight - (selected_qdq + selected_low_rank)).square())
    assert selected_error <= first_error


def test_svdquant_multi_round_materializes_known_later_winner(monkeypatch):
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    weight = torch.tensor(
        [
            [3.0, -1.0, 0.5, 2.0],
            [-2.0, 4.0, 1.5, -0.5],
            [0.25, -3.0, 2.5, 1.0],
        ]
    )
    layer = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.data_type = "int"
    layer.bits = 4
    layer.group_size = 4
    layer.sym = True
    layer.global_name = "model.layers.0.later_winner"

    first_u, first_s, first_vh = torch.linalg.svd(weight, full_matrices=False)
    first_down = first_vh[:1]
    first_up = first_u[:, :1] * first_s[:1].reshape(1, -1)
    first_residual = weight - first_up @ first_down
    first_qdq = first_residual + 10.0
    second_u, second_s, second_vh = torch.linalg.svd(weight - first_qdq, full_matrices=False)
    expected_down = second_vh[:1]
    expected_up = second_u[:, :1] * second_s[:1].reshape(1, -1)
    expected_residual = weight - expected_up @ expected_down
    qdq_calls = []

    def controlled_qdq(residual, scheme):
        qdq_calls.append(residual.clone())
        if len(qdq_calls) == 1:
            return residual + 10.0
        if len(qdq_calls) == 2:
            return residual
        return residual + 20.0

    monkeypatch.setattr(residual_module, "rtn_qdq_residual", controlled_qdq)
    block = torch.nn.Sequential(layer)
    transform = SVDQuantTransform(SVDQuantConfig(rank=1, residual_iters=3, low_rank_dtype="float32"))

    transform.pre_quantize_block(types.SimpleNamespace(block=block, block_name="model.layers.0"))

    assert len(qdq_calls) == 3
    torch.testing.assert_close(block[0].lora_down.weight, expected_down, rtol=0, atol=0)
    torch.testing.assert_close(block[0].lora_up.weight, expected_up, rtol=0, atol=0)
    torch.testing.assert_close(block[0].residual_linear.weight, expected_residual, rtol=0, atol=0)


def test_svdquant_multi_round_ranks_candidates_at_materialized_bf16_dtypes(monkeypatch):
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    layer = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.zero_()
    layer.data_type = "int"
    layer.bits = 4
    layer.group_size = 2
    layer.sym = True
    layer.global_name = "model.layers.0.bf16_ranking"
    svd_calls = []

    def controlled_svd(weight, full_matrices):
        svd_calls.append(weight.clone())
        factor = 1.01 if len(svd_calls) == 1 else 1.0
        u = torch.tensor([[factor, 0.0], [0.0, 1.0]], dtype=weight.dtype, device=weight.device)
        s = torch.tensor([1.0, 0.0], dtype=weight.dtype, device=weight.device)
        vh = torch.tensor([[factor, 0.0], [0.0, 1.0]], dtype=weight.dtype, device=weight.device)
        return u, s, vh

    qdq_dtypes = []

    def identity_qdq(residual, scheme):
        qdq_dtypes.append(residual.dtype)
        return residual

    monkeypatch.setattr(torch.linalg, "svd", controlled_svd)
    monkeypatch.setattr(residual_module, "rtn_qdq_residual", identity_qdq)
    block = torch.nn.Sequential(layer)
    transform = SVDQuantTransform(SVDQuantConfig(rank=1, residual_iters=2, low_rank_dtype="bfloat16"))

    transform.pre_quantize_block(types.SimpleNamespace(block=block, block_name="model.layers.0"))

    expected_down = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)
    expected_up = torch.tensor([[1.0], [0.0]], dtype=torch.bfloat16)
    expected_residual = torch.tensor([[-1.0, 0.0], [0.0, 0.0]], dtype=torch.bfloat16)
    assert qdq_dtypes == [torch.bfloat16, torch.bfloat16]
    assert svd_calls[1].dtype == torch.float32
    torch.testing.assert_close(
        svd_calls[1],
        torch.tensor([[1.0234375, 0.0], [0.0, 0.0]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(block[0].lora_down.weight, expected_down, rtol=0, atol=0)
    torch.testing.assert_close(block[0].lora_up.weight, expected_up, rtol=0, atol=0)
    torch.testing.assert_close(block[0].residual_linear.weight, expected_residual, rtol=0, atol=0)


def test_svdquant_early_stop_materializes_best_candidate_when_second_worsens(monkeypatch):
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    weight = torch.tensor(
        [
            [2.0, -1.0, 0.5, 3.0],
            [-2.5, 4.0, 1.0, -0.5],
            [0.25, -3.0, 2.5, 1.5],
        ]
    )
    layer = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.data_type = "int"
    layer.bits = 4
    layer.group_size = 4
    layer.sym = True
    layer.global_name = "model.layers.1.proj"

    qdq_calls = []

    def worsening_second_qdq(residual, scheme):
        qdq_calls.append(residual.clone())
        return residual if len(qdq_calls) == 1 else residual + 10.0

    monkeypatch.setattr(residual_module, "rtn_qdq_residual", worsening_second_qdq)
    block = torch.nn.Sequential(layer)
    transform = SVDQuantTransform(
        SVDQuantConfig(rank=1, residual_iters=5, residual_early_stop=True, low_rank_dtype="float32")
    )

    transform.pre_quantize_block(types.SimpleNamespace(block=block, block_name="model.layers.1"))

    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    expected_low_rank = (u[:, :1] * s[:1].reshape(1, -1)) @ vh[:1]
    assert len(qdq_calls) == 2
    torch.testing.assert_close(block[0].lora_up.weight @ block[0].lora_down.weight, expected_low_rank)
    torch.testing.assert_close(block[0].residual_linear.weight, weight - expected_low_rank)


def test_svdquant_multi_round_missing_scheme_names_source_module():
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    layer = torch.nn.Linear(4, 3, bias=False)
    layer.data_type = "int"
    layer.bits = 4
    layer.global_name = "model.layers.2.missing_scheme_proj"
    block = torch.nn.Sequential(layer)
    transform = SVDQuantTransform(SVDQuantConfig(rank=1, residual_iters=2))

    with pytest.raises(ValueError, match="model.layers.2.missing_scheme_proj"):
        transform.pre_quantize_block(types.SimpleNamespace(block=block, block_name="model.layers.2"))


def test_svdquant_multi_round_retains_finite_best_after_nonfinite_qdq(monkeypatch):
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    weight = torch.tensor([[2.0, -1.0, 0.5], [-2.5, 4.0, 1.0], [0.25, -3.0, 2.5]])
    layer = torch.nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.data_type = "int"
    layer.bits = 4
    layer.group_size = 3
    layer.sym = True
    layer.global_name = "model.layers.3.proj"
    qdq_calls = []

    def nonfinite_second_qdq(residual, scheme):
        qdq_calls.append(residual.clone())
        return residual if len(qdq_calls) == 1 else torch.full_like(residual, torch.nan)

    monkeypatch.setattr(residual_module, "rtn_qdq_residual", nonfinite_second_qdq)
    block = torch.nn.Sequential(layer)
    transform = SVDQuantTransform(SVDQuantConfig(rank=1, residual_iters=4, low_rank_dtype="float32"))

    transform.pre_quantize_block(types.SimpleNamespace(block=block, block_name="model.layers.3"))

    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    expected_low_rank = (u[:, :1] * s[:1].reshape(1, -1)) @ vh[:1]
    assert len(qdq_calls) == 2
    assert torch.isfinite(block[0].residual_linear.weight).all()
    torch.testing.assert_close(block[0].lora_up.weight @ block[0].lora_down.weight, expected_low_rank)
    torch.testing.assert_close(block[0].residual_linear.weight, weight - expected_low_rank)


def test_svdquant_multi_round_rejects_nonfinite_first_qdq_with_module_and_iteration(monkeypatch):
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    layer = torch.nn.Linear(3, 3, bias=False)
    layer.data_type = "int"
    layer.bits = 4
    layer.group_size = 3
    layer.sym = True
    layer.global_name = "model.layers.4.nonfinite_proj"
    monkeypatch.setattr(
        residual_module,
        "rtn_qdq_residual",
        lambda residual, scheme: torch.full_like(residual, torch.inf),
    )
    block = torch.nn.Sequential(layer)
    transform = SVDQuantTransform(SVDQuantConfig(rank=1, residual_iters=2))

    with pytest.raises(ValueError, match=r"model\.layers\.4\.nonfinite_proj.*iteration 1"):
        transform.pre_quantize_block(types.SimpleNamespace(block=block, block_name="model.layers.4"))
