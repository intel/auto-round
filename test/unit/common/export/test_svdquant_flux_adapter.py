import pytest
import torch

from auto_round.export.svdquant_adapters.flux import FluxSVDQuantNunchakuAdapter
from auto_round.export.svdquant_nunchaku import (
    SourceLinearRecord,
    SVDQuantExportConfig,
    SVDQuantLinearScheme,
    collect_svdquant_tensors,
    save_svdquant_nunchaku_safetensors,
)

SCHEME = SVDQuantLinearScheme("mx_fp4", 4, 32, True, "mx_fp4", 4, 32, True, True)


def _source(name, out_features=8, in_features=8, rank=2, seed=0, bias=True):
    generator = torch.Generator().manual_seed(seed)
    return SourceLinearRecord(
        name=name,
        residual_weight=torch.randn(out_features, in_features, generator=generator),
        lora_down=torch.randn(rank, in_features, generator=generator),
        lora_up=torch.randn(out_features, rank, generator=generator),
        smooth=torch.linspace(0.5, 1.5, in_features),
        smooth_orig=torch.linspace(0.75, 1.75, in_features),
        bias=torch.randn(out_features, generator=generator) if bias else None,
        scheme=SCHEME,
    )


def _effective(source):
    return (source.residual_weight + source.lora_up @ source.lora_down) * source.smooth


def _model(config=None):
    model = torch.nn.Module()
    model.config = config or {
        "num_layers": 1,
        "num_single_layers": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 4,
    }
    return model


def test_double_qkv_reconstructs_effective_sources_at_fixed_rank_in_order():
    sources = tuple(
        _source(f"transformer_blocks.0.attn.to_{name}", seed=index + 1) for index, name in enumerate(("q", "k", "v"))
    )
    adapter = FluxSVDQuantNunchakuAdapter(require_complete_model=False)

    (record,) = tuple(adapter.map_modules(_model(), sources))

    assert record.prefix == "transformer_blocks.0.qkv_proj"
    assert record.sources == sources
    assert record.lora_down.shape[0] == 2
    assert torch.equal(record.smooth, torch.ones(8))
    expected = torch.cat([_effective(source) for source in sources])
    actual = record.residual_weight + record.lora_up @ record.lora_down
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(record.bias, torch.cat([source.bias for source in sources]))
    assert all(
        tensor.device.type == "cpu" and tensor.is_contiguous()
        for tensor in (record.residual_weight, record.lora_down, record.lora_up)
    )


def test_double_qkv_preserves_shared_low_rank_and_smooth_without_redecomposition():
    shared_down = torch.randn(2, 8)
    shared_smooth = torch.linspace(0.5, 1.5, 8)
    shared_smooth_orig = torch.linspace(0.75, 1.75, 8)
    sources = tuple(
        _source(f"transformer_blocks.0.attn.to_{name}", seed=index + 21) for index, name in enumerate(("q", "k", "v"))
    )
    sources = tuple(
        SourceLinearRecord(
            name=source.name,
            residual_weight=source.residual_weight,
            lora_down=shared_down.clone(),
            lora_up=source.lora_up,
            smooth=shared_smooth.clone(),
            smooth_orig=shared_smooth_orig.clone(),
            bias=source.bias,
            scheme=source.scheme,
        )
        for source in sources
    )

    (record,) = tuple(FluxSVDQuantNunchakuAdapter(require_complete_model=False).map_modules(_model(), sources))

    torch.testing.assert_close(record.residual_weight, torch.cat([source.residual_weight for source in sources]))
    torch.testing.assert_close(record.lora_down, shared_down)
    torch.testing.assert_close(record.lora_up, torch.cat([source.lora_up for source in sources]))
    torch.testing.assert_close(record.smooth, shared_smooth)
    torch.testing.assert_close(record.smooth_orig, shared_smooth_orig)


def test_single_proj_out_splits_input_columns_and_keeps_bias_only_on_mlp():
    source = _source("single_transformer_blocks.0.proj_out", out_features=8, in_features=16, seed=10)
    model = _model({"num_layers": 0, "num_single_layers": 1, "num_attention_heads": 2, "attention_head_dim": 4})
    adapter = FluxSVDQuantNunchakuAdapter(require_complete_model=False)

    out_proj, mlp_fc2 = tuple(adapter.map_modules(model, (source,)))

    assert (out_proj.prefix, mlp_fc2.prefix) == (
        "single_transformer_blocks.0.out_proj",
        "single_transformer_blocks.0.mlp_fc2",
    )
    expected = _effective(source)
    out_effective = (out_proj.residual_weight + out_proj.lora_up @ out_proj.lora_down) * out_proj.smooth
    mlp_effective = (mlp_fc2.residual_weight + mlp_fc2.lora_up @ mlp_fc2.lora_down) * mlp_fc2.smooth
    torch.testing.assert_close(out_effective, expected[:, :8], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(mlp_effective, expected[:, 8:], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(out_proj.lora_down, source.lora_down[:, :8])
    torch.testing.assert_close(mlp_fc2.lora_down, source.lora_down[:, 8:])
    torch.testing.assert_close(out_proj.smooth, source.smooth[:8])
    torch.testing.assert_close(mlp_fc2.smooth, source.smooth[8:])
    assert out_proj.bias is None
    torch.testing.assert_close(mlp_fc2.bias, source.bias)


def _install(root, path, module):
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(current, part):
            current.add_module(part, torch.nn.Module())
        current = getattr(current, part)
    current.add_module(parts[-1], module)


def _install_parameter(root, name, tensor):
    module_name, parameter_name = name.rsplit(".", 1)
    current = root
    for part in module_name.split("."):
        if not hasattr(current, part):
            current.add_module(part, torch.nn.Module())
        current = getattr(current, part)
    current.register_parameter(parameter_name, torch.nn.Parameter(tensor))


def _small_complete_top_level_tensors():
    from auto_round.export.svdquant_adapters.flux import FLUX_TOP_LEVEL_TENSOR_KEYS

    return {
        key: torch.ones(2, dtype=torch.bfloat16) if key.endswith(".bias") else torch.ones(2, 2, dtype=torch.bfloat16)
        for key in FLUX_TOP_LEVEL_TENSOR_KEYS
    }


def _small_linear_tensors(prefix):
    return {
        f"{prefix}.qweight": torch.ones(1, 1, dtype=torch.int8),
        f"{prefix}.wscales": torch.ones(1, 1, dtype=torch.uint8),
        f"{prefix}.smooth": torch.ones(1, dtype=torch.bfloat16),
        f"{prefix}.smooth_orig": torch.ones(1, dtype=torch.bfloat16),
        f"{prefix}.lora_down": torch.ones(1, 1, dtype=torch.bfloat16),
        f"{prefix}.lora_up": torch.ones(1, 1, dtype=torch.bfloat16),
        f"{prefix}.bias": torch.ones(1, dtype=torch.bfloat16),
    }


def _small_adanorm_tensors(prefix):
    return {
        f"{prefix}.qweight": torch.ones(1, 1, dtype=torch.int32),
        f"{prefix}.wscales": torch.ones(1, 1, dtype=torch.bfloat16),
        f"{prefix}.wzeros": torch.ones(1, 1, dtype=torch.bfloat16),
        f"{prefix}.bias": torch.ones(1, dtype=torch.bfloat16),
    }


def _small_standard_complete_tensors():
    tensors = _small_complete_top_level_tensors()
    for index in range(19):
        block = f"transformer_blocks.{index}"
        for linear in (
            "qkv_proj",
            "qkv_proj_context",
            "out_proj",
            "out_proj_context",
            "mlp_fc1",
            "mlp_fc2",
            "mlp_context_fc1",
            "mlp_context_fc2",
        ):
            tensors.update(_small_linear_tensors(f"{block}.{linear}"))
        tensors.update(_small_adanorm_tensors(f"{block}.norm1.linear"))
        tensors.update(_small_adanorm_tensors(f"{block}.norm1_context.linear"))
        for norm in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            tensors[f"{block}.{norm}.weight"] = torch.ones(1, dtype=torch.bfloat16)
    for index in range(38):
        block = f"single_transformer_blocks.{index}"
        for linear in ("qkv_proj", "out_proj", "mlp_fc1", "mlp_fc2"):
            tensors.update(_small_linear_tensors(f"{block}.{linear}"))
        tensors.update(_small_adanorm_tensors(f"{block}.norm.linear"))
        for norm in ("norm_q", "norm_k"):
            tensors[f"{block}.{norm}.weight"] = torch.ones(1, dtype=torch.bfloat16)
    return tensors


def test_standard_complete_schema_validates_2604_tiny_tensors_without_model_allocation():
    model = _model({"num_layers": 19, "num_single_layers": 38})
    adapter = FluxSVDQuantNunchakuAdapter(require_complete_model=True)
    tensors = _small_standard_complete_tensors()

    assert len(tensors) == 2604
    adapter.validate(tensors, adapter.metadata(model, 32))


def test_complete_extra_collection_requires_and_copies_exact_top_level_parameters():
    model = _model({"num_layers": 0, "num_single_layers": 0})
    expected = _small_complete_top_level_tensors()
    for key, tensor in expected.items():
        _install_parameter(model, key, tensor)

    actual = FluxSVDQuantNunchakuAdapter(require_complete_model=True).extra_tensors(model)

    assert set(actual) == set(expected)
    assert all(tensor.dtype == torch.bfloat16 for tensor in actual.values())


def test_partial_flux_collect_and_save_roundtrip(tmp_path):
    from safetensors import safe_open

    from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear

    residual = torch.nn.Linear(32, 8)
    residual.data_type, residual.bits, residual.group_size, residual.sym = "mx_fp4", 4, 32, True
    residual.act_data_type, residual.act_bits, residual.act_group_size = "mx_fp4", 4, 32
    residual.act_sym, residual.act_dynamic = True, True
    wrapped = SVDQuantLinear(
        residual,
        torch.nn.Linear(32, 2, bias=False),
        torch.nn.Linear(2, 8, bias=False),
        torch.linspace(0.5, 1.5, 32),
    )
    model = _model({"num_layers": 1, "num_single_layers": 0})
    _install(model, "transformer_blocks.0.attn.to_out.0", wrapped)
    _install(model, "x_embedder", torch.nn.Linear(8, 8))
    adapter = FluxSVDQuantNunchakuAdapter(require_complete_model=False)
    config = SVDQuantExportConfig(runtime_loadable=True)

    collected = collect_svdquant_tensors(model, adapter=adapter, config=config)
    path = tmp_path / "flux.safetensors"
    save_svdquant_nunchaku_safetensors(model, str(path), adapter=adapter, config=config)

    with safe_open(path, framework="pt") as handle:
        assert set(handle.keys()) == set(collected)
        assert handle.metadata()["model_class"] == "NunchakuFluxTransformer2dModel"
        assert handle.get_tensor("x_embedder.weight").dtype == torch.bfloat16
