import inspect
import json

import pytest
import torch

from auto_round.export.svdquant_adapters.flux import (
    FluxSVDQuantNunchakuAdapter,
    flux_onefile_tensor_count,
)
from auto_round.export.svdquant_nunchaku import (
    IdentitySVDQuantModelAdapter,
    SVDQuantExportConfig,
    SVDQuantLinearScheme,
    SourceLinearRecord,
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
    assert all(
        tensor.device.type == "cpu" and tensor.is_contiguous()
        for tensor in (record.residual_weight, record.lora_down, record.lora_up)
    )


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
    torch.testing.assert_close(
        out_proj.residual_weight + out_proj.lora_up @ out_proj.lora_down, expected[:, :8], atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        mlp_fc2.residual_weight + mlp_fc2.lora_up @ mlp_fc2.lora_down, expected[:, 8:], atol=2e-5, rtol=2e-5
    )
    assert out_proj.bias is None
    torch.testing.assert_close(mlp_fc2.bias, source.bias)


@pytest.mark.parametrize(
    ("source_name", "target_name"),
    [
        ("transformer_blocks.0.attn.to_out.0", "transformer_blocks.0.out_proj"),
        ("transformer_blocks.0.ff.net.2.linear", "transformer_blocks.0.mlp_fc2"),
        ("transformer_blocks.0.ff_context.net.0.proj", "transformer_blocks.0.mlp_context_fc1"),
        ("single_transformer_blocks.0.proj_mlp", "single_transformer_blocks.0.mlp_fc1"),
    ],
)
def test_direct_maps_preserve_logical_records(source_name, target_name):
    source = _source(source_name)
    (record,) = tuple(FluxSVDQuantNunchakuAdapter(require_complete_model=False).map_modules(_model(), (source,)))
    assert record.prefix == target_name
    assert record.residual_weight is source.residual_weight
    assert record.lora_down is source.lora_down
    assert record.smooth is source.smooth


def _install(root, path, module):
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(current, part):
            current.add_module(part, torch.nn.Module())
        current = getattr(current, part)
    current.add_module(parts[-1], module)


def test_extra_tensors_pack_adanorm_copy_norms_and_top_level_bf16():
    model = _model()
    for name, splits in (
        ("transformer_blocks.0.norm1.linear", 6),
        ("transformer_blocks.0.norm1_context.linear", 6),
        ("single_transformer_blocks.0.norm.linear", 3),
    ):
        linear = torch.nn.Linear(1024, 12, bias=True, dtype=torch.bfloat16)
        _install(model, name, linear)
    for local_name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        norm = torch.nn.Module()
        norm.weight = torch.nn.Parameter(torch.randn(8))
        _install(model, f"transformer_blocks.0.attn.{local_name}", norm)
    _install(model, "x_embedder", torch.nn.Linear(8, 8))
    _install(model, "unrelated", torch.nn.Linear(8, 8))

    tensors = FluxSVDQuantNunchakuAdapter(require_complete_model=False).extra_tensors(model)

    for prefix in (
        "transformer_blocks.0.norm1.linear",
        "transformer_blocks.0.norm1_context.linear",
        "single_transformer_blocks.0.norm.linear",
    ):
        assert {f"{prefix}.{suffix}" for suffix in ("qweight", "wscales", "wzeros", "bias")} <= tensors.keys()
    assert tensors["transformer_blocks.0.norm_added_k.weight"].dtype == torch.bfloat16
    assert tensors["x_embedder.weight"].dtype == torch.bfloat16
    assert not any(key.startswith("unrelated.") for key in tensors)
    assert all(tensor.device.type == "cpu" and tensor.is_contiguous() for tensor in tensors.values())


def test_metadata_explicit_config_and_complete_mode_rejects_gaps():
    adapter = FluxSVDQuantNunchakuAdapter(config={"num_layers": 2, "num_single_layers": 0}, require_complete_model=True)
    metadata = adapter.metadata(_model(), 2)
    assert metadata["model_class"] == "NunchakuFluxTransformer2dModel"
    assert json.loads(metadata["config"])["num_layers"] == 2
    assert metadata["format"] == "pt" and metadata["comfy_config"] == "{}"
    with pytest.raises(ValueError, match="indices mismatch"):
        tuple(adapter.map_modules(_model(), (_source("transformer_blocks.1.attn.to_q"),)))


def test_standard_flux_onefile_count_formula():
    assert flux_onefile_tensor_count(19, 38) == 2604


def test_generic_export_merges_adapter_extras_and_rejects_duplicates():
    residual = torch.nn.Linear(32, 8)
    residual.data_type, residual.bits, residual.group_size, residual.sym = "mx_fp4", 4, 32, True
    residual.act_data_type, residual.act_bits, residual.act_group_size = "mx_fp4", 4, 32
    residual.act_sym, residual.act_dynamic = True, True
    from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear

    wrapped = SVDQuantLinear(
        residual, torch.nn.Linear(32, 2, bias=False), torch.nn.Linear(2, 8, bias=False), torch.ones(32)
    )
    model = torch.nn.Sequential(wrapped)

    class ExtraAdapter(IdentitySVDQuantModelAdapter):
        def extra_tensors(self, model):
            return {"passthrough.weight": torch.ones(3).tanh()}

    tensors = collect_svdquant_tensors(model, adapter=ExtraAdapter())
    assert torch.equal(tensors["passthrough.weight"], torch.ones(3).tanh())

    class DuplicateAdapter(IdentitySVDQuantModelAdapter):
        def extra_tensors(self, model):
            return {"0.bias": torch.ones(8)}

    with pytest.raises(ValueError, match="duplicate tensor key"):
        collect_svdquant_tensors(model, adapter=DuplicateAdapter())


def test_adapter_sources_have_no_external_runtime_imports():
    import auto_round.export.svdquant_adapters.flux as module

    source = inspect.getsource(module).lower()
    assert "import deepcompressor" not in source
    assert "import nunchaku" not in source


def test_partial_flux_collect_and_save_roundtrip(tmp_path):
    from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
    from safetensors import safe_open

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
