# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""CPU regressions for Nemotron-H Omni wrapper loading (Super VL / Bolt)."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import NemotronHConfig, conversion_mapping

from auto_round.modeling.nemotron_h_omni import (
    apply_nemotron_h_omni_post_load,
    prefer_remote_omni_class,
    register_nemotron_h_omni_checkpoint_mapping,
)
from auto_round.modeling.unfused_moe import apply_model_monkey_patches

_OMNI_AUTO_MAP = {"AutoModelForCausalLM": "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3"}


def _omni_config(text_model_type="nemotron_h_puzzle"):
    return {"model_type": "nemotron_h_omni", "auto_map": _OMNI_AUTO_MAP, "llm_config": {"model_type": text_model_type}}


@pytest.fixture
def mapping_registry(monkeypatch):
    """Isolate transformers' global conversion-mapping registry."""
    store = {}
    monkeypatch.setattr(conversion_mapping, "get_checkpoint_conversion_mapping", lambda t: list(store.get(t, [])))

    def register(model_type, mapping, overwrite=False):
        store[model_type] = list(mapping)

    monkeypatch.setattr(conversion_mapping, "register_checkpoint_conversion_mapping", register)
    return store


@pytest.mark.parametrize("text_model_type", ["nemotron_h_puzzle", "nemotron_h"])
def test_omni_rename_is_appended_once(mapping_registry, text_model_type):
    upstream = object()
    mapping_registry[text_model_type] = [upstream]
    assert register_nemotron_h_omni_checkpoint_mapping(_omni_config(text_model_type))
    assert register_nemotron_h_omni_checkpoint_mapping(_omni_config(text_model_type))
    rules = mapping_registry[text_model_type]
    assert rules[0] is upstream and len(rules) == 2
    assert rules[1].source_patterns == ["language_model.backbone."]
    assert rules[1].target_patterns == ["language_model.model."]


@pytest.mark.parametrize("config", [{"model_type": "nemotron_h"}, {"model_type": "qwen3_moe"}, None])
def test_omni_rename_skips_other_models(mapping_registry, config):
    assert not register_nemotron_h_omni_checkpoint_mapping(config)
    assert mapping_registry == {}


@pytest.mark.parametrize(
    "text_model_type,trust,overridden",
    [("nemotron_h_puzzle", True, True), ("nemotron_h_puzzle", False, False), ("nemotron_h", True, False)],
)
def test_remote_omni_class_name_mapping(mapping_registry, text_model_type, trust, overridden):
    """Transformers 5.20's native same-name class mapping must not rename the remote projector."""
    native = object()
    class_name = "NemotronH_Omni_Reasoning_V3"
    mapping_registry[class_name] = [native]
    config = {**_omni_config(text_model_type), "architectures": [class_name]}
    register_nemotron_h_omni_checkpoint_mapping(config, trust_remote_code=trust)
    rules = mapping_registry[class_name]
    if not overridden:
        assert rules == [native]
        return
    renamed = {}
    for src in ("mlp1.0.weight", "mlp1.1.weight", "mlp1.3.weight"):
        renamed[src] = next(r.rename_source_key(src)[0] for r in rules if r.rename_source_key(src)[0] != src)
    assert renamed == {
        "mlp1.0.weight": "vision_projector.mlp1.norm.weight",
        "mlp1.1.weight": "vision_projector.mlp1.linear1.weight",
        "mlp1.3.weight": "vision_projector.mlp1.linear2.weight",
    }


def test_remote_class_preferred_only_for_trusted_puzzle_omni():
    assert prefer_remote_omni_class(_omni_config(), trust_remote_code=True)
    assert prefer_remote_omni_class(SimpleNamespace(**_omni_config()), trust_remote_code=True)
    assert not prefer_remote_omni_class(_omni_config(), trust_remote_code=False)
    assert not prefer_remote_omni_class(_omni_config("nemotron_h"), trust_remote_code=True)
    assert not prefer_remote_omni_class({"model_type": "nemotron_h_omni", "llm_config": {}}, trust_remote_code=True)
    assert not prefer_remote_omni_class({"model_type": "nemotron_h", "auto_map": _OMNI_AUTO_MAP}, True)


def test_full_load_passes_root_level_trunk_rename():
    from auto_round.modeling.nemotron_h_omni import nemotron_h_omni_load_kwargs

    assert nemotron_h_omni_load_kwargs(_omni_config("nemotron_h")) == {
        "key_mapping": {r"^language_model\.backbone\.": "language_model.model."}
    }
    assert nemotron_h_omni_load_kwargs({"model_type": "nemotron_h"}) == {}


def test_disk_stream_resolves_names_needing_several_renames(monkeypatch):
    """RADIO's attention output projection needs two of its family's renames at once."""
    from transformers.core_model_loading import WeightRenaming

    from auto_round.utils import disk_stream_util

    rules = (
        WeightRenaming("radio_model.model.blocks", "encoder.layer").reverse_transform(),
        WeightRenaming("attn.proj", "attention.output.dense").reverse_transform(),
    )
    monkeypatch.setattr(disk_stream_util, "_index_model_types", lambda index: ("radio",))
    monkeypatch.setattr(disk_stream_util, "_reverse_renamings_for", lambda model_type: rules)
    monkeypatch.setattr(disk_stream_util, "_renamed_expert_candidates", lambda index, name: ())
    stored = {"vision_model.radio_model.model.blocks.0.attn.proj.weight", "vision_model.radio_model.model.blocks.0.x"}
    index = SimpleNamespace(has_tensor=stored.__contains__)
    resolve = disk_stream_util._resolve_checkpoint_name
    assert resolve(index, "vision_model.encoder.layer.0.attention.output.dense.weight") == (
        "vision_model.radio_model.model.blocks.0.attn.proj.weight"
    )
    assert resolve(index, "vision_model.encoder.layer.0.x") == "vision_model.radio_model.model.blocks.0.x"
    assert resolve(index, "vision_model.encoder.layer.1.attention.output.dense.weight") is None


def _wrapper(model_type, source_dir):
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        model_type=model_type,
        _name_or_path=str(source_dir),
        llm_config=SimpleNamespace(mamba_num_heads=4, mamba_head_dim=8, n_groups=2),
    )
    model.language_model = torch.nn.Module()
    model.language_model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.mixer = torch.nn.Module()
    layer.mixer.A_log = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    layer.mixer.gate = torch.nn.Module()
    layer.mixer.gate.register_buffer("e_score_correction_bias", torch.zeros(3, dtype=torch.bfloat16))
    model.language_model.model.layers = torch.nn.ModuleList([layer])
    return model


def test_post_load_restores_fp32_tensors_for_omni(tmp_path):
    a_log = torch.tensor([0.1, 0.2, 0.3, 0.4000001], dtype=torch.float32)
    bias = torch.tensor([1e-4, 2e-4, 3e-4], dtype=torch.float32)
    save_file(
        {
            "language_model.backbone.layers.0.mixer.A_log": a_log,
            "language_model.backbone.layers.0.mixer.gate.e_score_correction_bias": bias,
            "language_model.backbone.layers.0.mixer.in_proj.weight": torch.zeros(2, 2),
        },
        str(tmp_path / "model.safetensors"),
    )
    model = _wrapper("nemotron_h_omni", tmp_path)
    assert apply_nemotron_h_omni_post_load(model)["high_precision_restored"] == 2
    mixer = model.language_model.model.layers[0].mixer
    assert mixer.A_log.dtype == torch.float32 and torch.equal(mixer.A_log.data, a_log)
    assert torch.equal(mixer.gate.e_score_correction_bias, bias)
    # idempotent
    assert apply_nemotron_h_omni_post_load(model)["high_precision_restored"] == 0


def test_post_load_sets_group_size_per_instance_only(tmp_path):
    """The gated-norm group_size is restored on the instance; the class stays untouched."""

    class Zamba2RMSNormGated(torch.nn.Module):
        pass

    save_file({"language_model.backbone.layers.0.mixer.A_log": torch.ones(4)}, str(tmp_path / "model.safetensors"))
    model = _wrapper("nemotron_h_omni", tmp_path)
    model.language_model.model.layers[0].mixer.norm = Zamba2RMSNormGated()
    assert apply_nemotron_h_omni_post_load(model)["zamba2_patched"] == 1
    assert model.language_model.model.layers[0].mixer.norm.group_size == 4 * 8 // 2
    assert not hasattr(Zamba2RMSNormGated, "group_size")
    assert not hasattr(Zamba2RMSNormGated(), "group_size")


@pytest.mark.parametrize("model_type", ["nemotron_h", "nemotron_h_puzzle", "qwen3_moe"])
def test_post_load_is_noop_for_other_models(tmp_path, model_type):
    save_file({"language_model.backbone.layers.0.mixer.A_log": torch.ones(4)}, str(tmp_path / "model.safetensors"))
    model = _wrapper(model_type, tmp_path)
    assert apply_nemotron_h_omni_post_load(model) == {"zamba2_patched": 0, "high_precision_restored": 0}
    assert model.language_model.model.layers[0].mixer.A_log.dtype == torch.bfloat16


def test_plain_nemotron_h_text_model_is_not_patched(tmp_path):
    """Plain nemotron_h (e.g. Lightning) keeps the default fused-MoE path: no unfused-MoE patch."""
    NemotronHConfig(hidden_size=16, vocab_size=32, num_attention_heads=2, num_key_value_heads=2).save_pretrained(
        tmp_path
    )
    assert not apply_model_monkey_patches(str(tmp_path))


def test_nested_experts_config_gets_linear_experts_implementation():
    """Unfused experts inside a multimodal wrapper dispatch on their own nested config."""
    pytest.importorskip("transformers.integrations.moe")
    from transformers import NemotronHForCausalLM

    from auto_round.modeling.fused_moe.moe_experts_interface import (
        AUTO_ROUND_EXPERTS_IMPLS,
        prepare_model_for_moe_quantization,
    )

    text_config = NemotronHConfig(
        hidden_size=16,
        vocab_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        layers_block_type=["moe"],
        num_hidden_layers=1,
        moe_intermediate_size=12,
        moe_shared_expert_intermediate_size=12,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        num_nextn_predict_layers=0,
    )
    wrapper = torch.nn.Module()
    wrapper.config = SimpleNamespace(model_type="nemotron_h_omni")
    wrapper.language_model = NemotronHForCausalLM(text_config)
    assert prepare_model_for_moe_quantization(wrapper)
    impl = wrapper.config._experts_implementation
    assert impl in AUTO_ROUND_EXPERTS_IMPLS
    assert wrapper.language_model.config._experts_implementation == impl
    hidden = torch.randn(1, 3, 16)
    wrapper.language_model.model.layers[0].mixer(hidden)  # unfused experts run on the nested config


@pytest.mark.parametrize("processor_fails", [False, True])
def test_multimodal_disk_stream_keeps_meta_skeleton(monkeypatch, processor_fails):
    from unittest.mock import Mock

    from auto_round.context import model as context_module

    monkeypatch.setattr(context_module.ModelContext, "_instances", {})

    class Tiny(torch.nn.Module):
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.block = torch.nn.Linear(2, 2, device="meta")
            self.restored_ssm = torch.nn.Parameter(torch.ones(2))

    model, tokenizer, index = Tiny(), object(), object()
    processor = Mock(image_processor=object())
    load_processor = Mock(side_effect=ValueError("incompatible processor") if processor_fails else None)
    load_processor.return_value = processor
    full_load = Mock(side_effect=AssertionError("must not materialize the full checkpoint"))
    warning = Mock()
    materialize = Mock()
    monkeypatch.setattr(context_module, "is_diffusion_model", lambda *a, **kw: False)
    monkeypatch.setattr(context_module, "is_mllm_model", lambda *a, **kw: True)
    monkeypatch.setattr(context_module, "is_moe_model", lambda *a, **kw: False)
    monkeypatch.setattr(context_module, "install_debug_layer_config_patch", lambda: None)
    monkeypatch.setattr(context_module, "apply_nemotron_h_omni_post_load", lambda _: None)
    monkeypatch.setattr(context_module, "mllm_load_model", full_load)
    monkeypatch.setattr(context_module.logger, "warning", warning)
    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", load_processor)
    monkeypatch.setattr(context_module.ModelContext, "_peek_config", lambda _: None)
    monkeypatch.setattr(context_module.ModelContext, "_should_use_meta_skeleton", lambda *a: True)
    monkeypatch.setattr(context_module.ModelContext, "_build_disk_stream_model", lambda *a: (model, tokenizer, index))
    monkeypatch.setattr(context_module.ModelContext, "_materialize_disk_stream_non_block_params", materialize)
    monkeypatch.setattr(context_module.ModelContext, "_import_custom_moe_replacements", lambda *a: None)
    monkeypatch.setattr(
        context_module.ModelContext, "_set_amp_dtype", lambda self: setattr(self, "amp_dtype", torch.float32)
    )
    guard = Mock(side_effect=AssertionError("disk-stream skeleton must bypass the mixed-meta guard"))
    monkeypatch.setattr(context_module, "unsupported_meta_device", guard)
    context = context_module.ModelContext(model="tiny-nemotron", need_calib=False)
    assert context.model is model and context.tokenizer is tokenizer and context._disk_stream_index is index
    materialize.assert_called_once()
    full_load.assert_not_called()
    guard.assert_not_called()
    if processor_fails:
        assert context.processor is None and context.image_processor is None
        assert warning.call_args.kwargs["exc_info"] is True
    else:
        assert context.processor is processor and context.image_processor is processor.image_processor


def test_non_streamed_meta_model_still_rejected(monkeypatch):
    from auto_round.context import model as context_module

    monkeypatch.setattr(context_module.ModelContext, "_instances", {})

    monkeypatch.setattr(context_module.ModelContext, "_load_model", lambda _: None)
    monkeypatch.setattr(context_module, "unsupported_meta_device", lambda _: True)
    with pytest.raises(RuntimeError, match="does not support parameters on meta"):
        context_module.ModelContext(model=torch.nn.Linear(2, 2, device="meta"), need_calib=False)
