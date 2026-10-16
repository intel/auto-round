# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""CPU regressions for Nemotron-H routers and native/remote weight conversion."""

from contextlib import contextmanager

import pytest
import torch
from transformers import NemotronHConfig, NemotronHForCausalLM, PreTrainedConfig, PreTrainedModel, conversion_mapping
from transformers.core_model_loading import WeightRenaming

from auto_round.modeling.unfused_moe import (
    apply_model_monkey_patches,
    get_checkpoint_conversion_mapping_ar,
    get_model_conversion_mapping_ar,
)
from auto_round.modeling.unfused_moe.nemotron_h import LinearNemotronHMoE


def _config():
    return NemotronHConfig(
        hidden_size=16,
        vocab_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        layers_block_type=["moe"],
        moe_intermediate_size=12,
        moe_shared_expert_intermediate_size=12,
        moe_latent_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        num_nextn_predict_layers=0,
    )


@contextmanager
def _patched_mapping(monkeypatch):
    original = getattr(
        conversion_mapping,
        "orig_get_checkpoint_conversion_mapping",
        conversion_mapping.get_checkpoint_conversion_mapping,
    )
    original_model = getattr(
        conversion_mapping, "orig_get_model_conversion_mapping", conversion_mapping.get_model_conversion_mapping
    )
    with monkeypatch.context() as patch:
        patch.setattr(conversion_mapping, "orig_get_checkpoint_conversion_mapping", original, raising=False)
        patch.setattr(conversion_mapping, "orig_get_model_conversion_mapping", original_model, raising=False)
        patch.setattr(conversion_mapping, "get_checkpoint_conversion_mapping", get_checkpoint_conversion_mapping_ar)
        patch.setattr(conversion_mapping, "get_model_conversion_mapping", get_model_conversion_mapping_ar)
        yield


@pytest.mark.parametrize("legacy_router", [False, True])
def test_nemotron_router_output_and_backward(legacy_router, monkeypatch):
    block = LinearNemotronHMoE(_config())
    hidden = torch.randn(2, 3, 16, requires_grad=True)
    if legacy_router:
        # Transformers 5.12 returned just logits; 5.20 returns top-k results too.
        monkeypatch.setattr(
            block.gate,
            "forward",
            lambda value: torch.nn.functional.linear(value.reshape(-1, 16).float(), block.gate.weight.float()),
        )
    result = block(hidden)
    assert result.shape == hidden.shape
    assert block.experts[0].up_proj.weight.shape == (12, 8)
    assert torch.isfinite(result).all()
    result.square().sum().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    assert block.gate.weight.grad is not None and block.gate.weight.grad.abs().sum() > 0


def test_nemotron_native_checkpoint_roundtrip(tmp_path, monkeypatch):
    import transformers.modeling_utils as modeling_utils
    import transformers.models.nemotron_h.modeling_nemotron_h as modeling

    config = _config()
    config.save_pretrained(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(modeling, "NemotronHMoE", modeling.NemotronHMoE)
        patch.setitem(modeling.MIXER_TYPES, "moe", modeling.MIXER_TYPES["moe"])
        patch.setattr(modeling_utils, "get_model_conversion_mapping", modeling_utils.get_model_conversion_mapping)
        patch.setattr(
            modeling_utils,
            "get_checkpoint_conversion_mapping",
            getattr(modeling_utils, "get_checkpoint_conversion_mapping", None),
            raising=False,
        )
        with _patched_mapping(patch):
            assert apply_model_monkey_patches(str(tmp_path))
            model = NemotronHForCausalLM(config).eval()
            assert isinstance(model.model.layers[0].mixer, LinearNemotronHMoE)
            checkpoint = {
                key.replace("model.", "backbone.", 1) if key.startswith("model.") else key: value.clone()
                for key, value in model.state_dict().items()
            }
            from safetensors.torch import save_file

            save_file(checkpoint, str(tmp_path / "model.safetensors"))
            loaded, info = NemotronHForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
            assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
            for key, value in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(
                loaded(torch.tensor([[1, 2, 3]]), use_cache=False).logits,
                model(torch.tensor([[1, 2, 3]]), use_cache=False).logits,
            )


def test_remote_omni_mapping_does_not_use_native_class_override(monkeypatch):
    class NemotronH_Omni_Reasoning_V3(PreTrainedModel):
        def __init__(self):
            config = PreTrainedConfig()
            config.model_type = "nemotron_h_omni"
            super().__init__(config)
            self.vision_projector = torch.nn.Linear(2, 2)

    native = WeightRenaming(r"^mlp1\.0\.", "multi_modal_projector.layer_norm.")
    remote = WeightRenaming(r"^mlp1\.0\.", "vision_projector.mlp1.norm.")
    # Registering the model_type (as real NVIDIA remote code does) must not let
    # a same-named native class override the remote wrapper's module names.
    monkeypatch.setattr(conversion_mapping, "_checkpoint_conversion_mapping_cache", None)
    registered = getattr(conversion_mapping, "USER_REGISTERED_MAPPINGS", None)
    if registered is not None:
        monkeypatch.setattr(conversion_mapping, "USER_REGISTERED_MAPPINGS", set())
    conversion_mapping.register_checkpoint_conversion_mapping("NemotronH_Omni_Reasoning_V3", [native], overwrite=True)
    conversion_mapping.register_checkpoint_conversion_mapping("nemotron_h_omni", [remote], overwrite=True)
    with _patched_mapping(monkeypatch):
        mappings = conversion_mapping.get_model_conversion_mapping(NemotronH_Omni_Reasoning_V3(), add_legacy=False)
        assert [rule.target_patterns for rule in mappings] == [["vision_projector.mlp1.norm."]]
        # A subsequent native load still uses the native class mapping.
        assert get_checkpoint_conversion_mapping_ar("NemotronH_Omni_Reasoning_V3")[0].target_patterns == [
            "multi_modal_projector.layer_norm."
        ]


def test_preserved_mapping_appends_autoround_rules(monkeypatch):
    from auto_round.modeling.unfused_moe import MODEL_CONFIG

    keep = WeightRenaming("source", "kept")
    drop = WeightRenaming("expert", "dropped")
    extra = WeightRenaming("additional", "appended")
    monkeypatch.setitem(
        MODEL_CONFIG,
        "test_mapping",
        {
            "preserve_upstream_conversion_mapping": True,
            "drop_conversion_target_patterns": ["dropped"],
            "checkpoint_mapping": [extra],
        },
    )
    monkeypatch.setattr(
        conversion_mapping, "orig_get_checkpoint_conversion_mapping", lambda _: [keep, drop], raising=False
    )
    assert get_checkpoint_conversion_mapping_ar("test_mapping") == [keep, extra]


def test_puzzle_checkpoint_prefix_roundtrips_without_matching_language_model(monkeypatch):
    from types import SimpleNamespace

    from auto_round.utils.common import get_reverse_checkpoint_conversion_mapping, revert_checkpoint_conversion_mapping

    class NemotronH_Omni_Reasoning_V3(PreTrainedModel):
        def __init__(self):
            config = PreTrainedConfig()
            config.model_type = "nemotron_h_omni"
            super().__init__(config)
            self.vision_projector = torch.nn.Linear(2, 2)
            self.language_model = torch.nn.Linear(2, 2)
            self.language_model.config = SimpleNamespace(model_type="nemotron_h_puzzle")

    with _patched_mapping(monkeypatch):
        rules = conversion_mapping.get_model_conversion_mapping(NemotronH_Omni_Reasoning_V3(), add_legacy=False)
        puzzle_rule = rules[-1]
        assert puzzle_rule.source_patterns == ["language_model.backbone."]
        assert puzzle_rule.target_patterns == ["language_model.model."]
        assert puzzle_rule.scope_prefix is None
        model = SimpleNamespace(_weight_conversions=[puzzle_rule])
        reverse = get_reverse_checkpoint_conversion_mapping(model)
        assert (
            revert_checkpoint_conversion_mapping("language_model.model.layers.0.mixer.up_proj", reverse)
            == "language_model.backbone.layers.0.mixer.up_proj"
        )


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
    monkeypatch.setattr(context_module, "apply_post_load_fixups", lambda _: None)
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
