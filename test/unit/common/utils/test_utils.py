from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import auto_round.utils.device as auto_round_utils
from auto_round.utils.common import (
    compress_layer_names,
    expand_layer_config_for_weight_renames,
    get_checkpoint_conversion_mapping,
    get_reverse_checkpoint_conversion_mapping,
    preserve_original_visual_block_name,
    revert_checkpoint_conversion_mapping,
)
from auto_round.utils.model import is_code_model, map_nested_tensors


def test_map_nested_tensors_preserves_structure_without_mutating_input():
    original = {"args": (torch.tensor([1.0]),), "metadata": ["unchanged"]}

    mapped = map_nested_tensors(original, lambda tensor: tensor + 1)

    assert mapped is not original
    assert mapped["args"] is not original["args"]
    torch.testing.assert_close(mapped["args"][0], torch.tensor([2.0]))
    torch.testing.assert_close(original["args"][0], torch.tensor([1.0]))
    assert mapped["metadata"] == ["unchanged"]


@pytest.mark.parametrize(
    "model,config",
    [
        ("Qwen/Qwen3-Coder-30B", None),
        ("/models/CodeLlama-7b", None),
        ("bigcode/starcoder2-15b", None),
        ("generic/model", SimpleNamespace(architectures=["DeepseekCoderForCausalLM"])),
        ("generic/model", SimpleNamespace(finetuning_task="code-generation")),
        ("generic/model", SimpleNamespace(task_specific_params={"text-to-code": {}})),
    ],
)
def test_is_code_model(model, config):
    assert is_code_model(model, config)


@pytest.mark.parametrize(
    "model,config",
    [
        ("Qwen/Qwen3-4B", None),
        ("org/encoder-decoder", None),
        ("org/audio-codec", None),
        ("/srv/code/checkpoints/Llama-3", None),
        ("org/notstarcoder-model", None),
        ("generic/model", SimpleNamespace(_name_or_path="/srv/code/checkpoints/Llama-3")),
        ("generic/model", SimpleNamespace(architectures=["SomeEncoderDecoderModel"])),
    ],
)
def test_is_not_code_model(model, config):
    assert not is_code_model(model, config)


class TestPackingWithNumba:

    @patch.object(auto_round_utils, "_is_tbb_installed", lambda: False)
    def test_tbb_not_installed(self):
        assert auto_round_utils.is_tbb_available() is False, "`is_tbb_available` should return False."
        assert auto_round_utils.can_pack_with_numba() is False, "`can_pack_with_numba` should return False."

    @patch.object(auto_round_utils, "_is_tbb_installed", lambda: True)
    @patch.object(auto_round_utils, "_is_tbb_configured", lambda: False)
    def test_tbb_installed_but_not_configured_right(self):
        assert auto_round_utils.is_tbb_available() is False, "`is_tbb_available` should return False."
        assert auto_round_utils.can_pack_with_numba() is False, "`can_pack_with_numba` should return False."

    @patch.object(auto_round_utils, "is_numba_available", lambda: False)
    def test_numba_not_installed(self):
        assert auto_round_utils.can_pack_with_numba() is False, "`can_pack_with_numba` should return False."


def test_revert_checkpoint_conversion_mapping_handles_comma_separated_block_names():
    mapping = {
        r"^visual\.": "model.visual.",
        r"^model\.language_model\.layers": "model.layers",
    }

    converted = revert_checkpoint_conversion_mapping("visual.blocks,model.language_model.layers", mapping)

    assert converted == "model.visual.blocks,model.layers"


def test_revert_checkpoint_conversion_mapping_does_not_rewrite_quantized_tensor_suffixes():
    mapping = {"weight": [".weight_packed", ".weight_scale", ".weight_shape"]}

    assert (
        revert_checkpoint_conversion_mapping("model.layers.0.mlp.down_proj.weight", mapping)
        == "model.layers.0.mlp.down_proj.weight_packed"
    )
    assert (
        revert_checkpoint_conversion_mapping("model.layers.0.mlp.down_proj.weight_packed", mapping)
        == "model.layers.0.mlp.down_proj.weight_packed"
    )
    assert (
        revert_checkpoint_conversion_mapping("model.layers.0.mlp.down_proj.weight_scale", mapping)
        == "model.layers.0.mlp.down_proj.weight_scale"
    )


def test_revert_checkpoint_conversion_mapping_handles_backreference_patterns():
    mapping = {r"^model\.language_model\.(.+)$": r"model.\1"}

    converted = revert_checkpoint_conversion_mapping("model.language_model.layers.0.self_attn.q_proj.weight", mapping)

    assert converted == "model.layers.0.self_attn.q_proj.weight"


def test_get_reverse_checkpoint_conversion_mapping_falls_back_to_central_registry():
    # Models built from config (AutoRound's meta / disk-stream skeleton) are not
    # created through ``from_pretrained`` and therefore carry no
    # ``_weight_conversions``. The reverse map must still be recovered from
    # transformers' central per-family registry (transformers >= 5.x), otherwise
    # module-side names such as ``attn_hc.base`` are written to the checkpoint
    # verbatim and break reload (e.g. vLLM ``KeyError: 'layers.0.attn_hc.base'``).
    pytest.importorskip("transformers.conversion_mapping")
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping as _tf_mapping

    if not _tf_mapping("glm5_next"):
        pytest.skip("installed transformers has no glm5_next conversion mapping")

    model = SimpleNamespace(config=SimpleNamespace(model_type="glm5_next"))
    reverse_mapping = get_reverse_checkpoint_conversion_mapping(model)

    # attn_hc.base -> hc_attn_base is the exact rename that broke vLLM deployment.
    assert (
        revert_checkpoint_conversion_mapping("model.language_model.layers.0.attn_hc.base", reverse_mapping)
        == "model.language_model.layers.0.hc_attn_base"
    )
    assert (
        revert_checkpoint_conversion_mapping("model.language_model.layers.3.ffn_hc.scale", reverse_mapping)
        == "model.language_model.layers.3.hc_ffn_scale"
    )
    # The forget-gate submodule renames must reverse too, to avoid the next KeyError.
    assert (
        revert_checkpoint_conversion_mapping(
            "model.language_model.layers.0.self_attn.forget_gate.A_log", reverse_mapping
        )
        == "model.language_model.layers.0.self_attn.A_log"
    )


def test_expand_layer_config_for_weight_renames_to_model_names(monkeypatch):
    checkpoint_name = "language_model.model.layers.0.self_attn.q_proj"
    model_name = "model.language_model.layers.0.self_attn.q_proj"
    monkeypatch.setattr(
        "auto_round.utils.common.get_reverse_checkpoint_conversion_mapping",
        lambda model: {
            r"model.language_model": r"language_model",
            r"language_model": r"language_model.model",
        },
    )

    expanded = expand_layer_config_for_weight_renames(
        {checkpoint_name: {"bits": 4}},
        model=SimpleNamespace(),
        to_model_names=True,
    )

    assert expanded[checkpoint_name]["bits"] == 4
    assert expanded[model_name]["bits"] == 4


def test_expand_layer_config_for_deepseek_v4_actual_forward_mapping():
    pytest.importorskip("transformers.conversion_mapping")
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping as tf_mapping

    if not tf_mapping("deepseek_v4"):
        pytest.skip("installed transformers has no DeepSeek V4 mapping")
    model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4"))
    config = {
        "mlp.experts": {"scheme": "NVFP4"},
        "layers.3.attn.wq_a.weight": {"bits": 8},
        "layers.3.attn.compressor.wkv.weight": {"bits": 16},
        "layers.3.ffn.experts.0.w1.weight": {"bits": 4},
        "layers.3.ffn.experts.255.w2": {"bits": 8},
        "layers.3.ffn.experts.4.w3.weight": {"bits": 16},
    }
    expanded = expand_layer_config_for_weight_renames(config, model=model, to_model_names=True)
    assert expanded["mlp.experts"] == {"scheme": "NVFP4"}
    assert expanded["layers.3.self_attn.q_a_proj.weight"] == {"bits": 8}
    assert expanded["layers.3.self_attn.compressor.kv_proj.weight"] == {"bits": 16}
    assert expanded["layers.3.mlp.experts.0.gate_proj.weight"] == {"bits": 4}
    assert expanded["layers.3.mlp.experts.255.down_proj"] == {"bits": 8}
    assert expanded["layers.3.mlp.experts.4.up_proj.weight"] == {"bits": 16}
    assert "layers.3.mlp.experts.gate_up_proj" not in expanded
    assert all(expanded[k] == v for k, v in config.items())


def test_rename_only_mapping_preserves_default_tensor_conversions():
    pytest.importorskip("transformers.conversion_mapping")
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping as tf_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming

    rules = tf_mapping("deepseek_v4")
    if not rules:
        pytest.skip("installed transformers has no DeepSeek V4 mapping")
    model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4"))
    full = get_checkpoint_conversion_mapping(model)
    expected = {
        source: rule.target_patterns
        for rule in rules
        if isinstance(rule, WeightRenaming)
        for source in rule.source_patterns
    }
    converter_sources = {
        source for rule in rules if isinstance(rule, WeightConverter) for source in rule.source_patterns
    }
    assert converter_sources
    assert converter_sources <= full.keys()
    assert not (converter_sources & expected.keys())


@pytest.mark.parametrize("model_type", ["gemma4_unified", "qwen3_5_text", "qwen3_omni_moe", "glm5_next"])
def test_non_deepseek_aliases_keep_reverse_mapping_path(monkeypatch, model_type):
    # Real loaded models may carry scoped/runtime conversion rules that are
    # absent from the model-type registry. Do not replace this source of truth.
    reverse = {
        r"model.embed_vision.multimodal_embedder.embedding_projection": [r"model.embed_vision.embedding_projection"],
    }
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    monkeypatch.setattr("auto_round.utils.common.get_reverse_checkpoint_conversion_mapping", lambda model: reverse)

    def reject_forward(*args, **kwargs):
        raise AssertionError("Non-DeepSeek models must retain their existing reverse-map path")

    monkeypatch.setattr("auto_round.utils.common.get_checkpoint_conversion_mapping", reject_forward)
    source = "model.embed_vision.embedding_projection"
    target = "model.embed_vision.multimodal_embedder.embedding_projection"
    expanded = expand_layer_config_for_weight_renames({source: {"bits": 4}}, model=model, to_model_names=True)
    assert expanded[target] == {"bits": 4}


@pytest.mark.parametrize("family", ["gemma4_unified", "qwen3_5_text", "qwen3_omni_moe"])
def test_non_deepseek_actual_registry_aliases_match_legacy(family):
    pytest.importorskip("transformers.conversion_mapping")
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping as tf_mapping

    from auto_round.utils.common import apply_checkpoint_conversion_mapping

    if not tf_mapping(family):
        pytest.skip(f"installed transformers has no {family} conversion mapping")
    model = SimpleNamespace(config=SimpleNamespace(model_type=family))
    legacy = {}
    for source, targets in reversed(list(get_reverse_checkpoint_conversion_mapping(model).items())):
        for target in [targets] if isinstance(targets, str) else targets:
            legacy.setdefault(target, []).append(source)
    names = [
        "model.embed_vision.embedding_projection",
        "model.vision_embedder.patch_dense",
        "model.language_model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.experts.0.gate_proj",
    ]
    for name in names:
        expected = apply_checkpoint_conversion_mapping(name, legacy)
        expanded = expand_layer_config_for_weight_renames({name: {"bits": 4}}, model=model, to_model_names=True)
        assert expanded == {key: {"bits": 4} for key in (name, expected)}


def test_gemma4_visual_aliases_match_real_meta_model_modules():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4UnifiedForConditionalGeneration"):
        pytest.skip("installed transformers has no Gemma4 Unified model")
    from transformers.conversion_mapping import get_model_conversion_mapping

    # vision_config=None builds a text-only model; explicitly enable vision.
    config = transformers.Gemma4UnifiedConfig(vision_config={})
    with torch.device("meta"):
        model = transformers.Gemma4UnifiedForConditionalGeneration(config)
    model._weight_conversions = get_model_conversion_mapping(model, add_legacy=False)
    modules = dict(model.named_modules())
    pairs = {
        "model.embed_vision.embedding_projection": "model.embed_vision.multimodal_embedder.embedding_projection",
        "model.vision_embedder.patch_dense": "model.embed_vision.patch_dense",
        "model.vision_embedder.patch_ln1": "model.embed_vision.patch_ln1",
    }
    for source, target in pairs.items():
        expanded = expand_layer_config_for_weight_renames({source: {"bits": 4}}, model=model, to_model_names=True)
        assert target in modules
        assert expanded[target] == {"bits": 4}


def test_expand_layer_config_for_weight_renames_to_checkpoint_names(monkeypatch):
    model_name = "model.layers.0.attn_hc.base"
    checkpoint_name = "model.layers.0.hc_attn_base"
    monkeypatch.setattr(
        "auto_round.utils.common.get_reverse_checkpoint_conversion_mapping",
        lambda model: {r"attn_hc\.base$": "hc_attn_base"},
    )

    expanded = expand_layer_config_for_weight_renames(
        {model_name: {"bits": 4}},
        model_type="glm5_next",
        to_model_names=False,
    )

    assert expanded[model_name]["bits"] == 4
    assert expanded[checkpoint_name]["bits"] == 4


def test_preserve_original_visual_block_name():
    # Single visual block name
    assert preserve_original_visual_block_name("model.visual.blocks", "visual.blocks") == "model.visual.blocks"
    # Comma-separated: visual prefix is preserved and language-model keeps an
    # extra model.layers alias for runtimes that do not use the composite path.
    assert (
        preserve_original_visual_block_name(
            "model.visual.blocks,model.language_model.layers", "visual.blocks,model.layers"
        )
        == "model.visual.blocks,model.language_model.layers,model.layers"
    )
    # Direct language-model composite paths are preserved and also expose the
    # runtime alias used by sglang's text submodel.
    assert (
        preserve_original_visual_block_name("model.language_model.layers", "model.layers")
        == "model.language_model.layers,model.layers"
    )
    # Qwen-Omni renames ``thinker.model.layers`` to
    # ``thinker.model.language_model.layers`` in newer Transformers releases;
    # keep both forms for the runtime and checkpoint namespaces.
    assert (
        preserve_original_visual_block_name("thinker.model.language_model.layers", "thinker.model.layers")
        == "thinker.model.language_model.layers,thinker.model.layers"
    )
    assert (
        preserve_original_visual_block_name(
            "thinker.model.visual.blocks,thinker.model.audio_tower.layers",
            "thinker.visual.blocks,thinker.audio_tower.layers",
        )
        == "thinker.model.visual.blocks,thinker.visual.blocks,"
        "thinker.model.audio_tower.layers,thinker.audio_tower.layers"
    )


def test_preserve_original_mllm_language_block_name():
    assert (
        preserve_original_visual_block_name(
            "model.visual.blocks,model.language_model.layers",
            "model.visual.blocks,model.layers",
        )
        == "model.visual.blocks,model.language_model.layers,model.layers"
    )


class TestPredefinedIgnoreLayersBlockFilter:
    """Test the block-prefix filter in configure_layer_config."""

    @staticmethod
    def _make_compressor_stub(predefined_ignore_layers, quant_block_list):
        """Create a minimal stub that allows calling configure_layer_config."""
        from auto_round.compressors.base import BaseOrchestrator as BaseCompressor

        stub = object.__new__(BaseCompressor)
        # Minimal attributes required by configure_layer_config
        stub.ignore_layers = ""
        stub.quant_block_list = quant_block_list
        stub.is_auto_scheme = False
        stub.orig_scheme = "W4A16"
        stub.layer_config = None
        stub.quant_lm_head = False
        stub.scale_dtype = "fp16"

        # Mock compress_context (no gguf format)
        stub.compress_context = MagicMock()
        stub.compress_context.formats = None

        # Mock model_context with a fake model
        stub.model_context = MagicMock()
        stub.model_context.model = MagicMock()
        stub.model_context.is_mllm = False

        return stub

    @patch("auto_round.compressors.base.get_predefined_ignore_layers")
    @patch("auto_round.compressors.base.resolve_layer_config")
    @patch("auto_round.compressors.base._handle_special_schemes", return_value=None)
    def test_kimi_k25_ignore_layers_preserved(self, mock_handle, mock_set_lc, mock_get_predefined):
        """KIMI K2.5: vision_tower and mm_projector must end up in ignore_layers."""
        mock_get_predefined.return_value = ["vision_tower", "mm_projector"]
        mock_set_lc.return_value = {}

        stub = self._make_compressor_stub(
            predefined_ignore_layers=["vision_tower", "mm_projector"],
            quant_block_list=[["model.layers"]],
        )
        stub.configure_layer_config()

        assert (
            "vision_tower" in stub.ignore_layers
        ), f"vision_tower should be in ignore_layers, got: '{stub.ignore_layers}'"
        assert (
            "mm_projector" in stub.ignore_layers
        ), f"mm_projector should be in ignore_layers, got: '{stub.ignore_layers}'"

    @patch("auto_round.compressors.base.get_predefined_ignore_layers")
    @patch("auto_round.compressors.base.resolve_layer_config")
    @patch("auto_round.compressors.base._handle_special_schemes", return_value=None)
    def test_none_ignore_layers_accepts_predefined_layers(self, mock_handle, mock_set_lc, mock_get_predefined):
        mock_get_predefined.return_value = ["vision_tower", "mm_projector"]
        mock_set_lc.return_value = {}
        stub = self._make_compressor_stub(
            predefined_ignore_layers=["vision_tower", "mm_projector"],
            quant_block_list=[["model.layers"]],
        )
        stub.ignore_layers = None

        stub.configure_layer_config()

        assert stub.ignore_layers == "vision_tower,mm_projector"

    @patch("auto_round.compressors.base.get_predefined_ignore_layers")
    @patch("auto_round.compressors.base.resolve_layer_config")
    @patch("auto_round.compressors.base._handle_special_schemes", return_value=None)
    def test_step3p5_ignore_layers_preserved(self, mock_handle, mock_set_lc, mock_get_predefined):
        """step3p5: short pattern names must not be dropped by block filter."""
        predefined = ["g_proj", "moe.gate", "eh_proj", "shared_head", "layers.45"]
        mock_get_predefined.return_value = predefined
        mock_set_lc.return_value = {}

        stub = self._make_compressor_stub(
            predefined_ignore_layers=predefined,
            quant_block_list=[["model.layers"]],
        )
        stub.configure_layer_config()

        for name in predefined:
            assert name in stub.ignore_layers, f"'{name}' should be in ignore_layers, got: '{stub.ignore_layers}'"

    @patch("auto_round.compressors.base.get_predefined_ignore_layers")
    @patch("auto_round.compressors.base.resolve_layer_config")
    @patch("auto_round.compressors.base._handle_special_schemes", return_value=None)
    def test_longcat_classifier_preserved(self, mock_handle, mock_set_lc, mock_get_predefined):
        """Longcat: 'classifier' must not be dropped."""
        mock_get_predefined.return_value = ["classifier"]
        mock_set_lc.return_value = {}

        stub = self._make_compressor_stub(
            predefined_ignore_layers=["classifier"],
            quant_block_list=[["model.layers"]],
        )
        stub.configure_layer_config()

        assert "classifier" in stub.ignore_layers, f"classifier should be in ignore_layers, got: '{stub.ignore_layers}'"

    @patch("auto_round.compressors.base.get_predefined_ignore_layers")
    @patch("auto_round.compressors.base.resolve_layer_config")
    @patch("auto_round.compressors.base._handle_special_schemes", return_value=None)
    def test_glm_flash_full_path_inside_block_preserved(self, mock_handle, mock_set_lc, mock_get_predefined):
        """GLM Flash: model.layers.0.mlp (full path inside block) must be kept."""
        mock_get_predefined.return_value = ["model.layers.0.mlp"]
        mock_set_lc.return_value = {}

        stub = self._make_compressor_stub(
            predefined_ignore_layers=["model.layers.0.mlp"],
            quant_block_list=[["model.layers"]],
        )
        stub.configure_layer_config()

        assert (
            "model.layers.0.mlp" in stub.ignore_layers
        ), f"model.layers.0.mlp should be in ignore_layers, got: '{stub.ignore_layers}'"

    @patch("auto_round.compressors.base.get_predefined_ignore_layers")
    @patch("auto_round.compressors.base.resolve_layer_config")
    @patch("auto_round.compressors.base._handle_special_schemes", return_value=None)
    def test_mllm_with_multiple_block_groups(self, mock_handle, mock_set_lc, mock_get_predefined):
        """MLLM with vision + language blocks: ignore layers outside both are preserved."""
        mock_get_predefined.return_value = ["vision_tower", "mm_projector"]
        mock_set_lc.return_value = {}

        stub = self._make_compressor_stub(
            predefined_ignore_layers=["vision_tower", "mm_projector"],
            quant_block_list=[["model.visual.blocks"], ["model.layers"]],
        )
        stub.configure_layer_config()

        assert "vision_tower" in stub.ignore_layers
        assert "mm_projector" in stub.ignore_layers
