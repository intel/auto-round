from unittest.mock import MagicMock, patch

import pytest

import auto_round.utils.device as auto_round_utils
from auto_round.utils.common import (
    compress_layer_names,
    preserve_original_visual_block_name,
    revert_checkpoint_conversion_mapping,
)


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
        from auto_round.compressors.base import BaseCompressor

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
    @patch("auto_round.compressors.base.set_layer_config")
    @patch("auto_round.compressors.base._handle_special_schemes", return_value=None)
    def test_kimi_k25_ignore_layers_preserved(self, mock_handle, mock_set_lc, mock_get_predefined):
        """KIMI K2.5: vision_tower and mm_projector must end up in ignore_layers."""
        mock_get_predefined.return_value = ["vision_tower", "mm_projector"]
        mock_set_lc.return_value = ({}, False, None)

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
    @patch("auto_round.compressors.base.set_layer_config")
    @patch("auto_round.compressors.base._handle_special_schemes", return_value=None)
    def test_step3p5_ignore_layers_preserved(self, mock_handle, mock_set_lc, mock_get_predefined):
        """step3p5: short pattern names must not be dropped by block filter."""
        predefined = ["g_proj", "moe.gate", "eh_proj", "shared_head", "layers.45"]
        mock_get_predefined.return_value = predefined
        mock_set_lc.return_value = ({}, False, None)

        stub = self._make_compressor_stub(
            predefined_ignore_layers=predefined,
            quant_block_list=[["model.layers"]],
        )
        stub.configure_layer_config()

        for name in predefined:
            assert name in stub.ignore_layers, f"'{name}' should be in ignore_layers, got: '{stub.ignore_layers}'"

    @patch("auto_round.compressors.base.get_predefined_ignore_layers")
    @patch("auto_round.compressors.base.set_layer_config")
    @patch("auto_round.compressors.base._handle_special_schemes", return_value=None)
    def test_longcat_classifier_preserved(self, mock_handle, mock_set_lc, mock_get_predefined):
        """Longcat: 'classifier' must not be dropped."""
        mock_get_predefined.return_value = ["classifier"]
        mock_set_lc.return_value = ({}, False, None)

        stub = self._make_compressor_stub(
            predefined_ignore_layers=["classifier"],
            quant_block_list=[["model.layers"]],
        )
        stub.configure_layer_config()

        assert "classifier" in stub.ignore_layers, f"classifier should be in ignore_layers, got: '{stub.ignore_layers}'"

    @patch("auto_round.compressors.base.get_predefined_ignore_layers")
    @patch("auto_round.compressors.base.set_layer_config")
    @patch("auto_round.compressors.base._handle_special_schemes", return_value=None)
    def test_glm_flash_full_path_inside_block_preserved(self, mock_handle, mock_set_lc, mock_get_predefined):
        """GLM Flash: model.layers.0.mlp (full path inside block) must be kept."""
        mock_get_predefined.return_value = ["model.layers.0.mlp"]
        mock_set_lc.return_value = ({}, False, None)

        stub = self._make_compressor_stub(
            predefined_ignore_layers=["model.layers.0.mlp"],
            quant_block_list=[["model.layers"]],
        )
        stub.configure_layer_config()

        assert (
            "model.layers.0.mlp" in stub.ignore_layers
        ), f"model.layers.0.mlp should be in ignore_layers, got: '{stub.ignore_layers}'"

    @patch("auto_round.compressors.base.get_predefined_ignore_layers")
    @patch("auto_round.compressors.base.set_layer_config")
    @patch("auto_round.compressors.base._handle_special_schemes", return_value=None)
    def test_mllm_with_multiple_block_groups(self, mock_handle, mock_set_lc, mock_get_predefined):
        """MLLM with vision + language blocks: ignore layers outside both are preserved."""
        mock_get_predefined.return_value = ["vision_tower", "mm_projector"]
        mock_set_lc.return_value = ({}, False, None)

        stub = self._make_compressor_stub(
            predefined_ignore_layers=["vision_tower", "mm_projector"],
            quant_block_list=[["model.visual.blocks"], ["model.layers"]],
        )
        stub.configure_layer_config()

        assert "vision_tower" in stub.ignore_layers
        assert "mm_projector" in stub.ignore_layers
