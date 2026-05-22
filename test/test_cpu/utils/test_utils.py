from unittest.mock import patch

import auto_round.utils.device as auto_round_utils
from auto_round.utils.common import (
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


def test_preserve_original_visual_block_name():
    # Single visual block name
    assert preserve_original_visual_block_name("model.visual.blocks", "visual.blocks") == "model.visual.blocks"
    # Comma-separated: both multimodal block prefixes are preserved.
    assert (
        preserve_original_visual_block_name(
            "model.visual.blocks,model.language_model.layers", "visual.blocks,model.layers"
        )
        == "model.visual.blocks,model.language_model.layers"
    )
    # Direct language-model composite paths are also preserved.
    assert (
        preserve_original_visual_block_name("model.language_model.layers", "model.layers")
        == "model.language_model.layers"
    )


def test_preserve_original_mllm_language_block_name():
    assert (
        preserve_original_visual_block_name(
            "model.visual.blocks,model.language_model.layers",
            "model.visual.blocks,model.layers",
        )
        == "model.visual.blocks,model.language_model.layers"
    )
