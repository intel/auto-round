import pytest

from auto_round.utils.common import compress_layer_names


class TestCompressLayerNames:

    def test_simple_consecutive_layers(self):
        names = [f"model.layers.{i}.mlp.gate_proj" for i in range(4)]
        result = compress_layer_names(names)
        assert "model.layers.[0-3].mlp.gate_proj" in result

    def test_experts_same_suffix(self):
        names = [
            "mtp.layers.0.mlp.experts.248.up_proj",
            "mtp.layers.0.mlp.experts.87.up_proj",
            "mtp.layers.0.mlp.experts.23.up_proj",
        ]
        result = compress_layer_names(names)
        # Should group by expert number, not by layers index
        assert "mtp.layers.0.mlp.experts." in result
        assert ".up_proj" in result
        # Expert numbers should appear, not "layers.0"
        assert "23" in result and "87" in result and "248" in result

    def test_experts_multiple_suffixes(self):
        names = [
            "mtp.layers.0.mlp.experts.58.down_proj",
            "mtp.layers.0.mlp.experts.58.gate_proj",
            "mtp.layers.0.mlp.experts.37.up_proj",
            "mtp.layers.0.mlp.experts.37.down_proj",
        ]
        result = compress_layer_names(names)
        # down_proj and gate_proj should be separate groups
        assert "down_proj" in result
        assert "gate_proj" in result
        assert "up_proj" in result

    def test_experts_not_merged_across_layers(self):
        names = [
            "mtp.layers.0.mlp.experts.5.up_proj",
            "mtp.layers.1.mlp.experts.5.up_proj",
        ]
        result = compress_layer_names(names)
        # Different layer indices → different prefix → not merged into one group
        parts = [p.strip() for p in result.split(",")]
        assert len(parts) == 2

    def test_single_name(self):
        names = ["model.layers.3.mlp.gate_proj"]
        result = compress_layer_names(names)
        assert result == "model.layers.3.mlp.gate_proj"

    def test_non_consecutive_ranges(self):
        names = [f"model.layers.{i}.self_attn.q_proj" for i in [0, 1, 3, 4]]
        result = compress_layer_names(names)
        assert "[0-1,3-4]" in result

    def test_no_number_in_name(self):
        names = ["embed_tokens", "lm_head"]
        result = compress_layer_names(names)
        for name in names:
            assert name in result

    def test_empty_input(self):
        assert compress_layer_names([]) == ""

    def test_full_experts_example(self):
        """Reproduces the original bug report: expert index must be the grouped number."""
        names = [
            "mtp.layers.0.mlp.experts.248.up_proj",
            "mtp.layers.0.mlp.experts.87.up_proj",
            "mtp.layers.0.mlp.experts.58.down_proj",
            "mtp.layers.0.mlp.experts.37.up_proj",
            "mtp.layers.0.mlp.experts.37.down_proj",
        ]
        result = compress_layer_names(names)
        # All up_proj experts should be in one group
        assert "mtp.layers.0.mlp.experts.[" in result
        # The layer 0 index must NOT be the grouped number (i.e. no "[0]" grouping)
        assert "layers.[0]" not in result
