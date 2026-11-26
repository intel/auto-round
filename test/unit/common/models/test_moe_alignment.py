import pytest
import torch

from auto_round.data_type.nvfp import calculate_gparam
from auto_round.modeling.fused_moe import apply_replacements
from auto_round.utils.model import set_amax_for_all_moe_layers, set_amax_for_uncalibrated_experts


class TestNVFPMoeScaleRepair:
    @staticmethod
    def _projection(amax, data_type="nv_fp4_with_static_gs"):
        layer = torch.nn.Linear(16, 16, bias=False)
        layer.act_bits = 4
        layer.act_data_type = data_type
        if amax is not None:
            layer.act_max = torch.tensor([amax], dtype=torch.float32)
        return layer

    @pytest.mark.parametrize("name", ["Qwen3_5MoeSparseMoeBlock", "LinearQwen3_5MoeSparseMoeBlock"])
    def test_qwen35_repairs_only_missing_and_degenerate_experts(self, monkeypatch, name):
        monkeypatch.setenv("AR_ENABLE_UNIFY_MOE_INPUT_SCALE", "0")
        model = type(name, (torch.nn.Module,), {})()
        model.experts = torch.nn.ModuleList()
        for index, value in enumerate([2.0, 4.0, 0.0, 1e-40, None]):
            expert = torch.nn.Module()
            expert.gate_proj = self._projection(index + 1.0)
            expert.up_proj = self._projection(index + 1.0)
            expert.down_proj = self._projection(value)
            model.experts.append(expert)
        set_amax_for_all_moe_layers(model)
        assert [e.down_proj.act_max.item() for e in model.experts] == [2.0, 4.0, 4.0, 4.0, 4.0]
        assert [e.gate_proj.act_max.item() for e in model.experts] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert all(torch.isfinite(calculate_gparam(e.down_proj.act_max).reciprocal()) for e in model.experts)

    @pytest.mark.parametrize("amax", [float("nan"), float("inf"), -1.0])
    def test_corrupt_statistics_are_not_silently_replaced(self, amax):
        layers = [self._projection(2.0), self._projection(amax)]
        with pytest.raises(ValueError, match="finite, nonnegative"):
            set_amax_for_uncalibrated_experts(layers)

    def test_missing_valid_peer_requires_recalibration(self):
        with pytest.raises(ValueError, match="No valid NVFP.*recalibrate"):
            set_amax_for_uncalibrated_experts([self._projection(0.0), self._projection(1e-40)])

    def test_non_nvfp_zero_statistics_remain_unchanged(self):
        layers = [self._projection(2.0, "fp8"), self._projection(0.0, "fp8"), self._projection(None, "fp8")]
        missing = set_amax_for_uncalibrated_experts(layers)
        assert missing == [layers[2]]
        assert [layer.act_max.item() for layer in layers] == [2.0, 0.0, 2.0]

    def test_explicit_invalid_replacement_is_rejected(self):
        with pytest.raises(ValueError, match="replacement NVFP"):
            set_amax_for_uncalibrated_experts([self._projection(0.0)], set_amax_value=0.0)


def test_set_amax_for_all_moe_layers_direct(tiny_deepseek_v2_model_path_cpu):
    """Directly test set_amax_for_all_moe_layers unification logic."""
    import os

    from transformers import AutoModelForCausalLM

    os.environ["AR_ENABLE_UNIFY_MOE_INPUT_SCALE"] = "true"

    model = AutoModelForCausalLM.from_pretrained(tiny_deepseek_v2_model_path_cpu, trust_remote_code=False)
    apply_replacements(model)

    # Find the first MoE block and manually set different act_max values
    moe_block = None
    for name, module in model.named_modules():
        if hasattr(module, "experts") and len(list(module.experts)) > 0:
            moe_block = module
            break

    assert moe_block is not None, "Model should have MoE layers"

    # Manually set different act_max values to simulate post-calibration state
    experts = list(moe_block.experts)
    for i, expert in enumerate(experts):
        if hasattr(expert, "gate_proj"):
            expert.gate_proj.act_max = torch.tensor(float(i + 1), dtype=torch.float32)
        if hasattr(expert, "up_proj"):
            expert.up_proj.act_max = torch.tensor(float(i + 1) * 1.5, dtype=torch.float32)
        if hasattr(expert, "down_proj"):
            expert.down_proj.act_max = torch.tensor(float(i + 1) * 2.0, dtype=torch.float32)

    # Verify they are different before alignment
    gate_before = [expert.gate_proj.act_max.item() for expert in experts if hasattr(expert, "gate_proj")]
    up_before = [expert.up_proj.act_max.item() for expert in experts if hasattr(expert, "up_proj")]

    assert len(set(gate_before)) > 1, "gate_proj values should be different before alignment"
    assert len(set(up_before)) > 1, "up_proj values should be different before alignment"

    # Apply scale alignment
    set_amax_for_all_moe_layers(model, attr_name="act_max")

    # Verify they are unified after alignment
    gate_after = [expert.gate_proj.act_max.item() for expert in experts if hasattr(expert, "gate_proj")]
    up_after = [expert.up_proj.act_max.item() for expert in experts if hasattr(expert, "up_proj")]
    down_after = [expert.down_proj.act_max.item() for expert in experts if hasattr(expert, "down_proj")]

    # All gate_proj should have the same value (the maximum)
    assert len(set(gate_after)) == 1, f"gate_proj not unified: {gate_after}"
    assert gate_after[0] == max(gate_before), f"gate_proj should be max of {gate_before}"

    # All up_proj should have the same value (the maximum)
    assert len(set(up_after)) == 1, f"up_proj not unified: {up_after}"
    assert up_after[0] == max(up_before), f"up_proj should be max of {up_before}"

    print(f"✓ Successfully unified {len(gate_after)} experts:")
    print(f"  gate_proj: {gate_before} → {gate_after}")
    print(f"  up_proj: {up_before} → {up_after}")
    print(f"  down_proj: {down_after} (not unified - can differ)")
