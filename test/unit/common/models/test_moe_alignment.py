import torch

from auto_round.modeling.fused_moe import apply_replacements
from auto_round.utils.model import set_amax_for_all_moe_layers


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
