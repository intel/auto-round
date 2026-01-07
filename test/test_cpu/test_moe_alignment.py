import os
import shutil

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.utils.model import get_module, set_amax_for_all_moe_layers

from ..helpers import get_model_path

deepseek_v2_lite_path = get_model_path("deepseek-ai/DeepSeek-V2-Lite-Chat")


@pytest.fixture
def setup_deepseek_v2_lite():
    """Fixture to set up the DeepSeek-V2-Lite model for testing."""
    model_name = deepseek_v2_lite_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # Reduce layers for faster testing
    config.num_hidden_layers = 2
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    output_dir = "./tmp/test_moe_alignment_deepseek"
    return model, tokenizer, output_dir, config


def test_moe_scale_alignment_fp8_static(setup_deepseek_v2_lite):
    """Test that FP8_STATIC quantization unifies gate/up input scales across experts."""
    # Enable MoE scale unification explicitly
    os.environ["AR_ENABLE_UNIFY_MOE_INPUT_SCALE"] = "true"

    model, tokenizer, output_dir, config = setup_deepseek_v2_lite

    # Quantize with FP8_STATIC scheme
    autoround = AutoRound(
        model,
        tokenizer,
        scheme="FP8_STATIC",
        nsamples=4,
        iters=0,  # RTN for faster testing
        seqlen=32,
        ignore="self_attn,lm_head",
    )
    quantized_model, save_folder = autoround.quantize_and_save(format="auto_round", output_dir=output_dir)

    # Verify that the model has MoE layers
    has_moe = False
    for name, module in quantized_model.named_modules():
        if "experts" in name:
            has_moe = True
            break
    assert has_moe, "Model should have MoE layers"

    # Check that gate_proj and up_proj have unified act_max across all experts
    # Find the first MoE block
    for name, module in quantized_model.named_modules():
        if hasattr(module, "experts") and len(list(module.experts)) > 0:
            experts = list(module.experts)

            # Collect gate_proj act_max values
            gate_scales = []
            up_scales = []
            down_scales = []

            for expert in experts:
                if hasattr(expert, "gate_proj") and hasattr(expert.gate_proj, "input_scale"):
                    gate_scales.append(expert.gate_proj.input_scale)
                if hasattr(expert, "up_proj") and hasattr(expert.up_proj, "input_scale"):
                    up_scales.append(expert.up_proj.input_scale)
                if hasattr(expert, "down_proj") and hasattr(expert.down_proj, "input_scale"):
                    down_scales.append(expert.down_proj.input_scale)

            # Verify gate_proj scales are unified
            assert len(gate_scales) > 0, "No gate_proj scales found"
            gate_ref = gate_scales[0]
            for i, scale in enumerate(gate_scales):
                assert torch.allclose(
                    scale, gate_ref
                ), f"Expert {i} gate_proj.input_scale ({scale.item()}) != Expert 0 ({gate_ref.item()})"

            # Verify up_proj scales are unified
            assert len(up_scales) > 0, "No up_proj scales found"
            up_ref = up_scales[0]
            for i, scale in enumerate(up_scales):
                assert torch.allclose(
                    scale, up_ref
                ), f"Expert {i} up_proj.input_scale ({scale.item()}) != Expert 0 ({up_ref.item()})"

            print(f"✓ All {len(gate_scales)} experts have unified gate_proj.input_scale = {gate_ref.item()}")
            print(f"✓ All {len(up_scales)} experts have unified up_proj.input_scale = {up_ref.item()}")

            # down_proj scales can differ (not input projections)
            if len(down_scales) > 1:
                down_are_different = not all(torch.allclose(s, down_scales[0]) for s in down_scales)
                if down_are_different:
                    print("✓ down_proj.input_scale values correctly vary across experts (not unified)")

            break  # Only check the first MoE block

    # Clean up
    shutil.rmtree(output_dir, ignore_errors=True)


def test_set_amax_for_all_moe_layers_direct(setup_deepseek_v2_lite):
    """Directly test set_amax_for_all_moe_layers unification logic."""
    # Enable MoE scale unification explicitly
    os.environ["AR_ENABLE_UNIFY_MOE_INPUT_SCALE"] = "true"

    model, tokenizer, output_dir, config = setup_deepseek_v2_lite

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
