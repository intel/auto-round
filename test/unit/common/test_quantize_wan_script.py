import importlib.util
from pathlib import Path

import torch
from diffusers import WanTransformer3DModel

from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.export.svdquant_adapters.wan import WAN_SVDQUANT_TARGET_MODULES


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "quantize_wan_svdquant_nunchaku.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("quantize_wan_svdquant_nunchaku", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_child(module, path):
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def test_resolve_devices_accepts_multiple_explicit_devices():
    script = _load_script()

    assert script._resolve_devices("cpu,cpu") == (torch.device("cpu"), torch.device("cpu"))


def test_projection_level_scheduler_decomposes_complete_wan_block_on_cpu():
    script = _load_script()
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=32,
        in_channels=16,
        out_channels=16,
        text_dim=64,
        freq_dim=32,
        ffn_dim=128,
        num_layers=1,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=64,
    )

    script._decompose_blocks(
        model,
        rank=16,
        devices=(torch.device("cpu"), torch.device("cpu")),
        residual_iters=1,
    )

    for path in WAN_SVDQUANT_TARGET_MODULES:
        layer = _get_child(model.blocks[0], path)
        assert isinstance(layer, SVDQuantLinear)
        assert layer.residual_linear.data_type == "mx_fp_rceil"
        assert layer.residual_linear.act_data_type == "mx_fp_rceil"
        assert layer.residual_linear.weight.device.type == "cpu"
        assert layer.lora_down.weight.device.type == "cpu"
        assert layer.lora_up.weight.device.type == "cpu"
