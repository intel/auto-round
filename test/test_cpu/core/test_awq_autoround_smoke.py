"""Minimal runtime smoke for AWQ + AutoRound fusion."""

import torch

from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.transforms.awq.config import AWQConfig
from auto_round.algorithms.transforms.awq.quantizer import AWQQuantizer
from auto_round.compressors.entry import AutoRound


def test_awq_plus_autoround_quantize_smoke(tiny_opt_model_path):
    ar = AutoRound(
        [AWQConfig(n_grid=2), SignRoundConfig(iters=1)],
        tiny_opt_model_path,
        scheme="W4A16",
        nsamples=1,
        seqlen=8,
        low_cpu_mem_usage=False,
    )

    model, layer_config = ar.quantize()

    assert model is not None
    assert layer_config


def test_awq_mxfp_qdq_uses_v2_scale_search(monkeypatch):
    awq = AWQQuantizer(AWQConfig(bits=4, group_size=32, sym=True, data_type="mx_fp"))
    awq._use_v2_mx_scale_search = True

    layer = torch.nn.Linear(64, 8, bias=False)
    layer.global_name = "model.layers.0.mlp.down_proj"
    layer.imatrix = torch.ones(layer.in_features)
    awq.layer_config = {
        layer.global_name: {
            "bits": 4,
            "group_size": 32,
            "sym": True,
            "data_type": "mx_fp",
        }
    }

    calls = []

    def fake_search_mx_scale(weight, bits, qw=None):
        calls.append((weight.shape, bits, qw))
        return torch.ones(weight.shape[0], 1, device=weight.device, dtype=weight.dtype)

    monkeypatch.setattr("auto_round.algorithms.transforms.awq.quantizer.search_mx_scale", fake_search_mx_scale)

    qdq_weight = awq._quantize_dequantize_weight(layer, layer.weight.detach().float())

    assert qdq_weight is not None
    assert qdq_weight.shape == layer.weight.shape
    assert calls
    assert calls[0][1] == 4
    assert isinstance(calls[0][2], torch.Tensor)
