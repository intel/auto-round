"""Minimal runtime smoke for AWQ + AutoRound fusion."""

from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.transforms.awq.config import AWQConfig
from auto_round.compressors.entry import AutoRound


def test_awq_plus_autoround_quantize_smoke(tiny_opt_model_path):
    ar = AutoRound(
        tiny_opt_model_path,
        scheme="W4A16",
        alg_configs=[AWQConfig(n_grid=2), SignRoundConfig(iters=1)],
        nsamples=1,
        seqlen=8,
        low_cpu_mem_usage=False,
    )

    model, layer_config = ar.quantize()

    assert model is not None
    assert layer_config
