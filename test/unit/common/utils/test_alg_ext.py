import json

import pytest
import torch

from auto_round import AutoRound


def _local_calibration_dataset(tmp_path):
    dataset_path = tmp_path / "alg_ext_calibration.json"
    dataset_path.write_text(json.dumps(["Algorithm extension local calibration sample. " * 32]))
    return str(dataset_path)


def _asym_linear():
    layer = torch.nn.Linear(8, 4, bias=False)
    layer.bits = 2
    layer.group_size = 4
    layer.sym = False
    layer.data_type = "int"
    layer.act_bits = 16
    layer.scale_dtype = torch.float32
    return layer


@pytest.mark.parametrize("defer_search", [False, True])
def test_asym_alg_ext_wrapper_accepts_defer_search(defer_search):
    from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedAsymWrapperLinear

    wrapper = SignRoundOptimizedAsymWrapperLinear(
        _asym_linear(),
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        enable_torch_compile=False,
        device="cpu",
        defer_search=defer_search,
    )

    assert wrapper._asym_anchor
    assert isinstance(wrapper.min_scale, torch.nn.Parameter)
    assert isinstance(wrapper.max_scale, torch.nn.Parameter)
    assert torch.isfinite(wrapper.weight_min).all()
    assert torch.isfinite(wrapper.weight_max).all()


@pytest.mark.parametrize("batching_disabled", [False, True])
def test_asym_alg_ext_wrapper_keeps_search_inline(monkeypatch, batching_disabled):
    from auto_round.algorithms.quantization import search_dispatch
    from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedAsymWrapperLinear
    from auto_round.wrapper import wrapper_block

    monkeypatch.setattr(search_dispatch, "batched_search_disabled", lambda: batching_disabled)
    block = torch.nn.Sequential(_asym_linear(), _asym_linear())
    quantized, unquantized = wrapper_block(
        block,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        enable_torch_compile=False,
        device="cpu",
        wrapper_cls=SignRoundOptimizedAsymWrapperLinear,
    )

    assert quantized == ["0", "1"]
    assert unquantized == []
    assert not SignRoundOptimizedAsymWrapperLinear.supports_batched_search
    assert all(layer._asym_anchor for layer in block)


class TestAlgExt:
    def test_gguf_q4_alg_ext(self, tiny_qwen_model_path, tmp_path):
        """Exercise the GGUF algorithm-extension path not covered by CUDA matrix tests."""
        AutoRound(
            tiny_qwen_model_path,
            scheme="gguf:q4_k_s",
            iters=1,
            nsamples=1,
            seqlen=32,
            dataset=_local_calibration_dataset(tmp_path),
            enable_alg_ext=True,
        ).quantize()

    def test_alg_ext_import(self):
        from auto_round.algorithms.quantization.sign_roundv2 import SignRoundV2Quantizer

    def test_w2_asym_alg_ext(self, tiny_opt_model_path, tmp_path):
        """Asyme INT now uses the searched (scale, zp) anchor wrapper on the alg-ext path."""
        ar = AutoRound(
            tiny_opt_model_path,
            scheme="W2A16G64",
            sym=False,
            iters=2,
            nsamples=1,
            seqlen=32,
            dataset=_local_calibration_dataset(tmp_path),
            enable_alg_ext=True,
        )
        model, _ = ar.quantize()
        zps = [m.zp for m in model.modules() if hasattr(m, "zp") and isinstance(m.zp, __import__("torch").Tensor)]
        assert zps, "expected asym zero points on quantized layers"
        for zp in zps:
            assert zp.min() >= 0 and zp.max() <= 3

    def test_nvfp4_alg_ext(self, tiny_opt_model_path, tmp_path):
        """Keep NVFP4 as the representative format not exercised by the CUDA matrix."""
        AutoRound(
            tiny_opt_model_path,
            scheme="NVFP4",
            iters=1,
            nsamples=1,
            seqlen=32,
            dataset=_local_calibration_dataset(tmp_path),
            enable_alg_ext=True,
            enable_torch_compile=True,
        ).quantize()
