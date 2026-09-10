import json

import pytest

from auto_round import AutoRound


def _local_calibration_dataset(tmp_path):
    dataset_path = tmp_path / "alg_ext_calibration.json"
    dataset_path.write_text(json.dumps(["Algorithm extension local calibration sample. " * 32]))
    return str(dataset_path)


class TestAlgExt:
    @pytest.mark.timeout(60)
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

    @pytest.mark.timeout(60)
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
