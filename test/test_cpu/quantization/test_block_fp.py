import shutil
from math import ceil

import pytest
import torch

from auto_round import AutoRound
from auto_round.data_type.fp8 import quant_block_fp_sym
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad

from ...helpers import get_model_path


class TestAutoRoundBlockFP:
    @classmethod
    def setup_class(self):
        self.save_dir = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_invalid_scheme(self, tiny_qwen_model_path):
        model_name = tiny_qwen_model_path

        with pytest.raises(ValueError):
            scheme = {
                "bits": 8,
                "group_size": (128, 128),
                "data_type": "int",
                "act_bits": 16,
            }
            autoround = AutoRound(
                model_name,
                scheme=scheme,
                iters=2,
                seqlen=2,
            )

        with pytest.raises(NotImplementedError):
            scheme = {
                "bits": 8,
                "group_size": (128, 128),
                "data_type": "fp",
                "act_bits": 8,
                "act_data_type": "fp",
                "act_group_size": 128,
                "act_dynamic": False,
            }
            autoround = AutoRound(
                model_name,
                scheme=scheme,
                iters=2,
                seqlen=2,
            )

        with pytest.raises(ValueError):
            scheme = {
                "bits": 8,
                "group_size": (128, 128),
                "data_type": "fp",
                "act_bits": 8,
                "act_data_type": "fp",
                "act_group_size": (128, 128),
                "act_dynamic": True,
            }
            autoround = AutoRound(
                model_name,
                scheme=scheme,
                iters=2,
                seqlen=2,
            )

    def test_block_fp8_quant(self):
        data = torch.randn(256, 240)
        group_size = (128, 128)
        reshaped_data, orig_shape, pad_len = reshape_pad_tensor_by_group_size(data, group_size)
        assert list(reshaped_data.shape) == [2, 2, 128, 128]
        assert list(orig_shape) == [256, 240]
        assert pad_len == (0, 16)

        qdq_data, scale, _ = quant_block_fp_sym(data)
        M = ceil(data.shape[0] / 128)
        N = ceil(data.shape[1] / 128)
        scale_ref = torch.zeros(M, N)

        max_val = torch.finfo(torch.float8_e4m3fn).max
        for i in range(M):
            for j in range(N):
                scale_ref[i, j] = data[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128].abs().max() / max_val
        assert (scale == scale_ref).all()
