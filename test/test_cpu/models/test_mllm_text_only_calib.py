# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Regression test for calibrating a multimodal (VLM) checkpoint with a
non-MLLM (plain text) local calibration dataset when quant_nontext_module
is False.

Before this fix, any string dataset unconditionally fell through to
``get_mllm_dataloader``, which indexes ``MLLM_DATASET`` with the dataset's
value even when ``os.path.isfile(dataset)`` is what made it enter that
branch (a local file path is never actually a key in that registry) --
``KeyError: '<the local file path>'``. Since the vision/audio towers aren't
being quantized in this scenario, plain text-only calibration through the
standard (non-MLLM) dataloader is sufficient and now used instead.
"""

import json

import pytest

from auto_round import AutoRound


class TestMllmTextOnlyCalibration:
    @pytest.fixture(autouse=True)
    def setup_save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        import shutil

        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.fixture
    def local_text_calib_file(self, tmp_path):
        data = [{"text": "The quick brown fox jumps over the lazy dog. " * 40}] * 20
        path = tmp_path / "calib.json"
        with open(path, "w") as f:
            json.dump(data, f)
        return str(path)

    def test_local_text_dataset_with_quant_nontext_module_false(self, tiny_qwen_vl_model_path, local_text_calib_file):
        """A local (non-MLLM-registered) text file used to KeyError inside
        get_mllm_dataloader; it must now route through the standard text
        dataloader instead, since the vision tower isn't being quantized."""
        ar = AutoRound(
            model=tiny_qwen_vl_model_path,
            scheme="W4A16",
            iters=1,
            nsamples=1,
            seqlen=32,
            dataset=local_text_calib_file,
            quant_nontext_module=False,
        )
        _, layer_config = ar.quantize()

        quantized_language_layers = [
            name for name, cfg in layer_config.items() if "bits" in cfg and "language_model" in name
        ]
        assert quantized_language_layers, "expected language-model layers to be quantized"
