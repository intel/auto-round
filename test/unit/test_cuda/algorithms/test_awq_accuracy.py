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

"""Accuracy regression for AWQ on a real (non-tiny) model.

Uses lm_eval on the full OPT-125m model, so it's slow and only runs on one
representative device (cuda) -- functional AWQ coverage across all devices
lives in ``common/algorithms/test_awq.py``.
"""

import shutil
from test.helpers import evaluate_accuracy, opt_name_or_path

import pytest
from transformers import AutoTokenizer

from auto_round import AutoRound, AWQConfig


class TestAWQAccuracy:
    """AWQ accuracy evaluation on OPT-125m using lm_eval (lambada_openai, limit=50)."""

    @classmethod
    def setup_class(cls):
        cls.model_name = opt_name_or_path
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.mark.skip_ci(reason="Accuracy: Time-consuming lm_eval accuracy check; covered by nightly")
    def test_awq_w4a16_lmeval(self):
        """AWQ W4A16 on OPT-125m: lambada_openai accuracy check."""
        ar = AutoRound(
            self.model_name,
            scheme="W4A16",
            alg_configs=AWQConfig(),
            nsamples=32,
            seqlen=32,
            batch_size=8,
        )
        model, _ = ar.quantize()

        evaluate_accuracy(
            model,
            self.tokenizer,
            task="lambada_openai",
            threshold=0.3,
            batch_size="auto:8",
            limit=50,
        )
