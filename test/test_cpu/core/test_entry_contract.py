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

from auto_round.compressors import entry_contract
from auto_round.compressors.entry_contract import split_entry_kwargs


def test_split_entry_kwargs_partitions_owned_fields():
    processor = object()

    grouped = split_entry_kwargs(
        {"dataset": ["sample"], "scale_dtype": "fp32", "processor": processor, "model_free": False}
    )

    assert grouped["base"]["dataset"] == ["sample"]
    assert grouped["compressor"]["scale_dtype"] == "fp32"
    assert grouped["mllm"]["processor"] is processor
    assert grouped["route"]["model_free"] is False


def test_split_entry_kwargs_ignores_unknown_fields(monkeypatch):
    warnings = []
    monkeypatch.setattr(entry_contract.logger, "warning_once", lambda message, *args: warnings.append(message % args))

    grouped = split_entry_kwargs({"unknown_option": 1}, context="test entry")

    assert all(not values for values in grouped.values())
    assert "unknown_option" in warnings[0]
