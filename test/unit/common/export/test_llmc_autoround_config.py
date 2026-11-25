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
"""AutoRound hyper-parameters are recorded in llm-compressor checkpoints (#2469)."""

import json

import pytest

from auto_round.export.formats.compressed_tensors.utils import AUTOROUND_CONFIG_KEY, add_autoround_config


class TestAddAutoroundConfig:
    def test_records_tuning_args_and_version(self):
        qconfig = {"quant_method": "compressed-tensors", "provider": "auto-round"}
        serialization_dict = {
            "autoround_version": "0.17.0",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "iters": 200,
            "lr": 0.005,
            "nsamples": 128,
            "seqlen": 2048,
            "act_bits": None,
            "regex_config": {"foo": {"bits": 8}},
            "supported_types": ["Linear"],
        }
        out = add_autoround_config(qconfig, serialization_dict)
        assert out is qconfig
        ar = qconfig[AUTOROUND_CONFIG_KEY]
        # Structured per-layer and runtime fields remain usable JSON values.
        assert ar == {
            "autoround_version": "0.17.0",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "iters": 200,
            "lr": 0.005,
            "nsamples": 128,
            "seqlen": 2048,
            "regex_config": {"foo": {"bits": 8}},
        }
        assert "quant_block_list" not in ar
        assert "to_quant_block_names" not in ar
        assert "scale_dtype" not in ar
        assert "supported_types" not in ar
        assert "static_kv_dtype" not in ar
        assert "static_attention_dtype" not in ar
        assert "static_kv_granularity" not in ar
        assert "static_attention_granularity" not in ar
        # untouched top-level keys
        assert qconfig["quant_method"] == "compressed-tensors"
        json.dumps(qconfig)

    def test_keeps_applicable_static_dtype_and_json_values(self):
        import enum

        import torch

        class SampleEnum(enum.Enum):
            A = "alpha"

        qconfig = {"provider": "auto-round"}
        applicable_qconfig = qconfig | {
            "kv_cache_scheme": {"num_bits": 8},
            "attention_input_activations": {"num_bits": 8},
        }
        add_autoround_config(
            applicable_qconfig,
            {
                "scale_dtype": torch.float16,
                "group_size": (128, 128),
                "static_kv_dtype": torch.device("cpu"),
                "static_attention_dtype": SampleEnum.A,
                "static_kv_granularity": "head",
                "static_attention_granularity": "tensor",
            },
        )
        assert applicable_qconfig[AUTOROUND_CONFIG_KEY] == {
            "group_size": [128, 128],
            "static_kv_dtype": "cpu",
            "static_attention_dtype": "alpha",
            "static_kv_granularity": "head",
            "static_attention_granularity": "tensor",
        }

    def test_quantization_block_list_is_not_exported(self):
        qconfig = {"provider": "auto-round"}
        add_autoround_config(
            qconfig,
            {
                "quant_block_list": [["model.layers.0"]],
                "to_quant_block_names": ["model.layers"],
            },
        )
        assert AUTOROUND_CONFIG_KEY not in qconfig

    @pytest.mark.parametrize("serialization_dict", [None, {}, {"iters": None}])
    def test_noop_without_values(self, serialization_dict):
        qconfig = {"provider": "auto-round"}
        add_autoround_config(qconfig, serialization_dict)
        assert AUTOROUND_CONFIG_KEY not in qconfig

    def test_noop_without_auto_round_provider(self):
        qconfig = {"provider": "other"}
        add_autoround_config(qconfig, {"iters": 200, "autoround_version": "0.17.0"})
        assert AUTOROUND_CONFIG_KEY not in qconfig
        qconfig_missing = {}
        add_autoround_config(qconfig_missing, {"iters": 200})
        assert AUTOROUND_CONFIG_KEY not in qconfig_missing

    def test_compressed_tensors_ignores_extra_key(self):
        ct = pytest.importorskip("compressed_tensors.quantization")
        qconfig = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 128,
                    },
                }
            },
            "ignore": ["lm_head"],
            "provider": "auto-round",
        }
        add_autoround_config(qconfig, {"iters": 200, "autoround_version": "0.17.0"})
        parsed = ct.QuantizationConfig.model_validate(qconfig)
        assert parsed.config_groups["group_0"].weights.num_bits == 4


@pytest.mark.parametrize("scheme", ["W4A16", "MXFP4"])
def test_llmc_export_writes_autoround_config(tiny_opt_model_path, tmp_path, scheme):
    pytest.importorskip("compressed_tensors")
    from pathlib import Path

    from auto_round import AutoRound
    from auto_round.version import __version__

    ar = AutoRound(tiny_opt_model_path, scheme=scheme, iters=2, nsamples=2, seqlen=16, lr=0.01)
    _, export_dir = ar.quantize_and_save(output_dir=str(tmp_path), format="llm_compressor")
    config_path = Path(export_dir) / "config.json"
    assert config_path.is_file()
    with open(config_path) as f:
        qconfig = json.load(f)["quantization_config"]
    assert qconfig["quant_method"] == "compressed-tensors"
    ar_cfg = qconfig[AUTOROUND_CONFIG_KEY]
    assert ar_cfg["autoround_version"] == __version__
    assert ar_cfg["iters"] == 2
    assert ar_cfg["nsamples"] == 2
    assert ar_cfg["seqlen"] == 16
    assert ar_cfg["lr"] == 0.01
    assert ar_cfg["bits"] == 4
    assert "scale_dtype" not in ar_cfg
    assert "quant_block_list" not in ar_cfg
    assert "quantization_config" not in ar_cfg

    from transformers import AutoConfig

    AutoConfig.from_pretrained(str(export_dir))
    from compressed_tensors.quantization import QuantizationConfig

    QuantizationConfig.model_validate(qconfig)
