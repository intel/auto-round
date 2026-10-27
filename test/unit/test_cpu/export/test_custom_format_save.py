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

from pathlib import Path

import torch

from auto_round import AutoRound
from auto_round.export.formats import OutputFormat
from auto_round.utils import SUPPORTED_FORMATS


def test_custom_format_quantize_and_save(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import OPTConfig, OPTForCausalLM, PreTrainedTokenizerFast

    # Keep the extension acceptance test local and independent of model downloads.
    model_dir = tmp_path / "input_model"
    model = OPTForCausalLM(
        OPTConfig(
            vocab_size=128,
            hidden_size=64,
            ffn_dim=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=128,
        )
    )
    model.save_pretrained(model_dir)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(model_dir)
    tiny_opt_model_path = str(model_dir)

    name = "test_custom_save"
    calls = []

    @OutputFormat.register(name)
    class MarkerFormat(OutputFormat):
        def pack_layer(self, *args, **kwargs):
            pass

        def save_quantized_as_format(self, output_dir, *, model, **kwargs):
            calls.append((output_dir, model, kwargs))
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "export.marker").write_text(self.output_format)
            return model

    try:
        ar = AutoRound(
            tiny_opt_model_path,
            scheme="W4A16",
            iters=0,
            disable_opt_rtn=True,
            disable_model_free=True,
            low_cpu_mem_usage=False,
            device_map="0" if torch.cuda.is_available() else "cpu",
        )
        model, folder = ar.quantize_and_save(str(tmp_path), format=name)

        assert len(calls) == 1
        output_dir, saved_model, kwargs = calls[0]
        assert output_dir == folder
        assert Path(folder).is_relative_to(tmp_path)
        assert saved_model is model is ar.model
        assert kwargs["tokenizer"] is ar.tokenizer
        assert kwargs["layer_config"].keys() == ar.layer_config.keys()
        assert all(config["bits"] == ar.layer_config[name]["bits"] for name, config in kwargs["layer_config"].items())
        assert kwargs["inplace"] is True
        assert torch.device(kwargs["device"]).type in {"cpu", "cuda"}
        assert kwargs["serialization_dict"]["bits"] == 4
        assert kwargs["serialization_dict"]["data_type"] == "int"
        assert kwargs["layer_config"]
        assert (Path(folder) / "export.marker").read_text() == name
    finally:
        OutputFormat._format_list.pop(name, None)
    assert name not in SUPPORTED_FORMATS
