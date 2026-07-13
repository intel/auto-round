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

from types import SimpleNamespace

import pytest
import torch

from auto_round.formats import SVDQuantNunchakuFormat, get_formats
from auto_round.schemes import PRESET_SCHEMES


def test_get_formats_resolves_svdquant_nunchaku():
    compressor = SimpleNamespace(scheme="MXFP4", bits=4, group_size=32)

    formats = get_formats("svdquant_nunchaku", compressor)

    assert len(formats) == 1
    assert formats[0].format_name == "svdquant_nunchaku"


def test_svdquant_nunchaku_rejects_incompatible_structured_scheme():
    scheme = PRESET_SCHEMES["MXFP4"].copy()
    scheme.group_size = 64

    with pytest.raises(ValueError, match=r"group_size=64.*group_size=32"):
        SVDQuantNunchakuFormat.check_scheme_args(scheme)


def test_svdquant_nunchaku_accepts_mxfp4_structured_scheme():
    assert SVDQuantNunchakuFormat.check_scheme_args(PRESET_SCHEMES["MXFP4"].copy())


def test_get_formats_rejects_unsupported_svdquant_nunchaku_preset():
    compressor = SimpleNamespace(scheme="W4A16", bits=4, group_size=128)

    with pytest.raises(ValueError, match=r"svdquant_nunchaku.*only.*MXFP4.*W4A16"):
        get_formats("svdquant_nunchaku", compressor)


def test_save_quantized_delegates_only_explicit_exporter_kwargs(monkeypatch, tmp_path):
    import auto_round.export.svdquant_nunchaku as exporter

    compressor = SimpleNamespace(scheme="MXFP4", bits=4, group_size=32)
    output_format = get_formats("svdquant_nunchaku", compressor)[0]
    output_dir = tmp_path / "nested"
    model = torch.nn.Linear(2, 2)
    config = object()
    residual_provider = object()
    adapter = object()
    captured = {}

    def fake_export(export_model, output_path, **kwargs):
        captured.update(model=export_model, output_path=output_path, kwargs=kwargs)

    monkeypatch.setattr(exporter, "save_svdquant_nunchaku_safetensors", fake_export)

    result = output_format.save_quantized(
        output_dir,
        model=model,
        tokenizer=object(),
        layer_config={"ignored": True},
        inplace=False,
        device="cpu",
        serialization_dict={"ignored": True},
        processor=object(),
        config=config,
        residual_provider=residual_provider,
        adapter=adapter,
    )

    assert result is model
    assert captured == {
        "model": model,
        "output_path": str(output_dir / "model.safetensors"),
        "kwargs": {
            "config": config,
            "residual_provider": residual_provider,
            "adapter": adapter,
        },
    }
    assert not output_dir.exists()
