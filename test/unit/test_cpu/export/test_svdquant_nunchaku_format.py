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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.compressors.base import BaseCompressor
from auto_round.formats import SVDQuantNunchakuFormat, get_formats
from auto_round.schemes import PRESET_SCHEMES


def _mxfp4_compressor(**updates):
    values = PRESET_SCHEMES["MXFP4"].to_dict()
    values.update(scheme="MXFP4", **updates)
    return SimpleNamespace(**values)


def _toy_svd_model():
    model = torch.nn.Module()
    residual = torch.nn.Linear(32, 32)
    residual.data_type = "mx_fp4e2m1"
    residual.bits = 4
    residual.group_size = 32
    residual.sym = True
    residual.act_data_type = "mx_fp4e2m1"
    residual.act_bits = 4
    residual.act_group_size = 32
    residual.act_sym = True
    residual.act_dynamic = True
    model.svd = SVDQuantLinear(
        residual,
        torch.nn.Linear(32, 1, bias=False),
        torch.nn.Linear(1, 32, bias=False),
        torch.ones(32),
    )
    return model


def test_get_formats_resolves_full_model_svdquant_nunchaku_format():
    output_format = get_formats("svdquant_nunchaku", _mxfp4_compressor())[0]

    assert output_format.format_name == "svdquant_nunchaku"
    assert output_format.requires_full_model_export is True


def test_svdquant_nunchaku_rejects_incompatible_scheme():
    scheme = PRESET_SCHEMES["MXFP4"].copy()
    scheme.group_size = 64

    with pytest.raises(ValueError, match=r"group_size=64.*group_size=32"):
        SVDQuantNunchakuFormat.check_scheme_args(scheme)


def test_full_model_format_disables_immediate_packing_and_saving():
    output_format = get_formats("svdquant_nunchaku", _mxfp4_compressor())[0]
    compressor = SimpleNamespace(
        formats=[output_format],
        inplace=True,
        has_qlayer_outside_block=False,
        need_calib=True,
        model_context=SimpleNamespace(model=torch.nn.Module(), is_mllm=False),
        compress_context=SimpleNamespace(
            low_cpu_mem_usage=True,
            is_immediate_packing=True,
            is_immediate_saving=True,
        ),
        quantize_config=SimpleNamespace(data_type="mx_fp"),
        output_dir="unused",
        _ensure_shard_writer=lambda: pytest.fail("full-model export must not create a shard writer"),
    )

    BaseCompressor._adjust_immediate_packing_and_saving(compressor)

    assert compressor.compress_context.is_immediate_packing is False
    assert compressor.compress_context.is_immediate_saving is False


def test_format_uses_flux_adapter_and_diffusers_weight_name(monkeypatch, tmp_path):
    import auto_round.export.svdquant_nunchaku as exporter
    from auto_round.export.svdquant_adapters.flux import FluxSVDQuantNunchakuAdapter

    output_format = get_formats("svdquant_nunchaku", _mxfp4_compressor())[0]

    class FluxModel(torch.nn.Module):
        config = {"_class_name": "FluxTransformer2DModel", "num_layers": 0, "num_single_layers": 0}

        def save_config(self, output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            Path(output_dir, "config.json").write_text(json.dumps(self.config), encoding="utf-8")

    model = FluxModel()
    captured = {}

    def fake_export(export_model, output_path, *, config, residual_provider, adapter):
        captured.update(
            model=export_model,
            output_path=output_path,
            runtime_loadable=config.runtime_loadable,
            residual_provider=residual_provider,
            adapter=adapter,
        )

    monkeypatch.setattr(exporter, "save_svdquant_nunchaku_safetensors", fake_export)

    result = output_format.save_quantized(tmp_path, model=model, model_adapter="flux", device="cpu")

    assert result is model
    assert captured["output_path"] == str(tmp_path / "diffusion_pytorch_model.safetensors")
    assert captured["runtime_loadable"] is True
    assert isinstance(captured["adapter"], FluxSVDQuantNunchakuAdapter)
    assert (tmp_path / "config.json").is_file()


def test_format_rejects_models_without_runtime_adapter(tmp_path):
    output_format = get_formats("svdquant_nunchaku", _mxfp4_compressor())[0]

    with pytest.raises(ValueError, match="runtime model adapter"):
        output_format.save_quantized(tmp_path, model=torch.nn.Linear(2, 2), model_adapter="auto")


def test_format_rejects_incompatible_residual_override(monkeypatch, tmp_path):
    import auto_round.export.svdquant_nunchaku as exporter

    output_format = get_formats("svdquant_nunchaku", _mxfp4_compressor())[0]
    monkeypatch.setattr(
        exporter,
        "save_svdquant_nunchaku_safetensors",
        lambda *args, **kwargs: pytest.fail("exporter must not be called"),
    )

    with pytest.raises(ValueError, match=r"group_size=64.*group_size=32"):
        output_format.save_quantized(
            tmp_path,
            model=_toy_svd_model(),
            layer_config={"svd.residual_linear": {"group_size": 64}},
        )


def test_diffusion_save_exports_self_contained_nunchaku_pipeline(tmp_path):
    from auto_round.compressors.diffusion_mixin import DiffusionMixin

    class QuantizedTransformer(torch.nn.Module):
        def save_config(self, output_dir):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.json").write_text('{"_class_name": "FluxTransformer2DModel"}', encoding="utf-8")

    class Bf16Component:
        def save_pretrained(self, output_dir):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.json").write_text("{}", encoding="utf-8")
            (output_dir / "diffusion_pytorch_model.safetensors").touch()

    transformer = QuantizedTransformer()
    vae = Bf16Component()

    class Pipeline:
        def __init__(self):
            self.transformer = transformer
            self.vae = vae
            self.components = {"transformer": transformer, "vae": vae}

        def save_config(self, output_dir):
            Path(output_dir, "model_index.json").write_text(
                json.dumps(
                    {
                        "_class_name": "FluxPipeline",
                        "transformer": ["diffusers", "FluxTransformer2DModel"],
                        "vae": ["diffusers", "AutoencoderKL"],
                    }
                ),
                encoding="utf-8",
            )

    class ExportParent:
        def save_quantized(self, output_dir, **kwargs):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_file(
                {"probe": torch.zeros(1)},
                f"{output_dir}/diffusion_pytorch_model.safetensors",
                metadata={"model_class": "NunchakuFluxTransformer2dModel"},
            )
            return self.model_context.model

    class Compressor(DiffusionMixin, ExportParent):
        pass

    compressor = Compressor.__new__(Compressor)
    compressor.formats = [SimpleNamespace(format_name="svdquant_nunchaku")]
    compressor.model_context = SimpleNamespace(pipe=Pipeline(), model=transformer)
    compressor.compress_context = SimpleNamespace(is_immediate_saving=False)

    compressor.save_quantized(tmp_path)

    assert (tmp_path / "transformer" / "diffusion_pytorch_model.safetensors").is_file()
    assert (tmp_path / "vae" / "diffusion_pytorch_model.safetensors").is_file()
    model_index = json.loads((tmp_path / "model_index.json").read_text(encoding="utf-8"))
    assert model_index["transformer"] == ["nunchaku", "NunchakuFluxTransformer2dModel"]
    assert model_index["vae"] == ["diffusers", "AutoencoderKL"]


def test_diffusion_save_requires_runtime_model_class_metadata(tmp_path):
    from auto_round.compressors.diffusion_mixin import DiffusionMixin

    model = torch.nn.Module()

    class Pipeline:
        transformer = model
        components = {"transformer": model}

        def save_config(self, output_dir):
            Path(output_dir, "model_index.json").write_text(
                json.dumps({"transformer": ["diffusers", "FluxTransformer2DModel"]}), encoding="utf-8"
            )

    class ExportParent:
        def save_quantized(self, output_dir, **kwargs):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_file({"probe": torch.zeros(1)}, f"{output_dir}/diffusion_pytorch_model.safetensors")
            return model

    class Compressor(DiffusionMixin, ExportParent):
        pass

    compressor = Compressor.__new__(Compressor)
    compressor.formats = [SimpleNamespace(format_name="svdquant_nunchaku")]
    compressor.model_context = SimpleNamespace(pipe=Pipeline(), model=model)
    compressor.compress_context = SimpleNamespace(is_immediate_saving=False)

    with pytest.raises(ValueError, match="model_class"):
        compressor.save_quantized(tmp_path)
