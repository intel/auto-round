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


def test_get_formats_resolves_svdquant_nunchaku():
    compressor = SimpleNamespace(scheme="MXFP4", bits=4, group_size=32)

    formats = get_formats("svdquant_nunchaku", compressor)

    assert len(formats) == 1
    assert formats[0].format_name == "svdquant_nunchaku"


def test_nunchaku_pipeline_export_uses_full_diffusers_layout():
    from auto_round.compressors.diffusion_mixin import _is_nunchaku_pipeline_export

    formats = [SimpleNamespace(format_name="svdquant_nunchaku")]

    assert _is_nunchaku_pipeline_export(formats) is True
    assert _is_nunchaku_pipeline_export([SimpleNamespace(format_name="auto_round")]) is False


def test_nunchaku_pipeline_index_uses_onefile_model_class(tmp_path):
    from auto_round.compressors.diffusion_mixin import _rewrite_nunchaku_pipeline_index

    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    save_file(
        {"probe": torch.zeros(1)},
        transformer_dir / "diffusion_pytorch_model.safetensors",
        metadata={"model_class": "NunchakuFluxTransformer2dModel"},
    )
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "FluxPipeline",
                "transformer": ["diffusers", "FluxTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        ),
        encoding="utf-8",
    )

    _rewrite_nunchaku_pipeline_index(tmp_path, ["transformer"])

    model_index = json.loads((tmp_path / "model_index.json").read_text(encoding="utf-8"))
    assert model_index["transformer"] == ["nunchaku", "NunchakuFluxTransformer2dModel"]
    assert model_index["vae"] == ["diffusers", "AutoencoderKL"]


def test_nunchaku_pipeline_index_requires_model_class_metadata(tmp_path):
    from auto_round.compressors.diffusion_mixin import _rewrite_nunchaku_pipeline_index

    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    save_file({"probe": torch.zeros(1)}, transformer_dir / "diffusion_pytorch_model.safetensors")
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "FluxPipeline", "transformer": ["diffusers", "FluxTransformer2DModel"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_class"):
        _rewrite_nunchaku_pipeline_index(tmp_path, ["transformer"])


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
    source_transformer = Bf16Component()
    vae = Bf16Component()

    class Pipeline:
        def __init__(self):
            self.transformer = source_transformer
            self.vae = vae
            self.components = {"transformer": source_transformer, "vae": vae}

        def save_config(self, output_dir):
            output_dir = Path(output_dir)
            (output_dir / "model_index.json").write_text(
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
    compressor.model_context = SimpleNamespace(pipe=Pipeline(), model=transformer)
    compressor.compress_context = SimpleNamespace(is_immediate_saving=False)

    compressor.save_quantized(
        tmp_path,
        format=[SimpleNamespace(format_name="svdquant_nunchaku")],
    )

    assert (tmp_path / "transformer" / "diffusion_pytorch_model.safetensors").is_file()
    assert not (tmp_path / "transformer" / "model.safetensors").exists()
    assert (tmp_path / "transformer" / "config.json").is_file()
    assert (tmp_path / "vae" / "diffusion_pytorch_model.safetensors").is_file()
    model_index = json.loads((tmp_path / "model_index.json").read_text(encoding="utf-8"))
    assert model_index["transformer"] == ["nunchaku", "NunchakuFluxTransformer2dModel"]


def test_svdquant_nunchaku_resolves_flux_adapter_name(monkeypatch, tmp_path):
    import auto_round.export.svdquant_nunchaku as exporter
    from auto_round.export.svdquant_adapters.flux import FluxSVDQuantNunchakuAdapter

    output_format = get_formats("svdquant_nunchaku", _mxfp4_compressor())[0]
    model = torch.nn.Module()
    model.config = {"_class_name": "FluxTransformer2DModel", "num_layers": 0, "num_single_layers": 0}
    captured = {}

    def fake_export(model, output_path, *, config, residual_provider, adapter):
        captured["adapter"] = adapter
        captured["runtime_loadable"] = config.runtime_loadable

    monkeypatch.setattr(exporter, "save_svdquant_nunchaku_safetensors", fake_export)
    output_format.save_quantized(str(tmp_path), model=model, model_adapter="flux", device="cpu")

    assert isinstance(captured["adapter"], FluxSVDQuantNunchakuAdapter)
    assert captured["runtime_loadable"] is True


def test_svdquant_nunchaku_auto_detects_flux_adapter(monkeypatch, tmp_path):
    import auto_round.export.svdquant_nunchaku as exporter
    from auto_round.export.svdquant_adapters.flux import FluxSVDQuantNunchakuAdapter

    output_format = get_formats("svdquant_nunchaku", _mxfp4_compressor())[0]
    model = torch.nn.Module()
    model.config = {"_class_name": "FluxTransformer2DModel", "num_layers": 0, "num_single_layers": 0}
    captured = {}

    def fake_export(model, output_path, *, config, residual_provider, adapter):
        captured["adapter"] = adapter
        captured["runtime_loadable"] = config.runtime_loadable

    monkeypatch.setattr(exporter, "save_svdquant_nunchaku_safetensors", fake_export)
    output_format.save_quantized(str(tmp_path), model=model, model_adapter="auto", device="cpu")

    assert isinstance(captured["adapter"], FluxSVDQuantNunchakuAdapter)
    assert captured["runtime_loadable"] is True


def test_svdquant_nunchaku_rejects_incompatible_structured_scheme():
    scheme = PRESET_SCHEMES["MXFP4"].copy()
    scheme.group_size = 64

    with pytest.raises(ValueError, match=r"group_size=64.*group_size=32"):
        SVDQuantNunchakuFormat.check_scheme_args(scheme)


def test_svdquant_nunchaku_accepts_mxfp4_structured_scheme():
    assert SVDQuantNunchakuFormat.check_scheme_args(PRESET_SCHEMES["MXFP4"].copy())


def test_svdquant_nunchaku_accepts_canonical_e2m1_aliases():
    scheme = PRESET_SCHEMES["MXFP4"].copy()
    scheme.data_type = "mx_fp4e2m1"
    scheme.act_data_type = "mx_fp4"

    assert SVDQuantNunchakuFormat.check_scheme_args(scheme)


def test_get_formats_rejects_unsupported_svdquant_nunchaku_preset():
    compressor = SimpleNamespace(scheme="W4A16", bits=4, group_size=128)

    with pytest.raises(ValueError, match=r"svdquant_nunchaku.*only.*MXFP4.*W4A16"):
        get_formats("svdquant_nunchaku", compressor)


def test_get_formats_rejects_incompatible_resolved_compressor_scheme():
    compressor = _mxfp4_compressor(act_dynamic=False)

    with pytest.raises(ValueError, match=r"act_dynamic=False.*act_dynamic=True"):
        get_formats("svdquant_nunchaku", compressor)


def test_svdquant_nunchaku_disables_immediate_packing_for_low_memory_diffusion():
    class DiffusionTransformer(torch.nn.Module):
        pass

    output_format = get_formats(
        "svdquant_nunchaku", SimpleNamespace(scheme="MXFP4", bits=4, group_size=32)
    )[0]
    compressor = SimpleNamespace(
        formats=[output_format],
        inplace=True,
        has_qlayer_outside_block=False,
        need_calib=True,
        model_context=SimpleNamespace(model=DiffusionTransformer(), is_mllm=False),
        compress_context=SimpleNamespace(
            low_cpu_mem_usage=True,
            is_immediate_packing=True,
            is_immediate_saving=True,
        ),
        quantize_config=SimpleNamespace(data_type="mx_fp"),
        output_dir="unused",
        _ensure_shard_writer=lambda: pytest.fail("ShardWriter must not be created"),
    )

    BaseCompressor._adjust_immediate_packing_and_saving(compressor)

    assert compressor.compress_context.low_cpu_mem_usage is True
    assert compressor.compress_context.is_immediate_packing is False
    assert compressor.compress_context.is_immediate_saving is False


def test_save_quantized_without_output_dir_returns_model(monkeypatch):
    import auto_round.export.svdquant_nunchaku as exporter

    output_format = get_formats(
        "svdquant_nunchaku", SimpleNamespace(scheme="MXFP4", bits=4, group_size=32)
    )[0]
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        exporter,
        "save_svdquant_nunchaku_safetensors",
        lambda *args, **kwargs: pytest.fail("exporter must not be called"),
    )

    assert output_format.save_quantized(None, model=model) is model


def test_save_quantized_rejects_unknown_kwargs(tmp_path):
    output_format = get_formats(
        "svdquant_nunchaku", SimpleNamespace(scheme="MXFP4", bits=4, group_size=32)
    )[0]

    with pytest.raises(TypeError, match=r"unexpected keyword argument 'adaptor'"):
        output_format.save_quantized(tmp_path, model=torch.nn.Linear(2, 2), adaptor=object())


def test_save_quantized_rejects_incompatible_svd_residual_override(monkeypatch, tmp_path):
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


def test_save_quantized_allows_unrelated_exclusions_and_adanorm_overrides(monkeypatch, tmp_path):
    import auto_round.export.svdquant_nunchaku as exporter

    output_format = get_formats("svdquant_nunchaku", _mxfp4_compressor())[0]
    model = _toy_svd_model()
    model.excluded = torch.nn.Linear(32, 32)
    model.adanorm = torch.nn.Linear(32, 32)
    called = False

    def fake_export(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(exporter, "save_svdquant_nunchaku_safetensors", fake_export)

    output_format.save_quantized(
        tmp_path,
        model=model,
        layer_config={
            "excluded": {"bits": 16, "data_type": "fp"},
            "adanorm": "W4A16",
        },
    )

    assert called


def test_save_quantized_exports_real_tiny_svd_model(tmp_path):
    output_format = get_formats("svdquant_nunchaku", _mxfp4_compressor())[0]
    model = _toy_svd_model()

    result = output_format.save_quantized(tmp_path, model=model)

    assert result is model
    assert (tmp_path / "diffusion_pytorch_model.safetensors").is_file()


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
        image_processor=object(),
        quant_nontext_module=True,
        config=config,
        residual_provider=residual_provider,
        adapter=adapter,
    )

    assert result is model
    assert captured == {
        "model": model,
        "output_path": str(output_dir / "diffusion_pytorch_model.safetensors"),
        "kwargs": {
            "config": config,
            "residual_provider": residual_provider,
            "adapter": adapter,
        },
    }
    assert not output_dir.exists()


def test_save_quantized_accepts_documented_nunchaku_export_kwargs(monkeypatch, tmp_path):
    import auto_round.export.svdquant_nunchaku as exporter

    output_format = get_formats("svdquant_nunchaku", _mxfp4_compressor())[0]
    model = torch.nn.Linear(2, 2)
    model_adapter = object()
    captured = {}

    def fake_export(export_model, output_path, **kwargs):
        captured.update(model=export_model, output_path=output_path, kwargs=kwargs)

    monkeypatch.setattr(exporter, "save_svdquant_nunchaku_safetensors", fake_export)

    output_format.save_quantized(
        tmp_path,
        model=model,
        weight_dtype="mx_fp4e2m1",
        group_size=32,
        model_adapter=model_adapter,
    )

    config = captured["kwargs"]["config"]
    assert config.weight_dtype == "fp4_e2m1_all"
    assert config.group_size == 32
    assert config.runtime_loadable is True
    assert captured["kwargs"]["adapter"] is model_adapter
