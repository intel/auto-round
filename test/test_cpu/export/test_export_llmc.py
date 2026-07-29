import json
import os
import shutil

import pytest
import torch
from packaging import version
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.export.export_to_autogptq import export as autogptq_export
from auto_round.export.export_to_autoround import export as autoround_export
from auto_round.export.export_to_autoround import export_to_fp8 as autoround_fp8_export
from auto_round.export.export_to_awq import export as awq_export

from ...helpers import forbid_threaded_packing, get_model_path, opt_name_or_path, transformers_version


def _get_folder_size(path: str) -> float:
    """Return folder size in GB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # convert to GB


class TestAutoRound:

    @classmethod
    def setup_class(self):
        self.model_name = opt_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.parametrize(
        "iters,use_dataloader,scheme",
        [
            (0, False, "INT8"),  # RTN with new scheme name
            (1, True, "INT8"),  # tuning with new scheme name
            (0, False, "INT8_W8A8"),  # RTN with old scheme name (backward compat)
        ],
        ids=["rtn", "tuning", "rtn-old-scheme"],
    )
    def test_llmc_dynamic_wint8aint8_export(self, iters, use_dataloader, scheme, dataloader):
        from safetensors import safe_open

        dataset = dataloader if use_dataloader else None
        autoround = AutoRound(
            self.model_name,
            iters=iters,
            nsamples=2,
            seqlen=2,
            dataset=dataset,
            scheme=scheme,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="llm_compressor")
        with safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt") as f:
            assert "model.decoder.layers.8.self_attn.k_proj.weight_scale" in f.keys()
            assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype == torch.int8
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @pytest.mark.parametrize(
        "scheme,bits,group_size,sym",
        [
            ("W4A16", 4, 128, True),
            ("W4A16", 4, -1, True),
            ("W8A16", 8, -1, True),
        ],
    )
    def test_llmc_wint_a16_export(self, scheme, bits, group_size, sym):
        from safetensors import safe_open

        autoround = AutoRound(
            self.model_name,
            iters=2,
            nsamples=2,
            seqlen=2,
            scheme=scheme,
            bits=bits,
            group_size=group_size,
            sym=sym,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="llm_compressor")
        with safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt") as f:
            # weights must be packed as int32 (compressed-tensors stores both int4 and int8 as torch.int32)
            weight = f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight_packed")
            assert weight.dtype == torch.int32, f"Expected int32 weight for {scheme}, got {weight.dtype}"
            # weight_scale must be present and be a float tensor
            scale_key = "model.decoder.layers.8.self_attn.k_proj.weight_scale"
            assert scale_key in f.keys(), f"Missing {scale_key} for {scheme} export"
            scale = f.get_tensor(scale_key)
            assert scale.dtype in (
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ), f"Expected float weight_scale for {scheme}, got {scale.dtype}"
            # No input_scale should be present for weight-only quantization
            input_scale_keys = [k for k in f.keys() if k.endswith(".input_scale")]
            assert (
                len(input_scale_keys) == 0
            ), f"Expected no input_scale for weight-only {scheme}, but found: {input_scale_keys[:5]}"
        shutil.rmtree(quantized_model_path, ignore_errors=True)


@pytest.mark.parametrize(
    "format_name,export_module,sym",
    [
        ("auto_gptq", autogptq_export, False),
        ("auto_awq", awq_export, False),
        ("auto_round", autoround_export, True),
    ],
)
def test_weight_only_exports_pack_serially(tiny_opt_model_path, tmp_path, monkeypatch, format_name, export_module, sym):
    autoround = AutoRound(
        tiny_opt_model_path,
        bits=4,
        group_size=128,
        sym=sym,
        iters=0,
        disable_opt_rtn=True,
    )
    autoround.quantize()
    forbid_threaded_packing(monkeypatch, export_module)
    autoround.save_quantized(output_dir=tmp_path, inplace=False, format=format_name)
    assert os.path.exists(os.path.join(tmp_path, "config.json"))


def test_fp8_autoround_export_packs_serially(tiny_opt_model_path, tmp_path, monkeypatch):
    from safetensors import safe_open

    autoround = AutoRound(
        tiny_opt_model_path,
        bits=8,
        group_size=-1,
        iters=0,
        scheme="FP8_STATIC",
        nsamples=2,
        seqlen=2,
        static_kv_dtype="fp8",
    )
    autoround.quantize()
    forbid_threaded_packing(monkeypatch, autoround_fp8_export)
    autoround.save_quantized(output_dir=tmp_path, format="auto_round")
    with safe_open(os.path.join(tmp_path, "model.safetensors"), framework="pt") as f:
        assert "model.decoder.layers.0.self_attn.k_proj.weight_scale" in f.keys()


@pytest.mark.parametrize("low_cpu_mem_usage", [True, False])
def test_immediate_saving_mode(tiny_opt_model_path, tmp_path, low_cpu_mem_usage, caplog):
    """Verify that immediate_saving (triggered by low_cpu_mem_usage) produces a complete model output."""
    import logging

    output_dir = str(tmp_path / "output")
    with caplog.at_level(logging.DEBUG, logger="auto_round"):
        autoround = AutoRound(
            tiny_opt_model_path,
            scheme="MXFP4",
            iters=2,
            seqlen=2,
            nsamples=2,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=output_dir, format="llm_compressor")

    # No spurious "already exists" warning should be emitted
    conflict_messages = [r.message for r in caplog.records if "already exists" in r.message]
    assert len(conflict_messages) == 0, f"Unexpected conflict warnings: {conflict_messages}"

    # All essential files must exist regardless of immediate_saving mode
    assert os.path.exists(os.path.join(quantized_model_path, "config.json")), "config.json missing"
    assert os.path.exists(
        os.path.join(quantized_model_path, "quantization_config.json")
    ), "quantization_config.json missing"

    # Exactly 1 safetensors shard for this tiny model
    safetensor_files = [f for f in os.listdir(quantized_model_path) if f.endswith(".safetensors")]
    assert len(safetensor_files) == 1, f"Expected 1 safetensors file, got {len(safetensor_files)}: {safetensor_files}"

    # Tokenizer files must be present
    assert os.path.exists(os.path.join(quantized_model_path, "tokenizer_config.json")), "tokenizer_config.json missing"

    # Total file count: config.json, quantization_config.json, model.safetensors,
    # generation_config.json, tokenizer.json, tokenizer_config.json = 6
    all_files = os.listdir(quantized_model_path)
    assert len(all_files) == 6, f"Expected 6 files, got {len(all_files)}: {sorted(all_files)}"

    # Verify weights are loadable and non-empty
    from safetensors import safe_open

    with safe_open(os.path.join(quantized_model_path, safetensor_files[0]), framework="pt") as f:
        keys = f.keys()
        assert len(keys) > 0, "Safetensors file has no tensors"
