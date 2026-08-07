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

"""CPU CI tests for AWQ (Activation-Aware Weight Quantization) algorithm.

Covers:
- Normal LLM (OPT-125m): W4A16 quantization, inference, export args verification
- INT W8A8 export via llm_compressor format with config validation
- Tiny Qwen MoE: dynamic smoothing, quantized layer checks, quant config saving
"""

import json
import os
import shutil
from test.helpers import generate_prompt, get_model_path, opt_name_or_path, save_tiny_model

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AWQConfig, SignRoundConfig


class TestAWQNormalLLM:
    """AWQ quantization on a normal LLM (OPT-125m style tiny model).

    Tests W4A16 quantize → inference → export args validation.
    """

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_awq_w4a16_quantize_and_inference(self, tiny_opt_model_path):
        """W4A16 AWQ quantization produces valid layer_config and model can generate."""
        ar = AutoRound(
            tiny_opt_model_path,
            scheme="W4A16",
            alg_configs=AWQConfig(n_grid=1),
            n_grid=1,
            nsamples=2,
            seqlen=32,
            batch_size=2,
        )
        model, layer_config = ar.quantize()

        # Verify quantization happened
        assert model is not None
        assert len(layer_config) > 0
        for name, cfg in layer_config.items():
            assert cfg["bits"] == 4, f"Layer {name} expected bits=4, got {cfg['bits']}"

        # Inference smoke test
        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)
        output = generate_prompt(model, tokenizer, device="cpu")
        assert len(output) > 0, "Model should produce non-empty output"

    def test_awq_w4a16_export_auto_round_format(self, tiny_opt_model_path):
        """AWQ W4A16 export to auto_round format: verify quantization_config in saved config."""
        ar = AutoRound(
            tiny_opt_model_path,
            scheme="W4A16",
            alg_configs=AWQConfig(n_grid=1),
            n_grid=1,
            nsamples=2,
            seqlen=8,
            batch_size=2,
        )
        _, save_path = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        # Verify config.json has quantization_config
        config = AutoConfig.from_pretrained(save_path)
        qconfig = config.quantization_config
        assert qconfig is not None, "quantization_config should be present"
        assert qconfig["bits"] == 4
        assert qconfig["group_size"] == 128
        assert "auto-round" in qconfig["quant_method"]

    def test_awq_w4a16_export_args_check(self, tiny_opt_model_path):
        """Verify key export args in saved quantization_config match input parameters."""
        bits, group_size, sym = 8, 64, True
        ar = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            alg_configs=AWQConfig(n_grid=1),
            n_grid=1,
            nsamples=2,
            seqlen=8,
            batch_size=2,
        )
        _, save_path = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        config = AutoConfig.from_pretrained(save_path)
        qconfig = config.quantization_config
        assert qconfig["bits"] == bits
        assert qconfig["group_size"] == group_size
        assert qconfig["sym"] == sym


class TestAWQNonIntegerSchemes:
    """Regression: AWQ smoothing must run under non-integer schemes (MX/NV-FP).

    AWQ's grid-search / clip loss reproduces the block quantizer's weight QDQ.
    The reported failure mode was an end-to-end AWQ run raising
    under an MXFP/NVFP scheme.
    """

    @pytest.mark.parametrize("scheme", ["MXFP4", "NVFP4"])
    def test_awq_non_integer_scheme_smoke(self, tiny_opt_model_path, scheme):
        ar = AutoRound(
            tiny_opt_model_path,
            scheme=scheme,
            alg_configs=[AWQConfig(n_grid=1), SignRoundConfig()],
            n_grid=1,
            nsamples=2,
            seqlen=8,
            batch_size=2,
            dataset=["local AWQ calibration sample with enough tokens for quantization"] * 2,
        )
        model, layer_config = ar.quantize()

        assert model is not None
        assert len(layer_config) > 0
        for name, cfg in layer_config.items():
            assert cfg["bits"] == 4, f"Layer {name} expected bits=4, got {cfg['bits']}"
            assert cfg["act_bits"] == 4, f"Layer {name} expected act_bits=4, got {cfg['act_bits']}"


class TestAWQW8A8LLMCompressor:
    """AWQ INT W8A8 quantization with llm_compressor export format.

    Validates compressed-tensors config structure, num_bits, and key fields.
    """

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_awq_w8a8_llmc_export(self, tiny_opt_model_path):
        """W8A8 AWQ → llm_compressor: verify compressed-tensors metadata fields."""
        ar = AutoRound(
            tiny_opt_model_path,
            scheme="INT8",
            alg_configs=AWQConfig(n_grid=1),
            nsamples=2,
            seqlen=8,
            n_grid=1,
            batch_size=2,
        )
        _, save_path = ar.quantize_and_save(output_dir=self.save_dir, format="llm_compressor")

        config = AutoConfig.from_pretrained(save_path, trust_remote_code=True)
        qconfig = config.quantization_config

        assert qconfig["quant_method"] == "compressed-tensors"

        group0 = qconfig["config_groups"]["group_0"]
        # Weight args
        assert group0["weights"]["num_bits"] == 8
        assert group0["weights"]["type"] == "int"
        assert group0["weights"]["symmetric"] is True
        # Activation args
        assert group0["input_activations"]["num_bits"] == 8
        # Targets
        targets = group0.get("targets")
        assert targets is not None and len(targets) > 0

        # QuantLinear check: verify saved weights are int8 with per-channel scales
        from safetensors import safe_open

        st_files = [f for f in os.listdir(save_path) if f.endswith(".safetensors")]
        assert len(st_files) > 0, f"No safetensors files in {save_path}"
        with safe_open(os.path.join(save_path, st_files[0]), framework="pt") as f:
            weight = f.get_tensor("model.decoder.layers.0.self_attn.k_proj.weight")
            assert weight.dtype == torch.int8, f"Expected int8 weight, got {weight.dtype}"
            scale = f.get_tensor("model.decoder.layers.0.self_attn.k_proj.weight_scale")
            assert scale.shape[1] == 1, f"Expected per-channel scale shape (out, 1), got {scale.shape}"


class TestAWQMoE:
    """AWQ on a tiny MoE model: dynamic smoothing, layer checks, config saving.

    Uses the session-scoped tiny_qwen_moe_model_path fixture.
    """

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_awq_moe_dynamic_smoothing(self, tiny_qwen_moe_model_path):
        """AWQ dynamic smoothing should resolve mappings on a MoE model without error."""
        from auto_round.algorithms.transforms.awq.mappings import resolve_mappings

        model = AutoModelForCausalLM.from_pretrained(
            tiny_qwen_moe_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        resolved = resolve_mappings(model, user_mappings=None)

        # Must resolve a non-trivial number of mappings
        assert len(resolved) > 0, "Expected non-empty resolved mappings"

        # Verify smooth layers are unique (no duplicate smoothing targets)
        smooth_names = [r.smooth_name for r in resolved]
        assert len(smooth_names) == len(
            set(smooth_names)
        ), f"Duplicate smooth names: {[n for n in smooth_names if smooth_names.count(n) > 1]}"

        # Must have attention-related mappings (input_layernorm→qkv, v_proj→o_proj)
        n_layers = model.config.num_hidden_layers
        attn_smooths = [n for n in smooth_names if "input_layernorm" in n or "self_attn.v_proj" in n]
        assert len(attn_smooths) == 2 * n_layers, f"Expected {2 * n_layers} attn smooth layers, got {len(attn_smooths)}"

        # Shared expert up→down should resolve at block level
        if hasattr(model.model.layers[0].mlp, "shared_expert"):
            shared_smooths = [n for n in smooth_names if "shared_expert" in n]
            assert (
                len(shared_smooths) == n_layers
            ), f"Expected {n_layers} shared_expert smooth layers, got {len(shared_smooths)}"

        del model

    def test_awq_moe_skip_moe(self, tiny_qwen_moe_model_path):
        """skip_moe should drop routed-expert balance layers/mappings but keep dense paths."""
        import torch.nn as nn

        from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping, _drop_routed_experts

        model = nn.Module()
        ln = nn.LayerNorm(8)
        q = nn.Linear(8, 8, bias=False)
        shared_gate = nn.Linear(8, 16, bias=False)
        e0_gate = nn.Linear(8, 16, bias=False)
        e1_gate = nn.Linear(8, 16, bias=False)
        e0_up = nn.Linear(8, 16, bias=False)
        e0_down = nn.Linear(16, 8, bias=False)

        resolved = [
            ResolvedMapping(
                smooth_name="model.layers.0.input_layernorm",
                smooth_layer=ln,
                balance_names=["model.layers.0.self_attn.q_proj"],
                balance_layers=[q],
                parent_name="model.layers.0.self_attn",
                parent=nn.Module(),
            ),
            ResolvedMapping(
                smooth_name="model.layers.0.post_attention_layernorm",
                smooth_layer=ln,
                balance_names=[
                    "model.layers.0.mlp.shared_expert.gate_proj",
                    "model.layers.0.mlp.experts.0.gate_proj",
                    "model.layers.0.mlp.experts.1.gate_proj",
                ],
                balance_layers=[shared_gate, e0_gate, e1_gate],
                parent_name="model.layers.0.mlp",
                parent=nn.Module(),
            ),
            ResolvedMapping(
                smooth_name="model.layers.0.mlp.experts.0.up_proj",
                smooth_layer=e0_up,
                balance_names=["model.layers.0.mlp.experts.0.down_proj"],
                balance_layers=[e0_down],
                parent_name="model.layers.0.mlp.experts.0",
                parent=nn.Module(),
            ),
        ]

        kept = _drop_routed_experts(model, resolved)
        smooth_names = [m.smooth_name for m in kept]

        assert "model.layers.0.mlp.experts.0.up_proj" not in smooth_names
        assert "model.layers.0.input_layernorm" in smooth_names
        mixed = next(m for m in kept if m.smooth_name.endswith("post_attention_layernorm"))
        assert mixed.balance_names == ["model.layers.0.mlp.shared_expert.gate_proj"]

        del model

    def test_explicit_awq_mapping_preserves_activation_hook_target(self):
        """Custom AWQ mappings should keep activation_hook_target for non-standard balance inputs."""
        import torch.nn as nn

        from auto_round.algorithms.transforms.awq.mappings import resolve_mappings

        class TinyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.smooth = nn.Linear(4, 4, bias=False)
                self.hook = nn.Linear(4, 4, bias=False)
                self.balance = nn.Linear(4, 4, bias=False)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([TinyBlock()])

        resolved = resolve_mappings(
            TinyModel(),
            user_mappings=[
                {
                    "smooth_layer": "smooth$",
                    "balance_layers": ["balance$"],
                    "activation_hook_target": "hook",
                }
            ],
        )

        assert len(resolved) == 1
        assert resolved[0].activation_hook_target == "hook"

    def test_hybrid_attention_mapping_short_layer_types_falls_back(self):
        """Malformed hybrid configs should fall back instead of raising IndexError."""
        from types import SimpleNamespace

        import torch.nn as nn

        from auto_round.algorithms.transforms.awq.mappings import _build_hybrid_attention_mappings

        class BadHybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    layer_types=["full_attention", "linear_attention"],
                    num_hidden_layers=3,
                )

        assert _build_hybrid_attention_mappings(BadHybridModel()) is None

    def test_awq_ignored_layer_skips_mapping(self):
        """A mapping containing an ignore_layers / bits>=16 layer is skipped as one smooth-scale group."""
        import torch.nn as nn

        from auto_round.algorithms.transforms.awq.base import AWQTransform
        from auto_round.algorithms.transforms.awq.config import AWQConfig
        from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping

        transform = AWQTransform(AWQConfig(bits=4, group_size=128, sym=True, data_type="int"))

        ln = nn.LayerNorm(8)
        q = nn.Linear(8, 8, bias=False)
        q.global_name = "model.layers.0.self_attn.q_proj"
        k = nn.Linear(8, 8, bias=False)
        k.global_name = "model.layers.0.self_attn.k_proj"

        mapping = ResolvedMapping(
            smooth_name="model.layers.0.input_layernorm",
            smooth_layer=ln,
            balance_names=[q.global_name, k.global_name],
            balance_layers=[q, k],
            parent_name="model.layers.0.self_attn",
            parent=nn.Module(),
        )

        transform._qdq_tool.layer_config = {}
        assert transform._mapping_has_ignored_layer(mapping) is False

        transform._qdq_tool.layer_config = {q.global_name: {"bits": 4}, k.global_name: {"bits": 4}}
        assert transform._mapping_has_ignored_layer(mapping) is False

        transform._qdq_tool.layer_config = {q.global_name: {"bits": 4}, k.global_name: {"bits": 16}}
        assert transform._mapping_has_ignored_layer(mapping) is True
        assert transform._mapping_is_smoothable(mapping) is False

    def test_awq_mixed_balance_quant_params_skip_mapping(self, monkeypatch):
        """Balance layers sharing one AWQ scale must share the same resolved quantization params."""
        import torch.nn as nn

        from auto_round.algorithms.transforms.awq import base as awq_base
        from auto_round.algorithms.transforms.awq.base import AWQTransform
        from auto_round.algorithms.transforms.awq.config import AWQConfig
        from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping

        transform = AWQTransform(AWQConfig(bits=4, group_size=128, sym=True, data_type="int"))

        ln = nn.LayerNorm(8)
        q = nn.Linear(8, 8, bias=False)
        q.global_name = "model.layers.0.self_attn.q_proj"
        k = nn.Linear(8, 8, bias=False)
        k.global_name = "model.layers.0.self_attn.k_proj"

        mapping = ResolvedMapping(
            smooth_name="model.layers.0.input_layernorm",
            smooth_layer=ln,
            balance_names=[q.global_name, k.global_name],
            balance_layers=[q, k],
            parent_name="model.layers.0.self_attn",
            parent=nn.Module(),
        )

        warnings = []

        def fake_warning(message, *args, **kwargs):
            warnings.append(message % args if args else message)

        monkeypatch.setattr(awq_base.logger, "warning", fake_warning)

        transform._qdq_tool.layer_config = {
            q.global_name: {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"},
            k.global_name: {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"},
        }
        assert transform._mapping_has_mixed_quant_params(mapping) is False
        assert transform._mapping_is_smoothable(mapping) is True

        transform._qdq_tool.layer_config = {
            q.global_name: {
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "data_type": "int",
                "disable_opt_rtn": False,
            },
            k.global_name: {
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "data_type": "int",
                "disable_opt_rtn": True,
            },
        }
        assert transform._mapping_has_mixed_quant_params(mapping) is False
        assert transform._mapping_is_smoothable(mapping) is True

        transform._qdq_tool.layer_config = {
            q.global_name: {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"},
            k.global_name: {"bits": 4, "group_size": 128, "sym": True, "data_type": "mx_fp"},
        }
        assert transform._mapping_has_mixed_quant_params(mapping) is True
        assert transform._mapping_is_smoothable(mapping) is False
        assert any("different quantization parameters" in warning for warning in warnings)

    def test_awq_grid_search_uses_per_balance_layer_quant_func(self):
        """Direct grid search should pass each balance layer's own resolved quant function."""
        import torch.nn as nn

        from auto_round.algorithms.transforms.awq.base import AWQTransform
        from auto_round.algorithms.transforms.awq.config import AWQConfig
        from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping

        transform = AWQTransform(AWQConfig(bits=4, group_size=8, sym=True, data_type="int", duo_scaling=False))

        ln = nn.LayerNorm(8)
        q = nn.Linear(8, 8, bias=False)
        q.global_name = "model.layers.0.self_attn.q_proj"
        k = nn.Linear(8, 8, bias=False)
        k.global_name = "model.layers.0.self_attn.k_proj"
        mapping = ResolvedMapping(
            smooth_name="model.layers.0.input_layernorm",
            smooth_layer=ln,
            balance_names=[q.global_name, k.global_name],
            balance_layers=[q, k],
            parent_name="model.layers.0.self_attn",
            parent=nn.Module(),
        )

        transform._qdq_tool.layer_config = {
            q.global_name: {"bits": 4, "group_size": 8, "sym": True, "data_type": "int"},
            k.global_name: {"bits": 8, "group_size": 8, "sym": True, "data_type": "mx_fp"},
        }
        records = []

        def fake_resolve_quant_funcs(params):
            return f"{params['data_type']}_{params['bits']}", None

        def fake_qdq(weight, params, *, quant_func=None, opt_quant_func=None, imatrix=None):
            records.append((params["data_type"], params["bits"], quant_func))
            return weight

        transform._qdq_tool.resolve_quant_funcs = fake_resolve_quant_funcs
        transform._qdq_tool.qdq = fake_qdq

        transform._grid_search_scales(mapping, torch.ones(8))

        assert ("int", 4, "int_4") in records
        assert ("mx_fp", 8, "mx_fp_8") in records

    def test_awq_activation_stats_use_balance_layer_input_for_gated_mlp(self):
        """up_proj -> down_proj stats must use down_proj input, not raw up_proj output."""
        import torch.nn as nn

        from auto_round.algorithms.transforms.awq.base import AWQTransform
        from auto_round.algorithms.transforms.awq.config import AWQConfig
        from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping

        class TinyGatedBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_proj = nn.Linear(4, 6, bias=False)
                self.gate_proj = nn.Linear(4, 6, bias=False)
                self.down_proj = nn.Linear(6, 4, bias=False)

            def forward(self, x):
                return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = TinyGatedBlock()

            def forward(self, x):
                return self.block(x)

        torch.manual_seed(0)
        model = TinyModel()
        transform = AWQTransform(
            AWQConfig(bits=4, group_size=2, sym=True, data_type="int", apply_clip=True, clip_n_sample_token=32)
        )
        mapping = ResolvedMapping(
            smooth_name="block.up_proj",
            smooth_layer=model.block.up_proj,
            balance_names=["block.down_proj"],
            balance_layers=[model.block.down_proj],
            parent_name="block",
            parent=model.block,
        )
        transform._block_mappings = {"block": [mapping]}

        x = torch.randn(2, 3, 4)
        handles = transform._register_awq_hooks(model, model.block, "block")
        try:
            model(x)
        finally:
            for handle in handles:
                handle.remove()

        expected = torch.nn.functional.silu(model.block.gate_proj(x)) * model.block.up_proj(x)
        expected_feat = expected.detach().flatten(0, -2)
        raw_up_feat = model.block.up_proj(x).detach().flatten(0, -2)

        act_sum, act_count = transform._activation_stats["block.up_proj"]
        assert torch.allclose(act_sum, expected_feat.abs().sum(dim=0))
        assert act_count == expected_feat.shape[0]
        assert torch.allclose(transform._clip_input_feat["block.up_proj"], expected_feat.float().cpu())
        assert not torch.allclose(act_sum, raw_up_feat.abs().sum(dim=0))

    def test_awq_parent_cache_recursively_detaches_tensor_containers(self):
        """Parent replay cache should not retain tensors inside list/dict containers."""
        import torch.nn as nn

        from auto_round.algorithms.transforms.awq.base import AWQTransform
        from auto_round.algorithms.transforms.awq.config import AWQConfig
        from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping

        class ContainerParent(nn.Module):
            def __init__(self):
                super().__init__()
                self.smooth = nn.Linear(4, 4, bias=False)
                self.balance = nn.Linear(4, 4, bias=False)

            def forward(self, items=None, payload=None):
                return self.balance(items[0]) + self.balance(payload["x"])

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = ContainerParent()

            def forward(self, items=None, payload=None):
                return self.block(items=items, payload=payload)

        model = TinyModel()
        transform = AWQTransform(AWQConfig(bits=4, group_size=4, sym=True, data_type="int"))
        mapping = ResolvedMapping(
            smooth_name="block.smooth",
            smooth_layer=model.block.smooth,
            balance_names=["block.balance"],
            balance_layers=[model.block.balance],
            parent_name="block",
            parent=model.block,
        )
        transform._block_mappings = {"block": [mapping]}

        items = [torch.randn(1, 4, requires_grad=True)]
        payload = {"x": torch.randn(1, 4, requires_grad=True)}
        handles = transform._register_awq_hooks(model, model.block, "block")
        try:
            model(items=items, payload=payload)
        finally:
            for handle in handles:
                handle.remove()

        _, cached_kwargs = transform._parent_args_cache[model.block][0]
        assert cached_kwargs["items"] is not items
        assert cached_kwargs["payload"] is not payload
        assert cached_kwargs["items"][0].device.type == "cpu"
        assert cached_kwargs["payload"]["x"].device.type == "cpu"
        assert cached_kwargs["items"][0].requires_grad is False
        assert cached_kwargs["payload"]["x"].requires_grad is False

    def test_awq_parent_replay_microbatch_matches_full_batch(self):
        """AWQ parent replay microbatching should preserve exact parent-output loss."""
        import torch.nn as nn

        from auto_round.algorithms.transforms.awq.base import AWQTransform
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        class RecordingParent(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4, bias=False)
                self.batch_sizes = []

            def forward(self, hidden_states, *, position_ids=None, cache_position=None, payload=None):
                self.batch_sizes.append(hidden_states.shape[0])
                assert position_ids is not None and position_ids.shape[0] == hidden_states.shape[0]
                assert cache_position is not None and cache_position.shape == (3,)
                add = payload["add"] if payload is not None else 0
                return self.proj(hidden_states + add)

        torch.manual_seed(0)
        parent = RecordingParent()
        hidden_states = torch.randn(5, 3, 4)
        position_ids = torch.arange(15).reshape(5, 3)
        cache_position = torch.arange(3)
        payload = {"add": torch.randn(5, 3, 4)}
        kwargs_list = [((hidden_states,), {"position_ids": position_ids, "cache_position": cache_position, "payload": payload})]

        full = AWQTransform(AWQConfig(bits=4, group_size=4, sym=True, data_type="int"))
        micro = AWQTransform(AWQConfig(bits=4, group_size=4, sym=True, data_type="int", smooth_batch_size=2))

        full_outputs = full._run_parent_samples(parent, kwargs_list)
        parent.batch_sizes.clear()
        micro_outputs = micro._run_parent_samples(parent, kwargs_list)

        assert parent.batch_sizes == [2, 2, 1]
        assert len(full_outputs) == 1
        assert len(micro_outputs) == 3
        assert torch.allclose(torch.cat(micro_outputs, dim=0), full_outputs[0])

        ref_outputs = [out + 0.125 for out in micro_outputs]
        parent.batch_sizes.clear()
        streamed_loss = micro._compute_parent_loss(parent, kwargs_list, ref_outputs)
        expected_loss = micro._compute_loss(ref_outputs, micro_outputs)

        assert parent.batch_sizes == [2, 2, 1]
        assert streamed_loss == expected_loss

    def test_awq_smooth_seqlen_truncates_parent_forward_inputs(self):
        """smooth_seqlen replay cache should truncate matching sequence dimensions consistently."""
        from auto_round.algorithms.transforms.awq.base import _truncate_args_kwargs

        hidden_states = torch.zeros(1, 16, 8)
        position_ids = torch.arange(16).reshape(1, 16)
        attention_mask = torch.zeros(1, 1, 16, 16)
        rotary = (torch.zeros(1, 16, 8), torch.ones(1, 16, 8))

        args, kwargs = _truncate_args_kwargs(
            (hidden_states,),
            {"position_ids": position_ids, "attention_mask": attention_mask, "position_embeddings": rotary},
            seqlen=4,
        )

        assert args[0].shape == (1, 4, 8)
        assert kwargs["position_ids"].shape == (1, 4)
        assert kwargs["attention_mask"].shape == (1, 1, 4, 4)
        assert kwargs["position_embeddings"][0].shape == (1, 4, 8)
        assert kwargs["position_embeddings"][1].shape == (1, 4, 8)

    def test_awq_moe_quantized_layers_check(self, tiny_qwen_moe_model_path):
        """AWQ on MoE: expert layers should be quantized, gates/routers stay FP."""
        ar = AutoRound(
            tiny_qwen_moe_model_path,
            scheme="W4A16",
            alg_configs=AWQConfig(n_grid=1),
            n_grid=1,
            nsamples=2,
            seqlen=8,
            batch_size=2,
        )
        model, layer_config = ar.quantize()
        assert model is not None
        assert len(layer_config) > 0

        q4_layers = {k for k, v in layer_config.items() if v["bits"] == 4}
        fp_layers = {k for k, v in layer_config.items() if v["bits"] >= 16}
        other_layers = {k: v["bits"] for k, v in layer_config.items() if v["bits"] != 4 and v["bits"] < 16}

        # Tiny Qwen MoE: mlp.gate is a TopKRouter (not Linear) so it's not in layer_config.
        # Only mlp.shared_expert_gate (a Linear) stays FP → 1 FP gate layer per block.
        assert len(other_layers) == 0, f"Unexpected bit widths: {other_layers}"
        n_layers = model.config.num_hidden_layers
        assert len(fp_layers) == n_layers, (
            f"Expected {n_layers} FP gate layers (mlp.shared_expert_gate per block), "
            f"got {len(fp_layers)}: {sorted(fp_layers)}"
        )
        assert len(q4_layers) == len(layer_config) - len(
            fp_layers
        ), f"Expected {len(layer_config) - len(fp_layers)} W4 layers, got {len(q4_layers)}"

        for name in fp_layers:
            assert name.endswith("gate"), f"Unexpected FP layer: {name}"

    def test_awq_moe_save_quant_config(self, tiny_qwen_moe_model_path):
        """AWQ MoE: saved quantization_config should be consistent and loadable."""
        ar = AutoRound(
            tiny_qwen_moe_model_path,
            scheme="W4A16",
            alg_configs=AWQConfig(n_grid=1),
            n_grid=1,
            nsamples=2,
            seqlen=32,
            batch_size=2,
        )
        _, save_path = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        # Load and verify config
        config_path = os.path.join(save_path, "config.json")
        assert os.path.exists(config_path), f"config.json not found at {save_path}"

        with open(config_path, "r") as f:
            config_data = json.load(f)

        qconfig = config_data.get("quantization_config")
        assert qconfig is not None, "quantization_config missing from saved config.json"
        assert qconfig["bits"] == 4
        assert qconfig["group_size"] == 128
        assert "auto-round" in qconfig["quant_method"]


class TestAWQWeightClip:
    """AWQ weight-clip option (issue #1854).

    Validates the ``apply_clip`` AWQ preprocessing combined with downstream
    block quantizers, covering the two extensibility scenarios:
      - clip + RTN  (clip as the weight range, then round-to-nearest)
      - clip + SignRound (clip as initialization, then SignRound tuning)
    """

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_awq_clip_then_rtn(self, tiny_opt_model_path):
        """AWQ smooth+clip → RTN: produces a valid W4 model that can generate."""
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        ar = AutoRound(
            tiny_opt_model_path,
            alg_configs=[
                AWQConfig(
                    bits=4, group_size=128, sym=True, apply_clip=True, n_grid=2, clip_n_grid=2, clip_n_sample_token=8
                ),
                RTNConfig(disable_opt_rtn=True),
            ],
            nsamples=2,
            seqlen=8,
            batch_size=2,
        )
        model, layer_config = ar.quantize()

        assert model is not None
        assert len(layer_config) > 0
        for name, cfg in layer_config.items():
            assert cfg["bits"] == 4, f"Layer {name} expected bits=4, got {cfg['bits']}"

        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)
        output = generate_prompt(model, tokenizer, device="cpu")
        assert len(output) > 0, "Clipped model should produce non-empty output"

    def test_awq_clip_as_init_signround(self, tiny_opt_model_path):
        """clip_as_init: clip is kept on the model context and initializes SignRound's range."""
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        ar = AutoRound(
            tiny_opt_model_path,
            alg_configs=[
                AWQConfig(
                    bits=4,
                    group_size=128,
                    sym=True,
                    apply_clip=True,
                    clip_as_init=True,
                    n_grid=2,
                    clip_n_grid=2,
                    clip_n_sample_token=8,
                ),
                SignRoundConfig(iters=1),
            ],
            nsamples=2,
            seqlen=8,
            batch_size=2,
            enable_torch_compile=False,  # disable torch.compile to ensure CI pass, cannot reproduce in local env.
        )
        model, layer_config = ar.quantize()

        assert model is not None
        assert len(layer_config) > 0
        for name, cfg in layer_config.items():
            assert cfg["bits"] == 4, f"Layer {name} expected bits=4, got {cfg['bits']}"

        # The searched clip magnitudes are kept on the model context.
        clip_values = getattr(ar.model_context, "awq_clip_values", {})
        assert len(clip_values) > 0, "clip_as_init should record per-group clip values on the model context"

        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)
        output = generate_prompt(model, tokenizer, device="cpu")
        assert len(output) > 0, "clip_as_init model should produce non-empty output"


class TestAWQUseV2ScaleSearch:
    """Unit tests for AWQ's ``use_v2_scale_search`` detection and dispatch.

    The flag is True whenever the terminal block quantizer resolves to
    ``SignRoundV2Quantizer``. The per-data-type init_scale injection (int/mx/nv,
    sym only) is then handled by ``search_optimized_init_scale`` inside the QDQ
    path. Regression guard: the block-quantizer config does not expose
    ``_alg_cls``, so detection must go through the pipeline registry, not an
    ``_alg_cls`` string comparison (which always evaluated False before the fix).
    """

    @staticmethod
    def _make_compressor(block_config):
        import types

        return types.SimpleNamespace(quantize_config=block_config, alg_configs=[block_config])

    @staticmethod
    def _awq_transform():
        from auto_round.algorithms.transforms.awq.base import AWQTransform
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        return AWQTransform(AWQConfig(n_grid=1, apply_smooth=True))

    @staticmethod
    def _signroundv2_config(data_type=None):
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
        from auto_round.algorithms.registry import normalize_algorithm_config

        cfg = normalize_algorithm_config(SignRoundConfig(iters=1, enable_alg_ext=True))
        assert type(cfg).__name__ == "SignRoundV2Config"
        if data_type is not None:
            cfg.data_type = data_type
        return cfg

    # TODO weiwei monkey patch,recover better
    # def test_block_v2(self):
    #     """An RTN block quantizer must NOT be detected as V2."""
    #     from auto_round.algorithms.quantization.rtn.config import RTNConfig
    #
    #     q = self._awq_transform()
    #     compressor = self._make_compressor(RTNConfig())
    #     assert q._qdq_tool._block_quantizer_is_signroundv2(compressor) is False
    #
    #     compressor = self._make_compressor(self._signroundv2_config(data_type="mx_fp"))
    #     assert q._qdq_tool._block_quantizer_is_signroundv2(compressor) is True

    def test_use_v2_false_for_non_v2_block(self):
        """Gate is False when the block is not SignRoundV2, regardless of dtype."""
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        q = self._awq_transform()
        block = SignRoundConfig(iters=1)
        block.data_type = "mx_fp"
        compressor = self._make_compressor(block)
        assert q._qdq_tool._block_quantizer_is_signroundv2(compressor) is False

    def test_init_scale_dispatch_by_data_type(self):
        """``search_optimized_init_scale`` injects only for sym int/mx/nv."""
        import torch

        from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, search_optimized_init_scale

        for dt, gs in (("mx_fp4", 32),):
            weight = torch.randn(32, 64)
            weight_reshape, _, _ = reshape_pad_tensor_by_group_size(weight, gs)
            init_scale = search_optimized_init_scale(weight_reshape, dt, 4, None)
            assert init_scale is not None, dt
            assert init_scale.shape[0] == weight_reshape.shape[0]

        # Scalar imatrix sentinels mean uniform importance and must not enter
        # the tensor reshape path.
        scalar_init_scale = search_optimized_init_scale(weight_reshape, "mx_fp4", 4, 1.0)
        assert scalar_init_scale is not None
        assert scalar_init_scale.shape == init_scale.shape

        # asym int and *_dq are not part of the optimized init-scale path.
        assert search_optimized_init_scale(torch.randn(4, 128), "int_asym", 4, None) is None
        assert search_optimized_init_scale(torch.randn(4, 128), "int_sym_dq", 4, None) is None

    def test_nvfp4_opt_rtn_accepts_uniform_imatrix_sentinel(self):
        """AWQ's SignRound-V1 QDQ may call optimized RTN without a collected imatrix."""
        import torch

        from auto_round.data_type.nvfp import opt_rtn_fast_nvfp4

        weight = torch.randn(8, 32)
        qdq_weight, _, _ = opt_rtn_fast_nvfp4(weight, bits=4, group_size=16, imatrix=1.0)

        assert qdq_weight.shape == weight.shape
        assert torch.isfinite(qdq_weight).all()
