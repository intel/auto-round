# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""CPU-only grouped NVFP4 regressions using the actual Transformers class."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

from auto_round import AutoRound
from auto_round.export.formats.autoround.float import pack_layer as pack_autoround
from auto_round.export.formats.autoround.qlinear_fp import GroupedQuantLinear
from auto_round.export.formats.compressed_tensors.float import pack_layer as pack_ct
from auto_round.inference.convert_model import _create_quant_layer, _get_layer_features, _replace_by_quant_layers
from auto_round.modeling.grouped_linear import DeepseekV4GroupedLinear, grouped_linear_forward, validate_grouped_nvfp4
from auto_round.schemes import preset_name_to_scheme
from auto_round.utils import SUPPORTED_LAYER_TYPES
from auto_round.wrapper import WrapperLinear, WrapperWALayer, wrapper_block


def reference(x, weight, groups):
    # Independent per-group matmuls, not one dense linear.
    return torch.stack([x[..., g, :] @ w.T for g, w in enumerate(weight.chunk(groups))], dim=-2)


def decode(layer):
    packed = layer.weight_packed
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    lut = packed.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
    return lut[codes] * layer.weight_scale.float().repeat_interleave(16, 1) / layer.weight_global_scale


@unittest.skipIf(DeepseekV4GroupedLinear is None, "Transformers DeepSeek V4 is unavailable")
class TestGroupedNVFP4(unittest.TestCase):
    def layer(self):
        torch.manual_seed(17)
        layer = DeepseekV4GroupedLinear(32, 24, 3)
        scheme = preset_name_to_scheme("NVFP4")
        for name, value in vars(scheme).items():
            setattr(layer, name, value)
        layer.scale_dtype = torch.float32
        layer.act_max = torch.tensor(3.0)
        layer.global_name = "proj"
        return layer

    def wrapper(self, layer):
        return WrapperLinear(
            layer,
            enable_minmax_tuning=False,
            enable_round_tuning=False,
            enable_torch_compile=False,
            disable_opt_rtn=True,
            iters=0,
        )

    def test_actual_class_selected_and_wrapper_preserves_groups(self):
        layer = self.layer()
        self.assertIn(type(layer), SUPPORTED_LAYER_TYPES)
        self.assertEqual(_get_layer_features(layer), (32, 24))
        x = torch.randn(1, 4, 3, 32)
        torch.testing.assert_close(layer(x), reference(x, layer.weight, 3))
        model = torch.nn.Module()
        model.proj = layer
        selected, skipped = wrapper_block(
            model, False, False, enable_torch_compile=False, enable_round_tuning=False, iters=0, disable_opt_rtn=True
        )
        self.assertEqual(selected, ["proj"])
        self.assertEqual(skipped, [])
        wrapper = model.proj
        self.assertEqual(wrapper.n_groups, 3)
        for shape in ((3, 32), (1, 4, 3, 32), (2, 2, 3, 32)):
            x = torch.randn(shape)
            w, _, _ = wrapper._qdq_weight(wrapper.value, wrapper.min_scale, wrapper.max_scale)
            a, _, _ = wrapper._qdq_act(x, act_max=layer.act_max)
            torch.testing.assert_close(wrapper(x), reference(a, w, 3))
        restored = wrapper.unwrapper({})
        self.assertIsInstance(restored, WrapperWALayer)
        self.assertEqual(restored.n_groups, 3)
        self.assertIs(type(restored.orig_layer), DeepseekV4GroupedLinear)
        x = torch.randn(1, 4, 3, 32)
        a, _, _ = wrapper._qdq_act(x, act_max=layer.act_max)
        torch.testing.assert_close(restored(x), reference(a, restored.weight, 3))

    def test_rtn_packed_uint8_decode_and_grouped_export_forward(self):
        for pack in (pack_ct, pack_autoround):
            with self.subTest(packer=pack.__module__):
                layer = self.layer()
                model = torch.nn.Module()
                model.proj = self.wrapper(layer).unwrapper({})
                expected_weight = model.proj.weight.detach().clone()
                x = torch.randn(1, 4, 3, 32)
                expected = model.proj(x)
                if pack is pack_autoround:
                    pack("proj", model, "auto_round", device="cpu")
                else:
                    pack("proj", model, device="cpu")
                q = model.proj
                self.assertIsInstance(q, GroupedQuantLinear)
                self.assertEqual(q.n_groups, 3)
                self.assertEqual(q.weight_packed.dtype, torch.uint8)
                self.assertEqual(q.weight_packed.shape, (24, 16))
                self.assertEqual(q.weight_scale.shape, (24, 2))
                torch.testing.assert_close(decode(q), expected_weight, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(q(x), expected, rtol=1e-5, atol=1e-6)
                scheme = preset_name_to_scheme("NVFP4")
                reloaded = _create_quant_layer(self.layer(), "auto_round:torch_nvfp4", vars(scheme), 32, 24)
                reloaded.load_state_dict(q.state_dict())
                torch.testing.assert_close(reloaded(x), expected, rtol=1e-5, atol=1e-6)
                isolated = torch.zeros_like(x)
                isolated[..., 1, :] = x[..., 1, :]
                result = q(isolated)
                self.assertTrue(torch.equal(result[..., 0, :], torch.zeros_like(result[..., 0, :])))
                self.assertTrue(torch.equal(result[..., 2, :], torch.zeros_like(result[..., 2, :])))

    def test_weight_only_rtn_keeps_actual_grouped_forward(self):
        for pack in (pack_ct, pack_autoround):
            layer = self.layer()
            layer.act_bits = 16
            layer.act_data_type = "fp"
            restored = self.wrapper(layer).unwrapper({})
            self.assertIs(type(restored), DeepseekV4GroupedLinear)
            x = torch.randn(1, 4, 3, 32)
            expected = reference(x, restored.weight, 3)
            torch.testing.assert_close(restored(x), expected)
            model = torch.nn.Module()
            model.proj = restored
            if pack is pack_autoround:
                pack("proj", model, "auto_round", device="cpu")
            else:
                pack("proj", model, device="cpu")
            torch.testing.assert_close(model.proj(x), expected)
            scheme = preset_name_to_scheme("NVFP4")
            scheme.act_bits = 16
            reloaded = _create_quant_layer(self.layer(), "auto_round:torch_nvfp4", vars(scheme), 32, 24)
            reloaded.load_state_dict(model.proj.state_dict())
            torch.testing.assert_close(reloaded(x), expected)

    def test_unsupported_grouped_formats_fail_closed(self):
        layer = self.layer()
        with self.assertRaisesRegex(ValueError, "only NVFP4"):
            validate_grouped_nvfp4(layer, 4, 32, "mx_fp")
        with self.assertRaisesRegex(ValueError, "grouped reference backend"):
            _create_quant_layer(layer, "auto_round:fake", vars(preset_name_to_scheme("NVFP4")), 32, 24)
        layer.act_bits = 8
        with self.assertRaisesRegex(ValueError, "static NVFP4 activations"):
            validate_grouped_nvfp4(layer, 4, 16, "nv_fp")
        scheme = preset_name_to_scheme("NVFP4")
        scheme.act_bits = 8
        with self.assertRaisesRegex(ValueError, "static NVFP4 activations"):
            _create_quant_layer(self.layer(), "auto_round:torch_nvfp4", vars(scheme), 32, 24)

    def test_automatic_backend_preserves_groups_and_dense_backend_cache(self):
        from auto_round.experimental.qmodules.nvfp4 import NVFP4QuantLinear

        for act_bits in (4, 16):
            source = torch.nn.Module()
            layer = self.layer()
            layer.act_bits = act_bits
            source.proj = self.wrapper(layer).unwrapper({})
            pack_autoround("proj", source, "auto_round", device="cpu")
            model = torch.nn.Module()
            model.dense = torch.nn.Linear(32, 24, bias=False)
            model.proj = self.layer()
            model.dense_after = torch.nn.Linear(32, 24, bias=False)
            scheme = preset_name_to_scheme("NVFP4")
            scheme.act_bits = act_bits
            config = scheme
            used = _replace_by_quant_layers(
                model, {name: config for name in ("dense", "proj", "dense_after")}, "auto", "cpu", "auto_round"
            )
            self.assertIn("auto_round:torch_nvfp4", used)
            self.assertIsInstance(model.proj, NVFP4QuantLinear)
            self.assertNotIsInstance(model.dense, NVFP4QuantLinear)
            self.assertIs(type(model.dense_after), type(model.dense))
            model.proj.load_state_dict(source.proj.state_dict())
            x = torch.randn(2, 3, 32)
            torch.testing.assert_close(model.proj(x), source.proj(x))

    def test_rtn_uses_calibration_max_not_current_input_max(self):
        from auto_round.data_type.nvfp import calculate_gparam
        from auto_round.experimental.qmodules.nvfp4 import _nv_fp4_with_static_gs

        layer = self.layer()
        layer.act_max = torch.tensor(19.0)
        restored = self.wrapper(layer).unwrapper({})
        x = torch.linspace(-1, 1, 96).reshape(3, 32)
        expected_act, _, _ = _nv_fp4_with_static_gs(x, calculate_gparam(layer.act_max))
        actual_act = restored.qdq_input(x.clone())
        torch.testing.assert_close(actual_act, expected_act)
        dynamic_act, _, _ = _nv_fp4_with_static_gs(x, calculate_gparam(x))
        self.assertFalse(torch.equal(actual_act, dynamic_act))

    def test_rtn_explicit_global_scale_matches_both_exports(self):
        from auto_round.data_type.nvfp import calculate_gparam
        from auto_round.experimental.qmodules.nvfp4 import _nv_fp4_with_static_gs

        for pack in (pack_ct, pack_autoround):
            layer = self.layer()
            layer.input_global_scale = calculate_gparam(torch.tensor(19.0))
            wrapper = self.wrapper(layer)
            x = torch.linspace(-1, 1, 96).reshape(3, 32)
            expected_act, _, _ = _nv_fp4_with_static_gs(x, layer.input_global_scale)
            actual_act, _, _ = wrapper._qdq_act(x.clone(), act_max=layer.act_max)
            torch.testing.assert_close(actual_act, expected_act)
            model = torch.nn.Module()
            model.proj = wrapper.unwrapper({})
            torch.testing.assert_close(model.proj.qdq_input(x.clone()), expected_act)
            expected = model.proj(x.clone())
            if pack is pack_autoround:
                pack("proj", model, "auto_round", device="cpu")
            else:
                pack("proj", model, device="cpu")
            torch.testing.assert_close(model.proj(x), expected)

    def test_shape_guards_and_noncontiguous_empty_inputs(self):
        weight = self.layer().weight
        for x in (torch.randn(6, 16), torch.randn(2, 3, 16), torch.randn(96)):
            with self.assertRaises(ValueError):
                grouped_linear_forward(x, weight, 3)
        for groups in (0, -1, True):
            with self.assertRaises(ValueError):
                grouped_linear_forward(torch.randn(3, 32), weight, groups)
        x = torch.randn(2, 3, 5, 32).transpose(1, 2)
        torch.testing.assert_close(grouped_linear_forward(x, weight, 3), reference(x, weight, 3))
        self.assertEqual(grouped_linear_forward(torch.empty(0, 3, 32), weight, 3).shape, (0, 3, 8))
        layer = self.layer()
        layer.weight = torch.nn.Parameter(torch.randn(12, 64))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            validate_grouped_nvfp4(layer, 4, 16, "nv_fp")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            GroupedQuantLinear(4, 16, 32, 24, False, n_groups=5, data_type="nv_fp", act_bits=16)

    def test_packed_and_reloaded_compute_dtype(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            layer = self.layer().to(dtype)
            model = torch.nn.Module()
            model.proj = self.wrapper(layer).unwrapper({})
            pack_autoround("proj", model, "auto_round", device="cpu")
            q = model.proj
            x = torch.randn(2, 3, 32, dtype=dtype)
            scheme = preset_name_to_scheme("NVFP4")
            reloaded = _create_quant_layer(self.layer().to(dtype), "auto_round:torch_nvfp4", vars(scheme), 32, 24)
            reloaded.load_state_dict(q.state_dict())
            self.assertEqual(q(x).dtype, dtype)
            self.assertEqual(reloaded(x).dtype, dtype)
            torch.testing.assert_close(reloaded(x), q(x))

    def test_actual_tiny_model_rtn_and_export(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM, PreTrainedTokenizerFast

        from auto_round.experimental.qmodules.nvfp4 import _nv_fp4_with_static_gs

        torch.manual_seed(23)
        config = DeepseekV4Config(
            vocab_size=64,
            hidden_size=32,
            moe_intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            head_dim=16,
            q_lora_rank=16,
            o_groups=2,
            o_lora_rank=16,
            n_routed_experts=2,
            num_experts_per_tok=2,
            index_n_heads=2,
            index_head_dim=16,
            index_topk=2,
            hc_mult=2,
            hc_sinkhorn_iters=2,
            num_nextn_predict_layers=0,
            max_position_embeddings=32,
            sliding_window=8,
            layer_types=["sliding_attention"],
            mlp_layer_types=["hash_moe"],
            partial_rotary_factor=0.5,
            use_cache=False,
        )
        config._attn_implementation = "eager"
        model = DeepseekV4ForCausalLM(config).eval()
        name = "model.layers.0.self_attn.o_a_proj"
        original = model.model.layers[0].self_attn.o_a_proj.weight.detach().clone()
        tokens = torch.tensor([[2, 3, 4, 5]])
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "test": 2}, unk_token="<unk>")),
            unk_token="<unk>",
            eos_token="<eos>",
            pad_token="<eos>",
        )
        dataset = torch.utils.data.DataLoader([{"input_ids": tokens[0]}], batch_size=1)
        # MLP stays full precision: tiny hash-MoE calibration does not reliably
        # populate every expert's act_max. This exercises real attention RTN.
        ar = AutoRound(
            model,
            tokenizer=tokenizer,
            scheme="NVFP4",
            iters=0,
            disable_opt_rtn=True,
            disable_model_free=True,
            low_cpu_mem_usage=False,
            batch_size=1,
            ignore_layers="model.layers.0.mlp",
            nsamples=1,
            seqlen=4,
            dataset=dataset,
            device_map="cpu",
            enable_torch_compile=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            quantized, output = ar.quantize_and_save(directory, format="auto_round:llm_compressor")
            self.assertEqual(ar.layer_config[name]["bits"], 4)
            q = quantized.model.layers[0].self_attn.o_a_proj
            self.assertIsInstance(q, GroupedQuantLinear)
            self.assertEqual(q.n_groups, 2)
            self.assertEqual(q.weight_packed.dtype, torch.uint8)
            self.assertFalse(torch.equal(decode(q), original))
            self.assertLess(torch.linalg.vector_norm(decode(q) - original) / torch.linalg.vector_norm(original), 0.2)
            x = torch.randn(1, 4, 2, 32)
            a, _, _ = _nv_fp4_with_static_gs(x, q.input_global_scale)
            torch.testing.assert_close(q(x), reference(a, decode(q), 2))
            stored = {}
            for shard in Path(output).glob("*.safetensors"):
                with safe_open(str(shard), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if key.startswith(name + "."):
                            stored[key] = handle.get_tensor(key)
            self.assertEqual(stored[name + ".weight_packed"].dtype, torch.uint8)
            self.assertEqual(stored[name + ".weight_scale"].shape, (32, 2))
            self.assertEqual(stored[name + ".weight_global_scale"].numel(), 1)
            self.assertEqual(stored[name + ".input_global_scale"].numel(), 1)
            self.assertNotIn(name + ".weight", stored)
            self.assertFalse(any("weight_scale_inv" in key for key in stored))
            torch.testing.assert_close(stored[name + ".weight_packed"], q.weight_packed)


if __name__ == "__main__":
    unittest.main()
