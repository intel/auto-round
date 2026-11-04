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

import copy
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from auto_round import ParallelConfig


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
@pytest.mark.parametrize("axis", ["tp_size", "pp_size"])
def test_parallel_sizes_are_positive_integers(axis, value):
    with pytest.raises(ValueError, match=axis):
        ParallelConfig(**{axis: value})


def test_parallel_policy_validation(monkeypatch):
    assert ParallelConfig.from_value(None) == ParallelConfig()
    assert ParallelConfig.from_value({"tp_size": 2}).tp_size == 2
    with pytest.raises(TypeError, match="dp_size"):
        ParallelConfig(dp_size=2)
    with pytest.raises(NotImplementedError, match="Hybrid"):
        ParallelConfig(tp_size=2, pp_size=2)
    with pytest.raises(TypeError, match="parallel_config"):
        ParallelConfig.from_value("tp")
    with pytest.raises(ValueError, match="device_map"):
        ParallelConfig(tp_size=2).resolve_device_map({"": "cuda:0"})
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="torchrun"):
        ParallelConfig(tp_size=2).resolve_device_map(None)


def test_native_distributed_config_and_cli():
    from auto_round.cli.parser import build_quantize_parser

    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "DistributedConfig"):
        pytest.skip("requires native Transformers DistributedConfig")
    native = ParallelConfig(tp_size=2).to_transformers_config()
    assert native.tp_size == 2
    assert native.tp_plan == "auto"
    args = build_quantize_parser().parse_args(["local-model", "--tp_size", "2"])
    assert (args.tp_size, args.pp_size) == (2, 1)
    assert not hasattr(args, "dp_size")


@pytest.mark.parametrize("scheme", ["INT8_W8A8"])
def test_unsupported_tp_scheme_fails_before_loading(scheme):
    from auto_round import AutoRound

    with (
        patch.object(ParallelConfig, "resolve_device_map", return_value="cpu"),
        pytest.raises(NotImplementedError, match="activation quantization"),
    ):
        AutoRound(torch.nn.Linear(4, 4), scheme=scheme, alg_configs="qad", parallel_config={"tp_size": 2})


def test_unsupported_tp_algorithm_fails_before_loading():
    from auto_round import AutoRound
    from auto_round.algorithms.transforms.awq.config import AWQConfig

    with (
        patch.object(ParallelConfig, "resolve_device_map", return_value="cpu"),
        pytest.raises(NotImplementedError, match="AWQ TP clipping"),
    ):
        AutoRound(torch.nn.Linear(4, 4), alg_configs=AWQConfig(apply_clip=True), parallel_config={"tp_size": 2})


def test_multistage_tp_validates_later_algorithm_before_loading():
    from auto_round import AutoRound
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

    with (
        patch.object(ParallelConfig, "resolve_device_map", return_value="cpu"),
        pytest.raises(NotImplementedError, match="enable_lfq"),
    ):
        AutoRound(
            torch.nn.Linear(4, 4),
            alg_configs=[SignRoundConfig(), SignRoundConfig(enable_lfq=True)],
            parallel_config={"tp_size": 2},
        )


@pytest.mark.parametrize("data_type,group_size", [("mx_fp", 16), ("nv_fp", 32)])
def test_invalid_fp4_block_size(data_type, group_size):
    from auto_round.parallel.tp.policy import validate_tp_quantization
    from auto_round.schemes import QuantizationScheme

    with pytest.raises(NotImplementedError, match="group_size"):
        validate_tp_quantization(QuantizationScheme(data_type=data_type, group_size=group_size, act_bits=16))


@pytest.mark.parametrize("scheme", ["MXFP4", "MXFP4_RCEIL", "NVFP4"])
def test_tp_fp4_activation_policy(scheme):
    from auto_round.parallel.tp.policy import validate_tp_quantization
    from auto_round.schemes import preset_name_to_scheme

    config = preset_name_to_scheme(scheme).copy()
    validate_tp_quantization(config)
    config.act_group_size //= 2
    with pytest.raises(NotImplementedError, match="act_group_size"):
        validate_tp_quantization(config)


@pytest.mark.parametrize("override", [{"act_sym": False}, {"act_dynamic": False}, {"act_bits": 2}])
def test_invalid_tp_activation_options(override):
    from auto_round.parallel.tp.policy import validate_tp_quantization
    from auto_round.schemes import preset_name_to_scheme

    scheme = preset_name_to_scheme("MXFP4").copy()
    scheme.update_from_dict(override)
    with pytest.raises(NotImplementedError, match="requires act_bits"):
        validate_tp_quantization(scheme)


def _configure_linear(layer):
    for key, value in dict(bits=4, group_size=2, sym=True, data_type="int", act_bits=16).items():
        setattr(layer, key, value)
    layer.scale_dtype = torch.float32
    return layer.requires_grad_(False)


def _check_qdq_numerics(rank, mesh, symmetric, data_type="int", group_size=2, width=8, act_type=None, act_bits=4):
    from transformers.distributed.tensor_parallel import ColwiseParallel, RowwiseParallel

    from auto_round.parallel.tp.qdq import TensorParallelQDQLinear
    from auto_round.wrapper import WrapperLinear, WrapperWALayer

    torch.manual_seed(7)
    col = _configure_linear(torch.nn.Linear(width, width))
    row = _configure_linear(torch.nn.Linear(width, 4))
    col.sym = row.sym = symmetric
    col.data_type = row.data_type = data_type
    col.group_size = row.group_size = group_size
    if act_type:
        for layer in (col, row):
            layer.act_bits = act_bits
            layer.act_group_size = group_size
            layer.act_data_type = act_type
            layer.act_sym = True
            layer.act_dynamic = True
            if act_type == "nv_fp4_with_static_gs":
                layer.act_max = torch.tensor([20.0])
    # Distinct shard maxima catch accidentally computing NVFP global scales
    # independently per rank, for both output- and input-axis sharding.
    if data_type != "int":
        with torch.no_grad():
            col.weight[width // 2 :].mul_(17)
            row.weight[:, width // 2 :].mul_(17)
    reference_col = WrapperLinear(copy.deepcopy(col), enable_torch_compile=False)
    reference_row = WrapperLinear(copy.deepcopy(row), enable_torch_compile=False)
    for layer, style in ((col, ColwiseParallel(use_local_output=not bool(act_type))), (row, RowwiseParallel())):
        style.shard_param(layer, "weight", mesh)
        style.shard_param(layer, "bias", mesh)
        style.install_forward(layer, mesh)
    col_before = col.weight.to_local().clone()
    if act_type == "nv_fp4_with_static_gs" and rank == 0:
        col.act_max = row.act_max = torch.tensor([3.0])
    wrapped_col = TensorParallelQDQLinear(col, enable_torch_compile=False)
    wrapped_row = TensorParallelQDQLinear(row, enable_torch_compile=False)
    if act_type:
        for wrapper in (reference_col, reference_row, wrapped_col, wrapped_row):
            wrapper.act_max_scale.data.fill_(0.8)
    x = torch.randn(2, 3, width)
    actual_x = x.clone().requires_grad_(bool(act_type))
    expected_x = x.clone().requires_grad_(bool(act_type))
    expected = reference_row(reference_col(expected_x))
    actual = wrapped_row(wrapped_col(actual_x))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    # FP4 cases amplify half the weights by 17; reduction order differs in TP.
    torch.testing.assert_close(actual_x.grad, expected_x.grad, rtol=2e-4, atol=2e-5)
    grad_tolerance = dict(rtol=2e-4, atol=1e-4) if data_type != "int" else dict(rtol=3e-5, atol=3e-6)
    for key, parameter in wrapped_col.params.items():
        full_grad = reference_col.params[key].grad
        if full_grad is not None:
            expected_grad = full_grad if key.startswith("act_") else full_grad.chunk(2, dim=0)[rank]
            torch.testing.assert_close(parameter.grad, expected_grad, **grad_tolerance)
    for key, parameter in wrapped_row.params.items():
        full_grad = reference_row.params[key].grad
        if full_grad is not None:
            expected_grad = (
                full_grad
                if key.startswith("act_")
                else full_grad.reshape(4, width // group_size, -1).chunk(2, dim=1)[rank].reshape_as(parameter)
            )
            torch.testing.assert_close(parameter.grad, expected_grad, **grad_tolerance)
    torch.testing.assert_close(col.weight.to_local(), col_before)
    result_col = wrapped_col.unwrapper({key: value.detach() for key, value in wrapped_col.params.items()})
    result_row = wrapped_row.unwrapper({key: value.detach() for key, value in wrapped_row.params.items()})
    ref_col = reference_col.unwrapper({key: value.detach() for key, value in reference_col.params.items()})
    ref_row = reference_row.unwrapper({key: value.detach() for key, value in reference_row.params.items()})
    with torch.no_grad():
        torch.testing.assert_close(result_row(result_col(x)), ref_row(ref_col(x)), rtol=2e-5, atol=2e-6)
    if act_type:
        assert isinstance(result_col, WrapperWALayer)
        assert isinstance(result_row, WrapperWALayer)
        ref_col, ref_row = ref_col.orig_layer, ref_row.orig_layer
    for actual_layer, reference in ((col, ref_col), (row, ref_row)):
        torch.testing.assert_close(actual_layer.weight.full_tensor(), reference.weight)
        torch.testing.assert_close(actual_layer.scale.full_tensor(), reference.scale)
        if not symmetric:
            torch.testing.assert_close(actual_layer.zp.full_tensor(), reference.zp)
    if data_type != "int":
        from auto_round.parallel.tp.runtime import _gather_cpu_tensor

        if data_type == "nv_fp":
            torch.testing.assert_close(col.weight_global_scale, reference_col.weight_global_scale)
            torch.testing.assert_close(row.weight_global_scale, reference_row.weight_global_scale)
        # Tiny chunks force multiple transfers, including on the input axis.
        from torch.distributed.tensor import DTensor

        with patch.object(DTensor, "full_tensor", side_effect=AssertionError("GPU gather is forbidden")):
            full = _gather_cpu_tensor(row.weight, chunk_bytes=37)
        if rank == 0:
            torch.testing.assert_close(full, ref_row.weight)
        else:
            assert full is None
        invalid = _configure_linear(torch.nn.Linear(group_size * 3, 4, bias=False))
        invalid.group_size, invalid.data_type = group_size, data_type
        style = RowwiseParallel()
        style.shard_param(invalid, "weight", mesh)
        style.install_forward(invalid, mesh)
        with pytest.raises(ValueError, match="quantization group"):
            TensorParallelQDQLinear(invalid, enable_torch_compile=False)
        return
    row.group_size = -1
    with pytest.raises(ValueError, match="quantization group"):
        TensorParallelQDQLinear(row, enable_torch_compile=False)

    from auto_round.algorithms.composer import AlgorithmComposer

    block = torch.nn.Sequential(TensorParallelQDQLinear(col, enable_torch_compile=False))
    composer = object.__new__(AlgorithmComposer)

    def fail():
        raise RuntimeError("abort TP tuning")

    with pytest.raises(RuntimeError, match="abort TP tuning"):
        composer._run_tuning(block, fail)
    assert block[0] is col
    scalar_layer = _configure_linear(torch.nn.Linear(2, 2, bias=False))
    scalar_layer.sym = False
    scalar_reference = WrapperLinear(copy.deepcopy(scalar_layer), enable_torch_compile=False).unwrapper({})
    scalar_style = ColwiseParallel()
    scalar_style.shard_param(scalar_layer, "weight", mesh)
    scalar_style.install_forward(scalar_layer, mesh)
    scalar_result = TensorParallelQDQLinear(scalar_layer, enable_torch_compile=False).unwrapper({})
    torch.testing.assert_close(scalar_result.scale.full_tensor(), scalar_reference.scale.reshape(2, 1))
    torch.testing.assert_close(scalar_result.zp.full_tensor(), scalar_reference.zp.reshape(2, 1))


def _check_tp_activation_edges(rank, mesh):
    from torch.distributed.tensor import DTensor, Shard
    from transformers.distributed.tensor_parallel import RowwiseParallel

    from auto_round.data_type.nvfp import nv_fp4
    from auto_round.parallel.tp.activation import TensorParallelActivationQuantizer
    from auto_round.parallel.tp.qdq import TensorParallelQDQLinear

    weight = DTensor.from_local(torch.ones(4, 32), mesh, [Shard(1)], run_check=False)
    quantize = TensorParallelActivationQuantizer(nv_fp4, weight)
    for data in (torch.zeros(2, 64), torch.ones(2, 64)):
        data[:, 0] *= -1
        expected_x = data.clone().requires_grad_()
        local_x = data.chunk(2, dim=-1)[rank].clone().requires_grad_()
        distributed_x = DTensor.from_local(local_x, mesh, [Shard(-1)], run_check=False)
        expected, _, _ = nv_fp4(expected_x, group_size=16)
        with patch.object(DTensor, "full_tensor", side_effect=AssertionError("activation gather is forbidden")):
            actual, _, _ = quantize(distributed_x, group_size=16, data_type="nv_fp4")
        torch.testing.assert_close(actual.to_local(), expected.chunk(2, dim=-1)[rank])
        actual.to_local().sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(local_x.grad, expected_x.grad.chunk(2, dim=-1)[rank])
        assert torch.isfinite(local_x.grad).all()
    empty = DTensor.from_local(torch.empty(0, 32), mesh, [Shard(-1)], run_check=False)
    assert quantize(empty, group_size=16, data_type="nv_fp4")[0].numel() == 0

    invalid = _configure_linear(torch.nn.Linear(32, 4, bias=False))
    invalid.act_bits, invalid.act_group_size, invalid.act_data_type = 4, 32, "mx_fp"
    invalid.act_sym = invalid.act_dynamic = True
    RowwiseParallel().shard_param(invalid, "weight", mesh)
    with pytest.raises(ValueError, match="activation quantization group"):
        TensorParallelQDQLinear(invalid, enable_torch_compile=False)

    calibrated = _configure_linear(torch.nn.Linear(64, 4, bias=False))
    calibrated.data_type, calibrated.group_size = "nv_fp", 16
    calibrated.act_bits, calibrated.act_group_size, calibrated.act_data_type = 4, 16, "nv_fp4_with_static_gs"
    calibrated.act_sym = calibrated.act_dynamic = True
    calibrated.act_max = torch.empty(0) if rank == 0 else torch.tensor([20.0])
    RowwiseParallel().shard_param(calibrated, "weight", mesh)
    wrapped = TensorParallelQDQLinear(calibrated, enable_torch_compile=False)
    torch.testing.assert_close(wrapped.orig_layer.act_max, torch.tensor([20.0]))


def _check_tp_moe_numerics(rank, mesh, fp4=None):
    from transformers import Qwen3MoeConfig
    from transformers.distributed.tensor_parallel import MoeExpertsParallel, PackedColwiseParallel, RowwiseParallel
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    from auto_round.modeling.fused_moe.moe_experts_interface import prepare_model_for_moe_quantization
    from auto_round.parallel.tp.moe import prepare_tp_moe
    from auto_round.parallel.tp.qdq import TensorParallelQDQLinear
    from auto_round.parallel.tp.runtime import validate_tp_model
    from auto_round.wrapper import WrapperLinear

    torch.manual_seed(19)
    width = 64 if fp4 else 8
    group_size = {"mx_fp": 32, "nv_fp": 16}.get(fp4, 2)
    config = Qwen3MoeConfig(hidden_size=width, moe_intermediate_size=width, num_experts=4, num_experts_per_tok=2)
    native = Qwen3MoeExperts(config).requires_grad_(False)
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.normal_(std=0.1)
    reference = copy.deepcopy(native)
    with pytest.raises(NotImplementedError, match="DTensor"):
        prepare_tp_moe(reference)
    PackedColwiseParallel().shard_param(native, "gate_up_proj", mesh)
    RowwiseParallel().shard_param(native, "down_proj", mesh)
    MoeExpertsParallel().install_forward(native, mesh)
    original_forward = native.forward
    native._is_expert_parallel = True
    with pytest.raises(NotImplementedError, match="without EP"):
        prepare_tp_moe(native)
    native._is_expert_parallel = False
    packed_weight = native.gate_up_proj
    from torch.distributed.tensor import DTensor, Shard

    native.gate_up_proj = torch.nn.Parameter(
        DTensor.from_local(
            packed_weight.to_local(),
            mesh,
            [Shard(1)],
            run_check=False,
            shape=packed_weight.shape,
            stride=packed_weight.stride(),
        ),
        requires_grad=False,
    )
    with pytest.raises(NotImplementedError, match="packed_colwise"):
        prepare_tp_moe(native)
    native.gate_up_proj = packed_weight
    probe = torch.nn.Module()
    probe.register_parameter("weight", packed_weight)
    probe._tp_plan = {"q_proj": "packed_colwise"}
    with pytest.raises(NotImplementedError, match="packed_colwise"):
        validate_tp_model(probe, 2)
    probe._tp_plan = {}
    probe.add_module("experts", torch.nn.Sequential(torch.nn.Linear(width, width)))
    with pytest.raises(NotImplementedError, match="natively sharded"):
        validate_tp_model(probe, 2)
    x = torch.randn(4, width)
    routes = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 0]])  # expert 3 intentionally unused
    weights = torch.softmax(torch.randn(4, 2), dim=-1)
    expected = reference(x, routes, weights)
    torch.testing.assert_close(native(x, routes, weights), expected)
    with pytest.raises(NotImplementedError, match="DTensor"):
        prepare_model_for_moe_quantization(native)
    assert "gate_up_proj" in native._parameters  # rejection must not mutate weights
    prepare_tp_moe(native)
    assert native.forward is original_forward
    assert prepare_tp_moe(native) == []  # idempotent
    torch.testing.assert_close(native(x, routes, weights), expected)
    prepare_model_for_moe_quantization(reference, implementation="linear_loop")
    for index in range(4):
        for key in ("gate_proj", "up_proj", "down_proj"):
            actual_layer = _configure_linear(getattr(native.get_submodule(str(index)), key))
            ref_layer = _configure_linear(getattr(reference.get_submodule(str(index)), key))
            if fp4:
                for layer in (actual_layer, ref_layer):
                    layer.data_type = fp4
                    layer.group_size = layer.act_group_size = group_size
                    layer.act_bits = 4
                    layer.act_data_type = "nv_fp4_with_static_gs" if fp4 == "nv_fp" else "mx_fp"
                    layer.act_dynamic = layer.act_sym = True
                    if fp4 == "nv_fp":
                        layer.act_max = torch.tensor([2.0])
            native.get_submodule(str(index)).add_module(
                key, TensorParallelQDQLinear(actual_layer, enable_torch_compile=False)
            )
            reference.get_submodule(str(index)).add_module(key, WrapperLinear(ref_layer, enable_torch_compile=False))
    actual_x = x.clone().requires_grad_()
    expected_x = x.clone().requires_grad_()
    actual_weights = weights.clone().requires_grad_()
    expected_weights = weights.clone().requires_grad_()
    actual = native(actual_x, routes, actual_weights)
    expected = reference(expected_x, routes, expected_weights)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual_x.grad, expected_x.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual_weights.grad, expected_weights.grad, rtol=3e-5, atol=3e-6)
    for index in range(4):
        for key in ("gate_proj", "up_proj", "down_proj"):
            layer = getattr(native.get_submodule(str(index)), key)
            ref_layer = getattr(reference.get_submodule(str(index)), key)
            for name, parameter in layer.params.items():
                full_grad = ref_layer.params[name].grad
                if full_grad is None:
                    assert parameter.grad is None
                else:
                    axis = 1 if key == "down_proj" else 0
                    shard = (
                        full_grad
                        if name.startswith("act_")
                        else full_grad.reshape(width, width // group_size, -1)
                        .chunk(2, dim=axis)[rank]
                        .reshape_as(parameter)
                    )
                    torch.testing.assert_close(parameter.grad, shard, rtol=3e-5, atol=3e-6)
            native.get_submodule(str(index)).add_module(
                key, layer.unwrapper({k: p.detach() for k, p in layer.params.items()})
            )
            reference.get_submodule(str(index)).add_module(
                key, ref_layer.unwrapper({k: p.detach() for k, p in ref_layer.params.items()})
            )
    torch.testing.assert_close(native(x, routes, weights), reference(x, routes, weights))


def _check_stage_warm_start(rank, mesh, optimized):
    from transformers.distributed.tensor_parallel import ColwiseParallel, RowwiseParallel

    from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear
    from auto_round.parallel.tp.qdq import tensor_parallel_wrapper
    from auto_round.parallel.tp.runtime import is_dtensor
    from auto_round.wrapper import WrapperLinear, preserve_original_weights, store_warm_start_params_and_unwrap

    local_class = SignRoundOptimizedWrapperLinear if optimized else WrapperLinear
    wrapper_class = tensor_parallel_wrapper(local_class)
    assert tensor_parallel_wrapper(wrapper_class) is wrapper_class
    for style in (ColwiseParallel(), RowwiseParallel()):
        layer = _configure_linear(torch.nn.Linear(8, 8, bias=False))
        style.shard_param(layer, "weight", mesh)
        style.install_forward(layer, mesh)
        original = layer.weight.to_local().clone()
        block = torch.nn.Sequential(layer)
        with preserve_original_weights(block):
            wrapped = wrapper_class(layer, enable_torch_compile=False)
            with torch.no_grad():
                wrapped.value.fill_(0.1 * (rank + 1))
            best = {key: value.detach().clone() for key, value in wrapped.params.items()}
            block[0] = wrapped
            store_warm_start_params_and_unwrap(block, {"0": best})
            # Check before the outer snapshot restores the graph: the warm-start
            # helper itself must never replace a native layer with its local view.
            assert block[0] is layer
            assert is_dtensor(layer.weight)
        torch.testing.assert_close(layer.weight.to_local(), original)
        assert layer._ar_warm_start_wrapper_cls is local_class
        resumed = tensor_parallel_wrapper(layer._ar_warm_start_wrapper_cls)(layer, enable_torch_compile=False)
        for key, value in best.items():
            torch.testing.assert_close(resumed.params[key], value)
        if optimized:
            torch.testing.assert_close(resumed.init_scale.cpu(), layer._ar_warm_start["init_scale"])
        assert resumed.unwrapper(best) is layer
        assert is_dtensor(layer.weight)
        assert not hasattr(layer, "_ar_warm_start")
        assert not hasattr(layer, "_ar_warm_start_wrapper_cls")


def _check_export_state(model, exported):
    from torch.distributed.tensor import DTensor

    from auto_round.parallel.tp.activation import TensorParallelActivationQuantizer
    from auto_round.parallel.tp.runtime import _gather_cpu_tensor

    writer = dist.get_rank() == 0
    # Native MoE packed parameters alias the per-expert weights used for export.
    aliases = {
        f"{name}.{key}"
        for name, module in model.named_modules()
        if getattr(module, "_auto_round_tp_moe", False)
        for key in ("gate_up_proj", "down_proj")
    }
    state = {name: value for name, value in model.state_dict().items() if name not in aliases}
    exported_state = exported.state_dict() if writer else None
    if writer:
        assert state.keys() == exported_state.keys()
        assert exported.dtype == model.dtype

    def check_tensor(name, value, actual):
        expected = _gather_cpu_tensor(value)
        if writer:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda message: f"{name}: {message}")

    with patch.object(DTensor, "full_tensor", side_effect=AssertionError("GPU gather is forbidden")):
        for name, value in state.items():
            check_tensor(name, value, exported_state[name] if writer else None)
        exported_buffers = dict(exported.named_buffers()) if writer else None
        for name, value in model.named_buffers():
            check_tensor(name, value, exported_buffers[name] if writer else None)
        # Quantization metadata is not necessarily registered in state_dict.
        attributes = (
            "bits",
            "group_size",
            "sym",
            "data_type",
            "scale_dtype",
            "scale",
            "zp",
            "q_scale_thresh",
            "weight_global_scale",
            "act_bits",
            "act_data_type",
            "act_group_size",
            "act_sym",
            "act_dynamic",
            "act_max",
            "act_scale",
            "act_min_scale",
            "act_max_scale",
            "input_global_scale",
            "act_quant_func",
        )
        for name, module in model.named_modules():
            for key in attributes:
                if not hasattr(module, key):
                    continue
                expected = getattr(module, key)
                actual = getattr(exported.get_submodule(name), key) if writer else None
                if isinstance(expected, torch.Tensor):
                    check_tensor(f"{name}.{key}", expected, actual)
                else:
                    if isinstance(expected, TensorParallelActivationQuantizer):
                        expected = expected.quant_func
                    if writer:
                        assert actual == expected, f"{name}.{key} changed during export"


def _check_qad_lifecycle(
    rank,
    checkpoint,
    workspace,
    algorithm,
    fp4=None,
    activation=False,
    block_stages=(),
    lora_rank=0,
    round_tuning=True,
    dtype=torch.float32,
):
    from auto_round import AutoRound
    from auto_round.algorithms.quantization.qad.config import QADConfig
    from auto_round.algorithms.quantization.qat.config import QATConfig
    from auto_round.parallel.tp.runtime import gather_for_export, is_dtensor

    tokens = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    config_cls = QADConfig if algorithm == "qad" else QATConfig
    group_size = {"mx_fp": 32, "nv_fp": 16}.get(fp4, 8)
    compressor = AutoRound(
        checkpoint,
        scheme={"mx_fp": "MXFP4", "nv_fp": "NVFP4"}.get(fp4, "W4A16"),
        parallel_config=ParallelConfig(tp_size=2),
        alg_configs=[
            *block_stages,
            config_cls(
                iters=2,
                bits=4,
                group_size=group_size,
                act_bits=4 if activation else 16,
                act_data_type=({"mx_fp": "mx_fp", "nv_fp": "nv_fp4_with_static_gs"}[fp4] if activation else "float"),
                nsamples=2,
                topk=8,
                gradient_checkpointing=True,
                qad_workspace=workspace,
                qad_lora_rank=lora_rank,
                qad_enable_round_tuning=round_tuning,
            ),
        ],
        dataset=torch.utils.data.DataLoader(tokens, batch_size=1),
        nsamples=2,
        seqlen=4,
        batch_size=1,
        model_dtype=dtype,
        # Test FP32 and actual BF16 training explicitly, independent of CPU defaults.
        amp=dtype == torch.bfloat16,
        low_cpu_mem_usage=False,
        enable_torch_compile=False,
    )
    if block_stages:
        compressor.post_init()
        terminal = compressor.alg_composer.active_quantizer
        quantize_model = terminal.quantize_model

        def check_warm_start(model, *args, **kwargs):
            projection = model.model.layers[0].self_attn.q_proj
            assert is_dtensor(projection.weight)
            assert projection._ar_warm_start
            assert projection._ar_warm_start["value"].numel() == projection.weight.to_local().numel()
            return quantize_model(model, *args, **kwargs)

        terminal.quantize_model = check_warm_start
    model, _ = compressor.quantize()
    assert model.dtype == dtype
    projection = model.model.layers[0].self_attn.q_proj
    if activation:
        from auto_round.wrapper import WrapperWALayer

        assert isinstance(projection, WrapperWALayer)
        projection = projection.orig_layer
    assert is_dtensor(projection.weight)
    assert is_dtensor(projection.scale)
    assert not any(hasattr(module, "params") for module in model.modules())
    assert not any(hasattr(module, "_ar_warm_start") for module in model.modules())
    if model.config.model_type == "qwen3_moe":
        experts = model.model.layers[0].mlp.experts
        assert experts._auto_round_tp_moe
        assert experts.config._experts_implementation == "linear_loop"
        expert_projection = experts.get_submodule("0.gate_proj")
        if activation:
            expert_projection = expert_projection.orig_layer
        assert is_dtensor(expert_projection.scale)
    with torch.no_grad():
        logits = model(input_ids=tokens).logits
    assert torch.isfinite(logits).all()
    from torch.distributed.tensor import DTensor

    with patch.object(DTensor, "full_tensor", side_effect=AssertionError("GPU gather is forbidden")):
        exported = gather_for_export(model)
    _check_export_state(model, exported)
    if rank == 0:
        assert not any(is_dtensor(parameter) for parameter in exported.parameters())
        with torch.no_grad():
            exported_logits = exported(input_ids=tokens).logits
        if dtype == torch.bfloat16:
            assert exported_logits.dtype == logits.dtype == torch.bfloat16
            # Rowwise BF16 partial sums round before reduction, unlike a dense GEMM.
            # Bound absolute error by one BF16 epsilon at the peak logit magnitude;
            # near-zero logits must not turn harmless rounding into large relative errors.
            atol = torch.finfo(dtype).eps * logits.float().abs().amax().item()
            torch.testing.assert_close(exported_logits, logits, rtol=0, atol=atol)
        else:
            torch.testing.assert_close(exported_logits, logits, rtol=2e-4, atol=2e-5)
        exported_projection = exported.model.layers[0].self_attn.q_proj
        if activation:
            assert isinstance(exported_projection, WrapperWALayer)
            exported_projection = exported_projection.orig_layer
        assert exported_projection.scale.shape == (64, 64 // group_size)
        if fp4 == "nv_fp":
            torch.testing.assert_close(exported_projection.weight_global_scale, projection.weight_global_scale.cpu())
        if model.config.tie_word_embeddings:
            assert exported.lm_head.weight is exported.model.embed_tokens.weight
    quantizer = compressor.alg_composer.active_quantizer
    if algorithm == "qad":
        assert Path(workspace, f"tp_rank_{rank}").is_dir()
        # A cache missing only on rank one must rebuild collectively, rather
        # than leaving rank one waiting for a forward skipped by rank zero.
        samples = list(tokens)
        quantizer._build_teacher_cache_from_model(model, samples, "cpu")
        quantizer._teacher_cache.shutdown()
        if rank == 1:
            Path(quantizer._teacher_cache.workspace, "teacher_0.pt").unlink()
        dist.barrier()
        quantizer._build_teacher_cache_from_model(model, samples, "cpu")
        assert quantizer._teacher_cache.is_valid_for(0, tokens[:1])
        quantizer._teacher_cache.shutdown()
    else:
        assert quantizer._teacher_cache is None
    output = str(Path(workspace).parent / f"export_{Path(workspace).name}_{algorithm}")
    saved, folders = compressor.save_quantized(output, format="fake", return_folders=True)
    assert compressor.model_context.model is model
    if rank == 0:
        assert Path(folders, "config.json").is_file()
        assert not any(is_dtensor(parameter) for parameter in saved.parameters())
    else:
        assert saved is model
        assert folders == []
    with patch(
        "auto_round.export.formats.backends.fake.FakeFormat.save_quantized", side_effect=RuntimeError("writer failure")
    ), pytest.raises(RuntimeError, match="writer failure"):
        compressor.save_quantized(output, format="fake")
    assert compressor.model_context.model is model
    if fp4:
        # Select a second format explicitly after the fake-save checks.
        compressor.formats = "auto_round"
        packed, packed_folder = compressor.save_quantized(output + "_packed", format="auto_round", return_folders=True)
        assert compressor.model_context.model is model
        if rank == 0:
            assert Path(packed_folder, "config.json").is_file()
            assert list(Path(packed_folder).glob("*.safetensors"))
            packed_projection = packed.model.layers[0].self_attn.q_proj
            assert packed_projection.weight_packed.dtype == torch.uint8
            assert packed_projection.weight_packed.shape == (64, 32)
            assert packed_projection.weight_packed.device.type == "cpu"
            if model.config.model_type == "qwen3_moe":
                expert_projection = packed.model.layers[0].mlp.experts.get_submodule("0.gate_proj")
                assert expert_projection.weight_packed.shape == (128, 32)
                assert expert_projection.weight_packed.device.type == "cpu"
            if fp4 == "nv_fp":
                torch.testing.assert_close(
                    packed_projection.weight_global_scale.reshape(()), projection.weight_global_scale.cpu().reshape(())
                )
    dist.barrier()


def _distributed_worker(rank, rendezvous, checkpoint, qwen_checkpoint, moe_checkpoint, workspace, dtype):
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig, SignRoundV2Config

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        if dtype == torch.bfloat16:
            from auto_round.utils.device_manager import CpuARDevice

            # PyTorch can run these tiny models with software BF16 kernels even
            # without native BF16 instructions. Bypass only the FP32 fallback policy.
            with patch.object(CpuARDevice, "supports_bf16", return_value=True):
                _check_qad_lifecycle(rank, checkpoint, workspace, "qad", lora_rank=2, dtype=dtype)
                _check_qad_lifecycle(
                    rank,
                    qwen_checkpoint,
                    workspace + "_nv_act",
                    "qat",
                    fp4="nv_fp",
                    activation=True,
                    lora_rank=2,
                    dtype=dtype,
                )
            return
        mesh = dist.init_device_mesh("cpu", (2,))
        _check_stage_warm_start(rank, mesh, optimized=False)
        _check_stage_warm_start(rank, mesh, optimized=True)
        _check_qdq_numerics(rank, mesh, symmetric=True)
        _check_qdq_numerics(rank, mesh, symmetric=False)
        _check_qdq_numerics(rank, mesh, symmetric=True, data_type="mx_fp", group_size=32, width=128)
        _check_qdq_numerics(rank, mesh, symmetric=True, data_type="nv_fp", group_size=16, width=64)
        with patch.dict("os.environ", {"AR_ENABLE_ACT_MINMAX_TUNING": "1"}):
            for data_type, group_size, width, act_type in (
                ("mx_fp", 32, 128, "mx_fp"),
                ("mx_fp", 32, 128, "mx_fp_rceil"),
                ("nv_fp", 16, 64, "nv_fp4"),
                ("nv_fp", 16, 64, "nv_fp4_with_static_gs"),
            ):
                _check_qdq_numerics(rank, mesh, True, data_type, group_size, width, act_type)
            for act_bits in (6, 8):
                _check_qdq_numerics(rank, mesh, True, "mx_fp", 32, 128, "mx_fp", act_bits)
        _check_tp_activation_edges(rank, mesh)
        _check_tp_moe_numerics(rank, mesh)
        with patch.dict("os.environ", {"AR_ENABLE_ACT_MINMAX_TUNING": "1"}):
            _check_tp_moe_numerics(rank, mesh, fp4="mx_fp")
            _check_tp_moe_numerics(rank, mesh, fp4="nv_fp")
        _check_qad_lifecycle(rank, checkpoint, workspace, "qad", lora_rank=2)
        _check_qad_lifecycle(rank, qwen_checkpoint, workspace, "qat", lora_rank=2)
        _check_qad_lifecycle(
            rank,
            checkpoint,
            workspace + "_staged",
            "qad",
            block_stages=[SignRoundConfig(iters=2, group_size=8, lora_rank=2)],
            lora_rank=2,
            round_tuning=False,
        )
        _check_qad_lifecycle(
            rank,
            qwen_checkpoint,
            workspace + "_v2_staged",
            "qat",
            block_stages=[SignRoundV2Config(iters=2, group_size=8, lora_rank=2)],
            lora_rank=2,
        )
        _check_qad_lifecycle(
            rank,
            checkpoint,
            workspace + "_three_stages",
            "qad",
            block_stages=[SignRoundV2Config(iters=2, group_size=8), SignRoundConfig(iters=2, group_size=8)],
        )
        _check_qad_lifecycle(rank, checkpoint, workspace + "_mx", "qad", fp4="mx_fp", lora_rank=2)
        _check_qad_lifecycle(rank, qwen_checkpoint, workspace + "_nv", "qat", fp4="nv_fp", lora_rank=2)
        _check_qad_lifecycle(rank, moe_checkpoint, workspace + "_moe", "qad", lora_rank=2)
        _check_qad_lifecycle(rank, moe_checkpoint, workspace + "_moe_mx", "qat", fp4="mx_fp")
        _check_qad_lifecycle(rank, moe_checkpoint, workspace + "_moe_nv", "qat", fp4="nv_fp")
        _check_qad_lifecycle(rank, checkpoint, workspace + "_mx_act", "qad", fp4="mx_fp", activation=True, lora_rank=2)
        _check_qad_lifecycle(rank, qwen_checkpoint, workspace + "_nv_act", "qat", fp4="nv_fp", activation=True)
        _check_qad_lifecycle(rank, moe_checkpoint, workspace + "_moe_mx_act", "qat", fp4="mx_fp", activation=True)
        _check_qad_lifecycle(rank, moe_checkpoint, workspace + "_moe_nv_act", "qat", fp4="nv_fp", activation=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires distributed Gloo")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_two_rank_native_tp_qdq_and_qad(tmp_path, dtype):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "DistributedConfig"):
        pytest.skip("requires native Transformers DistributedConfig")
    pytest.importorskip("transformers.distributed.tensor_parallel")
    from tokenizers import Tokenizer, models
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
        Qwen3Config,
        Qwen3ForCausalLM,
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
    )

    checkpoint = tmp_path / "model"
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    config._attn_implementation = "eager"
    LlamaForCausalLM(config).save_pretrained(checkpoint)
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{str(i): i for i in range(4, 32)}}
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")),
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    ).save_pretrained(checkpoint)
    qwen_checkpoint = tmp_path / "qwen"
    qwen_config = Qwen3Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=16,
        tie_word_embeddings=True,
    )
    qwen_config._attn_implementation = "eager"
    Qwen3ForCausalLM(qwen_config).save_pretrained(qwen_checkpoint)
    PreTrainedTokenizerFast.from_pretrained(checkpoint).save_pretrained(qwen_checkpoint)
    moe_checkpoint = tmp_path / "moe"
    moe_config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=16,
    )
    moe_config._attn_implementation = "eager"
    Qwen3MoeForCausalLM(moe_config).save_pretrained(moe_checkpoint)
    PreTrainedTokenizerFast.from_pretrained(checkpoint).save_pretrained(moe_checkpoint)
    # conftest adds a relative '..' for interactive runs. Fresh workers must not
    # resolve that entry against a different cwd and shadow installed packages.
    with patch.object(sys, "path", [entry for entry in sys.path if Path(entry).is_absolute()]):
        context = mp.spawn(
            _distributed_worker,
            args=(
                str(tmp_path / "rendezvous"),
                str(checkpoint),
                str(qwen_checkpoint),
                str(moe_checkpoint),
                str(tmp_path / "teacher"),
                dtype,
            ),
            nprocs=2,
            join=False,
        )
    try:
        import time

        deadline = time.monotonic() + 240
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Native TP workers exceeded the test timeout.")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
