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
@pytest.mark.parametrize("axis", ["tp_size", "pp_size", "dp_size"])
def test_parallel_sizes_are_positive_integers(axis, value):
    with pytest.raises(ValueError, match=axis):
        ParallelConfig(**{axis: value})


def test_parallel_policy_validation(monkeypatch):
    assert ParallelConfig.from_value(None) == ParallelConfig()
    assert ParallelConfig.from_value({"tp_size": 2}).tp_size == 2
    with pytest.raises(NotImplementedError, match="DP"):
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
    assert (args.tp_size, args.pp_size, args.dp_size) == (2, 1, 1)


@pytest.mark.parametrize("scheme", ["MXFP4", "NVFP4", "INT8_W8A8"])
def test_unsupported_tp_scheme_fails_before_loading(scheme):
    from auto_round import AutoRound

    with (
        patch.object(ParallelConfig, "resolve_device_map", return_value="cpu"),
        pytest.raises(NotImplementedError, match="weight-only"),
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


@pytest.mark.parametrize("data_type,group_size", [("mx_fp", 16), ("nv_fp", 32)])
def test_invalid_fp4_block_size(data_type, group_size):
    from auto_round.parallel.policy import validate_tp_quantization
    from auto_round.schemes import QuantizationScheme

    with pytest.raises(NotImplementedError, match="group_size"):
        validate_tp_quantization(QuantizationScheme(data_type=data_type, group_size=group_size, act_bits=16))


def _configure_linear(layer):
    for key, value in dict(bits=4, group_size=2, sym=True, data_type="int", act_bits=16).items():
        setattr(layer, key, value)
    layer.scale_dtype = torch.float32
    return layer.requires_grad_(False)


def _check_qdq_numerics(rank, mesh, symmetric, data_type="int", group_size=2, width=8):
    from transformers.distributed.tensor_parallel import ColwiseParallel, RowwiseParallel

    from auto_round.parallel.qdq import TensorParallelQDQLinear
    from auto_round.wrapper import WrapperLinear

    torch.manual_seed(7)
    col = _configure_linear(torch.nn.Linear(width, width))
    row = _configure_linear(torch.nn.Linear(width, 4))
    col.sym = row.sym = symmetric
    col.data_type = row.data_type = data_type
    col.group_size = row.group_size = group_size
    # Distinct shard maxima catch accidentally computing NVFP global scales
    # independently per rank, for both output- and input-axis sharding.
    if data_type != "int":
        with torch.no_grad():
            col.weight[width // 2 :].mul_(17)
            row.weight[:, width // 2 :].mul_(17)
    reference_col = WrapperLinear(copy.deepcopy(col), enable_torch_compile=False)
    reference_row = WrapperLinear(copy.deepcopy(row), enable_torch_compile=False)
    for layer, style in ((col, ColwiseParallel()), (row, RowwiseParallel())):
        style.shard_param(layer, "weight", mesh)
        style.shard_param(layer, "bias", mesh)
        style.install_forward(layer, mesh)
    col_before = col.weight.to_local().clone()
    wrapped_col = TensorParallelQDQLinear(col, enable_torch_compile=False)
    wrapped_row = TensorParallelQDQLinear(row, enable_torch_compile=False)
    x = torch.randn(2, 3, width)
    expected = reference_row(reference_col(x))
    actual = wrapped_row(wrapped_col(x))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for key, parameter in wrapped_col.params.items():
        full_grad = reference_col.params[key].grad
        if full_grad is not None:
            torch.testing.assert_close(parameter.grad, full_grad.chunk(2, dim=0)[rank], rtol=3e-5, atol=3e-6)
    for key, parameter in wrapped_row.params.items():
        full_grad = reference_row.params[key].grad
        if full_grad is not None:
            expected_grad = full_grad.reshape(4, width // group_size, -1).chunk(2, dim=1)[rank].reshape_as(parameter)
            torch.testing.assert_close(parameter.grad, expected_grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(col.weight.to_local(), col_before)
    wrapped_col.unwrapper({key: value.detach() for key, value in wrapped_col.params.items()})
    wrapped_row.unwrapper({key: value.detach() for key, value in wrapped_row.params.items()})
    ref_col = reference_col.unwrapper({key: value.detach() for key, value in reference_col.params.items()})
    ref_row = reference_row.unwrapper({key: value.detach() for key, value in reference_row.params.items()})
    for actual_layer, reference in ((col, ref_col), (row, ref_row)):
        torch.testing.assert_close(actual_layer.weight.full_tensor(), reference.weight)
        torch.testing.assert_close(actual_layer.scale.full_tensor(), reference.scale)
        if not symmetric:
            torch.testing.assert_close(actual_layer.zp.full_tensor(), reference.zp)
    if data_type != "int":
        from auto_round.parallel.runtime import _gather_cpu_tensor

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


def _check_tp_moe_numerics(rank, mesh):
    from transformers import Qwen3MoeConfig
    from transformers.distributed.tensor_parallel import MoeExpertsParallel, PackedColwiseParallel, RowwiseParallel
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    from auto_round.modeling.fused_moe.moe_experts_interface import prepare_model_for_moe_quantization
    from auto_round.parallel.moe import prepare_tp_moe
    from auto_round.parallel.qdq import TensorParallelQDQLinear
    from auto_round.parallel.runtime import validate_tp_model
    from auto_round.wrapper import WrapperLinear

    torch.manual_seed(19)
    config = Qwen3MoeConfig(hidden_size=8, moe_intermediate_size=8, num_experts=4, num_experts_per_tok=2)
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
    probe.add_module("experts", torch.nn.Sequential(torch.nn.Linear(8, 8)))
    with pytest.raises(NotImplementedError, match="natively sharded"):
        validate_tp_model(probe, 2)
    x = torch.randn(4, 8)
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
                    shard = full_grad.reshape(8, 4, -1).chunk(2, dim=axis)[rank].reshape_as(parameter)
                    torch.testing.assert_close(parameter.grad, shard, rtol=3e-5, atol=3e-6)
            native.get_submodule(str(index)).add_module(
                key, layer.unwrapper({k: p.detach() for k, p in layer.params.items()})
            )
            reference.get_submodule(str(index)).add_module(
                key, ref_layer.unwrapper({k: p.detach() for k, p in ref_layer.params.items()})
            )
    torch.testing.assert_close(native(x, routes, weights), reference(x, routes, weights))


def _check_qad_lifecycle(rank, checkpoint, workspace, algorithm, fp4=None):
    from auto_round import AutoRound
    from auto_round.algorithms.quantization.qad.config import QADConfig
    from auto_round.algorithms.quantization.qat.config import QATConfig
    from auto_round.parallel.runtime import gather_for_export, is_dtensor

    tokens = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    config_cls = QADConfig if algorithm == "qad" else QATConfig
    group_size = {"mx_fp": 32, "nv_fp": 16}.get(fp4, 8)
    compressor = AutoRound(
        checkpoint,
        scheme={"mx_fp": "MXFP4", "nv_fp": "NVFP4"}.get(fp4, "W4A16"),
        parallel_config=ParallelConfig(tp_size=2),
        alg_configs=config_cls(
            iters=2,
            bits=4,
            group_size=group_size,
            act_bits=16,
            act_data_type="float",
            nsamples=2,
            topk=8,
            gradient_checkpointing=True,
            qad_workspace=workspace,
        ),
        dataset=torch.utils.data.DataLoader(tokens, batch_size=1),
        nsamples=2,
        seqlen=4,
        batch_size=1,
        model_dtype="float32",
        low_cpu_mem_usage=False,
        enable_torch_compile=False,
    )
    model, _ = compressor.quantize()
    projection = model.model.layers[0].self_attn.q_proj
    assert is_dtensor(projection.weight)
    assert is_dtensor(projection.scale)
    assert not any(hasattr(module, "params") for module in model.modules())
    if model.config.model_type == "qwen3_moe":
        experts = model.model.layers[0].mlp.experts
        assert experts._auto_round_tp_moe
        assert experts.config._experts_implementation == "linear_loop"
        assert is_dtensor(experts.get_submodule("0.gate_proj").scale)
    with torch.no_grad():
        logits = model(input_ids=tokens).logits
    assert torch.isfinite(logits).all()
    from torch.distributed.tensor import DTensor

    with patch.object(DTensor, "full_tensor", side_effect=AssertionError("GPU gather is forbidden")):
        exported = gather_for_export(model)
    if rank == 0:
        assert not any(is_dtensor(parameter) for parameter in exported.parameters())
        with torch.no_grad():
            torch.testing.assert_close(exported(input_ids=tokens).logits, logits, rtol=2e-4, atol=2e-5)
        assert exported.model.layers[0].self_attn.q_proj.scale.shape == (64, 64 // group_size)
        if fp4 == "nv_fp":
            torch.testing.assert_close(
                exported.model.layers[0].self_attn.q_proj.weight_global_scale, projection.weight_global_scale.cpu()
            )
        if model.config.tie_word_embeddings:
            assert exported.lm_head.weight is exported.model.embed_tokens.weight
    quantizer = compressor.alg_composer.block_quantizer
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


def _distributed_worker(rank, rendezvous, checkpoint, qwen_checkpoint, moe_checkpoint, workspace):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        mesh = dist.init_device_mesh("cpu", (2,))
        _check_qdq_numerics(rank, mesh, symmetric=True)
        _check_qdq_numerics(rank, mesh, symmetric=False)
        _check_qdq_numerics(rank, mesh, symmetric=True, data_type="mx_fp", group_size=32, width=128)
        _check_qdq_numerics(rank, mesh, symmetric=True, data_type="nv_fp", group_size=16, width=64)
        _check_tp_moe_numerics(rank, mesh)
        _check_qad_lifecycle(rank, checkpoint, workspace, "qad")
        _check_qad_lifecycle(rank, qwen_checkpoint, workspace, "qat")
        _check_qad_lifecycle(rank, checkpoint, workspace + "_mx", "qad", fp4="mx_fp")
        _check_qad_lifecycle(rank, qwen_checkpoint, workspace + "_nv", "qat", fp4="nv_fp")
        _check_qad_lifecycle(rank, moe_checkpoint, workspace + "_moe", "qad")
        _check_qad_lifecycle(rank, moe_checkpoint, workspace + "_moe_mx", "qat", fp4="mx_fp")
        _check_qad_lifecycle(rank, moe_checkpoint, workspace + "_moe_nv", "qat", fp4="nv_fp")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires distributed Gloo")
def test_two_rank_native_tp_qdq_and_qad(tmp_path):
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
            ),
            nprocs=2,
            join=False,
        )
    try:
        import time

        deadline = time.monotonic() + 180
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Native TP workers exceeded the test timeout.")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
