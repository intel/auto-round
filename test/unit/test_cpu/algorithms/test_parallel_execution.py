# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from auto_round.parallel.execution import ParallelExecution


def test_unadapted_algorithm_and_wrapper_fail_explicitly():
    from auto_round.algorithms.base import BaseAlgorithm
    from auto_round.parallel.tp.qdq import tensor_parallel_wrapper
    from auto_round.wrapper import WrapperLinear

    class CustomAlgorithm(BaseAlgorithm):
        pass

    class CustomWrapper(WrapperLinear):
        pass

    ParallelExecution().validate(CustomAlgorithm, SimpleNamespace())
    with pytest.raises(NotImplementedError, match="tensor-parallel contract"):
        ParallelExecution({"tp_size": 2}).validate(CustomAlgorithm, SimpleNamespace())
    with pytest.raises(NotImplementedError, match="TP wrapper contract"):
        tensor_parallel_wrapper(CustomWrapper)


def test_inherited_tp_support_requires_unchanged_execution_hooks():
    from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
    from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear
    from auto_round.parallel.tp.qdq import tensor_parallel_wrapper

    class SharedAlgorithm(SignRoundQuantizer):
        pass

    class CustomAlgorithm(SignRoundQuantizer):
        def quantize_block(self, *args, **kwargs):
            raise AssertionError("must validate before execution")

    class AdaptedAlgorithm(CustomAlgorithm):
        @classmethod
        def parallel_capabilities(cls):
            return super().parallel_capabilities()

    class CustomWrapper(SignRoundOptimizedWrapperLinear):
        pass

    class CustomTPWrapper(tensor_parallel_wrapper(SignRoundOptimizedWrapperLinear)):
        pass

    execution = ParallelExecution({"tp_size": 2})
    execution.validate(SharedAlgorithm, SimpleNamespace())
    execution.validate(AdaptedAlgorithm, SimpleNamespace())
    with pytest.raises(NotImplementedError, match="parallel-sensitive"):
        execution.validate(CustomAlgorithm, SimpleNamespace())
    with pytest.raises(NotImplementedError, match="TP wrapper contract"):
        tensor_parallel_wrapper(CustomWrapper)
    with pytest.raises(NotImplementedError, match="TP wrapper contract"):
        tensor_parallel_wrapper(CustomTPWrapper)


def test_algorithm_selects_parallel_execution_policy():
    from auto_round.algorithms.base import BaseAlgorithm

    class PlacementExecution(ParallelExecution):
        pass

    class PlacementAlgorithm(BaseAlgorithm):
        @classmethod
        def create_parallel_execution(cls, config, default_execution):
            return PlacementExecution(default_execution.config)

    default = ParallelExecution({"pp_size": 2})
    selected = default.for_algorithm(PlacementAlgorithm, SimpleNamespace())
    assert type(selected) is PlacementExecution
    assert selected.config == default.config
    assert default.for_algorithm(BaseAlgorithm, SimpleNamespace()) is default


@pytest.mark.parametrize("policy", [None, ParallelExecution(), ParallelExecution({"pp_size": 2})])
def test_algorithm_policy_cannot_silently_change_loaded_topology(policy):
    from auto_round.algorithms.base import BaseAlgorithm

    class InvalidAlgorithm(BaseAlgorithm):
        @classmethod
        def create_parallel_execution(cls, config, default_execution):
            return policy

    expected_error = TypeError if policy is None else NotImplementedError
    with pytest.raises(expected_error, match="ParallelExecution|resharding"):
        ParallelExecution({"tp_size": 2}).for_algorithm(InvalidAlgorithm, SimpleNamespace())


def test_pipeline_placement_keeps_signround_heuristic():
    execution = ParallelExecution({"pp_size": 2})
    block = torch.nn.Linear(8, 8)
    block.tuning_device = "cuda:1"
    manager = SimpleNamespace(device_map="auto", device_list=["cuda:0", "cuda:1"], device="cuda:0")
    inputs = [torch.randn(1, 2, 8)]
    with (
        patch("auto_round.parallel.execution.device_manager", manager),
        patch("auto_round.utils.is_auto_device_mapping", return_value=True),
        patch(
            "auto_round.utils.device.set_auto_device_map_for_block_with_tuning", return_value=(True, "cuda:1")
        ) as place,
        patch("accelerate.hooks.add_hook_to_module") as hook,
    ):
        result = execution.dispatch_block(block, inputs, {}, tuning=True, low_gpu_mem_usage=True, batch_size=4)
    place.assert_called_once_with(block, manager.device_list, inputs, True, 4, "cuda:0")
    hook.assert_called_once()
    assert result.block is block
    assert result.primary_device_at_risk
    assert result.loss_device == "cuda:1"


def test_tp_uses_native_backward_not_dp():
    block = torch.nn.Linear(4, 4)
    with patch("auto_round.parallel.distributed.setup_ddp_if_needed_", side_effect=AssertionError("not DP")):
        actual, synchronize = ParallelExecution({"tp_size": 2}).prepare_gradients(None, block)
        synchronize()
    assert actual is block


def test_local_execution_preserves_dp_rank_hooks(monkeypatch):
    from auto_round.parallel.dp import spmd

    sampler = object()
    rank = SimpleNamespace(
        index_sampler=Mock(return_value=sampler),
        all_reduce_weighted_mean=Mock(return_value=2.5),
        sync_gradients=Mock(),
    )
    monkeypatch.setattr(spmd, "current_rank", lambda: rank)
    execution = ParallelExecution()
    block = torch.nn.Linear(4, 4)
    actual, sync = execution.prepare_gradients(None, block)
    sync()
    assert actual is block
    rank.sync_gradients.assert_called_once_with(block)
    assert execution.index_sampler(16, 4) is sampler
    rank.index_sampler.assert_called_once_with(16, 4)
    assert execution.reduce_loss(3.0, 8) == 2.5
    rank.all_reduce_weighted_mean.assert_called_once_with(3.0, 8)


def test_tp_collection_and_tuning_never_create_dp_mirrors():
    from auto_round.parallel.dp.engine import ParallelPolicy

    execution = ParallelExecution({"tp_size": 2})
    quantizer = SimpleNamespace(
        compress_context=SimpleNamespace(parallel_policy=ParallelPolicy()),
        quantize_block=Mock(return_value={"tp": True}),
    )
    composer = SimpleNamespace(block_quantizer=quantizer)
    block = torch.nn.Linear(4, 4)
    inputs = [torch.randn(1, 2, 4)]
    with (
        patch("auto_round.parallel.dp.tune_parallel.TuneParallelContext.for_collection") as collect,
        patch("auto_round.parallel.dp.spmd.quantize_block_on_ranks") as tune,
        patch("auto_round.parallel.distributed.all_reduce_loss") as reduce_loss,
    ):
        ctx = execution.collection_context(composer, block, inputs)
        assert ctx.devices is None
        assert execution.quantize_block(quantizer, block, inputs, {}, inputs, None, None) == {"tp": True}
        assert execution.reduce_loss(3.0, 8) == 3.0
    collect.assert_not_called()
    tune.assert_not_called()
    reduce_loss.assert_not_called()
    quantizer.quantize_block.assert_called_once_with(block, inputs, {}, inputs, None, None, input_ids=None)


@pytest.mark.parametrize("engaged", [False, True])
def test_local_tuning_delegates_to_dp_engine(engaged):
    quantizer = SimpleNamespace(quantize_block=Mock(return_value="local"))
    with patch("auto_round.parallel.dp.spmd.quantize_block_on_ranks", return_value=(engaged, "replica")) as tune:
        result = ParallelExecution().quantize_block(quantizer, "block", "fp", {}, "out", None, "ctx")
    assert result == ("replica" if engaged else "local")
    tune.assert_called_once_with(quantizer, "block", "fp", {}, "out", None, "ctx", input_ids=None)
    assert quantizer.quantize_block.call_count == (0 if engaged else 1)


@pytest.mark.parametrize("topology", [{"tp_size": 2}, {"pp_size": 2}])
def test_stage_rejects_hybrid_dp_before_installing_hooks(topology):
    from auto_round.parallel.dp.engine import ParallelPolicy

    execution = ParallelExecution(topology)
    orchestrator = SimpleNamespace(compress_context=SimpleNamespace(parallel_policy=ParallelPolicy(world=2)))
    with pytest.raises(NotImplementedError, match="hybrid"), execution.stage(None, orchestrator):
        pytest.fail("must reject before entering the stage")


def _check_tensor_ops(mesh):
    from transformers.distributed.tensor_parallel import ColwiseParallel, RowwiseParallel

    from auto_round.algorithms.transforms.awq.smooth import compute_layer_means
    from auto_round.parallel.tp.tensor_ops import ParallelTensorOps

    torch.manual_seed(7)
    for style in (ColwiseParallel(), RowwiseParallel()):
        layer = torch.nn.Linear(8, 8, bias=False)
        reference = copy.deepcopy(layer)
        style.shard_param(layer, "weight", mesh)
        style.install_forward(layer, mesh)
        torch.testing.assert_close(
            ParallelTensorOps.input_channel_means([layer], 2), compute_layer_means([reference], 2)
        )
        scale = torch.linspace(0.5, 1.5, 8)
        with torch.no_grad():
            ParallelTensorOps.scale_inputs(layer, scale)
            reference.weight.mul_(scale.view(1, -1))
            ParallelTensorOps.divide_outputs(layer, scale)
            reference.weight.div_(scale.view(-1, 1))
        torch.testing.assert_close(layer.weight.full_tensor(), reference.weight)
        error = ParallelTensorOps.local(layer.weight).square().sum()
        torch.testing.assert_close(ParallelTensorOps.weight_sum(layer.weight, error), reference.weight.square().sum())


def _block_worker(rank, rendezvous, checkpoint, device_type="cpu"):
    from torch.distributed.tensor import DTensor

    from auto_round import AutoRound
    from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig, SignRoundV2Config
    from auto_round.algorithms.transforms.awq.config import AWQConfig
    from auto_round.parallel.tp.runtime import gather_for_export, is_dtensor
    from auto_round.wrapper import WrapperWALayer

    torch.set_num_threads(1)
    if device_type == "cuda":
        os.environ["LOCAL_RANK"] = str(rank)
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        sampler = SimpleNamespace(next_batch=lambda: [rank, rank + 2])
        assert ParallelExecution({"tp_size": 2}).sample_indices(sampler) == [0, 2]
        if device_type == "cpu":
            _check_tensor_ops(dist.init_device_mesh("cpu", (2,)))
        tokens = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
        configs = [
            SignRoundConfig(iters=2, group_size=8, not_use_best_mse=True),
            SignRoundV2Config(iters=2, group_size=8, not_use_best_mse=True),
            RTNConfig(group_size=8, disable_opt_rtn=True),
            OptimizedRTNConfig(group_size=8),
            [AWQConfig(n_grid=3, group_size=8), RTNConfig(group_size=8)],
            [AWQConfig(n_grid=3, group_size=8), SignRoundConfig(iters=2, group_size=8)],
            [AWQConfig(n_grid=3, group_size=8), SignRoundV2Config(iters=2, group_size=8)],
            [SignRoundConfig(iters=2, group_size=8), SignRoundConfig(iters=2, group_size=8)],
            [RTNConfig(group_size=8, disable_opt_rtn=True), SignRoundV2Config(iters=2, group_size=8)],
            [
                AWQConfig(n_grid=3, group_size=8),
                SignRoundV2Config(iters=2, group_size=8),
                SignRoundConfig(iters=2, group_size=8),
            ],
            AWQConfig(n_grid=3, group_size=8),
        ]
        if device_type == "cuda":
            configs = [configs[0], configs[5], configs[7]]
        else:
            for config_class, data_type, group_size, act_type in (
                (SignRoundConfig, "mx_fp", 32, "mx_fp"),
                (SignRoundV2Config, "nv_fp", 16, "nv_fp4_with_static_gs"),
                (RTNConfig, "nv_fp", 16, "nv_fp4_with_static_gs"),
                (OptimizedRTNConfig, "mx_fp", 32, "mx_fp"),
            ):
                tuning = (
                    {"iters": 2, "not_use_best_mse": True}
                    if config_class in (SignRoundConfig, SignRoundV2Config)
                    else {}
                )
                configs.append(
                    config_class(
                        bits=4,
                        group_size=group_size,
                        data_type=data_type,
                        act_bits=4,
                        act_group_size=group_size,
                        act_data_type=act_type,
                        act_sym=True,
                        act_dynamic=True,
                        **tuning,
                    )
                )
        original_full_tensor = DTensor.full_tensor

        def no_weight_gather(value, *args, **kwargs):
            assert tuple(value.shape) not in ((64, 64), (128, 64), (64, 128)), "full weight gather during tuning"
            return original_full_tensor(value, *args, **kwargs)

        for config in configs:
            compressor = AutoRound(
                checkpoint,
                parallel_config={"tp_size": 2},
                alg_configs=config,
                dataset=torch.utils.data.DataLoader(tokens, batch_size=1),
                nsamples=2,
                seqlen=4,
                batch_size=1,
                model_dtype="float32",
                # Loading in fp32 alone does not stop AMP from casting to bf16.
                # Compare TP reductions with full GEMMs at fp32 precision.
                amp=False,
                low_cpu_mem_usage=True,
                enable_torch_compile=False,
                format="auto_round",
            )
            compressor.post_init()
            assert not compressor.compress_context.is_immediate_packing
            assert not compressor.compress_context.is_immediate_saving
            dispatch = compressor.alg_composer.dispatch_block

            def check_residency(block, inputs, others):
                result = dispatch(block, inputs, others)
                if device_type == "cuda":
                    active = {id(parameter) for parameter in result.parameters()}
                    for candidate in compressor.model.model.layers:
                        for parameter in candidate.parameters():
                            assert parameter.device.type == ("cuda" if id(parameter) in active else "cpu")
                return result

            with (
                patch(
                    "auto_round.parallel.distributed.setup_ddp_if_needed_", side_effect=AssertionError("TP is not DP")
                ),
                patch.object(DTensor, "full_tensor", no_weight_gather),
                patch.object(compressor.alg_composer, "dispatch_block", check_residency),
            ):
                model, _ = compressor.quantize()
            assert model.dtype == torch.float32
            for block in model.model.layers:

                projection = block.self_attn.q_proj
                if isinstance(projection, WrapperWALayer):
                    projection = projection.orig_layer
                    assert projection.act_max_scale.device.type == "cpu"
                assert is_dtensor(projection.weight)
                assert is_dtensor(projection.scale)
                assert projection.weight.to_local().numel() * 2 == projection.weight.numel()
                assert all(parameter.device.type == "cpu" for parameter in block.parameters())
                assert projection.scale.device.type == "cpu"
            exported = gather_for_export(model)
            if device_type == "cuda":
                model.to(f"cuda:{rank}")
            inference_ids = tokens.to(f"cuda:{rank}" if device_type == "cuda" else "cpu")
            with torch.no_grad():
                logits = ParallelExecution.replicated(model(input_ids=inference_ids).logits)
            assert torch.isfinite(logits).all()
            if rank == 0:
                with torch.no_grad():
                    torch.testing.assert_close(exported(input_ids=tokens).logits, logits.cpu(), rtol=2e-4, atol=2e-5)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires distributed Gloo")
def test_two_rank_blockwise_quantization(tmp_path):
    checkpoint = _make_checkpoint(tmp_path)
    _run_workers(tmp_path, checkpoint)


def _make_checkpoint(tmp_path):
    from tokenizers import Tokenizer, models
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    checkpoint = tmp_path / "model"
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
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
    return checkpoint


def _run_workers(tmp_path, checkpoint, device_type="cpu"):
    with patch.object(sys, "path", [entry for entry in sys.path if Path(entry).is_absolute()]):
        workers = mp.spawn(
            _block_worker, args=(str(tmp_path / "rendezvous"), str(checkpoint), device_type), nprocs=2, join=False
        )
    try:
        deadline = time.monotonic() + 180
        while not workers.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("TP blockwise tuning timed out.")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
