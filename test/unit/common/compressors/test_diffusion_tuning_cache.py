import random
from types import SimpleNamespace

import pytest
import torch

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.algorithms.composer import BlockContext
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.compressors.diffusion.tuning_cache import DiffusionTuningCache, _auto_budget_bytes, _batch_plan, _nbytes
from auto_round.compressors.utils import IndexSampler
from auto_round.context.compress import CompressContext
from auto_round.schemes import W4A16
from auto_round.utils.device_manager import device_manager


@pytest.mark.parametrize("nsamples,global_batch", [(5, 1), (5, 2), (4, 4)])
def test_prefetch_plan_preserves_sampler_and_rng(nsamples, global_batch):
    state = random.getstate()
    try:
        random.seed(31)
        sampler = IndexSampler(nsamples, global_batch)
        before = random.getstate()
        predicted = list(_batch_plan(sampler, 8, 1))
        assert random.getstate() == before
        assert sampler.index == 0
        actual = [[i] for _ in range(8) for i in sampler.next_batch()]
        assert predicted == actual
    finally:
        random.setstate(state)


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(32, 32)

    def forward(self, hidden_states, temb=None):
        return self.proj(hidden_states) + temb


def test_tensor_backed_cache_selection():
    cache = DiffusionTuningCache()
    cache.runner = BlockForwardRunner(batch_size=1, device="cpu", cache_device="cpu", amp=False, is_diffusion=True)
    cache.inputs = {"hidden_states": torch.arange(24).reshape(3, 2, 4)}
    cache.others = {"temb": torch.arange(12).reshape(3, 4), "positional_inputs": []}
    cache.outputs = [torch.ones(1, 2, 4) * i for i in range(3)]
    inputs, others, reference = cache._select([1])
    torch.testing.assert_close(inputs["hidden_states"], cache.inputs["hidden_states"][1:2])
    torch.testing.assert_close(others["temb"], cache.others["temb"][1:2])
    torch.testing.assert_close(reference, cache.outputs[1])


def test_prefetched_forward_preserves_scalar_inputs():
    class ScaleBlock(torch.nn.Module):
        def forward(self, hidden_states, scale):
            return hidden_states * scale

    cache = DiffusionTuningCache()
    cache.runner = BlockForwardRunner(batch_size=1, device="cpu", cache_device="cpu", amp=False, is_diffusion=True)
    hidden = torch.ones(1, 2, 4)
    batch = ({"hidden_states": hidden, "scale": 2.0}, {"positional_inputs": []}, hidden)
    torch.testing.assert_close(cache.forward(ScaleBlock(), batch, "cpu"), 2 * hidden)


@pytest.mark.parametrize("shared", [True, False])
def test_prefetched_forward_preserves_list_conditioning(shared):
    class ConditionedBlock(torch.nn.Module):
        def forward(self, hidden_states, freqs):
            return hidden_states + freqs[0] + freqs[1]

    cache = DiffusionTuningCache()
    cache.runner = BlockForwardRunner(
        batch_size=1,
        device="cpu",
        cache_device="cpu",
        amp=False,
        is_diffusion=True,
        shared_cache_keys=("freqs",) if shared else (),
    )
    cache.inputs = {"hidden_states": [torch.ones(1, 2, 4)]}
    cache.others = {"freqs": [[torch.ones(1, 2, 4), torch.ones(1, 2, 4) * 2]], "positional_inputs": []}
    cache.outputs = cache.inputs["hidden_states"]
    expected = cache.runner.forward(ConditionedBlock(), cache.inputs, cache.others, [0], "cpu")
    batch = cache._select([0])
    torch.testing.assert_close(cache.forward(ConditionedBlock(), batch, "cpu"), expected)


def _tune(monkeypatch, device, budget, *, diffusion=True, low_mem=True, grad_acc=1, gap=-1, last=False, iters=8):
    torch.manual_seed(12)
    random.seed(34)
    monkeypatch.setattr(device_manager, "device", device)
    monkeypatch.setattr(device_manager, "_device_list", [device])
    block = TinyBlock().to(device).eval().requires_grad_(False)
    inputs = {"hidden_states": [torch.randn(1, 8, 32) for _ in range(5)]}
    others = {"temb": [torch.randn(1, 8, 32) for _ in range(5)], "positional_inputs": []}
    outputs = [
        block(x.to(device), t.to(device)).detach().cpu() for x, t in zip(inputs["hidden_states"], others["temb"])
    ]
    scheme = W4A16.copy()
    for key, value in scheme.to_dict().items():
        setattr(block.proj, key, value)
    block.proj.scale_dtype = torch.float32
    CompressContext.reset_context()
    cc = CompressContext(low_gpu_mem_usage=low_mem, enable_torch_compile=False)
    config = SignRoundConfig(
        scheme=scheme,
        iters=iters,
        lr=0.005,
        minmax_lr=0.005,
        gradient_accumulate_steps=grad_acc,
        dynamic_max_gap=gap,
        not_use_best_mse=last,
    )
    quantizer = SignRoundQuantizer(config)
    quantizer.bind(
        SimpleNamespace(
            model_context=SimpleNamespace(
                model=block,
                is_diffusion=diffusion,
                amp=False,
                amp_dtype=torch.float32,
                diffusion_tuning_cache_size=budget,
            ),
            compress_context=cc,
            calibration_context=SimpleNamespace(batch_size=1, batch_dim=0),
            scale_dtype=torch.float32,
            scheme_context=scheme,
        )
    )
    runner = BlockForwardRunner(
        batch_size=1, device=device, cache_device="cpu", amp=False, is_diffusion=True, enable_torch_compile=False
    )
    quantizer.bind_block_forward_runner(runner)
    trace = []
    get_loss = quantizer._get_loss

    def record_loss(pred, ref, indices, *args):
        loss = get_loss(pred, ref, indices, *args)
        trace.append((list(indices), loss.detach().cpu().item()))
        return loss

    quantizer._get_loss = record_loss
    best = quantizer.quantize_block(
        block,
        inputs,
        others,
        outputs,
        None,
        BlockContext(model=block, block_names=["blocks.0"], block_index=0, block_cnt=1, block_name="blocks.0"),
    )
    assert runner.cache_device == "cpu"
    return {k: v.detach().cpu().clone() for k, v in block.state_dict().items()}, trace, best


def test_cpu_does_not_enter_prefetch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU execution must not initialize CUDA prefetch")

    monkeypatch.setattr(DiffusionTuningCache, "create", forbidden)
    expected, trace, _ = _tune(monkeypatch, "cpu", 0)
    actual, other_trace, _ = _tune(monkeypatch, "cpu", 1)
    assert trace == other_trace
    assert all(torch.equal(v, actual[k]) for k, v in expected.items())


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
@pytest.mark.parametrize("budget", [0.01, "auto"])
@pytest.mark.parametrize("grad_acc,gap,last", [(1, -1, False), (2, -1, False), (1, 1, False), (1, -1, True)])
def test_cuda_tuning_equivalence_and_cleanup(monkeypatch, grad_acc, gap, last, budget):
    caches = []
    original = DiffusionTuningCache.create

    def capture(*args, **kwargs):
        cache = original(*args, **kwargs)
        if cache is not None:
            caches.append(cache)
        return cache

    monkeypatch.setattr(DiffusionTuningCache, "create", capture)
    expected, trace, _ = _tune(monkeypatch, "cuda:0", 0, grad_acc=grad_acc, gap=gap, last=last)
    actual, other_trace, best = _tune(monkeypatch, "cuda:0", budget, grad_acc=grad_acc, gap=gap, last=last)
    assert len(caches) == 1
    assert trace == other_trace
    assert all(torch.equal(v, actual[k]) for k, v in expected.items())
    assert caches[0].best["proj"]["value"].is_cuda
    assert not caches[0].thread.is_alive()
    assert caches[0].slots == []


@cuda
@pytest.mark.parametrize(
    "budget,diffusion,low_mem",
    [(0, True, True), (1, False, True), (1, True, False), ("auto", False, True), ("auto", True, False)],
)
def test_cuda_legacy_paths_do_not_create_cache(monkeypatch, budget, diffusion, low_mem):
    def forbidden(*args, **kwargs):
        pytest.fail("Legacy paths must not initialize prefetch")

    monkeypatch.setattr(DiffusionTuningCache, "create", forbidden)
    _tune(monkeypatch, "cuda:0", budget, diffusion=diffusion, low_mem=low_mem)


@cuda
def test_budget_fallback_keeps_quantization(monkeypatch):
    expected, trace, _ = _tune(monkeypatch, "cuda:0", 0)
    actual, other_trace, best = _tune(monkeypatch, "cuda:0", 1e-9)
    assert trace == other_trace
    assert all(torch.equal(v, actual[k]) for k, v in expected.items())
    assert not best["proj"]["value"].is_cuda


@cuda
def test_training_exception_closes_worker(monkeypatch):
    caches = []
    original = DiffusionTuningCache.create

    def capture(*args, **kwargs):
        cache = original(*args, **kwargs)
        caches.append(cache)
        return cache

    monkeypatch.setattr(DiffusionTuningCache, "create", capture)

    def fail(*args, **kwargs):
        raise RuntimeError("injected training failure")

    monkeypatch.setattr(SignRoundQuantizer, "_get_loss", fail)
    with pytest.raises(RuntimeError, match="injected training failure"):
        _tune(monkeypatch, "cuda:0", 0.01)
    assert len(caches) == 1
    assert not caches[0].thread.is_alive()
    assert caches[0].slots == []


def _cache_fixture():
    block = torch.nn.Linear(4, 4).cuda()
    block.orig_layer = object()
    block.params = {"value": torch.ones(4, device="cuda:0")}
    runner = BlockForwardRunner(batch_size=1, device="cuda:0", cache_device="cpu", amp=False, is_diffusion=True)
    inputs = {"hidden_states": [torch.full((1, 2, 4), float(i)) for i in range(3)]}
    outputs = [t.clone() for t in inputs["hidden_states"]]
    others = {"positional_inputs": []}
    sampler = IndexSampler(3, 1)
    sampler.indices = [0, 1, 2]
    return block, runner, inputs, others, outputs, sampler


@cuda
def test_staging_only_budget_and_dynamic_shape_fallback():
    block, runner, inputs, others, outputs, sampler = _cache_fixture()
    # The first batch fits, but the next batch has a different sequence length.
    inputs["hidden_states"][1] = torch.ones(1, 3, 4)
    outputs[1] = inputs["hidden_states"][1].clone()
    staging_bytes = 2 * (_nbytes(inputs["hidden_states"][0]) + _nbytes(outputs[0]))
    cache = DiffusionTuningCache.create(
        block, runner, inputs, others, outputs, sampler, 3, staging_bytes / 2**30, "cuda:0"
    )
    assert cache is not None and cache.best is None
    try:
        batch = cache.get(sampler.next_batch())
        torch.testing.assert_close(batch[0]["hidden_states"].cpu(), inputs["hidden_states"][0])
        assert cache.get(sampler.next_batch()) is None
        assert not cache.thread.is_alive()
    finally:
        cache.close()


@cuda
def test_allocation_oom_falls_back(monkeypatch):
    args = _cache_fixture()
    empty = torch.empty_like

    def fail_gpu(t, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            raise torch.OutOfMemoryError("injected allocation failure")
        return empty(t, **kwargs)

    monkeypatch.setattr(torch, "empty_like", fail_gpu)
    assert DiffusionTuningCache.create(*args, 3, 0.01, "cuda:0") is None


@cuda
def test_changed_sample_order_falls_back():
    args = _cache_fixture()
    cache = DiffusionTuningCache.create(*args, 3, 0.01, "cuda:0")
    try:
        assert cache.get([2]) is None
        assert not cache.thread.is_alive()
    finally:
        cache.close()


@pytest.mark.parametrize("free_gib,expected_gib", [(0.5, 0), (1, 0), (1.5, 0.5), (2, 1), (10, 5)])
def test_auto_budget_preserves_headroom(free_gib, expected_gib):
    assert _auto_budget_bytes(int(free_gib * 2**30)) == expected_gib * 2**30


@cuda
def test_auto_waits_for_full_optimizer_iteration(monkeypatch):
    steps = []
    original_step = SignRoundQuantizer._step
    original_create = DiffusionTuningCache.create

    def step(self, *args):
        result = original_step(self, *args)
        steps.append(1)
        return result

    def create(*args, **kwargs):
        assert len(steps) == 1
        return original_create(*args, **kwargs)

    monkeypatch.setattr(SignRoundQuantizer, "_step", step)
    monkeypatch.setattr(DiffusionTuningCache, "create", create)
    _tune(monkeypatch, "cuda:0", "auto", grad_acc=2)
    assert len(steps) == 8


@cuda
def test_auto_low_headroom_keeps_legacy_result(monkeypatch):
    expected, trace, _ = _tune(monkeypatch, "cuda:0", 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2**30, 32 * 2**30))
    actual, other_trace, best = _tune(monkeypatch, "cuda:0", "auto")
    assert trace == other_trace
    assert all(torch.equal(v, actual[k]) for k, v in expected.items())
    assert not best["proj"]["value"].is_cuda


@cuda
def test_auto_single_iteration_needs_no_cache(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("One iteration leaves nothing to prefetch")

    monkeypatch.setattr(DiffusionTuningCache, "create", forbidden)
    _tune(monkeypatch, "cuda:0", "auto", iters=1)
