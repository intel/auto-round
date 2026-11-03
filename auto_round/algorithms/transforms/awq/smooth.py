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
from __future__ import annotations

import contextlib
import inspect
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from auto_round.algorithms.transforms.smoothing.groups import SmoothGroup
from auto_round.data_type.utils import (
    compute_optimized_init_scale,
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
)
from auto_round.logger import logger

if TYPE_CHECKING:
    from auto_round.algorithms.transforms.awq.base import AWQTransform
    from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping


# Known normalization classes whose ``forward`` computes
# ``output = (1 + weight) * x_norm`` (Gemma-style "unit-offset" RMSNorm) rather
# than the standard ``output = weight * x_norm``. Folding an AWQ smoothing scale
# ``s`` into such a layer requires ``weight <- (1 + weight) / s - 1`` instead of
# ``weight <- weight / s``; using the wrong fold silently breaks AWQ's output
# invariance and severely degrades accuracy (e.g. Qwen3.5, Gemma2/3, Qwen3-Next).
_UNIT_OFFSET_RMSNORM_NAMES = frozenset(
    {
        "GemmaRMSNorm",
        "Gemma2RMSNorm",
        "Gemma3RMSNorm",
        "Gemma3TextRMSNorm",
        "Qwen3_5RMSNorm",
        "Qwen3_5MoeRMSNorm",
        "Qwen3NextRMSNorm",
    }
)

# Detects ``1 + self.weight`` / ``self.weight + 1`` in a norm's forward source.
_UNIT_OFFSET_SRC_RE = re.compile(r"1(\.0)?\s*\+\s*self\.weight|self\.weight(\.float\(\))?\s*\+\s*1")

# Cache the unit-offset decision per norm class to avoid repeated source parsing.
_unit_offset_cache: dict[type, bool] = {}


def _rmsnorm_has_unit_offset(module: torch.nn.Module) -> bool:
    """Return True if ``module`` applies a Gemma-style ``(1 + weight)`` gain.

    Uses a fast class-name allowlist, falling back to source inspection of the
    module's ``forward`` so newly-added Gemma-style norms are detected without a
    code change. Result is cached per class.
    """
    cls = type(module)
    cached = _unit_offset_cache.get(cls)
    if cached is not None:
        return cached
    result = cls.__name__ in _UNIT_OFFSET_RMSNORM_NAMES
    if not result:
        try:
            src = inspect.getsource(cls.forward)
            result = bool(_UNIT_OFFSET_SRC_RE.search(src))
        except (OSError, TypeError):
            result = False
    _unit_offset_cache[cls] = result
    return result


def smooth_group_from_mapping(mapping: ResolvedMapping) -> SmoothGroup:
    return SmoothGroup(
        key=mapping.smooth_name,
        projection_names=tuple(mapping.balance_names),
        projections=tuple(mapping.balance_layers),
        projection_input_key=mapping.activation_hook_target or mapping.balance_names[0],
        projection_input_module=mapping.balance_layers[0],
        evaluation_input_key=mapping.parent_name,
        evaluation_module=mapping.parent,
    )


class PerfSplit:
    """Wall-time buckets of one grid search for the ``[perf] awq grid split`` line.

    Disabled (zero cost) unless ``AR_PERF_COUNTERS`` is set. When enabled,
    each bucket ends with a device sync so it reports the device work it
    enqueued, not just the launch time.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            from auto_round import envs

            enabled = bool(envs.AR_PERF_COUNTERS)
        self.enabled = enabled
        self.walls: dict[str, float] = {}

    @contextlib.contextmanager
    def bucket(self, name: str, device: torch.device | None = None):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if device is not None and device.type != "cpu":
                from auto_round.utils.device_manager import get_ar_device

                get_ar_device(device).synchronize(device.index)
            self.walls[name] = self.walls.get(name, 0.0) + time.perf_counter() - t0

    def get(self, name: str) -> float:
        return self.walls.get(name, 0.0)


def candidate_scales(x_mean: torch.Tensor, ratio: float, use_duo: bool, w_mean: torch.Tensor | None) -> torch.Tensor:
    """One grid point's smoothing scales (shared by the serial and sharded searches)."""
    if use_duo:
        scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
    else:
        scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
    scales = scales / (scales.max() * scales.min()).sqrt()
    scales[torch.isinf(scales)] = 1
    scales[torch.isnan(scales)] = 1
    return scales


@dataclass
class MappingQDQ:
    """Per-mapping QDQ setup resolved once before a grid search."""

    params: dict
    quant_funcs: dict
    init_scales: dict
    w_mean: torch.Tensor | None


def resolve_mapping_qdq(owner: AWQTransform, mapping: ResolvedMapping) -> MappingQDQ:
    """Resolve the balance layers' quant params/functions at their current weights.

    Each balance layer keeps its own quant functions (a mapping requires
    compatible params, but per-layer resolution avoids hidden coupling to the
    first layer). Under ``v2_scale_search="reuse"`` the optimized init scale
    is computed once from the current weight and reused for every candidate;
    the final V2 quantizer still runs its own scale search.
    """
    device = mapping.balance_layers[0].weight.device
    params = {bl: owner._qdq_tool.resolve_params(bl) for bl in mapping.balance_layers}
    group_size = owner._normalize_group_size(params[mapping.balance_layers[0]]["group_size"], -1)
    w_mean = None
    if owner.duo_scaling is not False:
        w_mean = owner._compute_layer_means(mapping.balance_layers, group_size).to(device)
    quant_funcs = {bl: owner._qdq_tool.resolve_quant_funcs(params[bl]) for bl in mapping.balance_layers}
    init_scales = {}
    if owner._qdq_tool.v2_scale_search == "reuse":
        for bl in mapping.balance_layers:
            if quant_funcs[bl][1] is None:
                continue
            init_scales[bl] = compute_optimized_init_scale(
                bl.weight.data,
                params[bl]["data_type"],
                params[bl]["bits"],
                params[bl]["group_size"],
                imatrix=getattr(bl, "imatrix", None),
            )
    return MappingQDQ(params, quant_funcs, init_scales, w_mean)


class AWQSmoothStrategy:
    """AWQ statistics, ratio candidates, QDQ loss and scale folding policy."""

    def __init__(self, owner: AWQTransform, mapping: ResolvedMapping) -> None:
        self.owner = owner
        self.mapping = mapping
        self._candidates = None
        self._score = None
        self.perf = PerfSplit()

    def prepare(self, group: SmoothGroup, x_mean: torch.Tensor) -> None:
        mapping = self.mapping
        owner = self.owner
        device = mapping.balance_layers[0].weight.device
        x_mean = x_mean.to(device)

        qdq = resolve_mapping_qdq(owner, mapping)
        bl_params = qdq.params
        w_mean = qdq.w_mean

        parent_kwargs_list = owner._parent_args_cache.get(mapping.parent, [])
        use_parent_forward = len(parent_kwargs_list) > 0

        perf = self.perf
        if use_parent_forward:
            with perf.bucket("refs", device):
                fp16_outputs = owner._run_parent_samples(
                    mapping.parent,
                    parent_kwargs_list,
                    offload_to_cpu=owner._smooth_batch_size is not None,
                )
            if not fp16_outputs or all(f.numel() == 0 for f in fp16_outputs):
                use_parent_forward = False

        orig_state = {bl: bl.weight.data.clone() for bl in mapping.balance_layers}
        if not use_parent_forward:
            orig_weights = orig_state  # same reference is fine

        bl_quant_funcs = qdq.quant_funcs
        reused_init_scales = qdq.init_scales

        def candidates():
            for ratio, use_duo in owner._get_grid_search_params():
                yield ratio, candidate_scales(x_mean, ratio, use_duo, w_mean)

        def score(candidate):
            _, scales = candidate
            scales_view = scales.view(1, -1).to(device)
            if use_parent_forward:
                # Quantize each balance layer's smoothed weight and write the
                # de-smoothed result back, so the parent forward below sees the
                # weights the layer would actually compute with.
                with perf.bucket("qdq", device):
                    for bl in mapping.balance_layers:
                        quant_func, opt_quant_func = bl_quant_funcs[bl]
                        w_qdq = owner._qdq_tool.qdq(
                            orig_state[bl] * scales_view,
                            bl_params[bl],
                            quant_func=quant_func,
                            opt_quant_func=opt_quant_func,
                            imatrix=getattr(bl, "imatrix", None),
                            init_scale=reused_init_scales.get(bl),
                        )
                        bl.weight.data = (w_qdq / scales_view).to(bl.weight.dtype)

                with perf.bucket("replay", device):
                    total_loss = owner._compute_parent_loss(mapping.parent, parent_kwargs_list, fp16_outputs)
                for bl in mapping.balance_layers:
                    bl.weight.data.copy_(orig_state[bl])
            else:
                total_loss = 0.0
                for bl in mapping.balance_layers:
                    quant_func, opt_quant_func = bl_quant_funcs[bl]
                    w_orig = orig_weights[bl].to(device)
                    w_qdq = owner._qdq_tool.qdq(
                        w_orig * scales_view,
                        bl_params[bl],
                        quant_func=quant_func,
                        opt_quant_func=opt_quant_func,
                        imatrix=getattr(bl, "imatrix", None),
                        init_scale=reused_init_scales.get(bl),
                    )
                    total_loss += (w_orig - w_qdq / scales_view).pow(2).sum().item()

            return total_loss

        self._candidates = candidates
        self._score = score

    def candidates(self):
        return self._candidates()

    def score(self, candidate):
        return self._score(candidate)

    def apply(self, candidate) -> None:
        self.owner._apply_scales(self.mapping, candidate[1])

    def clear(self) -> None:
        # Drop closure-owned snapshots and outputs; shared parent calls remain
        # available to subsequent mappings and smoothing passes.
        self._candidates = None
        self._score = None


# Parents already reported as outside their block (one note per parent).
_OUT_OF_BLOCK_PARENTS: set[str] = set()


def _parent_shardable(mapping: ResolvedMapping, block_prefix: str | None) -> bool:
    """Whether the parent replay of ``mapping`` can run on private copies.

    The copy is the parent module, so every balance layer must live inside
    it, and the parent must belong to the block being processed (a parent
    outside the block may span devices or be arbitrarily large).
    """
    parent_name = mapping.parent_name or ""
    if not block_prefix or not (parent_name == block_prefix or parent_name.startswith(block_prefix + ".")):
        if parent_name not in _OUT_OF_BLOCK_PARENTS:
            _OUT_OF_BLOCK_PARENTS.add(parent_name)
            logger.info(
                "AWQ: parent '%s' lies outside block '%s'; its grid searches stay serial under --data_parallel.",
                parent_name,
                block_prefix,
            )
        return False
    inside = {id(m) for m in mapping.parent.modules()}
    return all(id(bl) in inside for bl in mapping.balance_layers)


# Returned by :func:`sharded_grid_search` when the search cannot shard.
SHARD_DECLINED = object()


@torch.no_grad()
def sharded_grid_search(owner: AWQTransform, mapping: ResolvedMapping, x_mean: torch.Tensor, block_prefix, seam):
    """Evaluate the smoothing grid with the parent replay sharded over ``seam``.

    The cached parent calls split into contiguous slices, one per device.
    Each device replays its slice on a private copy of the parent: reference
    outputs at the current weights first, then every candidate's QDQ weights.
    It returns the per-candidate loss SUMS plus its element count; the seam
    sums the partials, so ``sum / count`` is the serial per-candidate loss over
    all calls (up to fp32 summation order) and the selection is the serial
    one. Returns the best scales, ``None`` when no candidate is finite (as the
    serial search), or :data:`SHARD_DECLINED` when the search cannot shard
    (no cached parent calls, a parent outside the block, an empty replay).
    """
    from auto_round.algorithms.transforms.smoothing.errors import NoFiniteCandidateError
    from auto_round.algorithms.transforms.smoothing.search import search_candidates
    from auto_round.utils.model import move_to_device

    parent_kwargs_list = owner._parent_args_cache.get(mapping.parent, [])
    if not parent_kwargs_list or not _parent_shardable(mapping, block_prefix):
        return SHARD_DECLINED
    calls = [call for stored in parent_kwargs_list for call in owner._iter_parent_calls(*stored)]
    if len(calls) < 2:
        return SHARD_DECLINED

    device = mapping.balance_layers[0].weight.device
    x_mean = x_mean.to(device)
    qdq = resolve_mapping_qdq(owner, mapping)
    grid = list(owner._get_grid_search_params())
    scales_list = [candidate_scales(x_mean, ratio, use_duo, qdq.w_mean) for ratio, use_duo in grid]
    # smooth_batch_size bounds the replay memory: move each micro-call per
    # use and park the references on the host, like the serial replay
    bounded = owner._smooth_batch_size is not None
    perf_on = PerfSplit().enabled
    rank_perf: list[PerfSplit] = []

    def per_replica(parent, remap, calls_slice):
        pairs = [(bl, remap(bl)) for bl in mapping.balance_layers]
        dev = pairs[0][1].weight.device
        perf = PerfSplit(perf_on)
        rank_perf.append(perf)  # list.append is atomic under the GIL

        def _stage(call):
            args, kwargs = call
            return tuple(move_to_device(a, dev) for a in args), {k: move_to_device(v, dev) for k, v in kwargs.items()}

        # stage each call on the device ONCE: references and every
        # candidate reuse it (the cache is host-parked)
        with perf.bucket("stage", dev):
            staged = None if bounded else [_stage(c) for c in calls_slice]

        def _calls():
            return staged if staged is not None else (_stage(c) for c in calls_slice)

        refs = []
        with perf.bucket("refs", dev):
            for args, kwargs in _calls():
                ref = owner._normalize_parent_output(parent(*args, **kwargs)).detach()
                refs.append(ref.to("cpu") if bounded else ref)
        out = torch.zeros(len(grid) + 1, dtype=torch.float32, device=dev)
        out[-1] = float(sum(r.numel() for r in refs))
        orig = {r_bl: r_bl.weight.data.clone() for _bl, r_bl in pairs}
        init = {r_bl: qdq.init_scales[bl].to(dev) for bl, r_bl in pairs if bl in qdq.init_scales}
        try:
            for gi, scales in enumerate(scales_list):
                scales_view = scales.view(1, -1).to(dev)
                with perf.bucket("qdq", dev):
                    for bl, r_bl in pairs:
                        quant_func, opt_quant_func = qdq.quant_funcs[bl]
                        w_qdq = owner._qdq_tool.qdq(
                            orig[r_bl] * scales_view,
                            qdq.params[bl],
                            quant_func=quant_func,
                            opt_quant_func=opt_quant_func,
                            imatrix=getattr(r_bl, "imatrix", None),
                            init_scale=init.get(r_bl),
                        )
                        r_bl.weight.data = (w_qdq / scales_view).to(r_bl.weight.dtype)
                with perf.bucket("replay", dev):
                    loss = torch.zeros((), dtype=torch.float32, device=dev)
                    for (args, kwargs), ref in zip(_calls(), refs):
                        cand = owner._normalize_parent_output(parent(*args, **kwargs))
                        loss += torch.nn.functional.mse_loss(ref.to(dev).float(), cand.float(), reduction="sum")
                        del cand
                    out[gi] = loss
        finally:
            # the home slot replays on the home parent itself
            for r_bl, w in orig.items():
                r_bl.weight.data.copy_(w)
        return out

    _t0 = time.perf_counter()
    merged = seam.reduce(mapping.parent, per_replica, calls)
    if merged is None:
        return SHARD_DECLINED
    if perf_on and rank_perf:
        # per-bucket wall = the slowest device (the devices run concurrently)
        walls = {k: max(p.get(k) for p in rank_perf) for k in ("stage", "refs", "qdq", "replay")}
        logger.info(
            "[perf] awq grid split: mode=sharded name=%s world=%d calls=%d total=%.2fs build=%.2fs "
            "stage=%.2fs refs=%.2fs qdq=%.2fs replay=%.2fs",
            mapping.smooth_name,
            len(rank_perf),
            len(calls),
            time.perf_counter() - _t0,
            getattr(seam, "last_build_s", 0.0),
            walls["stage"],
            walls["refs"],
            walls["qdq"],
            walls["replay"],
        )
    merged = merged.cpu()
    count = float(merged[-1])
    if count <= 0:
        # no reference output survived anywhere: the serial search owns
        # the weight-space fallback
        return SHARD_DECLINED
    losses = (merged[:-1] / count).tolist()
    try:
        best, error = search_candidates(
            range(len(grid)),
            lambda i: losses[i],
            module_name=mapping.smooth_name,
            require_finite=False,
            log_search=False,
        )
    except NoFiniteCandidateError:
        logger.warning("AWQ: grid search failed for '%s': no finite error.", mapping.smooth_name)
        return None
    logger.debug(
        "AWQ '%s': best_ratio=%.2f, best_error=%.3e (sharded over %d calls)",
        mapping.smooth_name,
        grid[best][0],
        error,
        len(calls),
    )
    return scales_list[best].clone()


def get_grid_search_params(n_grid: int, duo_scaling: bool | str) -> list[tuple[float, bool]]:
    """Return (ratio, use_duo_scaling) tuples for the grid search."""
    match duo_scaling:
        case "both":
            n = max(int(n_grid / 2), 2)
            return [(idx / (n - 1), duo) for idx in range(n) for duo in [False, True]]
        case False:
            n = max(n_grid, 2)
            return [(idx / (n - 1), False) for idx in range(n)]
        case True:
            n = max(n_grid, 3)
            return [(0.0, False)] + [(idx / (n - 2), True) for idx in range(n - 1)]
        case _:
            raise ValueError(f"Unexpected duo_scaling value: {duo_scaling!r}")


def compute_layer_means(layers: list[torch.nn.Module], group_size: int) -> torch.Tensor:
    """Per-channel mean of normalised weights across all balance layers."""
    weight = torch.cat([m.weight.detach().float() for m in layers], dim=0)
    org_shape = weight.shape
    gs = group_size if group_size is not None and group_size > 0 else org_shape[1]
    weight, _, pad_len = reshape_pad_tensor_by_group_size(weight, gs)
    w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
    w_scale = revert_tensor_by_pad(w_scale, orig_shape=org_shape, pad_len=pad_len)
    return w_scale.mean(0)


@torch.no_grad()
def fold_scales_into_smooth_layer(smooth: torch.nn.Module, scales: torch.Tensor) -> None:
    """Divide a smooth layer's output by ``scales`` to offset balance scaling.

    Dispatches on the smooth layer's weight layout:

    * 1-D norm weight with a Gemma-style ``(1 + weight)`` gain: folded as
      ``weight <- (1 + weight) / s - 1`` to preserve output invariance.
    * 1-D standard norm weight: folded as ``weight <- weight / s``.
    * 2-D linear weight: its trailing ``s.numel()`` output rows are divided.

    Any bias is always divided by ``s``.
    """
    s = scales.to(smooth.weight.device)
    weight = smooth.weight.data
    if weight.ndim == 1:
        if _rmsnorm_has_unit_offset(smooth):
            weight.copy_((1.0 + weight) / s - 1.0)
        else:
            weight.div_(s)
    else:
        weight[-s.size(0) :].div_(s.view(-1, 1))

    if getattr(smooth, "bias", None) is not None:
        smooth.bias.data.div_(s)
