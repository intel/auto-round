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
"""RRQ inference module: multi-plane INT2 linear layer with dynamic precision.

Reference implementation.  Each plane is stored as its *dequantized*
float weight of shape ``(out_features, in_features)``.  ``forward`` accumulates
the first ``active_planes`` planes and runs a single linear op:

- active_planes=1: 2-bit (base only)
- active_planes=2: 4-bit (base + 1 residual)
- active_planes=3: 6-bit (base + 2 residuals)
- active_planes=4: 8-bit (base + 3 residuals)

This is a correctness reference implementation, not a performance kernel.
Packed-INT2 storage and fused kernels are a follow-up concern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_round.utils import logger

__all__ = [
    "RRQLinear",
    "materialize_rrq",
    "set_rrq_bits",
    "set_rrq_phase_bits",
    "set_rrq_random_residual",
    "set_rrq_weight_cache_budget",
]


# -- Dequantized-weight cache budget ------------------------------------------
#
# ``RRQLinear`` keeps packed INT2 planes on the device and dequantizes them on
# demand.  Retaining the dequantized weight per layer makes forwards fast but
# needs a dense copy of the whole model (a bf16 27B model is ~54 GB on top of
# the ~30 GB packed planes), which does not fit on a single 80 GB card.  A
# bounded budget lets small models cache automatically while large ones fall
# back to rebuilding per forward rather than failing with OOM.
_RRQ_CACHE_BUDGET_BYTES = 0  # 0 disables CUDA caching (set by load_rrq_model)
_RRQ_CACHE_USED_BYTES = 0


def set_rrq_phase_bits(
    model: nn.Module,
    prefill_bits: int | None = None,
    decode_bits: int | None = None,
    prefill_threshold: int = 1,
) -> dict:
    """Use one precision for prefill and another for decode.

    Prefill (prompt processing, and the verify step of speculative decoding) is
    compute bound: it pushes many tokens through each weight load, so the matmul
    dominates and a wider weight costs almost nothing extra.  Decode emits one
    token at a time and is bandwidth bound: every weight is streamed for a
    single token, so the weight width sets the speed.  Running prefill at a
    higher precision than decode therefore captures most of the accuracy of the
    higher precision at close to the decode cost of the lower one.

    The phase is inferred from the sequence length, which needs no plumbing
    through the model: a forward whose sequence length exceeds
    ``prefill_threshold`` is prefill, everything else is decode.  This works
    with vanilla ``transformers`` generation and ``model.generate``, because
    both feed ``(batch, seq)`` tensors where decode has ``seq == 1`` -- including
    batched decode, whose total token count equals the batch size but whose
    sequence length is still 1.

    Note that this is the *reference* implementation: both precisions are
    dequantized on demand, so the speedup is bounded by the packed-weight
    bandwidth.  The real win appears once a fused kernel selects the plane count
    from the token count without materializing the weight.

    Args:
        model: Model containing ``RRQLinear`` modules.
        prefill_bits: Effective bit-width during prefill (2/4/6/8).  ``None``
            keeps every plane the checkpoint holds.
        decode_bits: Effective bit-width during decode (2/4/6/8).
        prefill_threshold: Minimum sequence length that counts as prefill.  The
            default ``1`` means "more than one token per sequence": decode feeds
            one token per sequence, so anything longer is prefill.  Compare the
            sequence length (``x.shape[-2]``), not the total element count --
            batched decode has many tokens but a sequence length of 1.

    Returns:
        Dict with the resolved ``prefill_bits`` / ``decode_bits``.

    Raises:
        ValueError: If a bit-width is invalid or no RRQ layers exist.
    """
    valid_bits = {2, 4, 6, 8}
    for label, value in (("prefill_bits", prefill_bits), ("decode_bits", decode_bits)):
        if value is not None and value not in valid_bits:
            raise ValueError(f"{label} must be one of {sorted(valid_bits)} or None, got {value}")
    if prefill_threshold < 1:
        raise ValueError(f"prefill_threshold must be >= 1, got {prefill_threshold}")

    layers = [m for m in model.modules() if isinstance(m, RRQLinear)]
    if not layers:
        raise ValueError("No RRQLinear modules found in model.")

    resolved_prefill, resolved_decode = prefill_bits, decode_bits
    for module in layers:
        max_bits = module.num_planes * module.bits
        if resolved_prefill is None:
            resolved_prefill = max_bits
        if resolved_decode is None:
            resolved_decode = max_bits
        if max(resolved_prefill, resolved_decode) > max_bits:
            raise ValueError(
                f"Requested prefill={resolved_prefill}-bit / decode={resolved_decode}-bit "
                f"but a layer only holds {max_bits} bits of planes."
            )
        module.set_phase_bits(resolved_prefill, resolved_decode, prefill_threshold)

    logger.info(
        "RRQ phase precision: prefill=%d-bit, decode=%d-bit (prefill when >%d tokens).",
        resolved_prefill,
        resolved_decode,
        prefill_threshold,
    )
    return {"prefill_bits": resolved_prefill, "decode_bits": resolved_decode}


def set_rrq_weight_cache_budget(num_bytes: int | None = None) -> int:
    """Set how many bytes of dequantized weights may be cached on CUDA.

    The budget is all-or-nothing in practice: a single forward pass visits every
    RRQ layer, so holding only part of the dequantized weights does not speed
    anything up (the uncached layers dominate).  Callers should therefore pass
    the *total* footprint needed for the whole model, or ``0``.

    Args:
        num_bytes: Byte budget for cached dequantized weights.  ``None`` leaves
            caching disabled -- use :func:`required_rrq_cache_bytes` to size it
            from the model.  ``0`` disables caching (lowest memory, slowest).

    Returns:
        The budget that is now in effect.
    """
    global _RRQ_CACHE_BUDGET_BYTES, _RRQ_CACHE_USED_BYTES
    if num_bytes is None:
        num_bytes = 0
    _RRQ_CACHE_BUDGET_BYTES = max(0, int(num_bytes))
    _RRQ_CACHE_USED_BYTES = 0
    if _RRQ_CACHE_BUDGET_BYTES:
        logger.info("RRQ dequantized-weight cache budget set to %.1f GiB", _RRQ_CACHE_BUDGET_BYTES / 2**30)
    return _RRQ_CACHE_BUDGET_BYTES


def required_rrq_cache_bytes(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> int:
    """Bytes needed to cache the dequantized weight of every RRQ layer in ``model``.

    A layer with phase precision enabled needs **two** cached weights (one per
    phase), so it is counted twice; caching only one of them would force the
    other phase to rebuild on every step.

    Args:
        model: Model containing ``RRQLinear`` modules.
        dtype: Storage dtype the cache would use (bf16 by default).

    Returns:
        Total bytes for one dense copy of all RRQ layers, or ``0`` if the model
        has no RRQ layers.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    total = 0
    for module in model.modules():
        if isinstance(module, RRQLinear):
            per_layer = module.in_features * module.out_features * element_size
            slots = 1 if module.phase_bits is None else 2
            total += per_layer * slots
    return total


def enable_rrq_weight_cache_if_it_fits(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> bool:
    """Cache all dequantized weights when the device has room; otherwise disable.

    A partial cache cannot help: every forward visits every layer, so the
    uncached ones dominate.  This enables the cache only when the *whole* dense
    copy fits in the free device memory (keeping 20% headroom for activations),
    and otherwise leaves it disabled so the model still runs, just slower.

    Args:
        model: Model containing ``RRQLinear`` modules.
        dtype: Cache storage dtype.

    Returns:
        True if caching was enabled.
    """
    needed = required_rrq_cache_bytes(model, dtype)
    if needed == 0:
        set_rrq_weight_cache_budget(0)
        return False
    try:
        device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - parameterless model
        set_rrq_weight_cache_budget(0)
        return False
    if device.type != "cuda":
        # CPU always caches; the budget only gates CUDA.
        set_rrq_weight_cache_budget(0)
        return True

    free, _total = torch.cuda.mem_get_info(device)
    if needed > free * 0.8:
        set_rrq_weight_cache_budget(0)
        logger.info(
            "RRQ: %.1f GiB needed to cache dequantized weights but only %.1f GiB is free; "
            "running without the cache (slow). Call auto_round.inference.materialize_rrq(model) "
            "once for ~30x faster dense inference.",
            needed / 2**30,
            free / 2**30,
        )
        return False
    set_rrq_weight_cache_budget(needed)
    return True


def _rrq_cache_acquire(num_bytes: int) -> bool:
    """Reserve ``num_bytes`` of the cache budget; False if it does not fit."""
    global _RRQ_CACHE_USED_BYTES
    if _RRQ_CACHE_BUDGET_BYTES <= 0:
        return False
    if _RRQ_CACHE_USED_BYTES + num_bytes > _RRQ_CACHE_BUDGET_BYTES:
        return False
    _RRQ_CACHE_USED_BYTES += num_bytes
    return True


def _rrq_cache_release(num_bytes: int) -> None:
    """Return ``num_bytes`` to the cache budget."""
    global _RRQ_CACHE_USED_BYTES
    _RRQ_CACHE_USED_BYTES = max(0, _RRQ_CACHE_USED_BYTES - num_bytes)


class RRQLinear(nn.Module):
    """Multi-plane INT2 quantized linear layer with dynamic precision switching.

    Each plane is stored as its *dequantized* float weight of shape
    ``(out_features, in_features)``.  ``forward`` accumulates the first
    ``active_planes`` planes and runs a single linear op::

        - active_planes=1: 2-bit (base only)
        - active_planes=2: 4-bit (base + 1 residual)
        - active_planes=3: 6-bit (base + 2 residuals)
        - active_planes=4: 8-bit (base + 3 residuals)

    Buffers:
        ``rrq_qweight_k``  -- dequantized weight of plane ``k`` (float),
                             shape ``[out_features, in_features]``.
        ``rrq_scales_k``  : per-group scale of plane ``k`` (informational).
        ``rrq_zp_k``      : zero-point of plane ``k`` (asymmetric only), if any.
    """

    def __init__(
        self,
        in_features: int | None = None,
        out_features: int | None = None,
        num_planes: int = 4,
        bits: int = 2,
        bias: bool = True,
        base: nn.Module | None = None,
        residual_planes=None,
    ) -> None:
        super().__init__()

        if base is not None or residual_planes is not None:
            residual_planes = list(residual_planes or [])
            self._packed = True
            self.base = base
            self.num_planes = 1 + len(residual_planes)
            self.in_features = base.infeatures
            self.out_features = base.outfeatures
            self.bits = base.bits
            self.planes = nn.ModuleDict({f"rrq_{index}": plane for index, plane in enumerate(residual_planes, start=1)})
            self.active_planes = self.num_planes
            if isinstance(bias, torch.Tensor):
                self.register_buffer("bias", bias)
            else:
                self.register_parameter("bias", None)
            self._packed_weight = None
            self._packed_weight_planes = None
            # Phase-dependent precision (see ``set_rrq_phase_bits``).  ``None``
            # means "no phase switching": ``active_planes`` is used as is.
            self._prefill_planes = None
            self._decode_planes = None
            self._prefill_threshold = 1
            return

        self._packed = False
        self.in_features = in_features
        self.out_features = out_features
        self.num_planes = num_planes
        self.bits = bits
        self.active_planes = num_planes  # default: use all planes
        self._prefill_planes = None
        self._decode_planes = None
        self._prefill_threshold = 1

        # Dequantized weight of each plane, shape (out_features, in_features).
        for k in range(num_planes):
            self.register_buffer(
                f"rrq_qweight_{k}",
                torch.zeros((out_features, in_features), dtype=torch.float16),
            )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    def _resolve_forward_planes(self, x: torch.Tensor) -> int:
        """Pick the plane count for this forward, honouring phase precision.

        Decode feeds one token per sequence while prefill feeds the whole
        prompt, so the **sequence length** separates the two phases.  The length
        is read from the second-to-last dimension rather than the total element
        count, because batched decode is ``(batch, 1, hidden)``: its total token
        count is the batch size, which would be misread as prefill.  When phase
        precision is disabled (``_prefill_planes is None``) this is just
        ``active_planes``.

        Args:
            x: Input activations of shape ``(..., seq, in_features)``.

        Returns:
            The number of planes to accumulate for this forward.
        """
        if self._prefill_planes is None:
            return self.active_planes
        if x.dim() >= 2:
            seq_len = x.shape[-2]
        else:  # 1-D input: a single token
            seq_len = 1
        if seq_len > self._prefill_threshold:
            return self._prefill_planes
        return self._decode_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic plane selection.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        planes = self._resolve_forward_planes(x)
        if self._packed:
            # Dequant each active plane once, accumulate them into a single
            # weight, then run one matmul + bias. This replaces four separate
            # GEMMs (one per plane) with a single GEMM.
            weight = self._get_packed_weight(planes)
            out = torch.matmul(x, weight.to(x.dtype))
            out = out.to(x.dtype)
            # The base plane may own the bias; add it if present.
            if self.base.bias is not None:
                out = out + self.base.bias.to(out.dtype)
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out

        weight = self._dequantize(planes)
        out = F.linear(x.to(weight.dtype), weight)
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out

    # -- Packed (production) path helpers -----------------------------------------

    def _get_packed_weight(self, num_planes: int) -> torch.Tensor:
        """Build the dequantized accumulated weight for ``num_planes`` planes.

        Each plane is dequantized via :meth:`QuantLinear._dequantize` and the
        results are summed in float32 to avoid precision loss.

        The cache is keyed by plane count rather than holding a single slot,
        because phase precision alternates between two counts on every request
        (prefill then decode).  A single slot would evict the weight that the
        next phase needs, so both phases would rebuild on every step and the
        layer would be slower than plain active-plane selection.

        Caching policy: a dense copy of a large model does not fit alongside its
        packed representation (a bf16 copy of a 27B model is ~54 GB), so caching
        every layer unconditionally OOMs.  The cache is therefore bounded:

        - On CPU (unit tests, small reference models) the result is always
          cached -- repeated forwards are the common case and memory is ample.
        - On CUDA caching is allowed only while the global budget set by
          :func:`set_rrq_weight_cache_budget` has room.  ``load_rrq_model``
          sizes that budget from the free device memory, so models that fit are
          fast automatically and larger ones degrade to rebuilding per forward
          instead of failing.

        Note that a *partial* cache only helps when it covers a layer once per
        forward pass; for real deployment prefer :func:`materialize_rrq`, which
        converts the layers to dense ``nn.Linear`` once and is ~50x faster.
        """
        device = self.base.qweight.device
        if self._packed_weight is None:
            self._packed_weight = {}
            self._packed_weight_planes = set()
        cached = self._packed_weight.get(num_planes)
        if cached is not None and cached.device == device:
            return cached

        # float32 is used only for the running accumulation.
        weight = self.base._dequantize().float()
        for index in range(1, num_planes):
            weight = weight + self.planes[f"rrq_{index}"]._dequantize().float()

        # Cache in bf16 on CUDA: it halves the footprint (letting roughly twice
        # as many models cache) and matches the compute dtype, so the matmul
        # needs no extra conversion.  CPU keeps fp32 for exact reference numerics.
        store_dtype = torch.float32 if device.type != "cuda" else torch.bfloat16
        nbytes = weight.numel() * torch.tensor([], dtype=store_dtype).element_size()
        if self._can_cache(nbytes):
            stored = weight.to(store_dtype)
            self._packed_weight[num_planes] = stored
            self._packed_weight_planes.add(num_planes)
            return stored
        return weight

    def _can_cache(self, nbytes: int) -> bool:
        """Return whether ``nbytes`` of cache budget can be reserved."""
        if not torch.cuda.is_available():
            return True
        if self.base.qweight.device.type != "cuda":
            return True
        return _rrq_cache_acquire(nbytes)

    def _release_cached_weight(self) -> None:
        """Drop the cached weights and return their bytes to the budget."""
        if self._packed_weight:
            for stored in self._packed_weight.values():
                _rrq_cache_release(stored.numel() * stored.element_size())
        self._packed_weight = None
        self._packed_weight_planes = None

    def reset_packed_weight_cache(self) -> None:
        """Invalidate the cached weight (call after in-place weight changes)."""
        self._release_cached_weight()

    # -- Phase-dependent precision ------------------------------------------------

    def set_phase_bits(
        self,
        prefill_bits: int,
        decode_bits: int,
        prefill_threshold: int = 1,
    ) -> None:
        """Use different precisions for prefill and decode in this layer.

        Args:
            prefill_bits: Effective bit-width when the forward carries more than
                ``prefill_threshold`` tokens.
            decode_bits: Effective bit-width otherwise.
            prefill_threshold: Sequence length above which a forward counts as
                prefill.

        Raises:
            ValueError: If the bit-widths are not multiples of ``self.bits`` or
                exceed the planes available.
        """
        max_bits = self.num_planes * self.bits
        for label, value in (("prefill_bits", prefill_bits), ("decode_bits", decode_bits)):
            if value % self.bits != 0 or not self.bits <= value <= max_bits:
                raise ValueError(
                    f"{label} must be a multiple of {self.bits} in [ {self.bits}, {max_bits} ], got {value}"
                )
        if prefill_threshold < 1:
            raise ValueError(f"prefill_threshold must be >= 1, got {prefill_threshold}")

        self._prefill_planes = prefill_bits // self.bits
        self._decode_planes = decode_bits // self.bits
        self._prefill_threshold = prefill_threshold
        # Prefill dominates the visible precision, so report it as active.
        self.active_planes = self._prefill_planes
        # Drop (and refund) any cached weight so the budget reflects the new
        # slot count before use.
        self._release_cached_weight()

    def disable_phase_bits(self) -> None:
        """Turn off phase precision and fall back to ``active_planes``."""
        self._release_cached_weight()
        self._prefill_planes = None
        self._decode_planes = None
        self._prefill_threshold = 1
        self._packed_weight = None
        self._packed_weight_planes = None

    @property
    def phase_bits(self) -> tuple[int, int] | None:
        """``(prefill_bits, decode_bits)`` when enabled, else ``None``."""
        if self._prefill_planes is None:
            return None
        return self._prefill_planes * self.bits, self._decode_planes * self.bits

    # -- Non-packed (reference) path helpers --------------------------------------

    def _dequantize(self, num_planes: int) -> torch.Tensor:
        """Accumulate the first ``num_planes`` planes into a full weight tensor.

        Args:
            num_planes: Number of planes to accumulate (1 <= num_planes <= num_planes).

        Returns:
            Reconstructed weight tensor of shape ``(out_features, in_features)``.
        """
        assert 1 <= num_planes <= self.num_planes, f"active_planes must be in [1, {self.num_planes}], got {num_planes}"

        total = torch.zeros(
            self.out_features,
            self.in_features,
            dtype=torch.float32,
            device=self.rrq_qweight_0.device,
        )
        for k in range(num_planes):
            plane = getattr(self, f"rrq_qweight_{k}").to(torch.float32)
            total += plane

        return total

    def set_active_planes(self, num_planes: int) -> None:
        """Set how many planes are used (1..num_planes)."""
        if not 1 <= num_planes <= self.num_planes:
            raise ValueError(f"active_planes must be in [1, {self.num_planes}], got {num_planes}")
        self.active_planes = num_planes

    def set_active_bits(self, bits: int) -> None:
        """Compatibility wrapper for callers that select precision in bits."""
        if bits not in (2, 4, 6, 8) or bits % self.bits != 0:
            raise ValueError(f"active_bits must be one of 2/4/6/8, got {bits}")
        self.set_active_planes(bits // self.bits)

    @property
    def active_bits(self) -> int:
        return self.active_planes * self.bits

    @active_bits.setter
    def active_bits(self, bits: int) -> None:
        self.set_active_bits(bits)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, num_planes={self.num_planes}, "
            f"active_planes={self.active_planes}"
        )


def set_rrq_bits(model: nn.Module, bits: int) -> None:
    """Set the active precision for all RRQLinear layers in a model.

    This disables phase-dependent precision (see :func:`set_rrq_phase_bits`),
    because a single bit-width and a prefill/decode pair are contradictory
    settings; call ``set_rrq_phase_bits`` again afterwards if both are wanted.

    Args:
        model: Model containing ``RRQLinear`` modules.
        bits: Target effective bit-width (2, 4, 6, or 8).

    Raises:
        ValueError: If ``bits`` is not in {2, 4, 6, 8} or no RRQ layers exist.
    """
    valid_bits = {2, 4, 6, 8}
    if bits not in valid_bits:
        raise ValueError(f"RRQ supports only bits in {valid_bits}, got {bits}")

    num_planes = bits // 2  # each plane is 2 bits
    found = False
    disabled_phase = False
    for module in model.modules():
        if isinstance(module, RRQLinear):
            if num_planes > module.num_planes:
                raise ValueError(
                    f"Requested {bits}-bit precision ({num_planes} planes) but "
                    f"{module} only has {module.num_planes} planes."
                )
            if module.phase_bits is not None:
                module.disable_phase_bits()
                disabled_phase = True
            module.active_planes = num_planes
            found = True

    if not found:
        logger.warning("No RRQLinear modules found in model; set_rrq_bits had no effect.")
    elif disabled_phase:
        logger.info("set_rrq_bits disabled phase precision; call set_rrq_phase_bits to re-enable it.")


def set_rrq_random_residual(
    model: nn.Module,
    fraction: float = 0.5,
    seed: int = 0,
    high_bits: int = 4,
    low_bits: int = 2,
) -> int:
    """Randomly give a fraction of RRQLinear layers the higher precision.

    A reproducible (seeded) uniform sample of ``fraction`` of the RRQLinear
    layers is set to ``high_bits`` effective bits; the rest use ``low_bits``.
    This yields a mixed operating point between the two levels (e.g. ``0.5`` of
    the layers at 4-bit and the rest at 2-bit is an effective ~3-bit model).

    Args:
        model: Model containing ``RRQLinear`` modules.
        fraction: Fraction of layers assigned ``high_bits`` (0.0 <= f <= 1.0).
        seed: RNG seed for the layer selection (reproducible).
        high_bits: Effective bit-width for the selected layers (2/4/6/8).
        low_bits: Effective bit-width for the rest (2/4/6/8).

    Returns:
        The number of layers assigned ``high_bits``.

    Raises:
        ValueError: If ``fraction`` is out of range, bit-widths are invalid,
            or no RRQLinear modules exist.
    """
    import random

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0.0, 1.0], got {fraction}")
    valid_bits = {2, 4, 6, 8}
    if high_bits not in valid_bits or low_bits not in valid_bits:
        raise ValueError(f"high_bits/low_bits must be in {valid_bits}, got {high_bits}/{low_bits}")

    layers = [(name, m) for name, m in model.named_modules() if isinstance(m, RRQLinear)]
    if not layers:
        raise ValueError("No RRQLinear modules found in model.")

    high_planes = high_bits // 2
    low_planes = low_bits // 2
    for name, m in layers:
        if max(high_planes, low_planes) > m.num_planes:
            raise ValueError(
                f"Requested {max(high_bits, low_bits)}-bit but {name!r} only has " f"{m.num_planes} planes."
            )

    names = [name for name, _ in layers]
    n_high = round(fraction * len(names))
    keep = set(random.Random(seed).sample(names, n_high))
    for name, m in layers:
        if m.phase_bits is not None:
            m.disable_phase_bits()
        m.active_planes = high_planes if name in keep else low_planes

    logger.info(
        f"RRQ random residual: {n_high}/{len(names)} layers at {high_bits}-bit, "
        f"rest at {low_bits}-bit (fraction={fraction}, seed={seed})."
    )
    return n_high


def materialize_rrq(model: nn.Module, dtype: torch.dtype | None = None) -> int:
    """Convert every ``RRQLinear`` into a dense ``nn.Linear`` and free the planes.

    ``RRQLinear`` dequantizes its active planes on every forward.  That keeps
    memory low (only the packed INT2 planes are resident) but is roughly 50x
    slower than a plain matmul, because each layer rebuilds its weight tensor
    for every token.  Materializing once pays the dequantization a single time
    and then runs ordinary dense GEMMs.

    The conversion is numerically equivalent: the same float32 accumulation is
    performed, then rounded to ``dtype``.  Layers are converted one at a time
    and their packed planes are released immediately, so the transient peak
    stays close to the final dense footprint rather than double it.

    Trade-off: the dense weights need ~2 bytes/parameter (a 27B bf16 model is
    ~54 GB), against ~30 GB for the packed 4-plane checkpoint, and the result is
    no longer switchable between 2/4/6/8-bit -- it is frozen at the precision
    that was active when it was called.

    Args:
        model: Model whose ``RRQLinear`` modules should be materialized.
        dtype: Storage dtype for the dense layers.  Defaults to the model's own
            dtype, except on CPU where float32 is used so reference numerics
            stay exact.

    Returns:
        The number of layers that were converted.
    """
    import gc

    from auto_round.utils import set_module

    if dtype is None:
        # ``nn.Module`` itself has no ``dtype``; HuggingFace models do.  Fall
        # back to the first parameter's dtype, then to bf16.
        model_dtype = getattr(model, "dtype", None)
        if model_dtype is None:
            try:
                model_dtype = next(model.parameters()).dtype
            except StopIteration:  # pragma: no cover - parameterless model
                model_dtype = torch.bfloat16
        dtype = model_dtype if model_dtype.is_floating_point else torch.bfloat16
    try:
        first_device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - parameterless model
        first_device = torch.device("cpu")
    if first_device.type == "cpu":
        dtype = torch.float32

    names = [name for name, module in model.named_modules() if isinstance(module, RRQLinear)]
    for name in names:
        module = model.get_submodule(name)
        planes = module.active_planes

        weight = module.base._dequantize().float()
        for index in range(1, planes):
            weight = weight + module.planes[f"rrq_{index}"]._dequantize().float()

        bias = None
        if module.base.bias is not None:
            bias = module.base.bias.detach().to(dtype).clone()

        linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=bias is not None,
            device=weight.device,
            dtype=dtype,
        )
        with torch.no_grad():
            # ``_dequantize`` yields (in_features, out_features); nn.Linear
            # stores (out_features, in_features).
            linear.weight.copy_(weight.to(dtype).t())
            if bias is not None:
                linear.bias.copy_(bias)
        set_module(model, name, linear)

        # Release this layer's packed INT2 planes right away to keep the peak low.
        module._release_cached_weight()
        module.base = None
        module.planes = None
        module._prefill_planes = None
        module._decode_planes = None
        del weight, module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Materialized %d RRQ layers to dense %s linear layers", len(names), dtype)
    return len(names)
