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
"""Shared Hypothesis strategies and numeric oracles for the ``*_fuzz`` test folders.

Profiles (``ci`` / ``deep``) are registered in ``test/conftest.py``.  Tensor
strategies draw a compact *recipe* (shape, dtype, value pattern, seed, magnitude)
and materialize it with a seeded generator, so Hypothesis shrinks the recipe
instead of individual elements and failures stay readable and reproducible.
"""

import os
from dataclasses import dataclass

import torch
from hypothesis import note, settings
from hypothesis import strategies as st


def _detect_fuzz_device() -> str:
    override = os.environ.get("AR_FUZZ_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


# Tensor fuzz tests under ``test/unit/common`` run on the accelerator of the CI
# machine executing them, and on CPU in the CPU pipeline.
FUZZ_DEVICE = _detect_fuzz_device()

FLOAT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

TENSOR_PATTERNS = (
    "normal",
    "uniform",
    "zeros",
    "constant",
    "outlier",
    "positive",
    "negative",
    "tiny",
    "row_scaled",
    "sparse",
    "two_values",
)

# Column counts biased toward group-size boundaries (16/32/64/128) to exercise padding.
_INTERESTING_COLS = (1, 2, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 192, 256)


def scaled_settings(factor: float, **kwargs) -> settings:
    """Scale the active profile's example budget for properties that are expensive per example."""
    return settings(max_examples=max(1, int(settings.default.max_examples * factor)), **kwargs)


def weight_shapes(max_rows: int = 8, max_cols: int = 300):
    cols = st.one_of(st.sampled_from([c for c in _INTERESTING_COLS if c <= max_cols]), st.integers(1, max_cols))
    return st.tuples(st.integers(1, max_rows), cols)


def and_shapes(max_dims: int = 3, max_side: int = 40):
    """2-D or higher shapes; the grouping helpers index ``shape[1]`` and do not accept 1-D input."""
    return st.lists(st.integers(1, max_side), min_size=2, max_size=max(2, max_dims)).map(tuple)


def make_tensor(shape, dtype, pattern: str, seed: int, magnitude: float, device="cpu") -> torch.Tensor:
    """Deterministically build a float tensor following one of ``TENSOR_PATTERNS``."""
    g = torch.Generator().manual_seed(seed)
    shape = tuple(shape)
    randn = torch.randn(shape, generator=g, dtype=torch.float32)
    if pattern == "normal":
        t = randn
    elif pattern == "uniform":
        t = torch.rand(shape, generator=g) * 2 - 1
    elif pattern == "zeros":
        t = torch.zeros(shape)
    elif pattern == "constant":
        t = torch.full(shape, 1.0 if seed % 2 else -1.0)
    elif pattern == "outlier":
        t = randn.clone()
        flat = t.view(-1)
        flat[seed % flat.numel()] = 50.0 if seed % 2 else -50.0
    elif pattern == "positive":
        t = randn.abs()
    elif pattern == "negative":
        t = -randn.abs()
    elif pattern == "tiny":
        t = randn * 1e-6
    elif pattern == "row_scaled":
        exps = torch.randint(-3, 2, shape[:-1] + (1,), generator=g).to(torch.float32)
        t = randn * torch.pow(10.0, exps)
    elif pattern == "sparse":
        t = randn * (torch.rand(shape, generator=g) < 0.1)
    elif pattern == "two_values":
        t = torch.where(torch.rand(shape, generator=g) < 0.5, 0.75, -1.25)
    else:
        raise ValueError(f"unknown tensor pattern {pattern!r}")
    t = (t * magnitude).clamp(-6e4, 6e4)
    return t.to(dtype).to(device)


@st.composite
def tensors(
    draw,
    shapes=None,
    dtypes=FLOAT_DTYPES,
    patterns=TENSOR_PATTERNS,
    log10_magnitude=(-3, 2),
    device=None,
):
    """Draw a tensor recipe and materialize it on ``device`` (default: ``FUZZ_DEVICE``)."""
    shape = draw(shapes if shapes is not None else weight_shapes())
    dtype = draw(st.sampled_from(dtypes))
    pattern = draw(st.sampled_from(patterns))
    seed = draw(st.integers(0, 2**16))
    magnitude = 10.0 ** draw(st.integers(*log10_magnitude))
    note(f"tensor recipe: shape={tuple(shape)} dtype={dtype} pattern={pattern} seed={seed} magnitude={magnitude:g}")
    return make_tensor(shape, dtype, pattern, seed, magnitude, device=device or FUZZ_DEVICE)


def to_f64(t) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    return t.detach().to("cpu", torch.float64)


def dtype_slack(dtype: torch.dtype, ref: torch.Tensor) -> torch.Tensor:
    """Worst-case error from rounding ``ref`` (float64, cpu) into ``dtype``, incl. subnormal flush."""
    info = torch.finfo(dtype)
    return info.eps * ref.abs() + info.smallest_normal * info.eps


def assert_where(mask: torch.Tensor, message: str, **context: torch.Tensor) -> None:
    """Assert ``mask`` is all True; report the first violation with broadcast ``context`` values."""
    mask = mask.detach().cpu()
    if bool(mask.all()):
        return
    bad = (~mask).reshape(-1).nonzero()[0].item()
    details = []
    for name, value in context.items():
        value = torch.as_tensor(value).detach().cpu()
        details.append(f"{name}={value.broadcast_to(mask.shape).reshape(-1)[bad].item()!r}")
    raise AssertionError(f"{message}: {int((~mask).sum())}/{mask.numel()} violations; first: " + ", ".join(details))


def assert_shape_dtype_finite(qdq: torch.Tensor, ref: torch.Tensor) -> None:
    assert qdq.shape == ref.shape, f"shape changed: {tuple(ref.shape)} -> {tuple(qdq.shape)}"
    assert qdq.dtype == ref.dtype, f"dtype changed: {ref.dtype} -> {qdq.dtype}"
    assert qdq.device.type == ref.device.type, f"device changed: {ref.device} -> {qdq.device}"
    assert_where(torch.isfinite(qdq), "non-finite output for finite input", x=ref, qdq=qdq)


def assert_monotone_within_groups(xg: torch.Tensor, qg: torch.Tensor, dtype: torch.dtype) -> None:
    """Within each group (row of the float64 views), QDQ must be non-decreasing in its input."""
    order = xg.argsort(dim=-1)
    q_sorted = qg.gather(-1, order)
    steps = q_sorted[:, 1:] - q_sorted[:, :-1]
    assert_where(steps >= -dtype_slack(dtype, q_sorted[:, 1:]), "qdq is not monotone in its input within a group")


def assert_zero_and_sign_preserved(xg: torch.Tensor, qg: torch.Tensor) -> None:
    assert_where((xg != 0) | (qg == 0), "zero must quantize to zero", x=xg, qdq=qg)
    assert_where(xg * qg >= 0, "quantization flipped the sign", x=xg, qdq=qg)


@dataclass(frozen=True)
class FloatGrid:
    """Sign-magnitude minifloat with ``mbits`` explicit mantissa bits, subnormals below
    ``2**min_exp`` and saturation at ``max_norm``.  Uniform grids (MX-INT) are the
    special case where no value reaches the second binade."""

    mbits: int
    min_exp: int
    max_norm: float

    def _ulp(self, y: torch.Tensor) -> torch.Tensor:
        _, e = torch.frexp(y)
        e = torch.where(y == 0, torch.full_like(e, self.min_exp), (e - 1).clamp(min=self.min_exp))
        return torch.ldexp(torch.ones_like(y), e - self.mbits)

    def on_grid(self, values: torch.Tensor, rtol: float = 1e-6) -> torch.Tensor:
        """Elementwise: is ``values`` (float64) representable, up to relative ``rtol``?"""
        y = values.abs()
        k = y / self._ulp(y)
        return (y <= self.max_norm * (1 + rtol)) & ((k - k.round()).abs() <= rtol * k.clamp(min=1))

    def max_error(self, y: torch.Tensor) -> torch.Tensor:
        """Largest round-to-nearest + saturation error for magnitudes ``y`` (float64, >= 0)."""
        clipped = y.clamp(max=self.max_norm)
        return self._ulp(clipped) / 2 + (y - clipped)

    def snap(self, values: torch.Tensor) -> torch.Tensor:
        """Nearest representable value (float64), saturating at ``max_norm``."""
        y = values.abs().clamp(max=self.max_norm)
        ulp = self._ulp(y)
        return torch.sign(values) * ((y / ulp).round() * ulp).clamp(max=self.max_norm)


FP4_E2M1 = FloatGrid(mbits=1, min_exp=0, max_norm=6.0)
FP8_E4M3 = FloatGrid(mbits=3, min_exp=-6, max_norm=448.0)
FP8_E5M2 = FloatGrid(mbits=2, min_exp=-14, max_norm=57344.0)
UE5M3 = FloatGrid(mbits=3, min_exp=-14, max_norm=114688.0)


def mx_grid(ebits: int, mbits: int, max_norm: float) -> FloatGrid:
    """Element grid for an entry of ``auto_round.data_type.mxfp.MXFP_FORMAT_CACHE``.

    ``mbits`` there counts the sign and implicit bits, hence ``mbits - 2`` explicit bits.
    """
    min_exp = 0 if ebits == 0 else 2 - 2 ** (ebits - 1)
    return FloatGrid(mbits=mbits - 2, min_exp=min_exp, max_norm=max_norm)


def assert_on_scaled_grid(
    qdq: torch.Tensor, scale: torch.Tensor, grid: FloatGrid, dtype: torch.dtype, rtol: float = 0.0
) -> None:
    """``qdq / scale`` (float64, broadcastable) must lie on ``grid`` up to ``dtype`` output rounding."""
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    nearest = safe * grid.snap(qdq / safe)
    tol = dtype_slack(dtype, qdq) + rtol * qdq.abs()
    assert_where((qdq - nearest).abs() <= tol, "value off the quantization grid", qdq=qdq, scale=scale, nearest=nearest)


def group_view(t: torch.Tensor, group_size: int, pad_value: float = 0.0) -> torch.Tensor:
    """Independent re-implementation of the AutoRound grouping convention.

    ``-1``: one group per row; ``0``: the whole tensor is one group; a row shorter than
    ``group_size`` is one group; otherwise pad the last dim with ``pad_value`` and split into groups.
    """
    t = t.reshape(-1, t.shape[-1])
    if group_size == 0:
        return t.reshape(1, -1)
    if group_size == -1 or t.shape[-1] < group_size:
        return t
    pad = (-t.shape[-1]) % group_size
    if pad:
        t = torch.nn.functional.pad(t, (0, pad), value=pad_value)
    return t.reshape(-1, group_size)


Q_SCALE_THRESH = 1e-5


def wrapper_quant_kwargs(data_type, x, config, scale_dtype, mode, seed):
    """Fresh QDQ kwargs mirroring ``WrapperLinear._quant_call_kwargs``.

    ``mode="rtn"`` is the ``iters == 0`` layout (scalar ``v`` and margins, no ``tensor_min/max``);
    ``mode="tune"`` the tuning layout (per-element ``v``, per-group margins and clamped min/max),
    with the trainable values drawn from ``seed``.
    """
    from auto_round.data_type.utils import reshape_pad_tensor_by_group_size
    from auto_round.wrapper import get_scale_shape

    group_size = config["group_size"]
    kwargs = dict(
        scale_dtype=scale_dtype,
        data_type=data_type,
        q_scale_thresh=Q_SCALE_THRESH,
        imatrix=None,
        global_scale=None,
        init_scale=None,
        **config,
    )
    if mode == "rtn":
        one = torch.tensor(1.0, device=x.device)
        return dict(kwargs, v=torch.tensor(0.0, device=x.device), min_scale=one, max_scale=one.clone())
    g = torch.Generator().manual_seed(seed)
    reshaped = reshape_pad_tensor_by_group_size(x, group_size)[0]
    dims = (-2, -1) if isinstance(group_size, tuple) else (1,)
    scale_shape = get_scale_shape(x, group_size)
    return dict(
        kwargs,
        v=(torch.rand(reshaped.shape, generator=g) - 0.5).to(x.device),
        min_scale=torch.rand(scale_shape, generator=g).to(x.device),
        max_scale=torch.rand(scale_shape, generator=g).to(x.device),
        tensor_min=reshaped.amin(dim=dims).clamp(max=0),
        tensor_max=reshaped.amax(dim=dims).clamp(min=0),
    )
