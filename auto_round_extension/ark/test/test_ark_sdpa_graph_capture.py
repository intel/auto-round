# Copyright (c) 2025 Intel Corporation.
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
"""Tests that ARK SDPA kernels can be captured in a torch.xpu.XPUGraph.

Run from this directory:

    python -m pytest test_ark_sdpa_graph_capture.py -v -s

Requires the ARK extension (``auto_round_kernel``) to be built for the
current XPU runtime and an Intel GPU to be available.

What this exercises
-------------------
* Dense SDPA (``ark.sdpa``) — prefill (causal / non-causal, MHA & GQA) and
  decode.  The output tensor returned by ``ark.sdpa`` during capture lives
  in the graph's capture pool and keeps a stable device pointer across
  replays.  After mutating Q in-place and calling ``graph.replay()``, the
  same output buffer must hold the result for the new Q.
* Varlen (``ark.sdpa_varlen``) — currently only asserts that capture and
  one replay complete without raising.  Replay-numeric verification is not
  yet exercised because the varlen scratch allocation (``DeviceMemoryPool``)
  and grid sizing interact with graph replay in a way that requires
  further kernel-side validation.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parent))
import auto_round_kernel as ark  # noqa: E402
from ut_utils import is_xpu_available, reference_sdpa  # noqa: E402


def _graph_supported() -> bool:
    return all(hasattr(torch.xpu, n) for n in ("XPUGraph", "graph", "Stream"))


requires_graph = pytest.mark.skipif(
    not (is_xpu_available() and _graph_supported()),
    reason="requires XPU with torch.xpu.XPUGraph support",
)


# ---------------------------------------------------------------------------
# Capture helper
# ---------------------------------------------------------------------------


def _capture_dense_sdpa(q, k, v, is_causal, scale):
    """Warm up, then capture ark.sdpa into an XPUGraph.

    Returns (graph, output_tensor).  The output tensor lives in the
    XPU graph's capture pool and therefore retains a *stable* device
    address across replays, making it safe to compare after ``replay()``.
    """
    # --- eager warm-up (populates allocator pools, JITs kernels, etc.) ---
    with torch.no_grad():
        _ = ark.sdpa(q, k, v, is_causal=is_causal, scale=scale)
    torch.xpu.synchronize()

    # --- capture ---
    g = torch.xpu.XPUGraph()
    side = torch.xpu.Stream()
    cur = torch.xpu.current_stream()
    side.wait_stream(cur)
    with torch.no_grad(), torch.xpu.stream(side):
        torch.xpu.synchronize()
        with torch.xpu.graph(g):
            out = ark.sdpa(q, k, v, is_causal=is_causal, scale=scale)
    cur.wait_stream(side)
    torch.xpu.synchronize()
    return g, out


def _run_dense_case(b, hq, hkv, sq, skv, d, dtype, is_causal):
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(d)
    dev = "xpu:0"

    q = torch.randn(b, hq, sq, d, device=dev, dtype=dtype)
    k = torch.randn(b, hkv, skv, d, device=dev, dtype=dtype)
    v = torch.randn(b, hkv, skv, d, device=dev, dtype=dtype)
    q_new = torch.randn(b, hq, sq, d, device=dev, dtype=dtype)

    g, out = _capture_dense_sdpa(q, k, v, is_causal, scale)

    # --- replay with changed input ---
    q.copy_(q_new)
    torch.xpu.synchronize()
    g.replay()
    torch.xpu.synchronize()

    ref_new = reference_sdpa(q, k, v, is_causal=is_causal, scale=scale)
    diff = (out.float() - ref_new.float()).abs().max().item()
    assert torch.allclose(
        out.float(), ref_new.float(), atol=1e-2, rtol=1e-2
    ), f"dense SDPA graph replay mismatch: max_abs_diff={diff:.6f}"
    g.reset()


# ---------------------------------------------------------------------------
# Varlen helpers
# ---------------------------------------------------------------------------


def _build_varlen(batch, lengths_q, lengths_kv, hq, hkv, d, dtype):
    """Build varlen inputs: packed Q/K/V + cu_seqlens + max seqlens."""
    dev = "xpu:0"
    total_q = sum(lengths_q)
    total_kv = sum(lengths_kv)
    max_q = max(lengths_q)
    max_kv = max(lengths_kv)

    q = torch.randn(total_q, hq, d, device=dev, dtype=dtype)
    k = torch.randn(total_kv, hkv, d, device=dev, dtype=dtype)
    v = torch.randn(total_kv, hkv, d, device=dev, dtype=dtype)

    cu_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths_q), 0).tolist()), device=dev, dtype=torch.int64)
    cu_kv = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths_kv), 0).tolist()), device=dev, dtype=torch.int64)
    return q, k, v, cu_q, cu_kv, max_q, max_kv


def _run_varlen_capture(dtype, lengths_q, lengths_kv, hq=8, hkv=4, d=128):
    """Capture varlen SDPA in a graph and replay once — only checks no-throw."""
    torch.manual_seed(0)
    q, k, v, cu_q, cu_kv, max_q, max_kv = _build_varlen(len(lengths_q), lengths_q, lengths_kv, hq, hkv, d, dtype)
    scale = 1.0 / math.sqrt(d)

    def call():
        return ark.sdpa_varlen(q, k, v, cu_q, cu_kv, max_q, max_kv, is_causal=True, scale=scale)

    # Eager warm-up
    with torch.no_grad():
        _ = call()
    torch.xpu.synchronize()

    # Capture
    g = torch.xpu.XPUGraph()
    side, cur = torch.xpu.Stream(), torch.xpu.current_stream()
    side.wait_stream(cur)
    with torch.no_grad(), torch.xpu.stream(side):
        torch.xpu.synchronize()
        with torch.xpu.graph(g):
            call()
    cur.wait_stream(side)
    torch.xpu.synchronize()

    # One replay must not throw
    g.replay()
    torch.xpu.synchronize()
    g.reset()


# ---------------------------------------------------------------------------
# Dense SDPA graph-capture tests
# ---------------------------------------------------------------------------


@requires_graph
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
class TestSDPAGraphCapture:
    def test_prefill_causal(self, dtype, head_dim):
        _run_dense_case(2, 8, 8, 128, 128, head_dim, dtype, is_causal=True)

    def test_prefill_noncausal(self, dtype, head_dim):
        _run_dense_case(2, 8, 8, 128, 192, head_dim, dtype, is_causal=False)

    def test_prefill_gqa(self, dtype, head_dim):
        _run_dense_case(2, 32, 4, 64, 256, head_dim, dtype, is_causal=True)

    def test_decode(self, dtype, head_dim):
        _run_dense_case(2, 16, 4, 1, 64, head_dim, dtype, is_causal=True)


# ---------------------------------------------------------------------------
# Varlen graph-capture smoke tests
# ---------------------------------------------------------------------------


@requires_graph
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestSDPAVarlenGraphCapture:
    def test_capture_replay_smoke(self, dtype):
        """Capture + replay must not throw; numeric check TODO."""
        _run_varlen_capture(dtype, [128, 64, 96], [128, 64, 96])
