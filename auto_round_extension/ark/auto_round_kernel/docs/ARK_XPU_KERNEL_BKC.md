# AutoRound XPU Kernel BKC

Best Known Configuration for setting up and validating the `auto_round_kernel` XPU library on an Intel GPU host.

## Validated Configuration

| Component | Version / Detail |
|---|---|
| OS | Ubuntu / Linux with Intel GPU userspace drivers installed |
| GPU | Intel Arc Pro B60 class device |
| Access | User is in the `render` group |
| Python | 3.13.3 in the tested shell |
| uv | 0.11.23 |
| PyTorch | 2.11.0+xpu |
| torchvision | 0.26.0+xpu |
| torchaudio | 2.11.0+xpu |
| oneAPI | `/opt/intel/oneapi/setvars.sh` sourced |
| Build mode | `ARK_XPU=ON`, `ARK_SYCL_TLA=ON` |

## Prerequisites

1. Intel GPU drivers and device nodes are already installed system-wide.
2. The current user can access `/dev/dri/renderD*`.
3. `uv` is available.
4. The Intel oneAPI environment can be sourced.

Verify the GPU nodes:

```bash
ls -l /dev/dri/renderD*
```

Verify group membership:

```bash
id
```

You should see `render` in the group list.

## Create The XPU Python Environment

From the `auto_round_extension/ark/auto_round_kernel` directory, use the repo-local `uv` workflow:

```bash
uv sync
```

That creates the default `.venv` next to the project, unless you override `UV_PROJECT_ENVIRONMENT`.

## Build The XPU Kernel Library

From `auto_round_extension/ark/auto_round_kernel`:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B /tmp/ark-xbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icx \
  -DARK_XPU=ON \
  -DARK_SYCL_TLA=ON
cmake --build /tmp/ark-xbuild --target auto_round_kernel_xpu -j 4
```

If a repo-local `xbuild/` path is needed for import-time discovery, point it at the temp build:

```bash
ln -sfn /tmp/ark-xbuild xbuild
```

## Validate The Build

Run the XPU smoke test under the `render` group:

```bash
sg render -c 'uv run python test_xpu.py'
```

Run the sparse top-k benchmark in the same way:

```bash
sg render -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && \
  uv run python test/bench_sparse_topk.py \
  --seq-len 1024 --topk 1.0 0.5 --warmup 1 --iters 2'
```

For a larger run, keep the same `sg render` wrapper and increase `--seq-len`, `--warmup`, and `--iters` as needed.

## Fix `sg render -c` Friction

If `sg render -c ...` is required on every run, the usual issue is that the current shell session does not yet carry the `render` group.

Permanent fix:

```bash
sudo usermod -aG render "$USER"
```

Then start a fresh login session. On some hosts, opening a new terminal tab is not enough; use one of:

- full logout/login
- reconnect SSH
- reconnect the VS Code remote session

Verify that the new session has the right group before running XPU workloads:

```bash
id
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"
```

You should see `render` in `id`, and PyTorch should report XPU devices.

Short-term workaround without a full relogin:

```bash
su - "$USER"
cd /home/yiliu7/workspace/auto-round/auto_round_extension/ark
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
source .venv/bin/activate
python test/bench_sparse_topk.py --seq-len 1024 --topk 1.0 0.5 --warmup 1 --iters 2
```

If `sg render -c ...` still needs to be used, source oneAPI inside that shell too; otherwise the XPU extension may fail to load with `libiomp5.so` missing:

```bash
sg render -c 'bash -lc "
cd /home/yiliu7/workspace/auto-round/auto_round_extension/ark
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
source .venv/bin/activate
python test/bench_sparse_topk.py --seq-len 1024 --topk 1.0 0.5 --warmup 1 --iters 2
"'
```

## Reproduce The Current Triton-XPU Sparse Work

### Prefill executor regression test

This is the direct test for the new Triton-XPU sparse prefill executor added in this work:

```bash
cd /data/model/yiliu7/vllm-omni
source /opt/intel/oneapi/setvars.sh >/dev/null
uv run --no-sync python \
  /home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/test/test_triton_sparse_prefill_e2e.py
```

### Wan-shaped self-attention smoke

This validates the executor on Wan-like `NHD` self-attention tensors:

```bash
cd /data/model/yiliu7/vllm-omni
source /opt/intel/oneapi/setvars.sh >/dev/null
export SAGE_ATTN_XPU_SPARSE_KERNEL_BACKEND=triton_xpu_kernel
uv run --no-sync python - <<'PY'
import math
import sys
sys.path.insert(0, "/home/yiliu7/workspace/auto-round/auto_round_extension/ark")
import torch
import auto_round_kernel as ark

torch.manual_seed(20260629)
q = torch.randn((1, 256, 40, 128), device="xpu", dtype=torch.bfloat16)
k = torch.randn((1, 256, 40, 128), device="xpu", dtype=torch.bfloat16)
v = torch.randn((1, 256, 40, 128), device="xpu", dtype=torch.bfloat16)
out = ark.sparge_sage2_attn_meansim_topk_xpu(
    q, k, v,
    is_causal=False,
    scale=1.0 / math.sqrt(128),
    smooth_k=True,
    simthreshd1=-1.0,
    topk=0.5,
    attention_sink=False,
    tensor_layout="NHD",
)
torch.xpu.synchronize()
print("wan_self_smoke", tuple(out.shape), out.dtype, float(out.float().abs().mean().cpu()))
PY
```

## Expected Result

- `torch.xpu.is_available()` returns `True`
- `torch.xpu.device_count()` is non-zero
- `auto_round_kernel_xpu` builds successfully
- `bench_sparse_topk.py` completes and writes a CSV

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `XPU device count is zero` | Current shell is missing the `render` group | Add the user to `render`, start a fresh login session, and verify with `id`; use `sg render -c "..."` only as a temporary workaround |
| `XPU not available` | GPU driver stack or permissions issue | Check `/dev/dri/renderD*`, render group membership, and Intel GPU userspace packages |
| `libiomp5.so: cannot open shared object file` | oneAPI runtime libraries are not loaded in the current shell | Source `/opt/intel/oneapi/setvars.sh` inside the same shell that launches Python |
| `No space left on device` | Workspace or cache is on a full filesystem | Move the env or `uv` cache to a writable volume |
| `Unable to load XPU lib` | Build output not found | Make sure `xbuild/` points at the built `auto_round_kernel_xpu` artifact |
