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

## Expected Result

- `torch.xpu.is_available()` returns `True`
- `torch.xpu.device_count()` is non-zero
- `auto_round_kernel_xpu` builds successfully
- `bench_sparse_topk.py` completes and writes a CSV

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `XPU device count is zero` | Not running in `render` context | Use `sg render -c "..."` or log in with the correct groups |
| `XPU not available` | GPU driver stack or permissions issue | Check `/dev/dri/renderD*`, render group membership, and Intel GPU userspace packages |
| `No space left on device` | Workspace or cache is on a full filesystem | Move the env or `uv` cache to a writable volume |
| `Unable to load XPU lib` | Build output not found | Make sure `xbuild/` points at the built `auto_round_kernel_xpu` artifact |
