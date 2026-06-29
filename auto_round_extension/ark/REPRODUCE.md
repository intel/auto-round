# Reproduce

This file collects the exact commands used to validate the Triton-XPU sparse prefill work in this tree.

## 1. Build the XPU kernel library

```bash
cd /home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel
source /opt/intel/oneapi/setvars.sh
cmake -S . -B /tmp/ark-xbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icx \
  -DARK_XPU=ON \
  -DARK_SYCL_TLA=ON
cmake --build /tmp/ark-xbuild --target auto_round_kernel_xpu -j 4
ln -sfn /tmp/ark-xbuild xbuild
```

## 2. Run the Triton-XPU sparse prefill regression test

```bash
cd /data/model/yiliu7/vllm-omni
source /opt/intel/oneapi/setvars.sh >/dev/null
uv run --no-sync python \
  /home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/test/test_triton_sparse_prefill_e2e.py
```

## 3. Run the Wan-shaped self-attention smoke

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

## Related docs

- [ARK_XPU_KERNEL_BKC.md](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/docs/ARK_XPU_KERNEL_BKC.md)
