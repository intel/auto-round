# ARK XPU — int4 (S4) WOQ Prefill Path

**Question:** For an int4 weight doing WOQ on prefill (m>1), is it a fused dequant + joint_matrix, or two passes?

**Answer:** **Two passes.** The packed int4 blob is first unpacked into a full **int8 weight buffer in DRAM**, then a separate **s8×s8→s32 joint_matrix (XMX)** kernel consumes it. The `joint_matrix` op never sees int4. Nothing is fused.

## Dispatch chain (S4, m>1)

| Step | Function | Location | What it does |
|------|----------|----------|--------------|
| 0 | `woq_gemm` | `wrapper/include/xpu_wrapper.hpp:746` | Entry. Calls `woq_gemv` first; for m>1 it returns non-zero → fall through to GEMM. |
| 1 | `check_compute_type` / `can_comps8` | `xpu_wrapper.hpp:104` / `:96` | S4 passes the gate → `compute_type` stays `S8`, taking the `!= S8` check's else branch (`:763`). |
| 2 | `unpackq(BTLA_DTYPE::S8, …)` | `xpu_wrapper.hpp:387`, called at `:773` | **Pass 1.** Expands packed int4 → full **int8 scratch in global memory** (`bptr`, `k*n` bytes). The only 4→8 bit widening. |
| 3 | `DnnlWrapper::woq_s8` | `wrapper/include/dnnl_wrapper.hpp:229` | Orchestrates activation quant + the GEMM. |
| 3a | `sycl_dyn_quant_s8` | `dnnl_wrapper.hpp:237` | Dynamically quantizes activation A → int8 in DRAM. |
| 3b | `sycl_igemm_s8s8` | `dnnl_wrapper.hpp:139`, called at `:238` | **Pass 2.** Launches the XMX GEMM. |
| 4 | `Launcher<xmx::IKblockGemmDQCfg<T>, xmx::IKblockGemmDQCore>::run` | `dnnl_wrapper.hpp:160` | blocksize(128) ≠ k → per-k-block variant. (blocksize==k → `IGemmDQCore`, `:146`.) |
| 5 | `IKblockGemmDQCore` | `bestla/bestla/sycl/sycl_gemm.h:1026` | Matrix operands are `DT = int8_t` (`:1019`). `joint_matrix_load_checked` (`:1149` sub_b / `:1154` sub_a) reads the **pre-unpacked int8** weight; `joint_matrix_mad` (`:1158`) runs s8×s8→s32 on the DPAS/XMX array. |

**Key clarification:** the **"DQ" in `IGemmDQCore` dequantizes the int32 accumulator** back to fp via `scale_a × scale_b` — it does **not** unpack the weight. Weight unpack happened earlier in `unpackq`.

## Data flow

```
int4 blob ──unpackq(S8)──► int8 weight in DRAM ──┐
                                                 ├─► joint_matrix s8×s8→s32 ──► DQ (scale_a·scale_b) ──► C
activation ──sycl_dyn_quant_s8──► int8 act in DRAM┘
```

## Why it matters (roofline)

- Writes the full int8 weight (**2× the int4 blob's bytes**) to DRAM and reads it back into the XMX kernel — a global-memory round-trip a **fused** dequant-in-prologue kernel would avoid.
- That extra traffic + the launch barrier between the two kernels is a plausible contributor to the **~30%-of-peak** XMX utilization measured (S4 m512 = 65 TFLOPS vs ~233 INT8 TOPS roofline on Battlemage G21).
- A truly fused int4 path would unpack in the GEMM prologue (as bestla's CPU-side `WeightS4T` prologue does) and skip the int8 materialization entirely.

## Contrast: other weight types

- **S2** — same two-pass int8-XMX path as S4 (passes `can_comps8`). Only S2/S4 reach joint_matrix.
- **S3** — same two-pass *shape* but **fp, not int8**: `:756` forces `compute_type=F16` → `unpackq(acdt,…)` (`:761`) → `DnnlWrapper::gemm` fp DNNL GEMM (`:762`). **No XMX.** This is the fp-dequant fallback; int8-XMX S3 is a future optimization.
