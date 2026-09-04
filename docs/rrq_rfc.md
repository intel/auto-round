# RFC: Recurrent Residual Quantization (RRQ) INT2+2+2+2

- Status: Draft
- Authors: AutoRound contributors
- Target version: TBD
- Reviewers: AutoRound maintainers and runtime owners

## Abstract

This RFC proposes an experimental RRQ algorithm. RRQ encodes each eligible weight tensor as one INT2 base plane and three INT2 residual planes. At inference time, selecting base + 0–3 residual planes allows the weight to be reconstructed at 2, 4, 6, or 8 effective bits.

**Phased delivery strategy**:
1. **Phase 1**: RTN quantization with base model (INT2) and residual model (3×INT2) **stored separately**, each independently usable.
2. **Phase 2**: Support generating a residual model from an existing base model + original FP weights.
3. **Phase 3**: Add AutoRound sign-SGD tuning (OPT) so each round is truly optimal.

The first version strictly supports only `INT2+INT2+INT2+INT2`, weight-only linear layers, and eager PyTorch inference. Existing AutoRound, AutoGPTQ, AWQ, GGUF, vLLM, Triton, CUDA, XPU, HPU, MLX, and activation-quantization backends will reject RRQ and must not silently drop residual planes.

## Motivation

Currently, deploying W2A16 and W4A16 simultaneously requires maintaining two separate quantized models. RRQ retains the W2 representation and encodes the reconstruction error with three subsequent W2 planes. The maximum representation size is approximately four INT2 planes plus four sets of quantization metadata, while precision becomes a load-time and runtime choice.

RRQ is not a bit-shift representation. Each plane independently fits its own scale and zero point, which may reduce reconstruction error compared to direct fixed-bit quantization. This RFC does not assume unconditional accuracy or performance conclusions; both should be quantified by acceptance tests.

## Goals (Phased)

### Phase 1: RTN quantization + separate storage

- Add `RRQConfig` and the `rrq` algorithm alias.
- For each eligible layer, perform 4 sequential RTN rounds (`disable_opt_rtn=True`): `W → QDQ_W2 → E₁ → QDQ_W2 → E₂ → QDQ_W2 → E₃ → QDQ_W2`.
- **Separate storage**:
  - Base model: standard INT2 quantized model (fully compatible with existing W2A16 format, loadable by existing runtimes).
  - Residual model: independent 3×INT2 plane artifact (new `auto_round:rrq` format).
- Support combined loading in eager PyTorch: base + 0–3 residuals → 2/4/6/8-bit inference.
- Report actionable errors early for unsupported formats and options.

### Phase 2: Generate residual from existing base model

- Support generating a residual model from an exported base model (INT2 checkpoint) + original FP weights.
- No need to re-quantize the base; only compute `E₁ = W_fp − W_dequant_base` and perform 3 rounds of RTN INT2 quantization on the residual.
- Applicable to incremental upgrade scenarios for users who already have an INT2 quantized model.

### Phase 3: AutoRound tuning (OPT)

- Replace each RTN round with full AutoRound sign-SGD optimization (`iters > 0`).
- Each round targets the current residual, independently tuned to that round's optimum, with the prefix frozen entering the next round.
- Tuned base + residual combination yields accuracy superior to pure RTN decomposition.

## Non-Goals

- Arbitrary plane count, bit-width, data type, per-plane group size, or mixed-plane combinations.
- Fused GPU kernels, or claims of performance parity with fixed INT4/INT8 kernels.
- Packing base and residual into a single checkpoint (first version uses separate storage).
- First version does not support activation quantization, LoRA training, diffusion, MLLM, embeddings, convolutions, or Conv1D.
- Phase 1 does not pursue tuning accuracy improvement (pure RTN); tuning is left to Phase 3.

## Proposed API

### Phase 1: Full RRQ quantization (base + residual exported separately)

```python
from auto_round import AutoRound, RRQConfig

config = RRQConfig(group_size=128, sym=False)
autoround = AutoRound(model, tokenizer, alg_configs=config)
autoround.quantize()
# Base model: standard INT2, directly usable with existing W2A16 runtimes
autoround.save_quantized("./model-rrq-base", format="auto_round")
# Residual model: 3 INT2 planes
autoround.save_quantized("./model-rrq-residual", format="auto_round:rrq")
```

### Phase 2: Generate residual from existing base

```python
from auto_round import generate_rrq_residual

# Base quantized checkpoint + original FP weights
generate_rrq_residual(
    base_model="./model-rrq-base",   # Exported INT2 base
    raw_model="./Qwen3-8B",          # Original FP weights
    output_dir="./model-rrq-residual",
    group_size=128,
    sym=False,
)
```

### Phase 3: OPT tuning

```python
config = RRQConfig(group_size=128, sym=False, iters=200, lr=1e-3)
# Same as Phase 1 otherwise
```

`RRQConfig` fixes the following values; callers overriding them receive a `ValueError`:

| Field | Value |
| --- | --- |
| `bits` | `2` |
| `data_type` | `int` |
| `act_bits` | `16` |
| Residual plane count | `3` |
| Total planes | `4` |

`group_size` follows the existing scalar contract (`-1`, `0`, or positive integer). The first version uses the same `sym` value for all planes. The default value requires review; this RFC suggests `sym=False` because asymmetric INT2 handles non-zero-centered residual distributions more naturally. Layers excluded by the existing `check_to_quantized` rules remain in float.

The loaded RRQ model provides `set_rrq_bits(bits: Literal[2, 4, 6, 8])`, which atomically modifies the active plane count of all RRQ layers. Per-request mixed precision, per-layer strategies, and partial disk loading are left to future RFCs.

## Algorithm

RRQ employs **sequential optimization in 4 rounds**: each round takes the "current residual" as its optimization target, independently runs sign-SGD tuning to that round's optimum, and accumulates the prefix frozen into the next round. The overall flow is "2-bit optimized to optimum → residual re-optimized at 2-bit to optimum → …", for a total of 4 rounds.

Let the target weight be $W$, with plane count $K=4$. Let the converged prefix reconstruction be $A_0=0$ and the current residual be $E_0\equiv W$. Round $k$ ($k=0,1,2,3$) independently performs a full AutoRound tuning:

$$
\hat E_k,\, s_k,\, z_k \;\triangleq\; \arg\min_{v_k, m_k, c_k}\; \mathcal{L}\!\big(\, x\, \big(F\big(x,\, A_k + \mathrm{QDQ}_{\text{INT2}}(E_k;\, v_k, m_k, c_k)\big) + b\big)\, \big),\qquad
A_{k+1}\;\triangleq\; A_k + \hat E_k,\qquad
E_{k+1}\;\triangleq\; W - A_{k+1}.
$$

After each round's tuning converges, the plane's $(\hat E_k, s_k, z_k)$ is frozen and the next round's optimization target becomes the new residual $E_{k+1}$. The `round` inside QDQ uses straight-through estimation (STE), ensuring the current round's $v_k, m_k, c_k$ are differentiable.

Key point: the residual $E_k$ is always computed in the **original float weight domain** ($E_k = W - \sum_{j < k}\hat E_j$), not in the integer code domain; each round independently holds $s_k, z_k$. Each round is a full differentiable AutoRound optimization, operating on the true residual for scale/quantization.

Per-round semantics:

- Round 0 (base): targets $W$, full AutoRound tuning of $v_0$ + per-group min/max scale ($m_0, c_0$), yielding the optimal 2-bit representation $\hat E_0$.
- Round 1 (residual, 2+2 → 4-bit): targets $E_1 = W - \hat E_0$, runs AutoRound tuning for $v_1, m_1, c_1$; the converged result together with round 0's $(\hat E_0, s_0, z_0)$ constitutes the 4-bit prefix optimum.
- Rounds 2 / 3 follow the same pattern: independently tune targeting $E_k$, making $(\hat E_0 + \cdots + \hat E_k)$ optimal in that round's loss sense.

That is: 2-bit optimum + 4-bit prefix optimum + 6-bit prefix optimum + 8-bit prefix optimum, all obtained in one training pass.

Implementation reuses AutoRound's existing single-layer tuning loop per round: `wrapper_block` → calibration → sign-SGD iterations → `collect_best_params` → `unwrapper` to freeze that round's plane back to the module; the next round re-wraps and re-tunes on the frozen prefix. After all 4 rounds converge, the K planes' code/scale/zp are written back, producing a loadable `RRQLinear`; at inference time, prefix accumulation is performed as needed (1/2/3/4 planes → 2/4/6/8-bit).

Tuning overhead: 4 rounds of tuning, where round $k$'s forward pass requires $(k+1)$ weight QDQ operations, totaling ≈ $1+2+3+4=10$ QDQ operations (vs. 1 for a single-layer base alone). All are memory-bound element-wise operations, far smaller than matmul; acceptable for the correctness-first MVP, with fused kernel optimization as a follow-up.

## Model and Checkpoint ABI

RRQ uses **separate storage**: the base model and residual model are two independent artifacts, each independently loadable.

### Base Model (Phase 1 output)

The base model is a standard INT2 quantized model using the existing `auto_round` export format:

- `quantization_config` contains the existing `quant_method` value (e.g. `"auto_round"` / `"gptq"`).
- Contains `qweight` (packed INT2), `scales`, `qzeros` (if asymmetric).
- **Fully compatible with existing runtimes** (AutoGPTQ, vLLM, llama.cpp, etc.), no modifications required.
- Users can directly use the base model for 2-bit inference without RRQ support.

### Residual Model (Phase 1 output / Phase 2 standalone generation)

The residual model is the new `auto_round:rrq` format, containing 3 INT2 residual planes:

| buffer/attribute | shape | meaning |
| --- | --- | --- |
| `qweight_1` | packed INT2 | 1st residual plane (2+2→4-bit increment) |
| `qweight_2` | packed INT2 | 2nd residual plane (4+2→6-bit increment) |
| `qweight_3` | packed INT2 | 3rd residual plane (6+2→8-bit increment) |
| `scales_1` ... `scales_3` | existing scale layout | per-residual-plane group scale |
| `qzeros_1` ... `qzeros_3` | existing zero-point layout | per-plane zero point for asymmetric quantization |
| `rrq_format_version` | string metadata | ABI version, initially `"1"` |

Note: the residual model does **not** contain `qweight_0`/`scales_0` (base plane), because the base is provided by the separate base model.

The exported `quantization_config` (residual model) should contain:

```json
{
  "quant_method": "auto-round-rrq",
  "format_version": 1,
  "base_bits": 2,
  "residual_planes": [2, 2, 2],
  "supported_effective_bits": [4, 6, 8],
  "group_size": 128,
  "sym": false
}
```

### Inference Module `RRQLinear`

When combined loading (base + residual), an eager inference module `RRQLinear` is introduced:

| buffer/attribute | source | meaning |
| --- | --- | --- |
| `qweight_0`, `scales_0`, `qzeros_0` | base model | base plane |
| `qweight_1` ... `qweight_3` | residual model | residual planes |
| `scales_1` ... `scales_3`, `qzeros_1` ... `qzeros_3` | residual model | residual plane metadata |
| `rrq_active_planes` | runtime attribute, not persisted | current active plane count (1–4) |

`RRQLinear.forward(x)` dequantizes and accumulates the selected prefix planes, then calls `torch.nn.functional.linear`. This is a correctness reference implementation, not a performance kernel.

### Storage and Loading Rules

- Base model and residual model each use standard safetensors shards for storage.
- During combined loading, the loader verifies: (1) base model's `bits=2, group_size, sym` match the residual model; (2) residual plane count = 3; (3) shapes match. Any mismatch fails fast.
- Loading only the base model (without residuals) follows the existing standard INT2 inference path, without triggering RRQ logic.
- Non-RRQ backends (GGUF, MLX, etc) encountering a residual model with `quant_method="auto-round-rrq"` must raise an explicit error.

## Integration Steps (Phased)

### Phase 1: RTN quantization + separate storage

1. Add `auto_round.algorithms.quantization.rrq.config.RRQConfig`, register alias `rrq`, and export from `auto_round.__init__`. `RRQConfig` extends `QuantizationConfig`, fixing `bits=2, data_type="int", act_bits=16`, with a new field `num_residual_planes: int = 3` (fixed at 3 in the first version, not modifiable).
2. Implement `RRQRTNQuantizer(BaseQuantizer)`:
   - `quantize_block` performs 4 RTN rounds for each eligible linear layer:
     - Round 0: INT2 RTN on original weight $W$ → saved as base (`layer.weight` replaced with QDQ result, `layer.scale`/`layer.zp` are the base's scale/zp).
     - Rounds 1–3: compute residual $E_k = W_{\text{orig}} - A_k$, perform INT2 RTN on $E_k$, save results as `layer.rrq_residual_k` (k=1,2,3), containing `qweight_k`, `scales_k`, `qzeros_k`.
   - Note: the RTN stage does not require calibration data (`need_calib = False`), no `WrapperLinear` needed, directly perform QDQ on the weight tensor.
3. Export:
   - `save_quantized(format="auto_round")` outputs the base model (standard INT2, identical to existing W2A16).
   - `save_quantized(format="auto_round:rrq")` outputs the residual model (containing only `qweight_1/2/3`, `scales_1/2/3`, `qzeros_1/2/3`).
   - Both directories can exist independently; the residual model directory does not contain base data.
4. Add `RRQLinear` inference module + reusable INT2 pack/unpack helpers. `RRQLinear` accepts base + residual tensor combinations; `forward(x, active_planes)` dequantizes and accumulates prefix planes before calling `F.linear`.
5. Add combined loading entry: user specifies base model path + residual model path; the loader merges them into `RRQLinear` modules.
6. Non-RRQ backends (GGUF, MLX, AutoGPTQ, etc.) encountering `quant_method="auto-round-rrq"` must raise an explicit error.

### Phase 2: Generate residual from existing base

1. Add `generate_rrq_residual(base_model_path, raw_model_path, output_dir, ...)` utility function.
2. Flow: load base model's quantized tensors (`qweight_0`, `scales_0`, `qzeros_0`) → dequantize to get $\hat W_0$ → load original FP weights $W$ → compute $E_1 = W - \hat W_0$ → perform 3 rounds of RTN INT2 on $E_1$ → export residual model.
3. No calibration data needed; pure RTN.
4. Verification: base model's `bits=2, group_size, sym` must match the generated residual.

### Phase 3: AutoRound OPT tuning

1. Implement `RRQSignRoundQuantizer(BaseQuantizer)`:
   - Each round reuses `sign_round`'s single-layer sign-SGD tuning loop (`WrapperLinear` + `iters` iterations).
   - Round 0: target $W$, tune $v_0, m_0, c_0$ → freeze base plane.
   - Round $k$ (k=1,2,3): target $E_k = W - A_k$ ($A_k$ is the dequant result of the frozen prefix), tune $v_k, m_k, c_k$ → freeze.
   - In `forward`: $F.linear(x, A_k + QDQ_{\text{INT2}}(E_k; v_k, m_k, c_k), b)$, where $A_k$ is a frozen constant tensor (not participating in gradients); only the current round's QDQ parameters are differentiable.
2. Parameter naming: `value_k`, `min_scale_k`, `max_scale_k` (k=0..3), reusing existing per-layer lr grouping.
3. After convergence, `unwrapper` writes back K planes' code/scale/zp; export method same as Phase 1.
4. `RRQConfig` gains `iters`, `lr`, `minmax_lr` fields (defaults are RTN, i.e. `iters=0`); Phase 1 users who don't set these fields are equivalent to RTN.

## Verification and Acceptance Criteria

### Phase 1 Acceptance

On deterministic small tensors and tiny linear modules, unit tests must prove:

- Configuration rejects all non-`bits=2, data_type="int"` combinations and activation quantization.
- After 4 RTN rounds, the base model's `qweight`/`scales`/`qzeros` are bit-exact identical to direct `RTN W2A16` output.
- Residual model contains 3 planes of tensors with correct shapes and packed code range.
- For each prefix $p$ (1–4), $W_{\text{deq}}$ (base + first $p-1$ residuals' dequant accumulation) equals the explicit dequantization result within dtype tolerance.
- Residual norms are monotonically decreasing: $\|E_1\|_2 \geq \|E_2\|_2 \geq \|E_3\|_2 \geq \|E_4\|_2$.
- Base model can be independently loaded and inferred by existing W2A16 runtimes (without RRQ code).
- Residual model loaded independently without base must raise an error.
- `RRQLinear` output at 2/4/6/8 active bits equals `F.linear(x, W_reconstructed_p, bias)` within dtype tolerance.
- Non-RRQ backends give explicit rejection for `quant_method="auto-round-rrq"`.

### Phase 2 Acceptance

- `generate_rrq_residual` from existing base + FP weights produces a residual identical (bit-exact) to the Phase 1 full pipeline output.
- The generated residual model can be correctly combined-loaded and inferred.
- Mismatched base model `group_size`/`sym` triggers fail-fast error.

### Phase 3 Acceptance

- Sequential 4-round sign-SGD tuning is effective: each round's iteration receives non-zero gradients for that round's plane parameters.
- Prefix reconstruction norms $\|W - A_k\|_2$ are monotonically decreasing.
- 4-bit prefix ($A_2$) result differs from "single-plane direct 4-bit AutoRound", proving residual decomposition provides actual benefit.
- LM-eval accuracy: RRQ 2+2 prefix ≥ RTN W4 (same group_size/sym); RRQ 2+2+2 prefix ≥ RTN W6; RRQ 2+2+2+2 ≥ RTN W8.
- Initial CUDA benchmark: eager RRQ 4-bit vs W4A16 latency/memory comparison report. No performance SLA before fused kernels are available.

## Compatibility and Failure Behavior

RRQ is opt-in and does not change existing algorithm outputs. Checkpoints are self-describing via `quant_method` and `format_version`; generic W2/W4 implementations cannot load them. Loading on unsupported hardware or requesting unsupported bit-widths must throw errors, not degrade to arbitrary precision.

The first version's `set_rrq_bits` is model-level to prevent layers accidentally using different precision. It only accepts 2, 4, 6, 8 and maps to 1, 2, 3, 4 active planes; this choice is runtime state only and is not written to the checkpoint.

## Considered Alternatives

**Four independent model files.** Directly usable with existing runtimes, but users need to download/store 4 models separately, and base weights are redundantly stored in each. RRQ's base + residual separation design balances deployment flexibility (optional loading) and storage efficiency (base stored once).

**Packing four planes into a single synthetic INT8 code.** This loses independent scale/zero-point semantics, and 2/4/6-bit prefix selection requires runtime kernels to do bit-masking, increasing implementation complexity. Not adopted.

**Reusing SVDQuant residual support.** SVDQuant represents a low-rank FP branch + one residual branch (`residual_linear` + `lora_up/down`), not multi-layer cumulative quantization planes, and does not have RRQ's checkpoint/inference contract.

**Implementing fused kernels first.** This would delay correctness verification and fix the ABI before semantics are validated. The reference module establishes correctness first; runtime owners can add fused implementations under the same ABI.

**Joint optimization (single optimizer for all 4 planes simultaneously).** Theoretically achievable global optimum, but with 4 planes coupled to each other, sign-SGD's per-element search space explodes, convergence is hard to guarantee, and engineering complexity is significantly higher. Sequential per-round optimization (each round independently tuned to optimum) is more stable and naturally compatible with AutoRound's existing mechanisms.

## Open Questions for Review

1. Should the first version use the recommended asymmetric (`sym=False`), or symmetric (`sym=True`) to be closer to existing W2A16 deployment paths?
2. Is `auto_round:rrq` an acceptable residual format name, and is `auto-round-rrq` the expected checkpoint `quant_method` value?
3. During combined loading, should per-layer selection of different active planes be allowed (e.g. some layers at 8-bit, some at 2-bit), or does v1 only support global uniformity?
4. Should the residual model store a `base_model_hash` field for verifying base version consistency?
5. Should Phase 2's `generate_rrq_residual` command be a CLI subcommand (`auto-round generate-residual ...`) or a standalone Python script?
