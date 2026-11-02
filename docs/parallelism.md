# Parallel execution

AutoRound delegates tensor parallelism (TP) to Transformers. Transformers loads
the shards, installs its native forwards and owns layout redistribution. AutoRound
only adapts quantize-dequantize (QDQ) math to local weight shards. It does not
implement another tensor-parallel backend.

## Public configuration

`ParallelConfig` is independent of algorithm configs and accepts three positive
integer axes: `tp_size`, `pp_size`, and `dp_size` (all default to `1`). Python also
accepts an equivalent mapping through `parallel_config`.

```python
from auto_round import AutoRound, ParallelConfig
from auto_round.algorithms.quantization.qad.config import QADConfig

compressor = AutoRound(
    "Qwen/Qwen3-0.6B",
    scheme="W4A16",
    alg_configs=QADConfig(iters=200, group_size=128),
    parallel_config=ParallelConfig(tp_size=4),
    enable_torch_compile=False,
    low_cpu_mem_usage=False,
)
compressor.quantize_and_save("./quantized", format="auto_round")
```

Save the example as `train_qad_tp.py` and launch every TP rank:

```bash
torchrun --nproc-per-node 4 train_qad_tp.py
```

The CLI exposes the same axes. If `--tp_size` is omitted, it defaults to
torchrun's `WORLD_SIZE` (or `1` without torchrun), unless another axis is selected:

```bash
torchrun --nproc-per-node 4 -m auto_round quantize Qwen/Qwen3-0.6B --alg qad --tp_size 4 --scheme W4A16
torchrun --nproc-per-node 4 -m auto_round quantize Qwen/Qwen3-0.6B --alg auto_round --scheme W4A16 --group_size 32
torchrun --nproc-per-node 4 -m auto_round quantize Qwen/Qwen3-0.6B --alg awq,auto_round_v2 --scheme W4A16 --group_size 32
```

TP requires a Transformers release exposing `DistributedConfig` and the native
DTensor-based training path. The model config must define `base_model_tp_plan`.
AutoRound passes `DistributedConfig(tp_size=..., tp_plan="auto")` to native
loading, not the deprecated top-level `tp_plan` argument. A preloaded model must
already have native TP applied and must be accompanied by its tokenizer.

Do not pass a custom `device_map` with TP. AutoRound binds each process to
`LOCAL_RANK`; `WORLD_SIZE` must equal `tp_size`. Native loading initializes the
distributed mesh when necessary. AutoRound never destroys a caller-owned process
group. Fast interconnects such as NVLink are recommended.

## Supported scope

- Dense decoder-only language models using ordinary column-wise and row-wise
  Linear plans, including native embedding/output boundaries and replicated norms.
- Standard decoder-only fused MoE layouts described below, tested with Qwen3-MoE.
- Weight-only QAD/QAT, AutoRound/SignRound, AutoRound v2, RTN and optimized RTN.
- INT AWQ smoothing followed by RTN, optimized RTN, AutoRound or AutoRound v2.
  AWQ clipping/clip-as-init and FP4 AWQ smoothing are not supported yet.
- One quantizer stage, optionally preceded by supported block preprocessors.
  Whole-model quantizers cannot follow block preprocessors without a leading block stage.
- A one-dimensional TP mesh with evenly sized, nonempty weight shards.
- Positive integer group sizes. Row-wise shards must contain complete groups;
  `group_size=-1` is supported only for column-wise weights.
- Gradient checkpointing and the existing QAD optimizers. Joint tuning of
  non-quantized parameters, norm/bias tuning, activation quantization,
  `torch.compile`, non-standard fused MoE layouts and AutoScheme are not supported yet.
- Block-level LFQ/QAD losses and rotation preprocessors have not implemented
  their TP contracts. They fail explicitly, rather than falling back to replicated weights.

Unsupported combinations fail explicitly instead of silently running on one GPU
or changing the quantization scheme. The shared QDQ adapter is reusable, but
algorithms must declare their parallel capability before they can use native TP.

## Unified algorithm interface

Algorithms access `BaseAlgorithm.parallel`, a shared `ParallelExecution` injected
through `QuantizationRunContext`. Local execution, existing PP placement and native
TP use the same services; algorithm names are not used to select placement:

- `dispatch_block(...)` returns `BlockPlacement`: the placed block, loss device
  and primary-device memory-risk decision. SignRound's PP heuristic is unchanged.
- `prepare_gradients(...)` retains existing experimental DP synchronization for
  non-TP execution. TP uses native autograd and must not apply DDP to TP shards.
- `sample_indices(...)` broadcasts TP minibatch indices from rank zero, independent
  of rank-local RNG consumption. `offload_block(...)` releases both weight shards
  and quantization tensor metadata to CPU after each block.
- `replicated(...)` converts distributed **activations** to ordinary tensors at
  cache/loss boundaries while preserving autograd. It is not a weight-gather API.
- `tensor_ops` supplies local weight updates, channel-vector slicing, distributed
  channel statistics and scale folding. AWQ uses these for its custom weight math.

`parallel_capabilities()` defaults to rejecting TP. An algorithm using only the
shared services can inherit an already validated implementation's contract; a new
implementation with custom math must explicitly opt in and validate its layouts.
Overriding tuning, dispatch, calibration or preprocessing hooks requires a fresh
capability declaration; inherited support is not silently trusted for those changes.
`validate_parallel_config(config, execution)` rejects unsupported options before
execution. Capability checks cover both preprocessors and quantizers.

Custom QDQ wrappers also require `supports_tensor_parallel=True` in their own class. The
`tensor_parallel_wrapper(...)` factory composes shard transport/native forwards
with the algorithm's original local wrapper; it never replaces custom weight math
with standard rounding. Unadapted wrappers fail explicitly.

## Block-wise TP and CPU residency

For block-wise algorithms each rank keeps inactive **local weight shards** in CPU
memory. Only the current block's shards are moved to that rank's accelerator for
calibration, forward, backward and optimization, then offloaded to CPU. Calibration
forwards stream blocks the same way; non-block embedding/norm/output modules can
remain resident during that forward. No complete block weight is reconstructed.
Each rank uses the same samples and tuning indices. Channel statistics may be
reduced/gathered, but the weight matrices remain sharded throughout tuning.

This host-shard residency replaces the single-process disk offloader for TP.
Immediate packing/saving is disabled: export is collective after tuning and packs
only the writer's full CPU model. Cross-device search borrowing is disabled in
multi-process runs so one rank cannot borrow another rank's accelerator.
QAD/QAT retain their existing whole-model execution instead of this block schedule.
This changes calibration/tuning residency, not Transformers' native loading:
initial loading can still place all local shards on the accelerator before
offloading. Loading-time CPU streaming is not implemented by this execution service.
Before forwarding the returned TP model after block tuning, all ranks must move
their shards back to their rank-local accelerator; exported models are non-TP.

Offline two-rank tests cover dense Llama block-wise algorithms, AWQ channel
statistics/scale folding, export logits and rejection of full-weight gathers during
tuning. This does not establish block-wise MoE accuracy or CUDA/NCCL memory behavior.

## Weight-only FP4

The `MXFP4` and `NVFP4` presets default to W4A4. Native TP currently requires
explicit `act_bits=16, act_data_type="float"` overrides; W4A4 is rejected, not silently converted.
Set both fields: NVFP4's bit-encoded activation type otherwise restores `act_bits=4`.
For example, replace the scheme and algorithm configuration above with:

```python
from auto_round.algorithms.quantization.qad.config import QADConfig

scheme = "MXFP4"
algorithm = QADConfig(iters=200, act_bits=16, act_data_type="float", group_size=32)
```

Pass these as `scheme=scheme, alg_configs=algorithm` to `AutoRound`. For NVFP4,
use `scheme="NVFP4"` and `group_size=16`. Both formats require `bits=4` and
`sym=True`; other FP4 variants are not enabled by this TP adapter.
Row-wise local input widths must be divisible by the respective block size.
Incompatible layouts fail instead of splitting a block or changing its scale.

MXFP4 block scales are computed locally. NVFP4 computes a single global weight
scale per Linear using a scalar `MAX` reduction over the TP mesh, then performs
local block QDQ. It does not gather full weights to compute that scale. The
global scale is preserved for packed checkpoint export.

## Fused MoE and native TP

Standard concatenated, non-transposed, bias-free experts with 3-D `gate_up_proj`
and `down_proj` weights are supported when the native plan uses `packed_colwise`,
`rowwise` and `moe_tp_experts`. Intermediate shards must be even and group-aligned.
AutoRound splits each local packed gate/up shard into per-expert Linear views,
not entire global tensors. The original experts module and native TP forward
remain installed: they own input-gradient and output reductions. Expert Linear
children perform only local computation, avoiding duplicate TP collectives.

TP forces the experts `linear_loop` implementation instead of AutoRound's grouped
backend. Frozen fused parameters remain zero-copy aliases for native TP's layout
checks; CPU export omits these aliases and reconstructs the unfused expert tree.
Generic unfusing of DTensors is rejected before weight mutation. EP, transposed or
interleaved layouts, expert bias/post-norm and unsharded fused experts are not supported.

Two-process tests compare fused/unfused outputs, input/routing/QDQ gradients and
unused-expert behavior, then run Qwen3-MoE INT/MXFP4/NVFP4 weight-only tuning,
CPU export logits and FP4 packing. This does not validate fused-kernel performance
or CUDA/NCCL execution.

## Calibration and checkpoint export

Every TP rank consumes the same calibration samples and optimization indices.
Teacher forward runs on every rank. Each rank has its own teacher-cache directory
and cache hits are coordinated, so a missing shard cache cannot make ranks take
different collective paths. This intentionally duplicates the small teacher
Top-K cache, not the teacher's full logits.

Call `quantize()`, `quantize_and_save()` and `save_quantized()` on **all TP ranks**.
Checkpoint gathering is collective, but only rank zero writes files. Export
completion and writer failures are communicated to all participating ranks.
Export copies local shards to CPU first and transfers bounded byte chunks
(4 MiB each), rather than calling `DTensor.full_tensor()` on accelerator weights.
Only rank zero reconstructs full CPU tensors, and subsequent packing also runs
on CPU. Export reconstructs a non-TP CPU model with full weights, quantization metadata and
non-persistent buffers. The writer needs sufficient host RAM for that full model;
streaming/sharded quantized checkpoint export is not implemented. The original TP
model remains intact. Non-writer `save_quantized(return_folders=True)` calls return
the TP model and an empty folder list. Evaluate the exported model on rank zero
only after collective gathering has finished.

The CLI closes its training process groups on all TP ranks before rank-zero-only
evaluation and temporarily clears torchrun/Accelerate distributed state. This
applies to fake and packed checkpoint evaluation; Python APIs do not automatically
destroy caller-owned process groups.

Offline two-process CPU/Gloo tests cover FP4 forward and STE-gradient parity,
QAD/QAT lifecycle, plain export logits and real `auto_round` packed output.
CUDA/NCCL memory, performance and deployment-backend inference remain unvalidated.

## PP and future DP

`ParallelConfig(pp_size=N)` selects the existing single-process device-placement
path on the first `N` visible CUDA devices. It is not a microbatch pipeline
training schedule, and QAD does not yet support this path. Existing `device_map`
calls remain unchanged when no parallel axes are requested.

`dp_size` reserves a separate data-parallel axis; values greater than one and
hybrid TP/PP combinations currently raise `NotImplementedError`. Existing
experimental block-level DDP behavior is not replaced by this API. Future DP
support must explicitly manage data partitioning, replica groups and gradient
synchronization rather than treating a TP group as a DP group.
