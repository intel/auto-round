# Unitrace Usage Guide for Intel GPU Kernels — 2026-08-14

This is a general unitrace workflow for profiling SYCL/Level Zero kernels on Intel GPUs. It applies
to sparse SDPA and most other GPU kernels.

## 1. Environment

Use the project venv and source oneAPI before profiling:

```bash
source /opt/intel/oneapi/setvars.sh --force
export ZE_AFFINITY_MASK=5
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
```

`ZE_AFFINITY_MASK` chooses the GPU. `ZE_FLAT_DEVICE_HIERARCHY=FLAT` is recommended for hardware
metric profiling.

Useful checks:

```bash
unitrace --device-list
unitrace --metric-list
```

## 2. Find Hot Kernels

Start with device timing. This has lower overhead than full metric collection and identifies which
kernels matter.

```bash
unitrace \
  --device-timing --verbose \
  --output timing \
  <application> <args>
```

Use `--include-kernels` once you know a stable kernel-name substring:

```bash
unitrace \
  --device-timing --verbose \
  --include-kernels "<kernel-name-substring>" \
  --output timing_filtered \
  <application> <args>
```

Important: the command examples do not use `--` before the application command.

## 3. Kernel Properties: Spill, Private Memory, SLM, Registers

The key mode for spill information is:

```bash
unitrace \
  --device-timing --verbose \
  --include-kernels "<kernel-name-substring>" \
  --output kernel_props \
  <application> <args>
```

Look for the `=== Kernel Properties ===` table:

```text
Kernel, Compiled, SIMD, Number of Arguments, SLM Per Work Group,
Private Memory Per Thread, Spill Memory Per Thread, Register File Size Per Thread
```

Key fields:

| field | meaning |
|---|---|
| `SLM Per Work Group` | Shared local memory allocated per workgroup. |
| `Private Memory Per Thread` | Per-thread private memory allocation. |
| `Spill Memory Per Thread` | Compiler-reported spill/scratch memory per thread. This is the direct spill signal. |
| `Register File Size Per Thread` | Register file allocation reported by runtime/compiler metadata. |

Example interpretation:

```text
Spill Memory Per Thread = 0       -> no compiler-reported spill
Spill Memory Per Thread = 3840    -> compiler-reported private/scratch spill
```

SLM counters being zero does not rule out spill. Spill can be private/scratch memory and appear as
LSC/GPU memory traffic, not SLM traffic.

## 4. Compute Overview: ComputeBasic

Use `ComputeBasic` for the first hardware-counter pass.

```bash
unitrace \
  --device-timing --kernel-submission --metric-query --group ComputeBasic \
  --include-kernels "<kernel-name-substring>" \
  --output compute_basic \
  <application> <args>
```

Key metrics:

| metric | what it tells you |
|---|---|
| `GpuTime[ns]` | GPU time under the profiling pass. Use ratios more than absolute time. |
| `XVE_ACTIVE[%]` | Whether execution units are actively doing work. |
| `XVE_STALL[%]` | Time with threads loaded but no execution pipe active. |
| `XVE_THREADS_OCCUPANCY_ALL[%]` | Thread-slot occupancy. Low occupancy can indicate resource pressure. |
| `XVE_INST_EXECUTED_SEND_ALL` | SEND instruction pressure, usually memory/message traffic. |
| `GPU_MEMORY_BYTE_READ/WRITE` | Device-memory bytes. |
| `LOAD_STORE_CACHE_BYTE_READ/WRITE` | LSC traffic, useful as scratch/spill pressure proxy when spill is reported. |
| `L3_READ/WRITE/MISS/HIT` | L3 behavior and locality. |
| `SLM_BYTE_READ/WRITE` | SLM traffic only, not private/scratch spill traffic. |

Use this pass to classify the broad bottleneck: compute active, memory/SEND pressure, occupancy, or
cache behavior.

## 5. Stall Breakdown: VectorEngineStalls

Use this when `XVE_STALL[%]` is high.

```bash
unitrace \
  --device-timing --kernel-submission --metric-query --group VectorEngineStalls \
  --include-kernels "<kernel-name-substring>" \
  --output vector_stalls \
  <application> <args>
```

Common stall meanings:

| metric | likely meaning |
|---|---|
| `XVE_STALL_SBID[%]` | Scoreboard waits, often memory latency or dependent SEND results. |
| `XVE_STALL_ALUWR[%]` | ALU writeback/dependency pressure. |
| `XVE_STALL_INSTFETCH[%]` | Instruction fetch/code-size pressure. |
| `XVE_STALL_PIPESTALL[%]` | Structural hazards, often register pressure, accumulator hazards, or SEND holds. |
| `XVE_STALL_BARRIER[%]` | Synchronization/load imbalance. |
| `XVE_STALL_CONTROL[%]` | Control-flow overhead. |

High SBID plus high SEND/LSC/memory traffic usually points to memory-message latency or scratch/spill
pressure. Confirm spill with the kernel properties pass.

## 6. Memory Deep Dive: MemoryProfile

Use this when ComputeBasic suggests memory/SEND pressure.

```bash
unitrace \
  --device-timing --kernel-submission --metric-query --group MemoryProfile \
  --include-kernels "<kernel-name-substring>" \
  --output memory_profile \
  <application> <args>
```

Useful metrics:

| metric | what it tells you |
|---|---|
| `GPU_MEMORY_BYTE_READ/WRITE` | DRAM/device-memory traffic. |
| `LOAD_STORE_CACHE_BYTE_READ/WRITE` | LSC traffic. |
| `XVE_LOAD_STORE_CACHE_READ_MESSAGE_COUNT` | LSC read message count. |
| `XVE_LOAD_STORE_CACHE_WRITE_MESSAGE_COUNT` | LSC write message count. |
| `XVE_LOAD_STORE_CACHE_REGISTER_REQUEST_COUNT` | Payload/register transactions sent to LSC. |
| `XVE_LOAD_STORE_CACHE_REGISTER_RESPONSE_COUNT` | Payload/register responses from LSC. |
| `GPU_MEMORY_REQUEST_QUEUE_FULL[%]` | Memory request queue pressure. |
| `L3_BUSY[%]`, `L3_STALL[%]`, `L3_HIT`, `L3_MISS` | L3 saturation and locality. |

MemoryProfile still does not label individual messages as spill/fill. Combine it with
`Spill Memory Per Thread` from the kernel properties pass.

## 7. Timeline and Submission

Use kernel submission timing to separate launch/queue overhead from device execution:

```bash
unitrace \
  --device-timing --kernel-submission \
  --include-kernels "<kernel-name-substring>" \
  --output submission \
  <application> <args>
```

Use Chrome trace when host/device sequencing matters:

```bash
unitrace \
  --chrome-kernel-logging --chrome-sycl-logging --chrome-device-logging \
  --output trace \
  <application> <args>
```

Open the generated JSON in Perfetto.

## 8. Instruction-Level Stall Sampling

For instruction-level stall analysis, unitrace supports `--stall-sampling`, but it is only useful
when shader/debug dumps are available.

Typical flow:

```bash
# Build or run with shader dumps/debug info enabled where supported.
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=./shader_dump

unitrace \
  --stall-sampling \
  --include-kernels "<kernel-name-substring>" \
  --output stall_sampling \
  <application> <args>
```

This is the path to correlate stalls to instruction regions. For spill proof, kernel properties are
usually simpler; assembly dumps are useful when you need to inspect actual scratch send
instructions.

## 9. Output Files

`--output-dir-path` may not always place every output where expected with some unitrace builds.
If files appear in the current directory, move them after the run:

```bash
mkdir -p <result-dir>
find . -maxdepth 1 -type f -name '<output-prefix>*' -exec mv -t <result-dir> {} +
```

Common output patterns:

```text
<output>.<pid>                 timing/log output
<output>.metrics.<pid>         metric-query CSV-style output
<output>.json                  Chrome trace, when enabled
```

## 10. Practical Analysis Pattern

Use this order for most kernels:

1. `--device-timing --verbose`: identify hot kernels and check spill/private/SLM/register properties.
2. `--metric-query --group ComputeBasic`: broad compute, occupancy, SEND, memory and L3 counters.
3. `--metric-query --group VectorEngineStalls`: dominant stall reason.
4. `--metric-query --group MemoryProfile`: deeper memory/LSC/L3 analysis if SEND or memory pressure is high.
5. Chrome trace or stall sampling only if timeline or instruction-level evidence is needed.

Minimal command set:

```bash
# Properties and spill
unitrace --device-timing --verbose --include-kernels "<kernel>" --output props <app> <args>

# Broad counters
unitrace --device-timing --kernel-submission --metric-query --group ComputeBasic \
  --include-kernels "<kernel>" --output basic <app> <args>

# Stall reason
unitrace --device-timing --kernel-submission --metric-query --group VectorEngineStalls \
  --include-kernels "<kernel>" --output stalls <app> <args>

# Memory detail
unitrace --device-timing --kernel-submission --metric-query --group MemoryProfile \
  --include-kernels "<kernel>" --output memory <app> <args>
```

## 11. Reading Results Safely

- Metric profiling adds overhead. Use ratios, percentages, and relative A/B comparisons more than
  profiled wall time.
- Run the same benchmark shape for A/B comparisons.
- Use `--include-kernels` to reduce unrelated noise.
- The benchmark may call the target kernel multiple times even with `--iters 1`; aggregate rows by
  kernel name or compare matching rows consistently.
- `Spill Memory Per Thread` is the direct unitrace spill field. `LSC` and `GPU_MEMORY` counters are
  runtime symptoms, not direct proof by themselves.
- `SLM_BYTE_READ/WRITE = 0` only means no SLM traffic. It does not rule out private/scratch spill.

## 12. Sparse SDPA Example

For the BF16 sparse SDPA K32/K64 A/B, the important result came from combining:

```text
unitrace --device-timing --verbose
unitrace --metric-query --group ComputeBasic
unitrace --metric-query --group VectorEngineStalls
```

Observed:

| variant | spill/thread | XVE active | XVE stall | SBID stall | SEND instructions | GPU memory write |
|---|---:|---:|---:|---:|---:|---:|
| K32 | 0 B | 64.7% | 35.1% | 31.4% | 18.6G | 1.63 GB |
| K64 | 3840 B | 39.9% | 59.9% | 55.7% | 81.6G | 8.14 GB |

Conclusion: K64 has compiler-reported private/scratch spill and much higher SEND/SBID and memory
pressure. K32 avoids the spill and is the better selected-block microtile for this kernel.
