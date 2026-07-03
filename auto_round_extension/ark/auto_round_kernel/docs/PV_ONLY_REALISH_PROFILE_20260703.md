# PV-Only Realish Profile (2026-07-03)

## Scope

Profile the sparse `pv_only_realish` isolation mode to understand whether the
bad behavior is mostly:

- register spill / private-memory pressure
- SBID / memory dependency stall
- or some non-spill execution bottleneck

Baseline comparison:

- `qk_softmax_only`

Target shape:

- `B=1`
- `S=75000`
- `Hq=40`
- `Hkv=40`
- `D=128`
- `layout=NHD`
- `topk=0.5`
- `q_tile_override=64`

## Commands

```bash
OUTDIR=/home/yiliu4/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/profiles/pv_only_realish_profile_20260703
mkdir -p "${OUTDIR}"

ZE_AFFINITY_MASK=4 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
/home/yiliu4/workspace/pti-gpu/tools/unitrace/install_local/bin/unitrace \
  --stall-sampling \
  --include-kernels XeSparseSageFwdKernel \
  --output "${OUTDIR}/pv_only_realish_stall" \
  /home/yiliu4/workspace/auto-round/auto_round_extension/ark/.venv/bin/python \
  /home/yiliu4/workspace/vllm-omni-fork/scripts/profile_sparse_row_linear_kernel_only.py \
    --topk 0.5 \
    --tensor-layout NHD \
    --warmup 1 \
    --iters 1 \
    --seq-len 75000 \
    --num-heads-q 40 \
    --num-heads-kv 40 \
    --head-dim 128 \
    --profile-mode pv_only_realish

ZE_AFFINITY_MASK=4 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
/home/yiliu4/workspace/pti-gpu/tools/unitrace/install_local/bin/unitrace \
  --stall-sampling \
  --include-kernels XeSparseSageFwdKernel \
  --output "${OUTDIR}/qk_softmax_only_stall" \
  /home/yiliu4/workspace/auto-round/auto_round_extension/ark/.venv/bin/python \
  /home/yiliu4/workspace/vllm-omni-fork/scripts/profile_sparse_row_linear_kernel_only.py \
    --topk 0.5 \
    --tensor-layout NHD \
    --warmup 1 \
    --iters 1 \
    --seq-len 75000 \
    --num-heads-q 40 \
    --num-heads-kv 40 \
    --head-dim 128 \
    --profile-mode qk_softmax_only

ZE_AFFINITY_MASK=4 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
/home/yiliu4/workspace/pti-gpu/tools/unitrace/install_local/bin/unitrace \
  --metric-query -g ComputeBasic \
  --include-kernels XeSparseSageFwdKernel \
  --output "${OUTDIR}/pv_only_realish_metric" \
  /home/yiliu4/workspace/auto-round/auto_round_extension/ark/.venv/bin/python \
  /home/yiliu4/workspace/vllm-omni-fork/scripts/profile_sparse_row_linear_kernel_only.py \
    --topk 0.5 \
    --tensor-layout NHD \
    --warmup 1 \
    --iters 1 \
    --seq-len 75000 \
    --num-heads-q 40 \
    --num-heads-kv 40 \
    --head-dim 128 \
    --profile-mode pv_only_realish
```

## Results

### Kernel Resource Footprint

| Mode | Private/thread | Spill/thread | GRF/thread |
| --- | ---: | ---: | ---: |
| `pv_only_realish` | 4096 | 9600 | 256 |
| `qk_softmax_only` | 0 | 0 | 256 |

### Stall Sampling

| Mode | Dominant buckets |
| --- | --- |
| `pv_only_realish` | `SBID 74.58%`, `Sync 9.62%`, `Dist 6.07%`, `Active 7.11%` |
| `qk_softmax_only` | `Active 42.10%`, `SBID 38.64%`, `Dist 8.31%`, `Pipe 6.25%` |

Detailed `pv_only_realish` stall event mix:

- `SbidStall`: `29,783,073`
- `SyncStall`: `3,843,243`
- `Active`: `2,838,858`
- `DistStall`: `2,424,725`
- `InstrFetchStall`: `718,139`
- `ControlStall`: `305,927`
- `SendStall`: `13,053`
- `PipeStall`: `7,340`

### ComputeBasic (`pv_only_realish`)

- `GpuTime`: `7,994,612,003.5 ns`
- `XVE_ACTIVE`: `9.39%`
- `XVE_STALL`: `90.47%`
- `Occupancy`: `99.73%`
- `Multi-pipe`: `0.07%`
- `Read BW`: `195.58 GB/s`
- `Write BW`: `242.74 GB/s`

## Interpretation

This points to a spill-driven PV problem, not a QK-side problem.

Why:

1. `qk_softmax_only` is spill-free.
2. `pv_only_realish` jumps to `private=4096`, `spill=9600`.
3. `pv_only_realish` is dominated by `SBID` stall, not ALU pipe activity.
4. Occupancy remains high, so the issue is not under-occupancy.
5. `Multi-pipe` nearly disappears, which matches a kernel waiting on data /
   dependencies instead of keeping compute pipes fed.

Most likely meaning:

- the PV path keeps too much per-thread state alive
- that state spills into scratch/private memory
- later PV work repeatedly waits on those outstanding memory dependencies
- `SBID` becomes the visible symptom

## Actionable Conclusion

The next optimization pass should target PV-side live-state reduction first:

- reduce PV accumulator lifetime
- reduce temporary score-reorder state before PV MMA
- reduce sparse traversal metadata kept live across PV
- consider a more specialized one-row / one-head PV path for NHD profiling

This profile is strong evidence that the current sparse bottleneck is
PV-side spill/dependency pressure.
