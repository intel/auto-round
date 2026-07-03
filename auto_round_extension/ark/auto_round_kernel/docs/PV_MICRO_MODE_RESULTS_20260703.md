# PV Micro-Mode Results (2026-07-03)

## Goal

Identify whether the sparse PV spill is caused by:

- `P` reorder
- `V` load / reorder
- PV MMA / accumulator state
- or the combination of all of them

## Shape

- `B=1`
- `S=75000`
- `Hq=40`
- `Hkv=40`
- `D=128`
- `layout=NHD`
- `topk=0.5`
- `q_tile_override=64`
- PV dtype in this run: `FP16`

## Modes

- `pv_reorder_only`
- `pv_load_v_only`
- `pv_mma_only`
- `pv_reorder_plus_mma`

## Results

| Mode | Private/thread | Spill/thread | GRF/thread | GpuTime (ns) | XVE Active | XVE Stall | Multi-pipe | Read GB/s | Write GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pv_reorder_only` | 0 | 0 | 256 | 94,183,880 | 6.94% | 92.93% | 0.09% | 57.83 | 1.90 |
| `pv_load_v_only` | 0 | 0 | 256 | 543,300,078 | 53.69% | 45.58% | 0.90% | 9.38 | 0.34 |
| `pv_mma_only` | 0 | 7168 | 256 | 2,894,622,448 | 24.54% | 75.15% | 0.46% | 169.81 | 280.46 |
| `pv_reorder_plus_mma` | 0 | 7168 | 256 | 2,748,017,890 | 21.50% | 78.28% | 0.29% | 172.88 | 273.31 |

## Main Read

1. `P` reorder is not the spill source.

- `pv_reorder_only` compiles with `spill=0`

2. `V` load / reorder is not the spill source.

- `pv_load_v_only` compiles with `spill=0`

3. PV MMA is the first point where the spill appears.

- `pv_mma_only` jumps to `spill=7168`
- adding reorder on top does not materially increase spill further

## Conclusion

The current sparse PV spill is fundamentally a PV-MMA / accumulator-footprint
problem, not a reorder-only or V-load-only problem.

That means the next optimization pass should focus on:

- reducing `tArA` live state
- reducing the overlap between `tArP`, `tArV`, and `tArA`
- shrinking PV tile / `VTiles` work per pass if needed
- checking whether BF16 PV reduces the same spill compared with FP16 PV
