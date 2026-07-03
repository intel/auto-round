# Sparse Sage Row-Linear Plan

## Goal

Add a new sparse backend that keeps the current `lut + valid_block_num` interface but executes sparse prefill with **one sparse row per workgroup**.

This keeps the current frontier kernel intact and gives us an explicit backend that is structurally closer to the CUDA sparse path without introducing preprocess-side frontier merging.

## Why this path

The current sparse kernel usually runs multiple sparse rows inside one workgroup:

- `scale_block_size = 64`
- `q_tile = 256` -> `4` sparse rows per WG
- `q_tile = 128` -> `2` sparse rows per WG

That ownership model requires dynamic coordination in the hot loop:

- find the smallest next sparse block across active rows
- advance every row that matched that block
- repeat

For irregular preprocess-generated topk patterns, profiling showed that this control-heavy traversal is expensive.

The row-linear backend removes that coordination cost by changing ownership:

- one sparse row -> one WG
- one row-local `lut` stream -> one linear hot loop

## Initial implementation

The first implementation keeps risk low:

- reuse the existing sparse metadata contract
- reuse the existing sparse prefill kernel family
- expose a new explicit API:
  - `sage_sparse_row_linear(...)`
- route that API to the existing `q_tile = 64` sparse launcher

Because `scale_block_size = 64` in the current sparse path, `q_tile = 64` means:

- one sparse row per workgroup

So the first version already gives the intended ownership model without rewriting the sparse mainloop.

## API contract

The new backend should stay as close as possible to `sage_sparse(...)`:

- same Q/K/V tensors
- same `lut` and `valid_block_num`
- same scales
- same layout/stride behavior
- same prefill-only scope

Differences:

- explicit new entrypoint: `sage_sparse_row_linear(...)`
- initial implementation requires `quant_block_size == 64`
- initial implementation forces row-linear execution through `q_tile = 64`

## Implementation notes

### Python

Add:

- `sage_sparse_row_linear(...)`

Behavior:

- validate the same sparse contract as `sage_sparse(...)`
- require `quant_block_size == 64`
- call the new low-level binding

### C++ / binding

Add:

- `sdpa_impl_qks8_sparse_row_linear_pvhalf(...)`
- `sage_sparse_row_linear(...)` pybind wrapper

Behavior:

- keep parameter list aligned with `sage_sparse(...)`
- call the existing sparse prefill implementation with `q_tile_override = 64`
- reject conflicting overrides instead of silently changing behavior

## Success criteria

Correctness:

- matches dense masked Sage within current sparse tolerance
- matches `sage_sparse(...)` on equivalent prefill inputs

Performance:

- benchmark on the irregular preprocess-generated `topk=0.5` long-sequence case
- compare against:
  - dense Sage
  - current frontier sparse backend
  - row-linear sparse backend

## Next step after this slice

If the explicit row-linear backend shows better control behavior but still loses too much time in memory wait, the next optimization stage should happen inside the row-linear path:

- row-local K/V lookahead prefetch
- row-linear-specific pipeline tuning

That should be evaluated before attempting any larger structural rewrite.

## Current row-linear fast-path optimization

The next optimization inside the existing row-linear backend is a narrow kernel-only specialization:

- keep the same `lut + valid_block_num` preprocess contract
- keep the generic multi-row sparse traversal intact as fallback
- add a one-row sparse traversal fast path when the current kernel instance maps one sparse row to one workgroup

Implementation intent:

- replace the generic per-row frontier arrays with scalar state:
  - `row_ptr`
  - `row_valid`
  - `row_pos`
  - `row_cur_block`
  - optional scalar prefetch state
- preserve QK / softmax / PV math
- reduce scratch-heavy post-PV control/update code by removing multi-row scans and frontier min/select logic from the one-row case

Success signal:

- lower scratch traffic and lower spill-sensitive stall pressure in the row-linear `topk=0.5` long-sequence case
