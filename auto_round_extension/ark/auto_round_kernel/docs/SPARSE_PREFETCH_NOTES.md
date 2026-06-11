# Sparse Prefetch Notes

## Summary

The current ARK Sage mainloop enables K prefetch only for the dense traversal path and disables it for LUT-driven sparse traversal. This is a performance-design limitation, not a correctness requirement.

Relevant code:

- dense-only initial K prefetch:
  - `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp`
- dense-only rolling K prefetch:
  - `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp`
- sparse traversal:
  - `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp`

## What prefetch means here

At this level, `prefetch(...)` is a hardware/compiler hint on a global-memory address stream.

- The kernel still uses the same global tensor addresses later.
- `prefetch(...)` does not move data directly into MMA fragments or explicit shared memory.
- The real load into register/subgroup fragments still happens through `copy(...)`.

So the model is:

1. `prefetch(...)`
   - hint: bring a future global-memory line closer
   - typical destination is hardware-managed cache / prefetch buffer
2. `copy(...)`
   - actual movement into the register fragments used by `reorder(...)` and `gemm(...)`

This is why disabling K prefetch in sparse mode does not change correctness: `copy(...)` still loads the required K tile directly.

## Why dense prefetch is straightforward

Dense K traversal is affine and contiguous.

- The next block is trivial: `K + 1`
- A rolling window is easy: `K + Stages`
- Every workgroup follows the same simple progression
- Prefetch distance is uniform
- Addresses are regular and nearby in memory

That is why the current dense path can do:

- prefetch initial K tiles before the main loop
- prefetch future K tiles using `K_next = K + Stages`

## Why sparse LUT traversal is harder

The sparse path does know the selected block indices before launch, but that does **not** make the current dense prefetch scheme reusable.

The problem is not lack of knowledge. The problem is that the prefetch logic must be different.

### 1. The next block is not affine

Dense path:

- next tile is `K + 1`
- future tile is `K + Stages`

Sparse path:

- next tile comes from `lut_row`
- selected sequence may look like `[0, 3, 5]`
- the next selected block is `cur_block + lut_row[i + 1]`, not `K + 1`

So the current dense rolling prefetch rule is invalid for sparse traversal.

### 2. Each workgroup may follow a different sparse row

Sparse metadata is per `(batch, head, q_block)` row.

- workgroup A may visit `[0, 1, 2, 3]`
- workgroup B may visit `[0, 3, 5]`
- workgroup C may visit `[7, 8]`

That reduces the regularity and locality that dense prefetch benefits from.

### 3. Prefetch timing is less obvious

Dense path has a natural fixed lookahead distance.

Sparse path may jump over large gaps:

- a near jump may benefit from short-distance prefetch
- a far jump may need earlier prefetch
- too-early prefetch risks eviction before use

So a fixed dense-style `Stages` rule is not obviously optimal for sparse rows.

### 4. Decode makes it more complex

In sparse decode:

- a selected logical block may belong to cache or current KV
- the prefetch path must resolve that per selected block
- paged KV would add another physical mapping layer later

So sparse prefetch must compute the future selected logical block and then map it to the correct physical address stream.

## Why the current code disables dense K prefetch for sparse

The current dense K prefetch logic assumes contiguous traversal:

- initial prefetch fills the first dense K window
- rolling prefetch uses `K + Stages`

If reused unchanged for sparse traversal, it would:

- prefetch unselected K blocks
- waste bandwidth
- weaken cache usefulness
- make sparse performance analysis noisy

So the current implementation disables that dense K prefetch when `lut_row != nullptr`.

This is a performance simplification. The actual K tile is still loaded correctly through `copy(...)`.

## What a sparse-aware prefetch design would need

A proper sparse prefetch path should be driven by sparse row position, not dense K position.

Conceptually:

1. iterate the selected sparse row position `i`
2. look ahead to `i + prefetch_distance`
3. compute the future selected logical K block from `lut_row`
4. map that logical block to:
   - cache vs current KV
   - physical K tile index
5. prefetch that exact K tile

That means sparse prefetch needs its own scheduler instead of reusing:

- `K_next = K + Stages`

## Practical conclusion

- Current sparse path without K prefetch is correct.
- Prefetch is a performance optimization only.
- Dense prefetch works because traversal is regular and affine.
- Sparse traversal is harder because the next block is LUT-driven, row-dependent, and potentially discontinuous.
- A future sparse mainloop should add sparse-aware K prefetch instead of trying to reuse the dense prefetch rule unchanged.
