// SYCL MoE W4A8 -- cutlass-free front end (declarations + host helpers)
//
// Companion to `sycl_tla_moe_w4a8.hpp`. That header pulls in the whole
// cutlass-sycl / CuTe include set and defines every W4A8 kernel template, so a
// translation unit that includes it pays the full parse *and* instantiates
// whichever kernels it names. Concentrating all of them in one TU is what made
// the generated `sycl_tla_moe_w4a8.cpp` peak at ~4.2 GB of compiler RSS.
//
// This header carries the part the public entry points need and no kernels:
// the scratch pools, the host-side shape/environment helpers, POD parameter
// structs, and declarations of the per-variant `dispatch` entry points. It
// includes `sycl_tla_moe_decode.hpp` only, which is cutlass-free, so the
// dispatcher TU that includes *this* header compiles for almost nothing.
//
// Each declared entry point is defined in its own generated TU (see
// `sycl_tla_generation.cmake`), mirroring how `sycl_tla_moe_prefill_s4_-
// helpers.hpp` fans the S4 prefill tiles out across TUs. The split axes are
// documented next to the declarations below.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

// `env_flag_enabled`, `fill_expert_id_per_token`, `SG_SIZE` / `N_TILE`.
// Cutlass-free, unlike the DPAS headers.
#include "sycl_tla_moe_decode.hpp"
// `DeviceMemoryPool`, backing the scratch slabs below. Before the split this
// header's contents lived in `sycl_tla_moe_w4a8.hpp`, which picked the pool up
// transitively through `sycl_tla_moe_prefill_int_dpas.hpp`; that chain runs
// through cutlass, so the include has to be explicit here.
#include "utils.hpp"

namespace ark {
namespace moe_w4a8 {

using moe_decode_detail::N_TILE;
using moe_decode_detail::SG_SIZE;

// Symmetric int4 full range: 2^(bits-1). Matches `fullrange` in
// `xpu_wrapper.hpp`'s `packscale` rescale kernel.
constexpr float kInt4FullRange = 8.0f;
constexpr float kInt8Max = 127.0f;

// K elements decoded per work-item in the prepack kernel (one 32-bit word of
// packed nibbles). Requires `K % 8 == 0`, which the shape gate enforces.
constexpr int kPrepackOctet = 8;

// ---------------------------------------------------------------------------
// Scratch pools.
//
// The activation-quantization buffers (`[total_tokens, K]` int8 +
// `[total_tokens]` fp32) and the decode expert map (`[total_tokens]` int32)
// are recomputed on every call, so they come from persistent slabs instead of
// a hot-path `malloc_device`.
//
// The slabs are served from the extension-wide `DeviceMemoryPool`, which keys
// on the device UUID rather than on a `sycl::queue*`: a slab therefore follows
// the device and is immune to the caller destroying its queue and to a later
// queue landing on the same address. This mirrors how the int4 decode scratch
// is managed in `sycl_tla_moe_decode_scratch.{hpp,cpp}`.
//
// Slabs are intentionally never freed from a static destructor -- the SYCL
// context may already be torn down by then. `moe_w4a8_release_scratch`
// provides the explicit teardown (exposed to Python under the same name).
//
// Sharing one slab per device means these entry points must not be driven
// concurrently from two queues on one device, which matches every other
// `DeviceMemoryPool` slot.
// ---------------------------------------------------------------------------

// `DeviceMemoryPool` slots owned by the W4A8 path. Slots 0-7 belong to the
// dnnl / xpu / sycl-s8 / cpu wrappers and the SDPA kernels, slot 8 to the DPAS
// work-group counter, and slots 9-10 to the int4 decode scratch.
inline constexpr size_t kW4A8QactScratchLoc = 11;
inline constexpr size_t kW4A8ExpertMapScratchLoc = 12;

struct W4A8ScratchState {
  std::mutex mu;
  // Device key (`DeviceMemoryPool::get_device_key`) -> a queue handle on that
  // device, held *by value*: a `sycl::queue` is a reference-counted handle, so
  // keeping a copy guarantees the queue outlives the memory allocated against
  // it.
  std::map<size_t, sycl::queue> queues;
};

// Intentionally leaked, see above.
inline W4A8ScratchState& w4a8_scratch_state() {
  static W4A8ScratchState* s = new W4A8ScratchState();
  return *s;
}

// Acquire a slab from the shared pool, synchronizing first when the request
// grows it: `DeviceMemoryPool` frees the old pointer in place when it grows a
// slot, and in-flight kernels may still be reading the old slab, so the wait
// has to happen before the call rather than after.
//
// The caller must hold `W4A8ScratchState::mu`.
inline void* acquire_w4a8_slab(sycl::queue* q, size_t bytes, size_t buf_loc) {
  auto* pool = DeviceMemoryPool::Instance();
  const size_t held = pool->get_scratch_size(buf_loc, q);
  if (held != 0 && held < bytes) {
    q->wait();
  }
  void* ptr = pool->get_scratch_mem(bytes, buf_loc, q);
  if (ptr == nullptr) {
    // The pool records the slot before checking the result, so a failed
    // allocation leaves a {bytes, nullptr} entry behind that would satisfy
    // every later request of this size or smaller without ever retrying.
    // Drop it so the next call allocates again.
    pool->detach_scratch_mem(buf_loc, q);
    throw std::runtime_error("moe_gemm_w4a8: failed to allocate device scratch buffer");
  }
  auto& st = w4a8_scratch_state();
  const size_t key = pool->get_device_key(q);
  if (st.queues.find(key) == st.queues.end()) {
    st.queues.emplace(key, *q);
  }
  return ptr;
}

// Quantized activations + per-token scales.
inline uint8_t* acquire_qact_scratch(sycl::queue* q, size_t bytes) {
  if (q == nullptr) {
    throw std::invalid_argument("moe_gemm_w4a8: device scratch requires a non-null SYCL queue");
  }
  if (bytes == 0) return nullptr;
  auto& st = w4a8_scratch_state();
  std::lock_guard<std::mutex> lock(st.mu);
  return static_cast<uint8_t*>(acquire_w4a8_slab(q, bytes, kW4A8QactScratchLoc));
}

// Decode expert map (`[total_tokens]` int32).
inline int* acquire_expert_map_scratch(sycl::queue* q, size_t bytes) {
  if (q == nullptr) {
    throw std::invalid_argument("moe_gemm_w4a8: device scratch requires a non-null SYCL queue");
  }
  if (bytes == 0) return nullptr;
  auto& st = w4a8_scratch_state();
  std::lock_guard<std::mutex> lock(st.mu);
  return static_cast<int*>(acquire_w4a8_slab(q, bytes, kW4A8ExpertMapScratchLoc));
}


// Host-side helpers
// ---------------------------------------------------------------------------

// Resolve the effective AUTO_S8 re-scale block size.
//
// `requested <= 0` (the `group=-1` spelling) or any value that is not a valid
// block size falls back to `K`, i.e. one scale per output channel -- the
// maximum-efficiency shape. `ARK_MOE_W4A8_AUTO_S8` overrides the argument so
// benchmarks can sweep the block size without touching the caller.
inline int moe_w4a8_rescale_block_size(int K, int group_size, int requested) {
  int v = requested;
  const char* env = std::getenv("ARK_MOE_W4A8_AUTO_S8");
  if (env != nullptr) {
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end != env) v = static_cast<int>(parsed);
  }
  if (K <= 0) return K;
  if (v <= 0 || v >= K) return K;
  if (group_size > 0 && (v < group_size || v % group_size != 0)) return K;
  if (K % v != 0) return K;
  // The mainloop slices each block into 64-wide DPAS K tiles.
  if (v % 64 != 0) return K;
  return v;
}

// Shape preconditions shared by the prepack, prefill and decode paths.
inline bool moe_w4a8_shape_ok(int N, int K, int group_size) {
  if (N <= 0 || K <= 0 || group_size <= 0) return false;
  if (N % N_TILE != 0) return false;
  if (K % 64 != 0) return false;
  if (group_size % kPrepackOctet != 0) return false;
  if (K % group_size != 0) return false;
  return true;
}

// Token count at or below which the auto phase selection picks the decode
// GEMV. Mirrors `ARK_MOE_AUTO_DECODE_MAX_TOKENS` used by the Python `moe()`
// dispatcher; overridable with `ARK_MOE_W4A8_DECODE_MAX_TOKENS`.
inline int moe_w4a8_decode_max_tokens() {
  const char* env = std::getenv("ARK_MOE_W4A8_DECODE_MAX_TOKENS");
  if (env == nullptr) return 128;
  char* end = nullptr;
  const long parsed = std::strtol(env, &end, 10);
  if (end == env || parsed < 0) return 128;
  return static_cast<int>(parsed);
}

// Prefill tile selection.
//
// The ladder used to live inside `moe_w4a8_prefill_dispatch<ElementD>`, which
// meant the TU holding it instantiated all six policies. It is pure host
// arithmetic, so it moves here and the caller launches only the tile it picks
// -- that is what lets each policy live in its own translation unit.
//
// Rungs (measured on Arc Pro B60, see the sweep notes in
// `sycl_tla_moe_w4a8.hpp`): tiny M takes the 8-row tile, the 64-row tile
// covers M up to 128, and above that the choice is between the 256-wide and
// the 128-wide N tile depending on whether N splits evenly.
enum class W4A8PrefillTile { M8, M64, M128, M128N256, M256N128, Large };

// `ARK_MOE_W4A8_PREFILL_TILE` overrides the choice with an explicit `MxN` tile
// so it can be swept on hardware without a rebuild; anything unrecognised --
// including the default `auto` -- keeps the ladder.
inline W4A8PrefillTile moe_w4a8_prefill_select_tile(int A_avg_M, int N) {
  const char* tile_env = std::getenv("ARK_MOE_W4A8_PREFILL_TILE");
  if (tile_env != nullptr) {
    if (std::strcmp(tile_env, "8x128") == 0) return W4A8PrefillTile::M8;
    if (std::strcmp(tile_env, "64x128") == 0) return W4A8PrefillTile::M64;
    if (std::strcmp(tile_env, "128x128") == 0) return W4A8PrefillTile::M128;
    if (std::strcmp(tile_env, "128x256") == 0) return W4A8PrefillTile::M128N256;
    if (std::strcmp(tile_env, "256x128") == 0) return W4A8PrefillTile::M256N128;
    if (std::strcmp(tile_env, "256x256") == 0) return W4A8PrefillTile::Large;
  }

  if (A_avg_M < 16) return W4A8PrefillTile::M8;
  if (A_avg_M < 128) return W4A8PrefillTile::M64;
  if ((N % 256) == 0) return W4A8PrefillTile::M128N256;
  return W4A8PrefillTile::M128;
}

inline void moe_w4a8_release_scratch() {
  auto& st = w4a8_scratch_state();

  // Detach everything under the lock, then drop the lock before the device sync
  // and the frees: `wait()` blocks for an unbounded time and must not be held
  // across. Because the slabs are already out of the pool's tables, an acquire
  // that races in behind us allocates fresh ones instead of handing back a
  // pointer we are about to free.
  std::vector<std::pair<sycl::queue, void*>> pending;
  {
    std::lock_guard<std::mutex> lock(st.mu);
    auto* pool = DeviceMemoryPool::Instance();
    for (auto& kv : st.queues) {
      sycl::queue q = kv.second;
      for (size_t loc : {kW4A8QactScratchLoc, kW4A8ExpertMapScratchLoc}) {
        void* ptr = pool->detach_scratch_mem(loc, &q);
        if (ptr != nullptr) pending.emplace_back(q, ptr);
      }
    }
    st.queues.clear();
  }

  for (auto& item : pending) {
    item.first.wait();
    sycl::free(item.second, item.first);
  }
}

}  // namespace moe_w4a8

// ---------------------------------------------------------------------------
// Per-variant entry points.
//
// Every function below is defined in its own generated translation unit; this
// header only declares them, so including it never instantiates a kernel. The
// parameter structs are plain PODs of scalars and pointers -- deliberately no
// cutlass types -- so the declarations stay usable from a TU that has not seen
// CuTe.
//
// The split follows `sycl_tla_moe_prefill_s4_helpers.hpp`: one TU per
// (dtype x tile) for the DPAS prefill, which is where the compile cost is, and
// coarser grouping for the plain-SYCL kernels. Kernel counts per TU:
//
//   prefill_{f16,bf16}_*  : 1 DPAS kernel   (12 TUs)
//   decode_{f16,bf16}     : 7 GEMV kernels  (2 TUs)
//   quant_{f16,bf16}      : 11 quant kernels (2 TUs)
//   prepack_{f16,bf16}    : 2 rescale kernels (2 TUs)
//
// versus 52 kernels -- 12 of them DPAS -- in the single TU this replaced.
// ---------------------------------------------------------------------------
namespace moe_w4a8_detail {

// One-shot AUTO_S8 prepack: int4 + per-group scales -> int8 + block scales.
struct W4A8PrepackParams {
  sycl::queue* q = nullptr;
  const void* weights_s4 = nullptr;  // [E, N, K/2] packed nibbles
  const void* scales = nullptr;      // [E, N, K/group_size] act dtype
  int8_t* weights_s8 = nullptr;      // [E, N, K] out
  float* wscales = nullptr;          // [E, N, blks] out
  int num_experts = 0;
  int N = 0;
  int K = 0;
  int group_size = 0;
  int blocksize = 0;
  int blks = 0;
};

// Per-token activation quantization, optionally emitting the decode expert map
// in the same pass (`expert_map != nullptr`).
struct W4A8QuantParams {
  sycl::queue* q = nullptr;
  const void* activations = nullptr;  // [T, K] act dtype
  int8_t* qact = nullptr;             // [T, K] out
  float* ascale = nullptr;            // [T] out
  int total_tokens = 0;
  int K = 0;
  int* expert_map = nullptr;
  const int* num_tokens_per_expert = nullptr;
  int num_experts = 0;
};

// Decode GEMV (K-split or legacy, chosen inside the TU).
struct W4A8DecodeParams {
  sycl::queue* q = nullptr;
  const int8_t* qact = nullptr;
  const float* ascale = nullptr;
  const int8_t* weights = nullptr;
  const float* wscale = nullptr;
  void* outputs = nullptr;  // act dtype
  const int* expert_id_per_token = nullptr;
  int total_tokens = 0;
  int N = 0;
  int K = 0;
  int blocksize = 0;
  int blks = 0;
};

// Grouped prefill GEMM. The tile is chosen by `moe_w4a8_prefill_select_tile`
// before the call, so each TU below instantiates exactly one policy.
struct W4A8PrefillParams {
  sycl::queue* q = nullptr;
  const int8_t* qact = nullptr;
  const float* ascale = nullptr;
  const int8_t* weights = nullptr;
  const float* wscale = nullptr;
  void* outputs = nullptr;  // act dtype
  const int* num_tokens_per_expert = nullptr;
  int num_experts = 0;
  int N = 0;
  int K = 0;
  int blocksize = 0;
  int blks = 0;
  int total_tokens = 0;
  // Optional fused top-k reduction; `fused_out != nullptr` turns it on.
  const int* row_to_token = nullptr;
  const float* row_weight = nullptr;
  float* fused_out = nullptr;
  int fused_batch = 0;
};

void prepack_f16(const W4A8PrepackParams& params);
void prepack_bf16(const W4A8PrepackParams& params);

void quant_f16(const W4A8QuantParams& params);
void quant_bf16(const W4A8QuantParams& params);

void decode_f16(const W4A8DecodeParams& params);
void decode_bf16(const W4A8DecodeParams& params);

void prefill_f16_m8(const W4A8PrefillParams& params);
void prefill_f16_m64(const W4A8PrefillParams& params);
void prefill_f16_m128(const W4A8PrefillParams& params);
void prefill_f16_m128n256(const W4A8PrefillParams& params);
void prefill_f16_m256n128(const W4A8PrefillParams& params);
void prefill_f16_large(const W4A8PrefillParams& params);

void prefill_bf16_m8(const W4A8PrefillParams& params);
void prefill_bf16_m64(const W4A8PrefillParams& params);
void prefill_bf16_m128(const W4A8PrefillParams& params);
void prefill_bf16_m128n256(const W4A8PrefillParams& params);
void prefill_bf16_m256n128(const W4A8PrefillParams& params);
void prefill_bf16_large(const W4A8PrefillParams& params);

// ---------------------------------------------------------------------------
// Public entry point 1 -- one-shot AUTO_S8 prepack.
//
// Converts auto-round's packed int4-sym weights + per-group scales into the
// int8 weights + FP32 block scales the W4A8 kernels consume. Callers are
// expected to run this once per checkpoint and keep the results resident.
// ---------------------------------------------------------------------------
inline void moe_w4a8_prepack(sycl::queue* q, void* weights_s4, void* scales, void* weights_s8, void* wscales,
                             BTLA_DTYPE act_dtype, int num_experts, int N, int K, int group_size,
                             int rescale_group_size) {
  if (num_experts <= 0) return;
  if (!moe_w4a8::moe_w4a8_shape_ok(N, K, group_size)) {
    throw std::invalid_argument(
        "moe_w4a8_prepack: unsupported shape (need N % 16 == 0, K % 64 == 0, "
        "group_size % 8 == 0 and K % group_size == 0)");
  }
  if (weights_s4 == nullptr || scales == nullptr || weights_s8 == nullptr || wscales == nullptr) {
    throw std::invalid_argument("moe_w4a8_prepack: null buffer");
  }
  if (act_dtype != BTLA_DTYPE::F16 && act_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("moe_w4a8_prepack: act_dtype must be F16 or BF16");
  }

  W4A8PrepackParams params;
  params.q = q;
  params.weights_s4 = weights_s4;
  params.scales = scales;
  params.weights_s8 = static_cast<int8_t*>(weights_s8);
  params.wscales = static_cast<float*>(wscales);
  params.num_experts = num_experts;
  params.N = N;
  params.K = K;
  params.group_size = group_size;
  params.blocksize = moe_w4a8::moe_w4a8_rescale_block_size(K, group_size, rescale_group_size);
  params.blks = K / params.blocksize;

  if (act_dtype == BTLA_DTYPE::F16) {
    prepack_f16(params);
  } else {
    prepack_bf16(params);
  }
}

// ---------------------------------------------------------------------------
// Public entry point 2 -- W4A8 MoE GEMM (prefill + decode).
//
// `phase`: 0 = auto (decode when `total_tokens <=
// ARK_MOE_W4A8_DECODE_MAX_TOKENS`), 1 = force decode GEMV, 2 = force prefill
// grouped GEMM.
//
// Two optional call contracts trade interface work for DRAM traffic. Both are
// opt-in and the defaults are unchanged.
//
// Pre-quantized activations (`qact_in` + `ascale_in`)
// ---------------------------------------------------
// By default the call quantizes `[T, K]` itself: it reads the 16-bit
// activations, writes an int8 copy and the GEMM reads that copy back, i.e.
// `4 * T * K` bytes on top of the GEMM's own operands. On the down-projection
// that is 27% of everything the call moves -- and it is redundant, because the
// producer of those activations (the SiLU/gate elementwise kernel) already
// writes `[T, K]` once and could write int8 plus a per-row scale instead: the
// absmax it needs is a reduction over the row it is already holding. When both
// pointers are supplied all three streams disappear, along with a kernel
// launch. `ascale_in` is `[T]` fp32, `scale = absmax / 127`, matching what
// `launch_act_dynamic_quant` writes.
//
// Fused top-k reduction (`row_to_token` + `routing_weights` + `fused_out`)
// -----------------------------------------------------------------------
// See `MoEFusedReduce`. Prefill only, and the accumulator must be zeroed by
// the caller; `outputs` is then unused and may be null.
// ---------------------------------------------------------------------------
inline void moe_gemm_w4a8(sycl::queue* q, void* activations, void* weights_s8, void* wscales, void* outputs,
                          BTLA_DTYPE act_dtype, int N, int K, int rescale_block_size,
                          int* num_tokens_per_expert, int num_experts, int total_tokens, int phase,
                          const void* qact_in = nullptr, const float* ascale_in = nullptr,
                          const int* row_to_token = nullptr, const float* routing_weights = nullptr,
                          float* fused_out = nullptr, int fused_batch = 0) {
  if (total_tokens == 0 || num_experts <= 0) return;
  if (N % moe_w4a8::N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_w4a8: N must be a multiple of 16");
  }
  if (K % 64 != 0) {
    throw std::invalid_argument("moe_gemm_w4a8: K must be a multiple of 64");
  }
  if (rescale_block_size <= 0 || rescale_block_size > K || K % rescale_block_size != 0 ||
      rescale_block_size % 64 != 0) {
    throw std::invalid_argument(
        "moe_gemm_w4a8: rescale_block_size must be a multiple of 64 that divides K "
        "(use moe_w4a8_rescale_block_size to resolve it)");
  }
  if (act_dtype != BTLA_DTYPE::F16 && act_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("moe_gemm_w4a8: act_dtype must be F16 or BF16");
  }

  const bool prequantized = qact_in != nullptr || ascale_in != nullptr;
  if (prequantized && (qact_in == nullptr || ascale_in == nullptr)) {
    throw std::invalid_argument("moe_gemm_w4a8: pre-quantized activations need both qact and ascale");
  }
  if (!prequantized && activations == nullptr) {
    throw std::invalid_argument("moe_gemm_w4a8: null activations");
  }

  const bool fused_reduce = row_to_token != nullptr || routing_weights != nullptr || fused_out != nullptr;
  if (fused_reduce && (row_to_token == nullptr || routing_weights == nullptr || fused_out == nullptr ||
                       fused_batch <= 0)) {
    throw std::invalid_argument(
        "moe_gemm_w4a8: the fused top-k reduction needs row_to_token, routing_weights, a zeroed [batch, N] "
        "fp32 output and batch > 0");
  }
  if (!fused_reduce && outputs == nullptr) {
    throw std::invalid_argument("moe_gemm_w4a8: null outputs");
  }

  const bool is_f16 = act_dtype == BTLA_DTYPE::F16;
  const int blocksize = rescale_block_size;
  const int blks = K / blocksize;

  const bool use_decode =
      phase == 1 || (phase != 2 && total_tokens <= moe_w4a8::moe_w4a8_decode_max_tokens());

  if (fused_reduce && use_decode) {
    throw std::invalid_argument("moe_gemm_w4a8: the fused top-k reduction is prefill-only");
  }

  const int8_t* qact = static_cast<const int8_t*>(qact_in);
  const float* ascale = ascale_in;
  int8_t* qact_scratch = nullptr;
  float* ascale_scratch = nullptr;

  if (!prequantized) {
    // Quantized activations + per-token scales share one slab: `[T, K]` int8
    // followed by `[T]` fp32 (the int8 region is already 4-byte aligned because
    // K is a multiple of 64).
    const size_t qact_bytes = static_cast<size_t>(total_tokens) * static_cast<size_t>(K);
    const size_t scale_offset = (qact_bytes + sizeof(float) - 1) / sizeof(float) * sizeof(float);
    const size_t slab_bytes = scale_offset + static_cast<size_t>(total_tokens) * sizeof(float);
    uint8_t* slab = moe_w4a8::acquire_qact_scratch(q, slab_bytes);
    qact_scratch = reinterpret_cast<int8_t*>(slab);
    ascale_scratch = reinterpret_cast<float*>(slab + scale_offset);
    qact = qact_scratch;
    ascale = ascale_scratch;
  }

  // Decode consumes `expert_id_per_token`; the activation-quant kernel already
  // runs one sub-group per token, so it derives the map as well instead of
  // paying for a second launch (`fill_expert_id_per_token`) on a timeline where
  // one call is issued per generated token. Prefill passes nullptr and the scan
  // is not compiled into the work. With pre-quantized activations that kernel
  // does not run at all, so decode falls back to the standalone scan.
  int* expert_map = nullptr;
  if (use_decode) {
    expert_map = moe_w4a8::acquire_expert_map_scratch(q, static_cast<size_t>(total_tokens) * sizeof(int));
  }

  if (prequantized) {
    if (use_decode) {
      moe_decode_detail::fill_expert_id_per_token(q, expert_map, num_tokens_per_expert, num_experts,
                                                  total_tokens);
    }
  } else {
    W4A8QuantParams qp;
    qp.q = q;
    qp.activations = activations;
    qp.qact = qact_scratch;
    qp.ascale = ascale_scratch;
    qp.total_tokens = total_tokens;
    qp.K = K;
    qp.expert_map = expert_map;
    qp.num_tokens_per_expert = num_tokens_per_expert;
    qp.num_experts = num_experts;
    if (is_f16) {
      quant_f16(qp);
    } else {
      quant_bf16(qp);
    }
  }

  const auto* weights = static_cast<const int8_t*>(weights_s8);
  const auto* wscale = static_cast<const float*>(wscales);

  if (use_decode) {
    W4A8DecodeParams dp;
    dp.q = q;
    dp.qact = qact;
    dp.ascale = ascale;
    dp.weights = weights;
    dp.wscale = wscale;
    dp.outputs = outputs;
    dp.expert_id_per_token = expert_map;
    dp.total_tokens = total_tokens;
    dp.N = N;
    dp.K = K;
    dp.blocksize = blocksize;
    dp.blks = blks;
    if (is_f16) {
      decode_f16(dp);
    } else {
      decode_bf16(dp);
    }
    return;
  }

  W4A8PrefillParams pp;
  pp.q = q;
  pp.qact = qact;
  pp.ascale = ascale;
  pp.weights = weights;
  pp.wscale = wscale;
  pp.outputs = outputs;
  pp.num_tokens_per_expert = num_tokens_per_expert;
  pp.num_experts = num_experts;
  pp.N = N;
  pp.K = K;
  pp.blocksize = blocksize;
  pp.blks = blks;
  pp.total_tokens = total_tokens;
  if (fused_reduce) {
    pp.row_to_token = row_to_token;
    pp.row_weight = routing_weights;
    pp.fused_out = fused_out;
    pp.fused_batch = fused_batch;
  }

  // One `dispatch` symbol per (dtype, tile); only the selected one is linked
  // against a kernel-bearing TU, and none of them is instantiated here.
  switch (moe_w4a8::moe_w4a8_prefill_select_tile(total_tokens / num_experts, N)) {
    case moe_w4a8::W4A8PrefillTile::M8:
      if (is_f16) {
        prefill_f16_m8(pp);
      } else {
        prefill_bf16_m8(pp);
      }
      break;
    case moe_w4a8::W4A8PrefillTile::M64:
      if (is_f16) {
        prefill_f16_m64(pp);
      } else {
        prefill_bf16_m64(pp);
      }
      break;
    case moe_w4a8::W4A8PrefillTile::M128N256:
      if (is_f16) {
        prefill_f16_m128n256(pp);
      } else {
        prefill_bf16_m128n256(pp);
      }
      break;
    case moe_w4a8::W4A8PrefillTile::M256N128:
      if (is_f16) {
        prefill_f16_m256n128(pp);
      } else {
        prefill_bf16_m256n128(pp);
      }
      break;
    case moe_w4a8::W4A8PrefillTile::Large:
      if (is_f16) {
        prefill_f16_large(pp);
      } else {
        prefill_bf16_large(pp);
      }
      break;
    case moe_w4a8::W4A8PrefillTile::M128:
    default:
      if (is_f16) {
        prefill_f16_m128(pp);
      } else {
        prefill_bf16_m128(pp);
      }
      break;
  }
}

// Resolve the effective AUTO_S8 block size (host helper, also exported to
// Python so callers can size the `wscales` tensor consistently).
inline int moe_w4a8_rescale_block_size(int K, int group_size, int rescale_group_size) {
  return moe_w4a8::moe_w4a8_rescale_block_size(K, group_size, rescale_group_size);
}

// Free the W4A8 activation-quantization / expert-map scratch slabs.
inline void moe_w4a8_release_scratch() { moe_w4a8::moe_w4a8_release_scratch(); }

}  // namespace moe_w4a8_detail

}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
