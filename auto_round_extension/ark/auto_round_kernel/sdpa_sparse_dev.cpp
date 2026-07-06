// Sparse-development SDPA entrypoints.
// This file intentionally instantiates only the Sage dense/sparse prefill kernels
// used by the kernel benchmark workflow, so sparse-mainloop iteration avoids the
// full generated prefill/decode SDPA build.

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#include <cute/numeric/int.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <sycl/aliases.hpp>
#include "bestla/bestla.h"

#include <sycl/sycl.hpp>
#include "sycl_tla_sdpa.hpp"

namespace ark {

namespace {

using KernelLauncher = int (*)(detail::Options const& options);

constexpr int kSparseProfileModeFull = 0;
constexpr int kSparseProfileModePvMmaOnly = 9;

#if !defined(ARK_SPARSE_DEV_SPARSE_ONLY)
#if !defined(ARK_SPARSE_DEV_FOCUS_128_BF16)
int launch_prefill_kernel_f16_128_sage(detail::Options const& options) {
  return launch_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::half_t>(options);
}

int launch_prefill_kernel_f16_64_sage(detail::Options const& options) {
  return launch_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::half_t>(options);
}
#endif

int launch_prefill_kernel_bf16_128_sage(detail::Options const& options) {
  return launch_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}

#if !defined(ARK_SPARSE_DEV_FOCUS_128_BF16)
int launch_prefill_kernel_bf16_64_sage(detail::Options const& options) {
  return launch_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}
#endif
#endif

#if !defined(ARK_SPARSE_DEV_FOCUS_128_BF16)
int launch_prefill_kernel_f16_128_sparse_sage(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::half_t>(options);
}

int launch_prefill_kernel_f16_128_sparse_sage_qtile64(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_128_qtile64<cute::int8_t, cute::int8_t, cute::half_t>(options);
}

int launch_prefill_kernel_f16_128_sparse_sage_qtile128(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_128_qtile128<cute::int8_t, cute::int8_t, cute::half_t>(options);
}

int launch_prefill_kernel_f16_64_sparse_sage(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::half_t>(options);
}
#endif

int launch_prefill_kernel_bf16_128_sparse_sage(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}

int launch_prefill_kernel_bf16_128_sparse_sage_qtile64(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_128_qtile64<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}

int launch_prefill_kernel_bf16_128_sparse_sage_qtile128(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_128_qtile128<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}

#if !defined(ARK_SPARSE_DEV_FOCUS_128_BF16)
int launch_prefill_kernel_bf16_64_sparse_sage(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}
#endif

template <cutlass::sage::SparseProfileMode ProfileMode, typename ElementV>
int launch_prefill_kernel_sparse_sage_qtile64_profile(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_128_qtile64_profile<ProfileMode, cute::int8_t, cute::int8_t, ElementV>(
      options);
}

template <cutlass::sage::SparseProfileMode ProfileMode, typename ElementV>
int launch_prefill_kernel_sparse_sage_qtile64_profile_halfwidth(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_128_qtile64_profile_halfwidth<ProfileMode, cute::int8_t, cute::int8_t,
                                                                         ElementV>(options);
}

template <typename ElementV>
KernelLauncher select_sparse_sage_qtile64_profile_for_dtype(int sparse_profile_mode) {
  switch (sparse_profile_mode) {
    case kSparseProfileModePvMmaOnly:
      return &launch_prefill_kernel_sparse_sage_qtile64_profile<cutlass::sage::SparseProfileMode::PvMmaOnly, ElementV>;
    default:
      return nullptr;
  }
}

template <typename ElementV>
KernelLauncher select_sparse_sage_qtile64_profile_halfwidth_for_dtype(int sparse_profile_mode) {
  switch (sparse_profile_mode) {
    case kSparseProfileModeFull:
      return &launch_prefill_kernel_sparse_sage_qtile64_profile_halfwidth<cutlass::sage::SparseProfileMode::Full,
                                                                          ElementV>;
    case kSparseProfileModePvMmaOnly:
      return &launch_prefill_kernel_sparse_sage_qtile64_profile_halfwidth<cutlass::sage::SparseProfileMode::PvMmaOnly,
                                                                          ElementV>;
    default:
      return nullptr;
  }
}

#if !defined(ARK_SPARSE_DEV_SPARSE_ONLY)
KernelLauncher select_sage_prefill_launcher(BTLA_DTYPE pv_dtype, int head_dim) {
  switch (head_dim) {
    case 128:
#if defined(ARK_SPARSE_DEV_FOCUS_128_BF16)
      return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sage : nullptr;
#else
      return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sage : launch_prefill_kernel_f16_128_sage;
#endif
#if !defined(ARK_SPARSE_DEV_FOCUS_128_BF16)
    case 64:
      return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_64_sage : launch_prefill_kernel_f16_64_sage;
#endif
    default:
      return nullptr;
  }
}
#endif

KernelLauncher select_sparse_sage_prefill_launcher(BTLA_DTYPE pv_dtype, int head_dim, int q_tile_override,
                                                   int sparse_profile_mode = kSparseProfileModeFull) {
  if (sparse_profile_mode != kSparseProfileModeFull) {
    if (q_tile_override != 64 || head_dim != 128) return nullptr;
    return pv_dtype == BTLA_DTYPE::BF16 ? select_sparse_sage_qtile64_profile_for_dtype<cute::bfloat16_t>(sparse_profile_mode)
                                        : select_sparse_sage_qtile64_profile_for_dtype<cute::half_t>(sparse_profile_mode);
  }
  switch (head_dim) {
    case 128:
      if (q_tile_override == 64) {
#if defined(ARK_SPARSE_DEV_FOCUS_128_BF16)
        return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sparse_sage_qtile64 : nullptr;
#else
        return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sparse_sage_qtile64
                                            : launch_prefill_kernel_f16_128_sparse_sage_qtile64;
#endif
      }
      if (q_tile_override == 128) {
#if defined(ARK_SPARSE_DEV_FOCUS_128_BF16)
        return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sparse_sage_qtile128 : nullptr;
#else
        return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sparse_sage_qtile128
                                            : launch_prefill_kernel_f16_128_sparse_sage_qtile128;
#endif
      }
      if (q_tile_override != 0 && q_tile_override != 256) return nullptr;
#if defined(ARK_SPARSE_DEV_FOCUS_128_BF16)
      return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sparse_sage : nullptr;
#else
      return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sparse_sage
                                          : launch_prefill_kernel_f16_128_sparse_sage;
#endif
#if !defined(ARK_SPARSE_DEV_FOCUS_128_BF16)
    case 64:
      if (q_tile_override != 0 && q_tile_override != 128) return nullptr;
      return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_64_sparse_sage
                                          : launch_prefill_kernel_f16_64_sparse_sage;
#endif
    default:
      return nullptr;
  }
}

KernelLauncher select_sparse_sage_prefill_launcher_halfwidth(BTLA_DTYPE pv_dtype, int head_dim, int q_tile_override,
                                                             int sparse_profile_mode = kSparseProfileModeFull) {
  if (q_tile_override != 64 || head_dim != 128) return nullptr;
  return pv_dtype == BTLA_DTYPE::BF16
             ? select_sparse_sage_qtile64_profile_halfwidth_for_dtype<cute::bfloat16_t>(sparse_profile_mode)
             : select_sparse_sage_qtile64_profile_halfwidth_for_dtype<cute::half_t>(sparse_profile_mode);
}

detail::Options make_common_options(void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int q_stride_s,
                                    int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                                    int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h,
                                    int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b,
                                    int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
                                    int head_dim, float softmax_scale, bool is_causal) {
  if (q_stride_d != 1 || k_stride_d != 1 || v_stride_d != 1 || o_stride_d != 1) {
    throw std::invalid_argument("make_common_options: head-dim stride must be 1 for Q/K/V/O");
  }
  detail::Options options;
  options.q = Q_ptr;
  options.k = K_ptr;
  options.v = V_ptr;
  options.mask = mask;
  options.o = O_ptr;
  options.use_tensor_strides = true;
  options.q_stride_s = q_stride_s;
  options.q_stride_d = q_stride_d;
  options.q_stride_h = q_stride_h;
  options.q_stride_b = q_stride_b;
  options.k_stride_s = k_stride_s;
  options.k_stride_d = k_stride_d;
  options.k_stride_h = k_stride_h;
  options.k_stride_b = k_stride_b;
  options.v_stride_d = v_stride_d;
  options.v_stride_s = v_stride_s;
  options.v_stride_h = v_stride_h;
  options.v_stride_b = v_stride_b;
  options.o_stride_s = o_stride_s;
  options.o_stride_d = o_stride_d;
  options.o_stride_h = o_stride_h;
  options.o_stride_b = o_stride_b;
  options.batch = batch;
  options.num_heads_q = num_heads_q;
  options.num_heads_kv = num_heads_kv;
  options.seq_len_qo = seq_len_q;
  options.seq_len_kv = seq_len_kv;
  options.head_size_qk = head_dim;
  options.head_size_vo = head_dim;
  options.softmax_scale = softmax_scale;
  options.is_causal = is_causal;
  return options;
}

#if !defined(ARK_SPARSE_DEV_SPARSE_ONLY)
void sage_prefill(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
                  void* qscale, void* kscale, BTLA_DTYPE pv_dtype, int q_stride_s, int q_stride_d, int q_stride_h,
                  int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
                  int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h,
                  int o_stride_b, int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
                  int head_dim, float softmax_scale, bool is_causal) {
  detail::Options options =
      make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                          k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                          v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                          seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.scale_block_size = scale_block_size;
  options.qscale = qscale;
  options.kscale = kscale;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sage_prefill_launcher(pv_dtype, head_dim);
  if (launcher == nullptr) {
    throw std::runtime_error("Unsupported dense Sage dev config");
  }

  launcher(options);
}
#endif

void sparse_sage_prefill(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                         int scale_block_size, void* qscale, void* kscale, void* lut, void* valid_block_num,
                         int num_q_blocks, int num_k_blocks, int q_tile_override, BTLA_DTYPE pv_dtype, int q_stride_s,
                         int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                         int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h,
                         int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch,
                         int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
                         float softmax_scale, bool is_causal) {
  detail::Options options =
      make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                          k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                          v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                          seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.scale_block_size = scale_block_size;
  options.q_tile_override = q_tile_override;
  options.qscale = qscale;
  options.kscale = kscale;
  options.lut = static_cast<int const*>(lut);
  options.valid_block_num = static_cast<int const*>(valid_block_num);
  options.num_q_blocks = num_q_blocks;
  options.num_k_blocks = num_k_blocks;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sparse_sage_prefill_launcher(pv_dtype, head_dim, q_tile_override);
  if (launcher == nullptr) {
    throw std::runtime_error("Unsupported sparse Sage dev config");
  }

  launcher(options);
}

void sparse_sage_prefill_profile(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                                 int scale_block_size, void* qscale, void* kscale, void* lut, void* valid_block_num,
                                 int num_q_blocks, int num_k_blocks, int q_tile_override, int sparse_profile_mode,
                                 BTLA_DTYPE pv_dtype, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
                                 int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
                                 int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d,
                                 int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
                                 int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal) {
  detail::Options options =
      make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                          k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                          v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                          seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.scale_block_size = scale_block_size;
  options.q_tile_override = q_tile_override;
  options.sparse_profile_mode = sparse_profile_mode;
  options.qscale = qscale;
  options.kscale = kscale;
  options.lut = static_cast<int const*>(lut);
  options.valid_block_num = static_cast<int const*>(valid_block_num);
  options.num_q_blocks = num_q_blocks;
  options.num_k_blocks = num_k_blocks;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sparse_sage_prefill_launcher(pv_dtype, head_dim, q_tile_override, sparse_profile_mode);
  if (launcher == nullptr) {
    throw std::runtime_error("Unsupported sparse Sage dev profiling config");
  }

  launcher(options);
}

void sparse_sage_prefill_profile_halfwidth(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
    void* qscale, void* kscale, void* lut, void* valid_block_num, int num_q_blocks, int num_k_blocks,
    int q_tile_override, int sparse_profile_mode, BTLA_DTYPE pv_dtype, int q_stride_s, int q_stride_d, int q_stride_h,
    int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
    int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch,
    int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
    bool is_causal) {
  detail::Options options =
      make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                          k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                          v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                          seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.scale_block_size = scale_block_size;
  options.q_tile_override = q_tile_override;
  options.sparse_profile_mode = sparse_profile_mode;
  options.qscale = qscale;
  options.kscale = kscale;
  options.lut = static_cast<int const*>(lut);
  options.valid_block_num = static_cast<int const*>(valid_block_num);
  options.num_q_blocks = num_q_blocks;
  options.num_k_blocks = num_k_blocks;
  compat::set_default_queue(*q);

  KernelLauncher launcher =
      select_sparse_sage_prefill_launcher_halfwidth(pv_dtype, head_dim, q_tile_override, sparse_profile_mode);
  if (launcher == nullptr) {
    throw std::runtime_error("Unsupported sparse Sage dev halfwidth profiling config");
  }

  launcher(options);
}

void sparse_sage_prefill_halfwidth(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
    void* qscale, void* kscale, void* lut, void* valid_block_num, int num_q_blocks, int num_k_blocks,
    int q_tile_override, BTLA_DTYPE pv_dtype, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
    int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h,
    int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q,
    int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal) {
  detail::Options options =
      make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                          k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                          v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                          seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.scale_block_size = scale_block_size;
  options.q_tile_override = q_tile_override;
  options.qscale = qscale;
  options.kscale = kscale;
  options.lut = static_cast<int const*>(lut);
  options.valid_block_num = static_cast<int const*>(valid_block_num);
  options.num_q_blocks = num_q_blocks;
  options.num_k_blocks = num_k_blocks;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sparse_sage_prefill_launcher_halfwidth(pv_dtype, head_dim, q_tile_override);
  if (launcher == nullptr) {
    throw std::runtime_error("Unsupported sparse Sage dev halfwidth config");
  }

  launcher(options);
}

}  // namespace

void sdpa_impl_qks8_pvhalf(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                           int scale_block_size, void* qscale, void* kscale, int q_stride_s, int q_stride_d,
                           int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h,
                           int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
                           int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch,
                           int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
                           float softmax_scale, bool is_causal, BTLA_DTYPE pv_dtype) {
#if defined(ARK_SPARSE_DEV_SPARSE_ONLY)
  throw std::invalid_argument("sdpa_impl_qks8_pvhalf: dense path is disabled in sparse-only dev builds");
#else
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl_qks8_pvhalf: mask and is_causal cannot both be set");
  }
  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument("sdpa_impl_qks8_pvhalf: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (seq_len_q == 1) {
    throw std::invalid_argument("sdpa_impl_qks8_pvhalf: sparse-dev target does not build decode kernels");
  }
  if (pv_dtype != BTLA_DTYPE::F16 && pv_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("sdpa_impl_qks8_pvhalf: only F16 and BF16 are supported for V/O dtype");
  }

  sage_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, pv_dtype, q_stride_s,
               q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d,
               v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch,
               num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
#endif
}

void sdpa_impl_qks8_sparse_pvhalf(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                                  int scale_block_size, void* qscale, void* kscale, void* lut, void* valid_block_num,
                                  int num_q_blocks, int num_k_blocks, int q_tile_override, int q_stride_s,
                                  int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                                  int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h,
                                  int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b,
                                  int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
                                  int head_dim, float softmax_scale, bool is_causal, BTLA_DTYPE pv_dtype) {
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: mask and is_causal cannot both be set");
  }
  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (pv_dtype != BTLA_DTYPE::F16 && pv_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: only F16 and BF16 are supported for V/O dtype");
  }
  if (qscale == nullptr || kscale == nullptr) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: qscale and kscale must be provided");
  }
  if (lut == nullptr || valid_block_num == nullptr) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: lut and valid_block_num must be provided");
  }
  if (num_q_blocks <= 0 || num_k_blocks <= 0) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: num_q_blocks and num_k_blocks must be greater than 0");
  }
  if (scale_block_size <= 0) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: scale_block_size must be greater than 0");
  }

  sparse_sage_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, lut, valid_block_num,
                      num_q_blocks, num_k_blocks, q_tile_override, pv_dtype, q_stride_s, q_stride_d, q_stride_h,
                      q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                      v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                      seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
}

void sdpa_impl_qks8_sparse_row_linear_pvhalf(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
    void* qscale, void* kscale, void* lut, void* valid_block_num, int num_q_blocks, int num_k_blocks,
    int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
    int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s,
    int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
    int seq_len_kv, int head_dim, float softmax_scale, bool is_causal, BTLA_DTYPE pv_dtype) {
  int effective_q_tile_override = q_tile_override == 0 ? 64 : q_tile_override;
  sdpa_impl_qks8_sparse_pvhalf(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, lut,
                               valid_block_num, num_q_blocks, num_k_blocks, effective_q_tile_override, q_stride_s,
                               q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b,
                               v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h,
                               o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim,
                               softmax_scale, is_causal, pv_dtype);
}

void sdpa_impl_qks8_sparse_row_linear_profile(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
    void* qscale, void* kscale, void* lut, void* valid_block_num, int num_q_blocks, int num_k_blocks,
    int q_tile_override, int sparse_profile_mode, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
    int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h,
    int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q,
    int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
    BTLA_DTYPE pv_dtype) {
  int effective_q_tile_override = q_tile_override == 0 ? 64 : q_tile_override;
  if (sparse_profile_mode != kSparseProfileModePvMmaOnly) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_row_linear_profile: sparse-dev supports only pv_mma_only");
  }
  sparse_sage_prefill_profile(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, lut,
                              valid_block_num, num_q_blocks, num_k_blocks, effective_q_tile_override,
                              sparse_profile_mode, pv_dtype, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                              k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                              v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
                              num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
}

void sdpa_impl_qks8_sparse_row_linear_profile_pvhalf(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
    void* qscale, void* kscale, void* lut, void* valid_block_num, int num_q_blocks, int num_k_blocks,
    int q_tile_override, int sparse_profile_mode, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
    int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h,
    int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q,
    int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
    BTLA_DTYPE pv_dtype) {
  int effective_q_tile_override = q_tile_override == 0 ? 64 : q_tile_override;
  if (sparse_profile_mode != kSparseProfileModeFull && sparse_profile_mode != kSparseProfileModePvMmaOnly) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_row_linear_profile_pvhalf: sparse-dev supports only full and pv_mma_only");
  }
  sparse_sage_prefill_profile_halfwidth(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, lut,
                                        valid_block_num, num_q_blocks, num_k_blocks, effective_q_tile_override,
                                        sparse_profile_mode, pv_dtype, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                                        k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
                                        v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b,
                                        batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim,
                                        softmax_scale, is_causal);
}

void sdpa_impl_qks8_sparse_row_linear_halfwidth_pvhalf(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
    void* qscale, void* kscale, void* lut, void* valid_block_num, int num_q_blocks, int num_k_blocks,
    int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s,
    int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
    int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
    int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal, BTLA_DTYPE pv_dtype) {
  int effective_q_tile_override = q_tile_override == 0 ? 64 : q_tile_override;
  if (effective_q_tile_override != 64) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_row_linear_halfwidth_pvhalf: q_tile_override must be 0 or 64 for the halfwidth backend");
  }
  if (scale_block_size != 64) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_row_linear_halfwidth_pvhalf: scale_block_size must be 64 so one sparse row maps to one workgroup");
  }
  sparse_sage_prefill_halfwidth(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, lut,
                                valid_block_num, num_q_blocks, num_k_blocks, effective_q_tile_override, pv_dtype,
                                q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h,
                                k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d,
                                o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv,
                                head_dim, softmax_scale, is_causal);
}

}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
