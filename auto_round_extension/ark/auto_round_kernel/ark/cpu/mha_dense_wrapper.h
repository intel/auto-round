//  Copyright (c) 2026 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#pragma once

// -----------------------------------------------------------------------------
// ARK CPU flash-attention wrapper.
//
// This is a direct port of Neural Speed's BestLA attention wrapper
// (neural_speed/core/layers/mha_dense_wrapper.h) adapted to the BestLA snapshot
// vendored under auto_round_kernel/bestla. The eventual target is to land
// `mha_stable_interface_t` plus the `bestla_fusion_attn_forward` dtype
// specializations as the CPU SDPA runtime; the legacy scalar kernel in
// mha_dense.cpp is retained only as a temporary build-safety fallback and is not
// the long-term path.
//
// Phase 2, step 1 migrates the BestLA-independent softmax/epilogue building
// blocks that the stable interface composes:
//   * mha_exp_ref
//   * scale_write_back_t
//   * scale_track_max_t
//   * inplace_precompute_max_softmax_t
//   * activation_identity_t
//   * weight_base_t
//
// API-drift notes vs Neural Speed's BestLA:
//   * ARK's `kernel::wrapper::ScaleTrackMax::forward` takes an extra
//     `padding_type` argument (0=dense, 1=causal, 2=right-padding) that Neural
//     Speed folds into `causal_offset`. We surface it as `scale_track_max_t::
//     Param::padding_type` (default 0) so both the causal and right-padding
//     routes can be driven later without re-touching the call site.
//   * Neural Speed gates `exp` behind the `MHA_2ND_EXP` macro; we mirror it but
//     default it on to reuse BestLA's `kernel::ref::exp_ps_0_1`.
// -----------------------------------------------------------------------------

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "bestla/bestla.h"
#include "bestla/bestla_gemm.h"
#include "bestla/bestla_utils.h"
#include "bestla/kernel_ref.h"
#include "bestla/kernel_wrapper.h"

#include "mha_dense.h"

namespace ark::cpu {

// ---------------------------------------------------------------------------
// Legacy scalar helpers (used by the temporary mha_dense.cpp / sdpa.cpp path).
// These remain until the BestLA stable interface fully replaces the runtime.
// ---------------------------------------------------------------------------
size_t element_size(BTLA_DTYPE dtype);
float load_scalar(const void* base, size_t element_offset, BTLA_DTYPE dtype);
void store_scalar(void* base, size_t element_offset, BTLA_DTYPE dtype, float value);

// ---------------------------------------------------------------------------
// Neural-Speed-style BestLA attention components (Phase 2 migration).
// Namespace mirrors Neural Speed's ne_bestla::custom::mha grouping.
// ---------------------------------------------------------------------------
namespace bestla_mha {

using namespace bestla;  // NOLINT(build/namespaces): match Neural Speed wrapper

// Prefer BestLA's fast polynomial exp (matches Neural Speed's MHA_2ND_EXP path).
#ifndef ARK_MHA_2ND_EXP
#define ARK_MHA_2ND_EXP 1
#endif

inline float mha_exp_ref(float x) {
#if ARK_MHA_2ND_EXP
  return bestla::kernel::ref::exp_ps_0_1(x);
#else
  return std::exp(x);
#endif
}

/**
 * @brief Epilogue that scales the fp32 GEMM result (optionally per-row), casts
 * to the destination type and writes it back. Pure scalar; no ISA dependency.
 */
template <typename T_SRC, typename T_DST>
class scale_write_back_t {
 public:
  using SType = T_SRC;
  using DType = T_DST;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const float* scale;
    DType* dst;
    int ld_dst;
  };
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset,
                                  const int M, const int N, const Param& p, void* /* tmpcache */,
                                  size_t /* cachesize */) {
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto scale = p.scale + M_offset;
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)  //
        dst[i * p.ld_dst + j] = static_cast<DType>(scale[i] * src[i * src_step + j]);
    return BTLA_CODE::Success;
  }
};
using ScaleWriteBackFp32Bf16 = scale_write_back_t<float, utils::bf16>;
using ScaleWriteBackFp32Fp32 = scale_write_back_t<float, float>;
using ScaleWriteBackS32S8 = scale_write_back_t<int32_t, int8_t>;

/**
 * @brief Epilogue for the QK matmul: scales the scores, applies the causal /
 * right-padding mask and tracks the per-row running max (the m_i of the
 * flash-attention stable softmax).
 *
 * Adapts to ARK's BestLA `ScaleTrackMax::forward`, which carries an explicit
 * `padding_type` argument (see file header). `Param::padding_type` defaults to
 * dense (0); callers set 1 for causal or 2 for right padding.
 */
template <typename T_SRC, typename T_DST>
class scale_track_max_t {
 public:
  using DType = T_DST;
  using SType = T_SRC;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    DType* dst;
    DType* dst_max;
    int ld_dst;        // #elements
    float scale;
    int causal_offset;  // offset for causal mask; negative disables causal mask
    float alibi_slope;  // m-factor in the alibi paper (https://arxiv.org/abs/2108.12409)
    float tanh_scale;
    int padding_type = 0;  // ARK BestLA: 0=dense, 1=causal, 2=right-padding
  };
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset,
                                  const int M, const int N, const Param& p, void* tmpcache, size_t cachesize) {
    return bestla::kernel::wrapper::ScaleTrackMax<SType, DType>::template forward<ISA_T>(
        src, src_step, p.dst, p.dst_max, p.ld_dst, M_offset, N_offset, M, N, p.scale, p.causal_offset, p.alibi_slope,
        p.tanh_scale, p.padding_type, tmpcache, cachesize);
  }
};
using ScaleTrackMaxFp16Fp32 = scale_track_max_t<utils::fp16, float>;
using ScaleTrackMaxFp32Fp32 = scale_track_max_t<float, float>;
using ScaleTrackMaxS32Fp32 = scale_track_max_t<int32_t, float>;

/**
 * @brief In-place stable softmax over the score tile: subtracts the row max,
 * exponentiates, and accumulates the per-row exp-sum (the l_i of flash
 * attention). Delegates to BestLA's vectorized kernel.
 */
template <class SRC_T, class DST_T>
struct inplace_precompute_max_softmax_t {
  // n_size is the starting n-size when the causal mask is enabled.
  // src and dst may alias when sizeof(SRC_T) >= sizeof(DST_T) and ld is set.
  // s_max and expsum may alias.
  template <BTLA_ISA ISA_T>
  static inline void forward(int m_size, int n_size, int n_pad_size, bool is_causal, SRC_T* src, DST_T* dst,
                             const SRC_T* s_max, float* expsum, int ld_src, int ld_dst) {
    const auto ret = bestla::kernel::wrapper::InplacePrecomputeMaxSoftmax<SRC_T, DST_T>::template forward<ISA_T>(
        m_size, n_size, n_pad_size, is_causal, src, dst, s_max, expsum, ld_src, ld_dst);
    assert(ret == BTLA_CODE::Success);
    (void)ret;
  }
};

/**
 * @brief Activation prologue that passes the A-matrix straight through (used for
 * the already-laid-out P matrix of the PV matmul).
 */
template <class _GemmCore_T>
class activation_identity_t {
 public:
  using AType = typename _GemmCore_T::AType;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const AType* A;
    int lda;
  };
  activation_identity_t() = default;

  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size,
                                        int m_offset, int k_offset, void* /* tmpcache */, size_t /* cachesize */) {
    (void)m_size;
    (void)k_size;
    auto aptr = const_cast<AType*>(_param.A);
    *dstptr = aptr + m_offset * _param.lda + k_offset;
    *dststep = _param.lda;
    return BTLA_CODE::Success;
  }
};

/**
 * @brief Weight prologue that exposes a plain (row-major) B matrix to the GEMM,
 * padding the N dimension to the GemmCore NTILE when needed.
 */
template <class _GemmCore_T>
class weight_base_t {
 public:
  using BType = typename _GemmCore_T::BType;
  using SType = BType;
  struct Param {  // NOLINT(readability-identifier-naming): align with bestla name
    const SType* B;
    int ldb;
    bool is_padded;
  };
  weight_base_t() = default;
  template <BTLA_ISA ISA_T>
  static inline BTLA_CODE getWeight(BType** dst_ptr, int* dst_step, const Param& p, int k_size, int n_size,
                                    int k_offset, int n_offset, void* /* tmpcache */, size_t /* cachesize */) {
    if ((n_size % _GemmCore_T::NTILE == 0) && std::is_same<SType, BType>::value &&
        false) {  // TODO: use a gemm core that accepts a step for K, or reorder at runtime
      *dst_ptr = const_cast<SType*>(p.B) + k_offset * p.ldb + n_offset;
      *dst_step = p.ldb;
      return BTLA_CODE::Success;
    } else if (*dst_ptr != nullptr && std::is_same<SType, BType>::value) {
      const auto src = const_cast<SType*>(p.B) + k_offset * p.ldb + n_offset;
      const auto npad = utils::padto(n_size, _GemmCore_T::NTILE);
      *dst_step = npad;
      for (int k = 0; k < k_size; ++k) {
        std::memcpy(*dst_ptr + k * npad, src + k * p.ldb, sizeof(BType) * n_size);
        std::memset(*dst_ptr + k * npad + n_size, 0, sizeof(BType) * (npad - n_size));
      }
      return BTLA_CODE::Success;
    } else {
      assert(false);
      return BTLA_CODE::NotSupport;
    }
  }
};

}  // namespace bestla_mha
}  // namespace ark::cpu
