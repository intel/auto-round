//  Copyright (c) 2023 Intel Corporation
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

#ifdef BTLA_SYCL
#include <sycl/sycl.hpp>

#include "bestla/bestla_utils.h"
#include "sycl_utils.h"
#include "sycl_device.h"
#include "sycl_gemm.h"
#include "sycl_epilogue.h"
#include "sycl_prologue_a.h"
#include "sycl_prologue_b.h"

namespace bestla {
namespace sycl_wrapper {
template <template <class GCT> class ProAT, template <class GCT> class ProBT, template <class GCT> class EpiT,
          class GemmCoreT>
class Launcher {
 public:
  using GemmCore = GemmCoreT;
  using PrologueA = ProAT<GemmCore>;
  using PrologueB = ProBT<GemmCore>;
  using Epilogue = EpiT<GemmCore>;
  using AType = typename GemmCore::TA;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::TB;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::TC;
  using ACCType = typename GemmCore::TACC;
  using EpiParam = typename Epilogue::Param;
  struct Param {
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
  };
  template <bool debug = false>
  static inline sycl::event compute(sycl::queue* q, int m, int n, int k, const Param& _param) {
    sycl::range<2> group{GemmCore::WgM, GemmCore::WgN};
    auto A = _param.paramA.A;
    auto B = _param.paramB.B;
    auto C = _param.paramC.C;
    int lda = _param.paramA.lda;
    int ldb = _param.paramB.ldb;
    int ldc = _param.paramC.ldc;
    int m_pad = utils::padto(utils::updiv(m, GemmCore::TileM), GemmCore::WgM);
    sycl::range<2> problem{static_cast<size_t>(m_pad), static_cast<size_t>(n) / GemmCore::TileN};
    auto ev = q->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<BType, 1> slm_b(sycl::range(GemmCore::SLM_B_Size), cgh);
      cgh.parallel_for(sycl::nd_range<2>(problem, group),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(GemmCore::SgSize)]] {
                         sycl_utils::nd_item_helper<GemmCore> helper(it);
                         if constexpr (debug) {
                           compute_tile(k, B, ldb, slm_b, A, lda, C, ldc, it);
                         } else {
                           int m_tail = m - helper.sg_g_m();
                           m_tail = m_tail > GemmCore::TileM ? GemmCore::TileM : m_tail;
                           if (m_tail == GemmCore::TileM) {
                             compute_tile(k, B, ldb, slm_b, A, lda, C, ldc, it);
                           } else {
                             compute_tail(k, B, ldb, slm_b, A, lda, C, ldc, m_tail, it);
                           }
                         }
                       });
    });
    return ev;
  }

  static void compute_tile(int k, const BType* B, int ldb, const sycl::local_accessor<BType, 1>& slm_b, const AType* A,
                           int lda, CType* C, int ldc, sycl::nd_item<2>& it) {
    sycl_utils::nd_item_helper<GemmCore> helper(it);
    ACCType tmp[GemmCore::TileM * GemmCore::TileN];
    for (size_t im = 0; im < GemmCore::TileM; im++)
      for (size_t in = 0; in < GemmCore::TileN; in++) tmp[im * GemmCore::TileN + in] = ACCType(0.f);
    for (int i = 0; i < k; i += GemmCore::TileK) {
      PrologueB::getWeight({B, ldb}, slm_b, i, helper);
      it.barrier(sycl::access::fence_space::local_space);
      GemmCore::compute(&A[helper.item_g_m() * lda + i], lda, slm_b, tmp, helper);
      it.barrier(sycl::access::fence_space::local_space);
    }
    Epilogue::store({C, ldc}, tmp, helper);
  }

  static void compute_tail(int k, const BType* B, int ldb, const sycl::local_accessor<BType, 1>& slm_b, const AType* A,
                           int lda, CType* C, int ldc, int m_tail, sycl::nd_item<2>& it) {
    sycl_utils::nd_item_helper<GemmCore> helper(it);
    ACCType tmp[GemmCore::TileM * GemmCore::TileN];
    for (size_t im = 0; im < GemmCore::TileM; im++)
      for (size_t in = 0; in < GemmCore::TileN; in++) tmp[im * GemmCore::TileN + in] = ACCType(0.f);
    for (int i = 0; i < k; i += GemmCore::TileK) {
      PrologueB::getWeight({B, ldb}, slm_b, i, helper);
      it.barrier(sycl::access::fence_space::local_space);
      GemmCore::compute_mtail(&A[helper.item_g_m() * lda + i], lda, slm_b, tmp, helper, m_tail);
      it.barrier(sycl::access::fence_space::local_space);
    }
    Epilogue::store_tail({C, ldc}, tmp, helper, m_tail);
  }
};

template <template <class GCT> class ProAT, template <class GCT> class ProBT, template <class GCT> class EpiT,
          class GemmCoreT>
class LauncherWOQ {
 public:
  using GemmCore = GemmCoreT;
  using PrologueA = ProAT<GemmCore>;
  using PrologueB = ProBT<GemmCore>;
  using Epilogue = EpiT<GemmCore>;
  using AType = typename GemmCore::TA;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::TB;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::TC;
  using ACCType = typename GemmCore::TACC;
  using EpiParam = typename Epilogue::Param;
  struct Param {
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
  };
  static constexpr bool NeedFp8Lut = sycl_prologue_b::needs_fp8_lut<PrologueB>::value;

  template <typename ParamT>
  static inline void load_weight_tile(const ParamT& paramB, const sycl::local_accessor<BType, 1>& slm_b, int koffset,
                                      int blocksize, const sycl_utils::nd_item_helper<GemmCore>& helper,
                                      const sycl::local_accessor<BType, 1>* slm_lut) {
    if constexpr (NeedFp8Lut) {
      PrologueB::getWeight(paramB, slm_b, koffset, blocksize, helper, *slm_lut);
    } else {
      PrologueB::getWeight(paramB, slm_b, koffset, blocksize, helper);
    }
  }

  template <bool debug = false>
  static inline sycl::event compute(sycl::queue* q, int m, int n, int k, int blocksize, const Param& _param) {
    sycl::range<2> group{GemmCore::WgM, GemmCore::WgN};
    auto A = _param.paramA.A;
    auto B = _param.paramB.B;
    auto B_scale = _param.paramB.scale;
    auto C = _param.paramC.C;
    int lda = _param.paramA.lda;
    int ldb = _param.paramB.ldb;
    int ldc = _param.paramC.ldc;
    int m_pad = utils::padto(utils::updiv(m, GemmCore::TileM), GemmCore::WgM);
    sycl::range<2> problem{static_cast<size_t>(m_pad), static_cast<size_t>(n) / GemmCore::TileN};
    auto ev = q->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<BType, 1> slm_b(sycl::range(GemmCore::SLM_B_Size), cgh);
      [[maybe_unused]] sycl::local_accessor<BType, 1> slm_lut(sycl::range<1>(NeedFp8Lut ? 256 : 1), cgh);
      cgh.parallel_for(sycl::nd_range<2>(problem, group),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(GemmCore::SgSize)]] {
                         sycl_utils::nd_item_helper<GemmCore> helper(it);
                         if constexpr (debug) {
                           compute_tile(k, blocksize, B, B_scale, ldb, slm_b, A, lda, C, ldc, it,
                                        NeedFp8Lut ? &slm_lut : nullptr);
                         } else {
                           int m_tail = m - helper.sg_g_m();
                           m_tail = m_tail > GemmCore::TileM ? GemmCore::TileM : m_tail;
                           if (m_tail == GemmCore::TileM) {
                             compute_tile(k, blocksize, B, B_scale, ldb, slm_b, A, lda, C, ldc, it,
                                          NeedFp8Lut ? &slm_lut : nullptr);
                           } else {
                             compute_tail(k, blocksize, m_tail, B, B_scale, ldb, slm_b, A, lda, C, ldc, it,
                                          NeedFp8Lut ? &slm_lut : nullptr);
                           }
                         }
                       });
    });
    return ev;
  }

  template <typename ScaleT>
  static void compute_tile(int k, int blocksize, const uint8_t* B, const ScaleT* B_scale, int ldb,
                           const sycl::local_accessor<BType, 1>& slm_b, const AType* A, int lda, CType* C, int ldc,
                           sycl::nd_item<2>& it, const sycl::local_accessor<BType, 1>* slm_lut) {
    sycl_utils::nd_item_helper<GemmCore> helper(it);
    if constexpr (NeedFp8Lut) {
      PrologueB::init_slm_lut(*slm_lut, it);
    }
    ACCType tmp[GemmCore::TileM * GemmCore::TileN];
    for (size_t im = 0; im < GemmCore::TileM; im++)
      for (size_t in = 0; in < GemmCore::TileN; in++) tmp[im * GemmCore::TileN + in] = ACCType(0.f);
#pragma forceinline recursive
    for (int i = 0; i < k; i += GemmCore::TileK) {
      load_weight_tile(typename PrologueB::Param{B, B_scale, ldb}, slm_b, i, blocksize, helper, slm_lut);
      it.barrier(sycl::access::fence_space::local_space);
      GemmCore::compute(&A[helper.item_g_m() * k + i], k, slm_b, tmp, helper);
      it.barrier(sycl::access::fence_space::local_space);
    }
#pragma forceinline recursive
    Epilogue::store({C, ldc}, tmp, helper);
  }

  template <typename ScaleT>
  static void compute_tail(int k, int blocksize, int m_tail, const uint8_t* B, const ScaleT* B_scale, int ldb,
                           const sycl::local_accessor<BType, 1>& slm_b, const AType* A, int lda, CType* C, int ldc,
                           sycl::nd_item<2>& it, const sycl::local_accessor<BType, 1>* slm_lut) {
    sycl_utils::nd_item_helper<GemmCore> helper(it);
    if constexpr (NeedFp8Lut) {
      PrologueB::init_slm_lut(*slm_lut, it);
    }
    ACCType tmp[GemmCore::TileM * GemmCore::TileN];
    for (size_t im = 0; im < GemmCore::TileM; im++)
      for (size_t in = 0; in < GemmCore::TileN; in++) tmp[im * GemmCore::TileN + in] = ACCType(0.f);
#pragma forceinline recursive
    for (int i = 0; i < k; i += GemmCore::TileK) {
      load_weight_tile(typename PrologueB::Param{B, B_scale, ldb}, slm_b, i, blocksize, helper, slm_lut);
      it.barrier(sycl::access::fence_space::local_space);
      GemmCore::compute_mtail(&A[helper.item_g_m() * k + i], k, slm_b, tmp, helper, m_tail);
      it.barrier(sycl::access::fence_space::local_space);
    }
#pragma forceinline recursive
    Epilogue::store_tail({C, ldc}, tmp, helper, m_tail);
  }
};

struct SDPAParam {
  void *Q, *K, *V;
  void* O;
  void* mask;
  int seq, seq_kv, hn_q, hn_kv, h_dim, h_vdim;
  float scale;
};

struct HGemmCfg {
  static int constexpr sg_size = 16;
  static int constexpr sg_row = 8;
  static int constexpr sg_col = 4;
  static int constexpr TM = 16;
  static int constexpr TN = 16;
  static int constexpr TK = 16;
  static int constexpr SGM = 2;
  static int constexpr SGN = 4;
  static int constexpr UnrollK = 2;
  static int constexpr PrefetchDis = 3;
  static int constexpr G_NROW = 3;
  static int constexpr REPEAT = 1;
  static int constexpr GRF = 256;
  using DT = sycl::half;
  using DT_ACC = sycl::half;
  using DT_MASK = bool;
  using Param = sycl_gemm::GemmParam;
};

template <typename TP, typename CFG>
struct SDPA {
  using DT = CFG::DT;
  using DT1 = CFG::DT_ACC;
  using Param = CFG::Param;
  static int constexpr sg_size = CFG::sg_size;
  static int constexpr sg_row = CFG::sg_row;
  static int constexpr sg_col = CFG::sg_col;
  static int constexpr TM = CFG::TM;
  static int constexpr TN = CFG::TN;
  static int constexpr TK = CFG::TK;
  static int constexpr SGM = CFG::SGM;
  static int constexpr SGN = CFG::SGN;
  static int constexpr UnrollK = CFG::UnrollK;
  static int constexpr PrefetchDis = CFG::PrefetchDis;
  static int constexpr G_NROW = CFG::G_NROW;

  static int constexpr wg_size = sg_size * sg_row * sg_col;
  static int constexpr BM = TM * sg_row * SGM;
  static int constexpr BM_STRIDE = BM;
  static int constexpr BN = TN * sg_col * SGN;
  static int constexpr BN_STRIDE = BN;

  Param param;
  TP mProp;

  int g_ncnt, g_mcnt;
  size_t wg_repeat, aligned_wg;
  SDPA(TP Prop, CFG _cfg, Param _param) : mProp(Prop) {
    param = _param;

    g_ncnt = (param.n + BN - 1) / BN;
    g_mcnt = (param.m + BM - 1) / BM;
    wg_repeat = g_mcnt * g_ncnt;
    size_t m_tail = g_mcnt % G_NROW;
    size_t g_m_aligned = g_mcnt - m_tail;
    aligned_wg = g_m_aligned * g_ncnt;
  }

  inline sycl::nd_range<2> get_range() { return sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}); }

  void operator() [[sycl::reqd_sub_group_size(sg_size)]] (sycl::nd_item<2> it) const {
    auto sg = it.get_sub_group();
    int g_id = it.get_group(0);
    int g_n = g_id % g_ncnt;
    int g_m = g_id / g_ncnt;
    if (g_id < aligned_wg) {
      int g_m_ = g_id % G_NROW;
      g_id /= G_NROW;
      g_n = g_id % g_ncnt;
      g_id /= g_ncnt;
      g_m = g_id * G_NROW + g_m_;
    }
    int sgSize = sg.get_local_range()[0];
    int sgGroupId = sg.get_group_id()[0];
    int sggid_col = sgGroupId % sg_col;
    int sggid_row = sgGroupId / sg_col;
    int sgId = sg.get_local_id()[0];
    int g_idn = sggid_col * TN * SGN + g_n * BN;
    int g_idm = sggid_row * TM * SGM + g_m * BM;
    auto m = param.m;
    auto n = param.n;
    auto k = param.k;
    if (g_idn >= n || g_idm >= m) return;
    auto lda = param.lda;
    auto ldb = param.ldb;
    auto ldc = param.ldc;
    auto Awg_d = (DT*)param.A_d + (size_t)g_idm * lda;
    auto Bwg_d = (DT*)param.B_d + (size_t)g_idn * ldb;
    auto Cwg_d = (DT1*)param.C_d + (size_t)g_idm * ldc + g_idn;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;
    auto pA = syclex::annotated_ptr{
        Awg_d, syclex::properties{syclintelex::read_hint<
                   syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L1>,
                   syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};

    auto pB = syclex::annotated_ptr{
        Bwg_d, syclex::properties{syclintelex::read_hint<
                   syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                   syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};

    auto pC = syclex::annotated_ptr{
        Cwg_d, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                   syclintelex::cache_mode::write_through, syclex::cache_level::L1, syclex::cache_level::L3>>}};
    sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::a, TM, TK, layout::row_major>
        sub_a[SGM];
    sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::b, TK, TN, layout::col_major>
        sub_b[SGN];
    sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT1, use::accumulator, TM, TN>
        sub_c[SGM * SGN];

    DT1 zero = (DT1)0;
    if constexpr (!std::is_same_v<DT1, int32_t>) {
#pragma unroll(1)
      for (int in = 0; in < SGN; in++) {
        auto biasv = param.Bias ? *((DT1*)param.Bias + g_idn + in * TN + sgId) : zero;
#pragma unroll
        for (int im = 0; im < SGM; im++) {
          joint_matrix_fill(sg, sub_c[im * SGN + in], biasv);
        }
      }
    } else {
#pragma unroll
      for (int im = 0; im < SGM; im++)
#pragma unroll
        for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], DT1(0));
    }
    sycl_gemm::xmx::Prefetcher<CFG>::next(sg, 0, m, n, k, g_idm, g_idn, sggid_col, sg_col, sggid_row, sg_row, lda, ldb, Awg_d, Bwg_d);
    sycl::group_barrier(sg);

#pragma unroll(1)
    for (size_t ik = 0; ik < k; ik += TK * UnrollK) {
#pragma unroll
      for (size_t ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
        for (int in = 0; in < SGN; in++) {
          sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(sg, sub_b[in], pB, ldb, n - g_idn, k,
                                                                            in * TN, ik + TK * ikk);
        }
#pragma unroll
        for (int im = 0; im < SGM; im++) {
          sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(sg, sub_a[im], pA, lda, m - g_idm, k,
                                                                            im * TM, ik + TK * ikk);
#pragma unroll
          for (int in = 0; in < SGN; in++) {
            joint_matrix_mad(sg, sub_c[im * SGN + in], sub_a[im], sub_b[in], sub_c[im * SGN + in]);
          }
        }
      }
      sycl_gemm::xmx::Prefetcher<CFG>::next(sg, ik, m, n, k, g_idm, g_idn, sggid_col, sg_col, sggid_row, sg_row, lda, ldb, Awg_d,
                            Bwg_d);
    }
    sycl::group_barrier(sg);
#pragma unroll
    for (int im = 0; im < SGM; im++) {
#pragma unroll
      for (int in = 0; in < SGN; in++) {
        sycl::ext::intel::experimental::matrix::joint_matrix_store_checked(
            sg, sub_c[im * SGN + in], pC, ldc, layout::row_major, m - g_idm, n - g_idn, im * TM, in * TN);
      }
    }
  }

  auto get(syclex::properties_tag) const { return mProp; }
};
}  // namespace sycl_wrapper
}  // namespace bestla
#endif
