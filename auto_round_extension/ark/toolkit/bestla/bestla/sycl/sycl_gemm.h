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
#include <array>

#include "bestla/bestla_utils.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace bestla {

namespace syclex = sycl::ext::oneapi::experimental;
namespace syclintelex = sycl::ext::intel::experimental;
namespace sycl_gemm {

struct GemmParam {
  void *A_d, *B_d;
  void* C_d;
  int m, n, k, lda, ldb, ldc;
  void* Bias = nullptr;
};
namespace xve {

template <class ConfigT>
class GemmCoreSharedB {
 public:
  using Helper_t = sycl_utils::and_item_helper<GemmCoreSharedB<ConfigT>>;
  static int constexpr SgSize = ConfigT::sg_size;
  static int constexpr WgM = ConfigT::wg_m;
  static int constexpr WgN = ConfigT::wg_n;
  static int constexpr SgNStride = WgN / SgSize;
  static int constexpr WgWorkers = WgM * WgN;
  static int constexpr SgCount = WgWorkers / SgSize;
  static int constexpr TileM = ConfigT::sg_m;
  static int constexpr TileN = ConfigT::sg_n;
  static int constexpr TileK = ConfigT::sg_k;
  static int constexpr UnrollK = ConfigT::unroll_k;
  static int constexpr WgNEle = WgN * TileN;
  static int constexpr SLM_B_STRIDE = (WgNEle + 8);
  static int constexpr WgMEle = WgM * TileM;
  static int constexpr SgNEle = SgSize * TileN;
  static int constexpr SLM_B_Size = SLM_B_STRIDE * TileK;
  static int constexpr SLM_A_Size = 0;

  using TA = typename ConfigT::data_type_a;
  using TB = typename ConfigT::data_type_b;
  using TC = typename ConfigT::data_type_c;
  using TACC = typename ConfigT::data_type_acc;

  using SLM_B_Acc = sycl::local_accessor<TB, 1>;

  using AType = TA;
  using BType = TB;
  using CType = TC;
  static auto constexpr NTILE = WgNEle;
  static auto constexpr MTILE = WgMEle;
  static auto constexpr KTILE = TileK;
  static auto constexpr PACK_ROW = 1;
  static int constexpr PREFERRED_N = NTILE;
  static auto constexpr ISA = BTLA_ISA::ISA_COUNT;
  static auto constexpr ID = 0;

  static inline void compute(const TA* aptr, int lda, const SLM_B_Acc& bacc, TACC* accptr, const Helper_t& helper) {
#pragma unroll(1)
    for (int ik = 0; ik < TileK; ik += UnrollK) {
      int constexpr MReg = TileM / SgSize;
      TA regA[UnrollK * MReg];
      for (int im = 0; im < MReg; im++) {
        *(sycl::vec<TA, UnrollK>*)&regA[im * UnrollK] =
            *(sycl::vec<TA, UnrollK>*)&aptr[(helper.sg_id() + im * SgSize) * lda + ik];
      }
#pragma unroll
      for (int ikk = 0; ikk < UnrollK; ikk++) {
        TB tmpB[TileN];
#pragma unroll
        for (int in = 0; in < TileN; in++) {
          tmpB[in] = bacc[helper.sg_idx_n() * SgNEle + helper.sg_id() * TileN + in + (ik + ikk) * SLM_B_STRIDE];
        }
#pragma unroll
        for (size_t im = 0; im < TileM; im++) {
          auto tmpA = sycl::select_from_group(helper.sg, regA[ikk + im / SgSize * UnrollK], im % SgSize);
#pragma unroll
          for (size_t in = 0; in < TileN; in++) {
            accptr[im * TileN + in] += tmpA * tmpB[in];
          }
        }
      }
    }
  }

  static inline void compute_mtail(const TA* aptr, int lda, const SLM_B_Acc& bacc, TACC* accptr, const Helper_t& helper,
                                   int& m_tail) {
    if (m_tail > 0) {
#pragma unroll(1)
      for (int ik = 0; ik < TileK; ik += UnrollK) {
        int constexpr MReg = TileM / SgSize;
        TA regA[UnrollK * MReg];
        for (int im = 0; im < MReg; im++) {
          if ((helper.sg_id() + im * SgSize) < m_tail) {
            *(sycl::vec<TA, UnrollK>*)&regA[im * UnrollK] =
                *(sycl::vec<TA, UnrollK>*)&aptr[(helper.sg_id() + im * SgSize) * lda + ik];
          }
        }

#pragma unroll
        for (int ikk = 0; ikk < UnrollK; ikk++) {
          TB tmpB[TileN];
#pragma unroll
          for (int in = 0; in < TileN; in++) {
            tmpB[in] = bacc[helper.sg_idx_n() * SgNEle + helper.sg_id() * TileN + in + (ik + ikk) * SLM_B_STRIDE];
          }
          for (size_t im = 0; im < TileM; im++) {
            auto tmpA = sycl::select_from_group(helper.sg, regA[ikk + im / SgSize * UnrollK], im % SgSize);
#pragma unroll
            for (size_t in = 0; in < TileN; in++) {
              accptr[im * TileN + in] += tmpA * tmpB[in];
            }
          }
        }
      }
    }
  }
};

class Config_Fp32Fp32Fp32 {
 public:
  static int constexpr sg_size = 32;
  static int constexpr sg_m = 32;
  static int constexpr sg_n = 1;
  static int constexpr sg_k = 16;
  static int constexpr unroll_k = 4;
  static int constexpr wg_m = 8;
  static int constexpr wg_n = 64;

  using data_type_a = float;
  using data_type_b = float;
  using data_type_c = float;
  using data_type_acc = float;
};

class Config_Bf16Bf16Bf16 {
 public:
  static int constexpr sg_size = 32;
  static int constexpr sg_m = 32;
  static int constexpr sg_n = 1;
  static int constexpr sg_k = 16;
  static int constexpr unroll_k = 4;
  static int constexpr wg_m = 8;
  static int constexpr wg_n = 64;

  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = sycl::ext::oneapi::bfloat16;
  using data_type_acc = sycl::ext::oneapi::bfloat16;
};

class Config_Fp16Fp16Fp16 {
 public:
  static int constexpr sg_size = 32;
  static int constexpr sg_m = 32;
  static int constexpr sg_n = 2;
  static int constexpr sg_k = 16;
  static int constexpr unroll_k = 8;
  static int constexpr wg_m = 4;
  static int constexpr wg_n = 64;

  using data_type_a = sycl::half;
  using data_type_b = sycl::half;
  using data_type_c = sycl::half;
  using data_type_acc = sycl::half;
};
using DefaultSGemmCore = GemmCoreSharedB<Config_Fp32Fp32Fp32>;
using DefaultHGemmCore = GemmCoreSharedB<Config_Fp16Fp16Fp16>;
using DefaultHGemmCoreBF16 = GemmCoreSharedB<Config_Bf16Bf16Bf16>;

template <class ConfigT>
class GemmCoreSharedBT {
 public:
  using Helper_t = sycl_utils::and_item_helper<GemmCoreSharedBT<ConfigT>>;
  static int constexpr SgSize = ConfigT::sg_size;
  static int constexpr WgM = ConfigT::wg_m;
  static int constexpr WgN = ConfigT::wg_n;
  static int constexpr SgNStride = WgN / SgSize;
  static int constexpr WgWorkers = WgM * WgN;
  static int constexpr SgCount = WgWorkers / SgSize;
  static int constexpr TileM = ConfigT::sg_m;
  static int constexpr TileN = ConfigT::sg_n;
  static int constexpr TileK = ConfigT::sg_k;
  static int constexpr UnrollK = ConfigT::unroll_k;
  static int constexpr WgNEle = WgN * TileN;
  static int constexpr SLM_B_STRIDE = WgNEle;
  static int constexpr WgMEle = WgM * TileM;
  static int constexpr SgNEle = SgSize * TileN;
  static int constexpr SLM_B_Size = SLM_B_STRIDE * TileK;
  static int constexpr SLM_A_Size = 0;

  using TA = typename ConfigT::data_type_a;
  using TB = typename ConfigT::data_type_b;
  using TC = typename ConfigT::data_type_c;
  using TACC = typename ConfigT::data_type_acc;

  using SLM_B_Acc = sycl::local_accessor<TB, 1>;

  using AType = TA;
  using BType = TB;
  using CType = TC;
  static auto constexpr NTILE = WgNEle;
  static auto constexpr MTILE = WgMEle;
  static auto constexpr KTILE = TileK;
  static auto constexpr PACK_ROW = 1;
  static int constexpr PREFERRED_N = NTILE;
  static auto constexpr ISA = BTLA_ISA::ISA_COUNT;
  static auto constexpr ID = 0;

  static inline void compute(const TA* aptr, int lda, const SLM_B_Acc& bacc, TACC* accptr, const Helper_t& helper) {
#pragma unroll(1)
    for (int ik = 0; ik < TileK; ik += UnrollK) {
      int constexpr MReg = TileM / SgSize;
      TA regA[UnrollK * MReg];
      for (int im = 0; im < MReg; im++) {
        *(sycl::vec<TA, UnrollK>*)&regA[im * UnrollK] =
            *(sycl::vec<TA, UnrollK>*)&aptr[(helper.sg_id() + im * SgSize) * lda + ik];
      }
      TB tmpB[UnrollK * TileN];
#pragma unroll
      for (int in = 0; in < TileN; in++) {
        *(sycl::vec<TB, UnrollK>*)&tmpB[in * UnrollK] =
            *(sycl::vec<TB, UnrollK>*)&bacc[(helper.sg_idx_n() * SgNEle + helper.sg_id() * TileN + in) * UnrollK +
                                            ik * SLM_B_STRIDE];
      }
      constexpr int unrk = std::is_same_v<TB, float> ? 8 : 8;
#pragma unroll(unrk)
      for (int ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
        for (size_t im = 0; im < TileM; im++) {
          auto tmpA = sycl::select_from_group(helper.sg, regA[ikk + im / SgSize * UnrollK], im % SgSize);
#pragma unroll
          for (size_t in = 0; in < TileN; in++) {
            accptr[im * TileN + in] += tmpA * tmpB[in * UnrollK + ikk];
          }
        }
      }
    }
  }

  static inline void compute_mtail(const TA* aptr, int lda, const SLM_B_Acc& bacc, TACC* accptr, const Helper_t& helper,
                                   int& m_tail) {
    if (m_tail > 0) {
#pragma unroll(1)
      for (int ik = 0; ik < TileK; ik += UnrollK) {
        int constexpr MReg = TileM / SgSize;
        TA regA[UnrollK * MReg];
#pragma unroll
        for (int im = 0; im < MReg; im++) {
          if ((helper.sg_id() + im * SgSize) < m_tail) {
            *(sycl::vec<TA, UnrollK>*)&regA[im * UnrollK] =
                *(sycl::vec<TA, UnrollK>*)&aptr[(helper.sg_id() + im * SgSize) * lda + ik];
          }
        }
        TB tmpB[UnrollK * TileN];
#pragma unroll
        for (int in = 0; in < TileN; in++) {
          *(sycl::vec<TB, UnrollK>*)&tmpB[in * UnrollK] =
              *(sycl::vec<TB, UnrollK>*)&bacc[(helper.sg_idx_n() * SgNEle + helper.sg_id() * TileN + in) * UnrollK +
                                              ik * SLM_B_STRIDE];
        }
        constexpr int unrk = std::is_same_v<TB, float> ? 4 : 8;
#pragma unroll(unrk)
        for (int ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
          for (size_t im = 0; im < TileM; im++) {
            auto tmpA = sycl::select_from_group(helper.sg, regA[ikk + im / SgSize * UnrollK], im % SgSize);
#pragma unroll
            for (size_t in = 0; in < TileN; in++) {
              accptr[im * TileN + in] += tmpA * tmpB[in * UnrollK + ikk];
            }
          }
        }
      }
    }
  }
};

class Config_Fp32Fp32Fp32T {
 public:
  static int constexpr sg_size = 32;
  static int constexpr sg_m = 32;
  static int constexpr sg_n = 1;
  static int constexpr sg_k = 64;
  static int constexpr unroll_k = 8;
  static int constexpr wg_m = 4;
  static int constexpr wg_n = 64;

  using data_type_a = float;
  using data_type_b = float;
  using data_type_c = float;
  using data_type_acc = float;
};

class Config_Fp16Fp16Fp16T {
 public:
  static int constexpr sg_size = 32;
  static int constexpr sg_m = 32;
  static int constexpr sg_n = 1;
  static int constexpr sg_k = 64;
  static int constexpr unroll_k = 8;
  static int constexpr wg_m = 4;
  static int constexpr wg_n = 64;

  using data_type_a = sycl::half;
  using data_type_b = sycl::half;
  using data_type_c = sycl::half;
  using data_type_acc = sycl::half;
};
using DefaultSGemmCoreT = GemmCoreSharedBT<Config_Fp32Fp32Fp32T>;
using DefaultHGemmCoreT = GemmCoreSharedBT<Config_Fp16Fp16Fp16T>;
using DefaultHGemmCoreBF16T = GemmCoreSharedBT<Config_Bf16Bf16Bf16>;

template <typename CFG>
struct Prefetcher : CFG {
  using DT = CFG::DT;

  static int constexpr SGM = CFG::SGM;
  static int constexpr SGN = CFG::SGN;
  static int constexpr TM = CFG::TM;
  static int constexpr TN = CFG::TN;
  static int constexpr TK = CFG::TK;
  static int constexpr UnrollK = CFG::UnrollK;
  static int constexpr PrefetchDis = CFG::PrefetchDis;

  static inline void next(sycl::sub_group sg, int ik, int m, int n, int k, int g_idm, int g_idn, int sggid_col,
                          int sg_col, int sggid_row, int sg_row, int lda, int ldb, const DT* Awg_d, const DT* Bwg_d) {
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;

    if (ik + TK * UnrollK * (PrefetchDis + 1) <= k) {
      for (int im = sggid_col; im < SGM; im += sg_col)
        if ((g_idm + im * TM + TM) <= m)
          joint_matrix_prefetch<TM, TK * UnrollK>(
              sg, Awg_d + (im * TM) * lda + ik + TK * UnrollK * PrefetchDis, lda, layout::row_major,
              sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L1});
      for (int in = sggid_row; in < SGN; in += sg_row)
        if ((g_idn + in * TN + TN) <= n)
          joint_matrix_prefetch<TN, TK * UnrollK>(
              sg, Bwg_d + (in * TN) * ldb + ik + TK * UnrollK * PrefetchDis, ldb, layout::row_major,
              sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L1});
    }
  }
};
struct SGemmCfg {
  static int constexpr sg_size = 16;
  static int constexpr sg_row = 8;
  static int constexpr sg_col = 4;
  static int constexpr TM = 16;
  static int constexpr TN = 16;
  static int constexpr TK = 16;
  static int constexpr SGM = 2;
  static int constexpr SGN = 2;
  static int constexpr UnrollK = 2;
  static int constexpr PrefetchDis = 3;
  static int constexpr G_NROW = 3;
  static int constexpr GRF = 256;
  using DT = float;
  using DT_ACC = float;
  using Param = GemmParam;
};

template <typename TP, typename CFG>
struct GemmCore {
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
  GemmCore(TP Prop, CFG _cfg, Param _param) : mProp(Prop) {
    param = _param;

    g_ncnt = (param.n + BN - 1) / BN;
    g_mcnt = (param.m + BM - 1) / BM;
    wg_repeat = g_mcnt * g_ncnt;
    size_t m_tail = g_mcnt % G_NROW;
    size_t g_m_aligned = g_mcnt - m_tail;
    aligned_wg = g_m_aligned * g_ncnt;
  }

  inline sycl::and_range<2> get_range() { return sycl::and_range<2>({wg_repeat, wg_size}, {1, wg_size}); }

  void operator() [[sycl::reqd_sub_group_size(sg_size)]] (sycl::and_item<2> it) const {
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
    DT sub_a[SGM * TM], sub_b[SGN * TK];
    DT1 sub_c[SGM * TM * SGN];

    DT1 zero = (DT1)0;
    if (param.Bias) {
      auto bptr = (DT1*)param.Bias + g_idn;
#pragma unroll
      for (int in = 0; in < SGN; in++) {
        if ((in * TN + sgId) + g_idn >= n)
#pragma unroll
          for (int im = 0; im < SGM * TM; im++) sub_c[im * SGN + in] = zero;
        else
#pragma unroll
          for (int im = 0; im < SGM * TM; im++) sub_c[im * SGN + in] = bptr[in * TN + sgId];
      }

    } else {
#pragma unroll
      for (int im = 0; im < SGM * TM; im++)
#pragma unroll
        for (int in = 0; in < SGN; in++) sub_c[im * SGN + in] = zero;
    }

    // Prefetcher<CFG>::next(sg, 0, m, n, k, g_idm, g_idn, sggid_col, sg_col, sggid_row, sg_row, lda, ldb, Awg_d,
    // Bwg_d);
    sycl::group_barrier(sg);

#pragma unroll(1)
    for (size_t ik = 0; ik < k; ik += TK * UnrollK) {
#pragma unroll
      for (size_t ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
        for (int _ik = 0; _ik < TK; _ik++) {
          if (ik + TK * ikk + _ik >= k) {
#pragma unroll
            for (int in = 0; in < SGN; in++) sub_b[in * TK + _ik] = zero;
          } else {
#pragma unroll
            for (int in = 0; in < SGN; in++) {
              if ((in * TN + sgId) + g_idn >= n) break;
              sub_b[in * TK + _ik] = pB[(in * TN + sgId) * ldb + ik + TK * ikk + _ik];
            }
          }
        }
        if (ik + TK * ikk + sgId < k)
#pragma unroll
          for (int im = 0; im < SGM * TM; im++) {
            if (im + g_idm >= m) break;
            sub_a[im] = pA[im * lda + ik + TK * ikk + sgId];
          }
        else
#pragma unroll
          for (int im = 0; im < SGM * TM; im++) {
            sub_a[im] = zero;
          }
        sycl::group_barrier(sg);
#pragma unroll
        for (int im = 0; im < SGM * TM; im++) {
#pragma unroll
          for (int in = 0; in < SGN; in++) {
#pragma unroll
            for (int _ik = 0; _ik < TK; _ik++) {
              sub_c[im * SGN + in] += sycl::select_from_group(sg, sub_a[im], _ik) * sub_b[in * TK + _ik];
            }
          }
        }
      }
      // Prefetcher<CFG>::next(sg, ik, m, n, k, g_idm, g_idn, sggid_col, sg_col, sggid_row, sg_row, lda, ldb, Awg_d,
      //                       Bwg_d);
      sycl::group_barrier(sg);
    }
#pragma unroll
    for (int im = 0; im < SGM * TM; im++) {
      if (im + g_idm >= m) break;
#pragma unroll
      for (int in = 0; in < SGN; in++) {
        if ((in * TN + sgId) + g_idn >= n) break;
        pC[im * ldc + in * TN + sgId] = sub_c[im * SGN + in];
      }
    }
  }

  auto get(syclex::properties_tag) const { return mProp; }
};
}  // namespace xve

namespace xmx {
template <typename CFG>
struct Prefetcher : CFG {
  using DT = CFG::DT;

  static int constexpr SGM = CFG::SGM;
  static int constexpr SGN = CFG::SGN;
  static int constexpr TM = CFG::TM;
  static int constexpr TN = CFG::TN;
  static int constexpr TK = CFG::TK;
  static int constexpr UnrollK = CFG::UnrollK;
  static int constexpr PrefetchDis = CFG::PrefetchDis;

  static inline void next(sycl::sub_group sg, int ik, int m, int n, int k, int g_idm, int g_idn, int sggid_col,
                          int sg_col, int sggid_row, int sg_row, int lda, int ldb, const DT* Awg_d, const DT* Bwg_d) {
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;

    if (ik + TK * UnrollK * (PrefetchDis + 1) <= k) {
      for (int im = sggid_col; im < SGM; im += sg_col)
        if ((g_idm + im * TM + TM) <= m)
          joint_matrix_prefetch<TM, TK * UnrollK>(
              sg, Awg_d + (im * TM) * lda + ik + TK * UnrollK * PrefetchDis, lda, layout::row_major,
              sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L1});
      for (int in = sggid_row; in < SGN; in += sg_row)
        if ((g_idn + in * TN + TN) <= n)
          joint_matrix_prefetch<TN, TK * UnrollK>(
              sg, Bwg_d + (in * TN) * ldb + ik + TK * UnrollK * PrefetchDis, ldb, layout::row_major,
              sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L1});
    }
  }
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
  static int constexpr GRF = 256;
  using DT = sycl::half;
  using DT_ACC = sycl::half;
  using Param = GemmParam;
};

struct HGemmAfp32Cfg {
  static int constexpr sg_size = 16;
  static int constexpr sg_row = 8;
  static int constexpr sg_col = 4;
  static int constexpr TM = 16;
  static int constexpr TN = 16;
  static int constexpr TK = 16;
  static int constexpr SGM = 2;
  static int constexpr SGN = 2;
  static int constexpr UnrollK = 2;
  static int constexpr PrefetchDis = 3;
  static int constexpr G_NROW = 3;
  static int constexpr GRF = 256;
  using DT = sycl::half;
  using DT_ACC = float;
  using Param = GemmParam;
};

struct HGemmBf16Cfg {
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
  static int constexpr G_NROW = 2;
  static int constexpr GRF = 256;
  using DT = sycl::ext::oneapi::bfloat16;
  using DT_ACC = sycl::ext::oneapi::bfloat16;
  using Param = GemmParam;
};

struct HGemmBf16Fp32Cfg {
  static int constexpr sg_size = 16;
  static int constexpr sg_row = 8;
  static int constexpr sg_col = 4;
  static int constexpr TM = 16;
  static int constexpr TN = 16;
  static int constexpr TK = 16;
  static int constexpr SGM = 2;
  static int constexpr SGN = 2;
  static int constexpr UnrollK = 2;
  static int constexpr PrefetchDis = 3;
  static int constexpr G_NROW = 2;
  static int constexpr GRF = 256;
  using DT = sycl::ext::oneapi::bfloat16;
  using DT_ACC = float;
  using Param = GemmParam;
};

struct IGemmCfg {
  static int constexpr sg_size = 16;
  static int constexpr sg_row = 8;
  static int constexpr sg_col = 4;
  static int constexpr TM = 8;
  static int constexpr TN = 16;
  static int constexpr TK = 32;
  static int constexpr SGM = 4;
  static int constexpr SGN = 4;
  static int constexpr UnrollK = 2;
  static int constexpr PrefetchDis = 3;
  static int constexpr G_NROW = 3;
  static int constexpr GRF = 256;
  using DT = int8_t;
  using DT_ACC = int32_t;
  using Param = GemmParam;
};

template <typename TP, typename CFG>
struct GemmCore {
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
  GemmCore(TP Prop, CFG _cfg, Param _param) : mProp(Prop) {
    param = _param;

    g_ncnt = (param.n + BN - 1) / BN;
    g_mcnt = (param.m + BM - 1) / BM;
    wg_repeat = g_mcnt * g_ncnt;
    size_t m_tail = g_mcnt % G_NROW;
    size_t g_m_aligned = g_mcnt - m_tail;
    aligned_wg = g_m_aligned * g_ncnt;
  }

  inline sycl::and_range<2> get_range() { return sycl::and_range<2>({wg_repeat, wg_size}, {1, wg_size}); }

  void operator() [[sycl::reqd_sub_group_size(sg_size)]] (sycl::and_item<2> it) const {
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
    Prefetcher<CFG>::next(sg, 0, m, n, k, g_idm, g_idn, sggid_col, sg_col, sggid_row, sg_row, lda, ldb, Awg_d, Bwg_d);
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
      Prefetcher<CFG>::next(sg, ik, m, n, k, g_idm, g_idn, sggid_col, sg_col, sggid_row, sg_row, lda, ldb, Awg_d,
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

struct IGemmDQParam : GemmParam {
  void *scaleA, *scaleB;
};

template <typename T>
struct IGemmDQCfg {
  static int constexpr sg_size = 16;
  static int constexpr sg_row = 8;
  static int constexpr sg_col = 4;
  static int constexpr TM = 8;
  static int constexpr TN = 16;
  static int constexpr TK = 32;
  static int constexpr SGM = 4;
  static int constexpr SGN = 4;
  static int constexpr UnrollK = 2;
  static int constexpr PrefetchDis = 3;
  static int constexpr G_NROW = 3;
  static int constexpr GRF = 256;
  using DT = int8_t;
  using DT_ACC = int32_t;
  using DQT = T;
  using Param = IGemmDQParam;
};

template <typename TP, typename CFG>
struct IGemmDQCore {
  using DT = CFG::DT;
  using DT1 = CFG::DT_ACC;
  using DQT = CFG::DQT;
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
  IGemmDQCore(TP Prop, CFG _cfg, Param _param) : mProp(Prop) {
    param = _param;

    g_ncnt = (param.n + BN - 1) / BN;
    g_mcnt = (param.m + BM - 1) / BM;
    wg_repeat = g_mcnt * g_ncnt;
    size_t m_tail = g_mcnt % G_NROW;
    size_t g_m_aligned = g_mcnt - m_tail;
    aligned_wg = g_m_aligned * g_ncnt;
  }

  inline sycl::and_range<2> get_range() { return sycl::and_range<2>({wg_repeat, wg_size}, {1, wg_size}); }

  void operator() [[sycl::reqd_sub_group_size(sg_size)]] (sycl::and_item<2> it) const {
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
    auto Cwg_d = (DQT*)param.C_d + (size_t)g_idm * ldc + g_idn;
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

#pragma unroll
    for (int im = 0; im < SGM; im++)
#pragma unroll
      for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], DT1(0));
    Prefetcher<CFG>::next(sg, 0, m, n, k, g_idm, g_idn, sggid_col, sg_col, sggid_row, sg_row, lda, ldb, Awg_d, Bwg_d);
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
      Prefetcher<CFG>::next(sg, ik, m, n, k, g_idm, g_idn, sggid_col, sg_col, sggid_row, sg_row, lda, ldb, Awg_d,
                            Bwg_d);
    }
    sycl::group_barrier(sg);
    auto scaleA = (DQT*)param.scaleA;
    auto scaleB = (DQT*)param.scaleB;
    auto bias = (DQT*)param.Bias;

    DQT sBs[SGN], bBs[SGN];
#pragma unroll
    for (int in = 0; in < SGN; in++) {
      if (in * TN + g_idn + sgId >= n) break;
      sBs[in] = scaleB[in * TN + g_idn + sgId];
      bBs[in] = bias ? bias[in * TN + g_idn + sgId] : DQT(0);
    }

#pragma unroll
    for (int im = 0; im < SGM; im++) {
#pragma unroll
      for (int imm = 0; imm < TM; imm++) {
        if (im * TM + g_idm + imm >= m) break;
        auto scale2 = scaleA[im * TM + g_idm + imm];
#pragma unroll
        for (int in = 0; in < SGN; in++) {
          auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(sg, sub_c[im * SGN + in]);
          Cwg_d[(im * TM + imm) * ldc + sgId + in * TN] =
              static_cast<float>(wi_data_c[imm]) * sBs[in] * scale2 + bBs[in];
        }
      }
    }
  }

  auto get(syclex::properties_tag) const { return mProp; }
};

struct IKblockGemmDQParam : IGemmDQParam {
  int blocksize;
};

template <typename T>
struct IKblockGemmDQCfg {
  static int constexpr sg_size = 16;
  static int constexpr sg_row = 8;
  static int constexpr sg_col = 4;
  static int constexpr TM = 8;
  static int constexpr TN = 16;
  static int constexpr TK = 32;
  static int constexpr SGM = 4;
  static int constexpr SGN = 2;
  static int constexpr UnrollK = 2;
  static int constexpr PrefetchDis = 3;
  static int constexpr G_NROW = 3;
  static int constexpr GRF = 256;
  using DT = int8_t;
  using DT_ACC = int32_t;
  using DQT = T;
  using Param = IKblockGemmDQParam;
};

template <typename TP, typename CFG>
struct IKblockGemmDQCore {
  using DT = CFG::DT;
  using DT1 = CFG::DT_ACC;
  using DQT = CFG::DQT;
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

  int g_ncnt, g_mcnt, blks;
  size_t wg_repeat, aligned_wg;
  IKblockGemmDQCore(TP Prop, CFG _cfg, Param _param) : mProp(Prop) {
    param = _param;

    g_ncnt = (param.n + BN - 1) / BN;
    g_mcnt = (param.m + BM - 1) / BM;
    blks = (_param.k) / _param.blocksize;
    wg_repeat = g_mcnt * g_ncnt;
    size_t m_tail = g_mcnt % G_NROW;
    size_t g_m_aligned = g_mcnt - m_tail;
    aligned_wg = g_m_aligned * g_ncnt;
  }

  inline sycl::and_range<2> get_range() { return sycl::and_range<2>({wg_repeat, wg_size}, {1, wg_size}); }

  void operator() [[sycl::reqd_sub_group_size(sg_size)]] (sycl::and_item<2> it) const {
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
    int blocksize = param.blocksize;
    auto Awg_d = (DT*)param.A_d + (size_t)g_idm * lda;
    auto Bwg_d = (DT*)param.B_d + (size_t)g_idn * ldb;
    auto Cwg_d = (DQT*)param.C_d + (size_t)g_idm * ldc + g_idn;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;
    auto pA = syclex::annotated_ptr{
        Awg_d, syclex::properties{syclintelex::read_hint<
                   syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                   syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
    auto pSA = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
        (DQT*)param.scaleA);
    auto pB = syclex::annotated_ptr{
        Bwg_d, syclex::properties{syclintelex::read_hint<
                   syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                   syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
    auto pSB = syclex::annotated_ptr{
        (DQT*)param.scaleB, syclex::properties{syclintelex::read_hint<
                                syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};
    auto pC = syclex::annotated_ptr{
        Cwg_d, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                   syclintelex::cache_mode::write_through, syclex::cache_level::L1, syclex::cache_level::L3>>}};
    sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::a, TM, TK, layout::row_major>
        sub_a[SGM];
    sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::b, TK, TN, layout::col_major>
        sub_b[SGN];
    sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT1, use::accumulator, TM, TN>
        sub_c[SGM * SGN];
    float acc_c[SGM * SGN * TM];

    DT1 zero = (DT1)0;
#pragma unroll
    for (int in = 0; in < SGN; in++) {
#pragma unroll
      for (int im = 0; im < SGM; im++) {
#pragma unroll
        for (int imm = 0; imm < TM; imm++) {
          acc_c[(im * TM + imm) * SGN + in] = zero;
        }
      }
    }
    Prefetcher<CFG>::next(sg, 0, m, n, k, g_idm, g_idn, sggid_col, sg_col, sggid_row, sg_row, lda, ldb, Awg_d, Bwg_d);
    sycl::group_barrier(sg);
    for (int ib = 0; ib < blks; ib++) {
#pragma unroll
      for (int im = 0; im < SGM; im++)
#pragma unroll
        for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], 0);
#pragma unroll(1)
      for (int ik = ib * blocksize; ik < (ib + 1) * blocksize; ik += TK * UnrollK) {
#pragma unroll
        for (int ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
          for (int in = 0; in < SGN; in++) {
            sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(sg, sub_b[in], pB, ldb, n - g_idn, k,
                                                                              in * TN, ik + ikk * TK);
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
        Prefetcher<CFG>::next(sg, ik, m, n, k, g_idm, g_idn, sggid_col, sg_col, sggid_row, sg_row, lda, ldb, Awg_d,
                              Bwg_d);
      }
      sycl::group_barrier(sg);
      DQT sBs[SGN];
#pragma unroll
      for (int in = 0; in < SGN; in++) {
        if (in * TN + g_idn + sgId >= n) break;
        sBs[in] = pSB[(in * TN + g_idn + sgId) * blks + ib];
      }
#pragma unroll
      for (int im = 0; im < SGM; im++) {
#pragma unroll
        for (int in = 0; in < SGN; in++) {
          auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(sg, sub_c[im * SGN + in]);
          for (int i = 0; i < wi_data_c.length(); i++) {
            auto element = wi_data_c[i];
            acc_c[(im * TM + i) * SGN + in] += static_cast<float>(element) * static_cast<float>(sBs[in]);
          }
        }
      }
      sycl::group_barrier(sg);
    }
    DQT bBs[SGN];
#pragma unroll
    for (int in = 0; in < SGN; in++) {
      if (in * TN + g_idn + sgId >= n) break;
      bBs[in] = param.Bias ? ((DQT*)param.Bias)[in * TN + g_idn + sgId] : DQT(0);
    }

#pragma unroll
    for (int im = 0; im < SGM; im++) {
#pragma unroll
      for (int imm = 0; imm < TM; imm++) {
        if (im * TM + g_idm + imm >= m) break;
        auto scale2 = pSA[im * TM + g_idm + imm];
#pragma unroll
        for (int in = 0; in < SGN; in++) {
          if (in * TN + g_idn + sgId >= n) break;
          pC[(im * TM + imm) * ldc + sgId + in * TN] = acc_c[(im * TM + imm) * SGN + in] * scale2 + bBs[in];
        }
      }
    }
  }

  auto get(syclex::properties_tag) const { return mProp; }
};

}  // namespace xmx

template <typename CFG, template <typename AT, typename BT> typename KER>
struct Launcher {
  static inline sycl::event run(sycl::queue* q, const typename CFG::Param& _param) {
    auto ker = [&](sycl::handler& cgh) {
      syclex::properties prop{syclintelex::grf_size<CFG::GRF>};
      KER largeker(prop, CFG(), _param);
      cgh.parallel_for(largeker.get_range(), largeker);
    };
    return q->submit(ker);
  }
};
}  // namespace sycl_gemm
}  // namespace bestla
#endif
