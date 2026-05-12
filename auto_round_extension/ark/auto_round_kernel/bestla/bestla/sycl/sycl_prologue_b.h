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

#include "bestla/sycl/fp8_lut.h"

namespace bestla {
namespace sycl_prologue_b {

template <typename SrcT>
struct ParamWeightBase {
  const SrcT* B;
  int ldb;
};

template <typename ScaleT>
struct ParamWeightS4 {
  const uint8_t* B;
  const ScaleT* scale;
  int ldb;
  const void* bias = nullptr;
  const int8_t* zp = nullptr;
};

template <typename ScaleT>
struct ParamWeightS8 {
  const int8_t* B;
  const ScaleT* scale;
  int ldb;
  const void* bias = nullptr;
  const int8_t* zp = nullptr;
};

template <typename ScaleT>
struct ParamWeightF8 {
  const uint8_t* B;
  const ScaleT* scale;
  int ldb;
  const void* bias = nullptr;
};

template <typename ScaleT>
struct ParamWeightS4Ext : ParamWeightS4<ScaleT> {
  const ScaleT* scaleExt = nullptr;
};

#define INT2x4_TO_0(src) (static_cast<int8_t>(src & (uint8_t)0x03))
#define INT2x4_TO_1(src) (static_cast<int8_t>((src >> (uint8_t)2) & (uint8_t)0x03))
#define INT2x4_TO_2(src) (static_cast<int8_t>((src >> (uint8_t)4) & (uint8_t)0x03))
#define INT2x4_TO_3(src) (static_cast<int8_t>(src >> (uint8_t)6))
#define INT4x2_TO_LO(src) (static_cast<int8_t>(src & (uint8_t)0x0f))
#define INT4x2_TO_HI(src) (static_cast<int8_t>(src >> (uint8_t)4))

template <typename ScaleT>
class WeightS2T {
 public:
  using Param = ParamWeightS4Ext<ScaleT>;

  static __attribute__((always_inline)) inline int8_t int2x4_to_0(const uint8_t& src) {
    return INT2x4_TO_0(src);
  }

  static __attribute__((always_inline)) inline int8_t int2x4_to_1(const uint8_t& src) {
    return INT2x4_TO_1(src);
  }

  static __attribute__((always_inline)) inline int8_t int2x4_to_2(const uint8_t& src) {
    return INT2x4_TO_2(src);
  }

  static __attribute__((always_inline)) inline int8_t int2x4_to_3(const uint8_t& src) {
    return INT2x4_TO_3(src);
  }

  struct CfgDequantF32 {
    static int constexpr SgSize = 32;
    static int constexpr TileK = 16;
    static int constexpr TileN = 1;
    static int constexpr Unroll = 8;
  };

  struct CfgDequantF16 {
    static int constexpr SgSize = 32;
    static int constexpr TileK = 16;
    static int constexpr TileN = 1;
    static int constexpr Unroll = 8;
  };

  template <typename Cfg, typename BType>
  static inline sycl::event dequant(int n, int k, int blocksize, const Param& in, BType* outptr, sycl::queue* q) {
    int constexpr SgSize = Cfg::SgSize;
    int constexpr TileK = Cfg::TileK;
    int constexpr TileN = Cfg::TileN;
    static_assert(TileN == 1);
    int constexpr Unroll = Cfg::Unroll;
    int constexpr SubGroupK = SgSize * TileK;
    int constexpr GroupK = SgSize * TileK;
    assert(blocksize % Unroll == 0);
    int nsg_k = utils::updiv(k, GroupK);
    int nsg_n = n;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{static_cast<size_t>(nsg_n) * nsg_k * SgSize};
    auto B_d = in.B;
    auto S_d = in.scale;
    auto zp_d = in.zp;
    int ldb = in.ldb;
    int ldbn = in.ldb * blocksize;
    auto deq_kernel = [&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            int sg_id = sg.get_local_id()[0];
            int sg_group_id = sg.get_group_id()[0];
            int g_idx_n = g_idx / nsg_k;
            int g_idx_k = g_idx % nsg_k;
            int g_n = g_idx_n * 1;
            int g_k = g_idx_k * GroupK;
            int sg_k = g_k + sg_group_id * SubGroupK;
            int sg_k_remain = k - sg_k;
            auto sptr = S_d + sg_k / blocksize + g_n * ldb;
            auto zpptr = zp_d ? zp_d + sg_k / blocksize + g_n * ldb : nullptr;
            auto bptr = B_d + (sg_k + g_n * ldbn) / 4;
            auto dbptr = outptr + sg_k + g_n * k;
#pragma unroll
            for (int ik = 0; ik < TileK; ik += Unroll) {
              if (ik * SgSize + sg_id * Unroll >= sg_k_remain) break;
              BType dst[Unroll];
              BType scale = sptr[(ik * SgSize + sg_id * Unroll) / blocksize];
              int8_t zp = zpptr ? zpptr[(ik * SgSize + sg_id * Unroll) / blocksize] + int8_t(2) : int8_t(2);
#pragma unroll
              for (int ir = 0; ir < Unroll; ir += 4) {
                uint8_t srcu8 = *(bptr + (ik * SgSize + sg_id * Unroll + ir) / 4);
                dst[ir] = static_cast<BType>(int2x4_to_0(srcu8) - zp) * scale;
                dst[ir + 1] = static_cast<BType>(int2x4_to_1(srcu8) - zp) * scale;
                dst[ir + 2] = static_cast<BType>(int2x4_to_2(srcu8) - zp) * scale;
                dst[ir + 3] = static_cast<BType>(int2x4_to_3(srcu8) - zp) * scale;
              }
              *(sycl::vec<BType, Unroll>*)&dbptr[ik * SgSize + sg_id * Unroll] = *(sycl::vec<BType, Unroll>*)dst;
            }
          });
    };
    return q->submit(deq_kernel);
  }

  struct CfgDequantS8 {
    static int constexpr SgSize = 32;
    static int constexpr TileK = 16;
    static int constexpr TileN = 1;
    static int constexpr Unroll = 16;
    static bool constexpr Rescale = 0;
  };

  struct CfgDequantS8Rescale {
    static int constexpr SgSize = 32;
    static int constexpr TileK = 32;
    static int constexpr TileN = 1;
    static int constexpr Unroll = 16;
    static bool constexpr Rescale = 1;
  };

  template <class Cfg>
  static inline sycl::event dequantS8(int n, int k, int blocksize, const Param& in, int8_t* outptr, sycl::queue* q,
                                      int newblocksize = -1) {
    int constexpr SgSize = Cfg::SgSize;
    int constexpr TileK = Cfg::TileK;
    int constexpr TileN = Cfg::TileN;
    int constexpr Rescale = Cfg::Rescale;
    static_assert(TileN == 1);
    int constexpr Unroll = Cfg::Unroll;
    int constexpr SubGroupK = SgSize * TileK;
    int constexpr GroupK = SgSize * TileK;
    assert(blocksize % Unroll == 0);
    int nsg_k = utils::updiv(k, GroupK);
    int nsg_n = n;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{static_cast<size_t>(nsg_n) * nsg_k * SgSize};
    auto B_d = in.B;
    auto S_d = in.scale;
    auto Sext_d = in.scaleExt;
    auto zp_d = in.zp;
    int ldb = in.ldb;
    int ldbn = in.ldb * blocksize;
    int newblks = newblocksize == -1 ? 1 : k / newblocksize;
    auto deq_kernel = [&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            int sg_id = sg.get_local_id()[0];
            int sg_group_id = sg.get_group_id()[0];
            int g_idx_n = g_idx / nsg_k;
            int g_idx_k = g_idx % nsg_k;
            int g_n = g_idx_n * 1;
            int g_k = g_idx_k * GroupK;
            int sg_k = g_k + sg_group_id * SubGroupK;
            int sg_k_remain = k - sg_k;
            auto sptr = S_d + sg_k / blocksize + g_n * ldb;
            auto newsptr = Sext_d + sg_k / newblocksize + g_n * newblks;
            auto zpptr = zp_d ? zp_d + sg_k / blocksize + g_n * ldb : nullptr;
            auto bptr = B_d + (sg_k + g_n * ldbn) / 4;
            auto dbptr = outptr + sg_k + g_n * k;

#pragma unroll
            for (int ik = 0; ik < TileK; ik += Unroll) {
              if (ik * SgSize + sg_id * Unroll >= sg_k_remain) break;
              int8_t zp = zpptr ? zpptr[(ik * SgSize + sg_id * Unroll) / blocksize] + int8_t(2) : int8_t(2);
              int8_t dst[Unroll];
              if constexpr (Rescale) {
                float scaleN = (float)newsptr[(ik * SgSize + sg_id * Unroll) / newblocksize];
                float scale = (float)sptr[(ik * SgSize + sg_id * Unroll) / blocksize] / scaleN;
#pragma unroll
                for (int ir = 0; ir < Unroll; ir += 4) {
                  uint8_t srcu8 = *(bptr + (ik * SgSize + sg_id * Unroll + ir) / 4);
                  dst[ir] = std::round(static_cast<float>(int2x4_to_0(srcu8) - zp) * scale);
                  dst[ir + 1] = std::round(static_cast<float>(int2x4_to_1(srcu8) - zp) * scale);
                  dst[ir + 2] = std::round(static_cast<float>(int2x4_to_2(srcu8) - zp) * scale);
                  dst[ir + 3] = std::round(static_cast<float>(int2x4_to_3(srcu8) - zp) * scale);
                }
              } else {
#pragma unroll
                for (int ir = 0; ir < Unroll; ir += 4) {
                  uint8_t srcu8 = *(bptr + (ik * SgSize + sg_id * Unroll + ir) / 4);
                  dst[ir] = static_cast<float>(int2x4_to_0(srcu8) - zp);
                  dst[ir + 1] = static_cast<float>(int2x4_to_1(srcu8) - zp);
                  dst[ir + 2] = static_cast<float>(int2x4_to_2(srcu8) - zp);
                  dst[ir + 3] = static_cast<float>(int2x4_to_3(srcu8) - zp);
                }
              }
              *(sycl::vec<int8_t, Unroll>*)&dbptr[ik * SgSize + sg_id * Unroll] = *(sycl::vec<int8_t, Unroll>*)dst;
            }
          });
    };
    return q->submit(deq_kernel);
  }

  struct CfgGemvF32 {
    static int constexpr SgSize = 32;
    static int constexpr TileK0 = 8;
    static int constexpr TileK1 = 8;
    static int constexpr TileM = 1;
    static int constexpr Unroll = 4;
    static int constexpr PrefetchDis = 2;
  };

  struct CfgGemvF16 {
    static int constexpr SgSize = 32;
    static int constexpr TileK0 = 16;
    static int constexpr TileK1 = 8;
    static int constexpr TileM = 1;
    static int constexpr Unroll = 4;
    static int constexpr PrefetchDis = 2;
  };

  template <typename Cfg, typename T>
  static sycl::event gemv(const T* A, const Param& paramB, T* C, int n, int k, int blocksize, sycl::queue* q) {
    auto B = paramB.B;
    auto B_scale = paramB.scale;
    auto zp_d = paramB.zp;
    auto bias = (const T*)paramB.bias;
    int ldb = paramB.ldb;
    int constexpr SgSize = Cfg::SgSize;
    int constexpr TileK0 = Cfg::TileK0;
    int constexpr TileK1 = Cfg::TileK1;
    int constexpr TileM = Cfg::TileM;
    int constexpr Unroll = Cfg::Unroll;
    int constexpr PrefetchDis = Cfg::PrefetchDis;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{static_cast<size_t>(n) * SgSize};
    assert(k % TileK1 == 0);
    int constexpr SGK0 = TileK0 * SgSize * Unroll;
    int constexpr SGK1 = TileK1 * SgSize;
    int k_0 = utils::padto_le(k, SGK0);
    auto ev = q->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            int sg_id = sg.get_local_id()[0];
            int g_n = g_idx;
            auto sptr = B_scale + g_n * ldb;
            auto zpptr = zp_d ? zp_d + g_n * ldb : nullptr;
            auto bptr = B + g_n * k / 4;
            auto aptr = A;
            auto cptr = C + g_n;
            T tmpAcc = 0.f;
            int i = 0;
            if (i + SGK0 * PrefetchDis + SGK0 < k)
              for (int j = 0; j < PrefetchDis; j++) {
                sycl::ext::oneapi::experimental::joint_prefetch(sg, bptr + j * SGK0 / 4, SGK0 / 4);
              }

            sycl::group_barrier(sg);
            for (; i < k_0; i += SGK0) {
#pragma unroll
              for (size_t ir = 0; ir < Unroll; ir++) {
                auto tmpu8 = *(sycl::vec<uint8_t, TileK0 / 4>*)(bptr + sg_id * TileK0 / 4);
                T scale = *(sptr + sg_id * TileK0 / blocksize);
                int8_t zp = zpptr ? *(zpptr + sg_id * TileK0 / blocksize) + int8_t(2) : int8_t(2);
                T tmpA[TileK0];
                *(sycl::vec<T, TileK0>*)tmpA = *(sycl::vec<T, TileK0>*)&aptr[sg_id * TileK0];
                T tacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int ikk = 0; ikk < TileK0; ikk += 4) {
                  auto pack = tmpu8[ikk / 4];
                  tacc[0] += tmpA[ikk + 0] * static_cast<T>(int2x4_to_0(pack) - zp);
                  tacc[1] += tmpA[ikk + 1] * static_cast<T>(int2x4_to_1(pack) - zp);
                  tacc[2] += tmpA[ikk + 2] * static_cast<T>(int2x4_to_2(pack) - zp);
                  tacc[3] += tmpA[ikk + 3] * static_cast<T>(int2x4_to_3(pack) - zp);
                }
                tmpAcc += (tacc[0] + tacc[1] + tacc[2] + tacc[3]) * scale;
                sptr += TileK0 * SgSize / blocksize;
                zpptr += zpptr ? TileK0 * SgSize / blocksize : 0;
                aptr += TileK0 * SgSize;
                bptr += TileK0 * SgSize / 4;
              }
              if (i + SGK0 * PrefetchDis + SGK0 < k)
                sycl::ext::oneapi::experimental::joint_prefetch(sg, bptr + SGK0 * PrefetchDis / 4, SGK0 / 4);
              sycl::group_barrier(sg);
            }
            if (i < k) {
              for (; i < k; i += SGK1) {
#pragma unroll
                for (size_t ir = 0; ir < 1; ir++) {
                  if (i + ir * TileK1 * SgSize + sg_id * TileK1 >= k) break;
                  auto tmpu8 = *(sycl::vec<uint8_t, TileK1 / 4>*)(bptr + sg_id * TileK1 / 4);
                  T scale = *(sptr + sg_id * TileK1 / blocksize);
                  int8_t zp = zpptr ? *(zpptr + sg_id * TileK1 / blocksize) + int8_t(2) : int8_t(2);
                  T tmpA[TileK1];
                  *(sycl::vec<T, TileK1>*)tmpA = *(sycl::vec<T, TileK1>*)&aptr[sg_id * TileK1];
                  T tacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                  for (int ikk = 0; ikk < TileK1; ikk += 4) {
                    auto pack = tmpu8[ikk / 4];
                    tacc[0] += tmpA[ikk + 0] * static_cast<T>(int2x4_to_0(pack) - zp);
                    tacc[1] += tmpA[ikk + 1] * static_cast<T>(int2x4_to_1(pack) - zp);
                    tacc[2] += tmpA[ikk + 2] * static_cast<T>(int2x4_to_2(pack) - zp);
                    tacc[3] += tmpA[ikk + 3] * static_cast<T>(int2x4_to_3(pack) - zp);
                  }
                  tmpAcc += (tacc[0] + tacc[1] + tacc[2] + tacc[3]) * scale;
                  sptr += TileK1 * SgSize / blocksize;
                  zpptr += zpptr ? TileK1 * SgSize / blocksize : 0;
                  aptr += TileK1 * SgSize;
                  bptr += TileK1 * SgSize / 4;
                }
                sycl::group_barrier(sg);
              }
            }
            sycl::group_barrier(sg);
            auto sum = sycl::reduce_over_group(sg, tmpAcc, std::plus<>());
            if (bias) sum += bias[g_n];
            if (sg_id == 0) {
              *cptr = sum;
            }
          });
    });
    return ev;
  }
};

template <typename ScaleT>
class WeightS4T {
 public:
  using Param = ParamWeightS4Ext<ScaleT>;

  static __attribute__((always_inline)) inline int8_t int4x2_to_low(const uint8_t& src) {
    return static_cast<int8_t>(src & (uint8_t)0x0f) - (int8_t)8;
  }

  static __attribute__((always_inline)) inline int8_t int4x2_to_high(const uint8_t& src) {
    return static_cast<int8_t>(src >> (uint8_t)4) - (int8_t)8;
  }

  struct CfgDequantF32 {
    static int constexpr SgSize = 32;
    static int constexpr TileK = 16;
    static int constexpr TileN = 1;
    static int constexpr Unroll = 4;
  };

  struct CfgDequantF16 {
    static int constexpr SgSize = 32;
    static int constexpr TileK = 16;
    static int constexpr TileN = 1;
    static int constexpr Unroll = 4;
  };

  template <typename Cfg, typename BType>
  static inline sycl::event dequant(int n, int k, int blocksize, const Param& in, BType* outptr, sycl::queue* q) {
    int constexpr SgSize = Cfg::SgSize;
    int constexpr TileK = Cfg::TileK;
    int constexpr TileN = Cfg::TileN;
    static_assert(TileN == 1);
    int constexpr Unroll = Cfg::Unroll;
    int constexpr SubGroupK = SgSize * TileK;
    int constexpr GroupK = SgSize * TileK;
    assert(blocksize % Unroll == 0);
    int nsg_k = utils::updiv(k, GroupK);
    int nsg_n = n;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{static_cast<size_t>(nsg_n) * nsg_k * SgSize};
    auto B_d = in.B;
    auto S_d = in.scale;
    auto zp_d = in.zp;
    int ldb = in.ldb;
    int ldbn = in.ldb * blocksize;
    auto deq_kernel = [&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            int sg_id = sg.get_local_id()[0];
            int sg_group_id = sg.get_group_id()[0];
            int g_idx_n = g_idx / nsg_k;
            int g_idx_k = g_idx % nsg_k;
            int g_n = g_idx_n * 1;
            int g_k = g_idx_k * GroupK;
            int sg_k = g_k + sg_group_id * SubGroupK;
            int sg_k_remain = k - sg_k;
            auto sptr = S_d + sg_k / blocksize + g_n * ldb;
            auto zpptr = zp_d ? zp_d + sg_k / blocksize + g_n * ldb : nullptr;
            auto bptr = B_d + (sg_k + g_n * ldbn) / 2;
            auto dbptr = outptr + sg_k + g_n * k;
#pragma unroll
            for (int ik = 0; ik < TileK; ik += Unroll) {
              if (ik * SgSize + sg_id * Unroll >= sg_k_remain) break;
              BType dst[Unroll];
              BType scale = sptr[(ik * SgSize + sg_id * Unroll) / blocksize];
              int8_t zp = zpptr ? zpptr[(ik * SgSize + sg_id * Unroll) / blocksize] + int8_t(8) : int8_t(8);
#pragma unroll
              for (int ir = 0; ir < Unroll; ir += 2) {
                uint8_t srcu8 = *(bptr + (ik * SgSize + sg_id * Unroll + ir) / 2);
                dst[ir] = static_cast<BType>(INT4x2_TO_LO(srcu8) - zp) * scale;
                dst[ir + 1] = static_cast<BType>(INT4x2_TO_HI(srcu8) - zp) * scale;
              }
              *(sycl::vec<BType, Unroll>*)&dbptr[ik * SgSize + sg_id * Unroll] = *(sycl::vec<BType, Unroll>*)dst;
            }
          });
    };
    return q->submit(deq_kernel);
  }

  struct CfgDequantS8 {
    static int constexpr SgSize = 32;
    static int constexpr TileK = 16;
    static int constexpr TileN = 1;
    static int constexpr Unroll = 8;
    static bool constexpr Rescale = 0;
  };

  struct CfgDequantS8Rescale {
    static int constexpr SgSize = 32;
    static int constexpr TileK = 32;
    static int constexpr TileN = 1;
    static int constexpr Unroll = 16;
    static bool constexpr Rescale = 1;
  };

  template <class Cfg>
  static inline sycl::event dequantS8(int n, int k, int blocksize, const Param& in, int8_t* outptr, sycl::queue* q,
                                      int newblocksize = -1) {
    int constexpr SgSize = Cfg::SgSize;
    int constexpr TileK = Cfg::TileK;
    int constexpr TileN = Cfg::TileN;
    int constexpr Rescale = Cfg::Rescale;
    static_assert(TileN == 1);
    int constexpr Unroll = Cfg::Unroll;
    int constexpr SubGroupK = SgSize * TileK;
    int constexpr GroupK = SgSize * TileK;
    assert(blocksize % Unroll == 0);
    int nsg_k = utils::updiv(k, GroupK);
    int nsg_n = n;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{static_cast<size_t>(nsg_n) * nsg_k * SgSize};
    auto B_d = in.B;
    auto S_d = in.scale;
    auto Sext_d = in.scaleExt;
    auto zp_d = in.zp;
    int ldb = in.ldb;
    int ldbn = in.ldb * blocksize;
    int newblks = newblocksize == -1 ? 1 : k / newblocksize;
    auto deq_kernel = [&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            int sg_id = sg.get_local_id()[0];
            int sg_group_id = sg.get_group_id()[0];
            int g_idx_n = g_idx / nsg_k;
            int g_idx_k = g_idx % nsg_k;
            int g_n = g_idx_n * 1;
            int g_k = g_idx_k * GroupK;
            int sg_k = g_k + sg_group_id * SubGroupK;
            int sg_k_remain = k - sg_k;
            auto sptr = S_d + sg_k / blocksize + g_n * ldb;
            auto newsptr = Sext_d + sg_k / newblocksize + g_n * newblks;
            auto zpptr = zp_d ? zp_d + sg_k / blocksize + g_n * ldb : nullptr;
            auto bptr = B_d + (sg_k + g_n * ldbn) / 2;
            auto dbptr = outptr + sg_k + g_n * k;

#pragma unroll
            for (int ik = 0; ik < TileK; ik += Unroll) {
              if (ik * SgSize + sg_id * Unroll >= sg_k_remain) break;
              int8_t zp = zpptr ? zpptr[(ik * SgSize + sg_id * Unroll) / blocksize] + int8_t(8) : int8_t(8);
              int8_t dst[Unroll];
              if constexpr (Rescale) {
                float scaleN = (float)newsptr[(ik * SgSize + sg_id * Unroll) / newblocksize];
                float scale = (float)sptr[(ik * SgSize + sg_id * Unroll) / blocksize] / scaleN;
#pragma unroll
                for (int ir = 0; ir < Unroll; ir += 2) {
                  uint8_t srcu8 = *(bptr + (ik * SgSize + sg_id * Unroll + ir) / 2);
                  dst[ir] = std::round(static_cast<float>(INT4x2_TO_LO(srcu8) - zp) * scale);
                  dst[ir + 1] = std::round(static_cast<float>(INT4x2_TO_HI(srcu8) - zp) * scale);
                }
              } else {
#pragma unroll
                for (int ir = 0; ir < Unroll; ir += 2) {
                  uint8_t srcu8 = *(bptr + (ik * SgSize + sg_id * Unroll + ir) / 2);
                  dst[ir] = static_cast<float>(INT4x2_TO_LO(srcu8) - zp);
                  dst[ir + 1] = static_cast<float>(INT4x2_TO_HI(srcu8) - zp);
                }
              }
              *(sycl::vec<int8_t, Unroll>*)&dbptr[ik * SgSize + sg_id * Unroll] = *(sycl::vec<int8_t, Unroll>*)dst;
            }
          });
    };
    return q->submit(deq_kernel);
  }

  struct CfgGemvF32 {
    static int constexpr SgSize = 32;
    static int constexpr TileK0 = 8;
    static int constexpr TileK1 = 8;
    static int constexpr TileM = 1;
    static int constexpr Unroll = 2;
    static int constexpr PrefetchDis = 2;
  };

  struct CfgGemvF16 {
    static int constexpr SgSize = 32;
    static int constexpr TileK0 = 8;
    static int constexpr TileK1 = 8;
    static int constexpr TileM = 1;
    static int constexpr Unroll = 2;
    static int constexpr PrefetchDis = 2;
  };

  template <typename Cfg, typename T>
  static sycl::event gemv(const T* A, const Param& paramB, T* C, int n, int k, int blocksize, sycl::queue* q) {
    auto B = paramB.B;
    auto B_scale = paramB.scale;
    auto zp_d = paramB.zp;
    auto bias = (const T*)paramB.bias;
    int ldb = paramB.ldb;
    int constexpr SgSize = Cfg::SgSize;
    int constexpr TileK0 = Cfg::TileK0;
    int constexpr TileK1 = Cfg::TileK1;
    int constexpr TileM = Cfg::TileM;
    int constexpr Unroll = Cfg::Unroll;
    int constexpr PrefetchDis = Cfg::PrefetchDis;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{static_cast<size_t>(n) * SgSize};
    assert(k % TileK1 == 0);
    int constexpr SGK0 = TileK0 * SgSize * Unroll;
    int constexpr SGK1 = TileK1 * SgSize;
    int k_0 = utils::padto_le(k, SGK0);
    auto ev = q->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            int sg_id = sg.get_local_id()[0];
            int g_n = g_idx;
            auto sptr = B_scale + g_n * ldb;
            auto zpptr = zp_d ? zp_d + g_n * ldb : nullptr;
            auto bptr = B + g_n * k / 2;
            auto aptr = A;
            auto cptr = C + g_n;
            T tmpAcc = 0.f;
            int i = 0;
            if (i + SGK0 * PrefetchDis + SGK0 < k)
              for (int j = 0; j < PrefetchDis; j++) {
                sycl::ext::oneapi::experimental::joint_prefetch(sg, bptr + j * SGK0, SGK0 / 2);
              }

            sycl::group_barrier(sg);
            for (; i < k_0; i += SGK0) {
#pragma unroll
              for (size_t ir = 0; ir < Unroll; ir++) {
                auto tmpu8 = *(sycl::vec<uint8_t, TileK0 / 2>*)(bptr + sg_id * TileK0 / 2);
                T scale = *(sptr + sg_id * TileK0 / blocksize);
                int8_t zp = zpptr ? *(zpptr + sg_id * TileK0 / blocksize) + int8_t(8) : int8_t(8);
                T tmpA[TileK0];
                *(sycl::vec<T, TileK0>*)tmpA = *(sycl::vec<T, TileK0>*)&aptr[sg_id * TileK0];
                T tacc[2] = {0.f, 0.f};
#pragma unroll
                for (int ikk = 0; ikk < TileK0; ikk += 2) {
                  tacc[0] += tmpA[ikk + 0] * static_cast<T>(INT4x2_TO_LO(tmpu8[ikk / 2]) - zp);
                  tacc[1] += tmpA[ikk + 1] * static_cast<T>(INT4x2_TO_HI(tmpu8[ikk / 2]) - zp);
                }
                tmpAcc += (tacc[0] + tacc[1]) * scale;
                sptr += TileK0 * SgSize / blocksize;
                zpptr += zpptr ? TileK0 * SgSize / blocksize : 0;
                aptr += TileK0 * SgSize;
                bptr += TileK0 * SgSize / 2;
              }
              if (i + SGK0 * PrefetchDis + SGK0 < k)
                sycl::ext::oneapi::experimental::joint_prefetch(sg, bptr + i + SGK0 * PrefetchDis, SGK0 / 2);
              sycl::group_barrier(sg);
            }
            if (i < k) {
              for (; i < k; i += SGK1) {
#pragma unroll
                for (size_t ir = 0; ir < 1; ir++) {
                  if (i + ir * TileK1 * SgSize + sg_id * TileK1 >= k) break;
                  auto tmpu8 = *(sycl::vec<uint8_t, TileK1 / 2>*)(bptr + sg_id * TileK1 / 2);
                  T scale = *(sptr + sg_id * TileK1 / blocksize);
                  int8_t zp = zpptr ? *(zpptr + sg_id * TileK1 / blocksize) + int8_t(8) : int8_t(8);
                  T tmpA[TileK1];
                  *(sycl::vec<T, TileK1>*)tmpA = *(sycl::vec<T, TileK1>*)&aptr[sg_id * TileK1];
                  T tacc[2] = {0.f, 0.f};
#pragma unroll
                  for (int ikk = 0; ikk < TileK1; ikk += 2) {
                    tacc[0] += tmpA[ikk + 0] * static_cast<T>(INT4x2_TO_LO(tmpu8[ikk / 2]) - zp);
                    tacc[1] += tmpA[ikk + 1] * static_cast<T>(INT4x2_TO_HI(tmpu8[ikk / 2]) - zp);
                  }
                  tmpAcc += (tacc[0] + tacc[1]) * scale;
                  sptr += TileK1 * SgSize / blocksize;
                  zpptr += zpptr ? TileK0 * SgSize / blocksize : 0;
                  aptr += TileK1 * SgSize;
                  bptr += TileK1 * SgSize / 2;
                }
                sycl::group_barrier(sg);
              }
            }
            sycl::group_barrier(sg);
            auto sum = sycl::reduce_over_group(sg, tmpAcc, std::plus<>());
            if (bias) sum += bias[g_n];
            if (sg_id == 0) {
              *cptr = sum;
            }
          });
    });
    return ev;
  }
};

template <typename ScaleT>
class WeightS8T {
 public:
  using Param = ParamWeightS8<ScaleT>;

  struct CfgDequantF32 {
    static int constexpr SgSize = 16;
    static int constexpr TileK = 8;
    static int constexpr TileN = 1;
    static int constexpr Unroll = 4;
  };

  struct CfgDequantF16 {
    static int constexpr SgSize = 32;
    static int constexpr TileK = 8;
    static int constexpr TileN = 1;
    static int constexpr Unroll = 4;
  };

  template <typename Cfg, typename BType>
  static inline sycl::event dequant(int n, int k, int blocksize, const Param& in, BType* outptr, sycl::queue* q) {
    int constexpr SgSize = Cfg::SgSize;
    int constexpr TileK = Cfg::TileK;
    int constexpr Unroll = Cfg::Unroll;
    int constexpr SubGroupK = SgSize * TileK;
    int constexpr GroupK = SgSize * TileK;
    assert(blocksize % Unroll == 0);

    int nsg_k = utils::updiv(k, GroupK);
    int nsg_n = n;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{static_cast<size_t>(nsg_n) * nsg_k * SgSize};
    auto B_d = in.B;
    auto S_d = in.scale;
    auto zp_d = in.zp;
    int ldb = in.ldb;
    int ldbn = in.ldb * blocksize;
    auto deq_kernel = [&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            int sg_id = sg.get_local_id()[0];
            int sg_group_id = sg.get_group_id()[0];
            int g_idx_n = g_idx / nsg_k;
            int g_idx_k = g_idx % nsg_k;
            int g_n = g_idx_n * 1;
            int g_k = g_idx_k * GroupK;
            int sg_k = g_k + sg_group_id * SubGroupK;
            int sg_k_remain = k - sg_k;
            auto sptr = S_d + sg_k / blocksize + g_n * ldb;
            auto zpptr = zp_d ? zp_d + sg_k / blocksize + g_n * ldb : nullptr;
            auto bptr = B_d + (sg_k + g_n * ldbn);
            auto dbptr = outptr + sg_k + g_n * k;
#pragma unroll
            for (int ik = 0; ik < TileK; ik += Unroll) {
              if (ik * SgSize + sg_id * Unroll >= sg_k_remain) break;
              BType dst[Unroll];
              BType scale = sptr[(ik * SgSize + sg_id * Unroll) / blocksize];
              int8_t zp = zpptr ? zpptr[(ik * SgSize + sg_id * Unroll) / blocksize] : int8_t(0);
              int8_t srcs8[Unroll];
              *(sycl::vec<int8_t, Unroll>*)srcs8 = *(sycl::vec<int8_t, Unroll>*)(bptr + (ik * SgSize + sg_id * Unroll));
#pragma unroll
              for (int ir = 0; ir < Unroll; ir += 1) {
                dst[ir] = (BType)((int16_t)srcs8[ir] - (int16_t)zp) * scale;
              }
              *(sycl::vec<BType, Unroll>*)&dbptr[ik * SgSize + sg_id * Unroll] = *(sycl::vec<BType, Unroll>*)dst;
            }
          });
    };
    return q->submit(deq_kernel);
  }

  struct CfgGemvF32 {
    static int constexpr SgSize = 32;
    static int constexpr TileK0 = 16;
    static int constexpr TileK1 = 4;
    static int constexpr TileM = 1;
  };

  struct CfgGemvF16 {
    static int constexpr SgSize = 32;
    static int constexpr TileK0 = 16;
    static int constexpr TileK1 = 4;
    static int constexpr TileM = 1;
  };

  template <typename Cfg, typename T>
  static sycl::event gemv(const T* A, const Param& paramB, T* C, int n, int k, int blocksize, sycl::queue* q) {
    auto B = paramB.B;
    auto B_scale = paramB.scale;
    auto zp_d = paramB.zp;
    auto bias = (const T*)paramB.bias;
    int ldb = paramB.ldb;
    int constexpr SgSize = Cfg::SgSize;
    int constexpr TileK0 = Cfg::TileK0;
    int constexpr TileK1 = Cfg::TileK1;
    int constexpr TileM = Cfg::TileM;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{static_cast<size_t>(n) * SgSize};
    int constexpr SGK0 = TileK0 * SgSize;
    int constexpr SGK1 = TileK1 * SgSize;
    int k_0 = utils::padto_le(k, SGK0);
    auto ev = q->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<1>(problem, group),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
                         int g_idx = it.get_group(0);
                         auto sg = it.get_sub_group();
                         int sg_id = sg.get_local_id()[0];
                         int g_n = g_idx;
                         auto sptr = B_scale + g_n * ldb;
                         auto zpptr = zp_d ? zp_d + g_n * ldb : nullptr;
                         auto bptr = B + g_n * k;
                         auto aptr = A;
                         auto cptr = C + g_n;
                         T tmpAcc = 0.f;
                         int i = 0;
                         for (; i < k_0; i += SGK0) {
                           auto tmps8 = *(sycl::vec<int8_t, TileK0>*)(bptr + sg_id * TileK0);
                           T scale = *(sptr + sg_id * TileK0 / blocksize);
                           int16_t zp = zpptr ? *(zpptr + sg_id * TileK0 / blocksize) : int16_t(0);
#pragma unroll
                           for (int ikk = 0; ikk < TileK0; ikk += 1) {
                             tmpAcc += T(aptr[sg_id * TileK0 + ikk]) * T((int16_t)tmps8[ikk] - zp) * scale;
                           }
                           sptr += SGK0 / blocksize;
                           zpptr += zpptr ? SGK0 / blocksize : 0;
                           aptr += SGK0;
                           bptr += SGK0;
                         }
                         if (i < k) {
                           for (; i < k; i += SGK1) {
                             if (i + sg_id * TileK1 >= k) break;
                             auto tmps8 = *(sycl::vec<int8_t, TileK1>*)(bptr + sg_id * TileK1);
                             T scale = *(sptr + sg_id * TileK1 / blocksize);
                             int16_t zp = zpptr ? *(zpptr + sg_id * TileK1 / blocksize) : int16_t(0);
#pragma unroll
                             for (int ikk = 0; ikk < TileK1; ikk += 1) {
                               tmpAcc += T(aptr[sg_id * TileK1 + ikk + 0]) * T((int16_t)tmps8[ikk] - zp) * scale;
                             }

                             sptr += SGK1 / blocksize;
                             zpptr += zpptr ? SGK1 / blocksize : 0;
                             aptr += SGK1;
                             bptr += SGK1;
                           }
                         }

                         auto sum = sycl::reduce_over_group(sg, tmpAcc, std::plus<>());
                         if (bias) sum += bias[g_n];
                         if (sg_id == 0) {
                           *cptr = sum;
                         }
                       });
    });
    return ev;
  }
};

// FP8 weight dequantization and GEMM support
// FP8 can be E4M3 or E5M2 format, stored as uint8_t
template <typename ScaleT, bool E4M3 = true>
class WeightF8T {
 public:
  using Param = ParamWeightF8<ScaleT>;
  static constexpr bool IsE4M3 = E4M3;

  static inline float scale_to_f32(const ScaleT& s) {
    if constexpr (std::is_same_v<ScaleT, bestla::utils::f8>) {
      return s.tofloat();
    } else {
      return static_cast<float>(s);
    }
  }

  template <typename ItemT, typename BType>
  static inline void init_slm_lut(const sycl::local_accessor<BType, 1>& slm_lut, const ItemT& it) {
    int local_id = it.get_local_id(0);
    int local_range = it.get_local_range(0);
    const float* lut_data = E4M3 ? fp8_lut::lut_e4m3_128.data() : fp8_lut::lut_e5m2_128.data();
    for (int i = local_id; i < 128; i += local_range) {
      slm_lut[i] = BType(lut_data[i]);
    }
    sycl::group_barrier(it.get_group());
  }

  class Cfg {
   public:
    static int constexpr SgSize = 32;
    static int constexpr TileK = 16;
    static int constexpr TileN = 1;
  };

  template <typename Cfg, typename BType>
  static inline sycl::event dequant(int n, int k, int blocksize, const Param& in, BType* outptr, sycl::queue* q) {
    int constexpr SgSize = Cfg::SgSize;
    int constexpr TileK = Cfg::TileK;
    int constexpr SubGroupK = SgSize * TileK;
    int constexpr GroupK = SgSize * TileK;
    assert(blocksize % TileK == 0);

    int nsg_k = utils::updiv(k, GroupK);
    int nsg_n = n;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{static_cast<size_t>(nsg_n) * nsg_k * SgSize};
    auto B_d = in.B;
    auto S_d = in.scale;
    int ldb = in.ldb;
    int ldbn = in.ldb * blocksize;
    auto deq_kernel = [&](sycl::handler& cgh) {
      sycl::local_accessor<BType, 1> slm_lut(sycl::range<1>(128), cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
            init_slm_lut(slm_lut, it);
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            int sg_id = sg.get_local_id()[0];
            int sg_group_id = sg.get_group_id()[0];
            int g_idx_n = g_idx / nsg_k;
            int g_idx_k = g_idx % nsg_k;
            int g_n = g_idx_n * 1;
            int g_k = g_idx_k * GroupK;
            int sg_k = g_k + sg_group_id * SubGroupK;
            int sg_k_remain = k - sg_k;
            if (sg_k_remain >= SubGroupK) {
              auto sptr = S_d + sg_k / blocksize + g_n * ldb;
              auto bptr = B_d + (sg_k + g_n * ldbn);
              auto dbptr = outptr + sg_k + g_n * k;
              int constexpr Unroll = 4;
#pragma unroll
              for (int ik = 0; ik < TileK; ik += Unroll) {
                BType dst[Unroll];
                BType scale = static_cast<BType>(scale_to_f32(sptr[(ik * SgSize + sg_id * Unroll) / blocksize]));
#pragma unroll
                for (int ir = 0; ir < Unroll; ir += 1) {
                  uint8_t srcfp8 = *(bptr + (ik * SgSize + sg_id * Unroll + ir));
                  BType val = slm_lut[srcfp8 & 0x7F];
                  if (srcfp8 & 0x80) val = -val;
                  dst[ir] = val * scale;
                }
                *(sycl::vec<BType, Unroll>*)&dbptr[ik * SgSize + sg_id * Unroll] = *(sycl::vec<BType, Unroll>*)dst;
              }
            } else {
              auto sptr = S_d + sg_k / blocksize + g_n * ldb;
              auto bptr = B_d + (sg_k + g_n * ldbn);
              auto dbptr = outptr + sg_k + g_n * k;
              int constexpr Unroll = 4;
              for (int ik = 0; ik < TileK; ik += Unroll) {
                if (ik * SgSize + sg_id * Unroll >= sg_k_remain) break;
                BType dst[Unroll];
                BType scale = static_cast<BType>(scale_to_f32(sptr[(ik * SgSize + sg_id * Unroll) / blocksize]));
#pragma unroll
                for (int ir = 0; ir < Unroll; ir += 1) {
                  uint8_t srcfp8 = *(bptr + (ik * SgSize + sg_id * Unroll + ir));
                  BType val = slm_lut[srcfp8 & 0x7F];
                  if (srcfp8 & 0x80) val = -val;
                  dst[ir] = val * scale;
                }
                *(sycl::vec<BType, Unroll>*)&dbptr[ik * SgSize + sg_id * Unroll] = *(sycl::vec<BType, Unroll>*)dst;
              }
            }
          });
    };
    return q->submit(deq_kernel);
  }

  template <typename T>
  static inline sycl::event gemv(const T* A, const Param& paramB, T* C, int n, int k, int blocksize, sycl::queue* q) {
    auto B = paramB.B;
    auto B_scale = paramB.scale;
    int ldb = paramB.ldb;
    const T* bias_ptr = reinterpret_cast<const T*>(paramB.bias);
    int constexpr SgSize = 32;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{static_cast<size_t>(n) * SgSize};

    // Configuration based on data type
    constexpr bool is_fp32 = std::is_same_v<T, float>;
    int constexpr TileK0 = is_fp32 ? 8 : 16;  // Reduce tile size for FP32
    int constexpr TileK1 = 8;
    // FP32: Use 8 accumulators (match TileK0) and 4x unroll for max parallelism
    // FP16: Use 4 accumulators and 2x unroll (balanced)
    int constexpr Unroll0 = is_fp32 ? 4 : 2;
    int constexpr NumAcc = is_fp32 ? 8 : 4;

    int constexpr SGK0 = TileK0 * SgSize;
    int constexpr SGK1 = TileK1 * SgSize;
    int k_0 = utils::padto_le(k, SGK0 * Unroll0);
    int k_1 = utils::padto_le(k, SGK1);
    auto ev = q->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<T, 1> slm_lut(sycl::range<1>(128), cgh);
      cgh.parallel_for(sycl::nd_range<1>(problem, group),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
                         init_slm_lut(slm_lut, it);
                         int g_idx = it.get_group(0);
                         auto sg = it.get_sub_group();
                         int sg_id = sg.get_local_id()[0];
                         int g_n = g_idx;
                         auto sptr = B_scale + g_n * ldb;
                         auto bptr = B + g_n * k;
                         auto aptr = A;
                         auto cptr = C + g_n;

                         // Multiple accumulators
                         T tmpAcc[NumAcc];
#pragma unroll
                         for (int i = 0; i < NumAcc; ++i) tmpAcc[i] = 0.f;

                         int i = 0;
                         // Main loop with 2x unrolling + vectorized LUT lookup
                         for (; i < k_0; i += SGK0 * Unroll0) {
#pragma unroll
                           for (int iu = 0; iu < Unroll0; iu++) {
                             auto tmpfp8 = *(sycl::vec<uint8_t, TileK0>*)(bptr + sg_id * TileK0);
                             T scale = static_cast<T>(scale_to_f32(*(sptr + sg_id * TileK0 / blocksize)));
                             auto tmpA = *(sycl::vec<T, TileK0>*)&aptr[sg_id * TileK0];

                             // Vectorized LUT lookup + fused scale multiply
                             T tmpB[TileK0];
#pragma unroll
                             for (int ikk = 0; ikk < TileK0; ikk += 1) {
                               uint8_t u8 = tmpfp8[ikk];
                               T val = slm_lut[u8 & 0x7F];
                               if (u8 & 0x80) val = -val;
                               tmpB[ikk] = val * scale;
                             }

            // Fused multiply-accumulate with multiple accumulators
#pragma unroll
                             for (int ikk = 0; ikk < TileK0; ikk += NumAcc) {
#pragma unroll
                               for (int ia = 0; ia < NumAcc; ++ia) {
                                 tmpAcc[ia] += T(tmpA[ikk + ia]) * tmpB[ikk + ia];
                               }
                             }
                             sptr += SGK0 / blocksize;
                             aptr += SGK0;
                             bptr += SGK0;
                           }
                         }
                         // Secondary loop with larger TileK1
                         for (; i < k_1; i += SGK1) {
                           auto tmpfp8 = *(sycl::vec<uint8_t, TileK1>*)(bptr + sg_id * TileK1);
                           T scale = static_cast<T>(scale_to_f32(*(sptr + sg_id * TileK1 / blocksize)));
                           auto tmpA = *(sycl::vec<T, TileK1>*)&aptr[sg_id * TileK1];

                           // Vectorized LUT lookup + fused scale multiply
                           T tmpB[TileK1];
#pragma unroll
                           for (int ikk = 0; ikk < TileK1; ikk += 1) {
                             uint8_t u8 = tmpfp8[ikk];
                             T val = slm_lut[u8 & 0x7F];
                             if (u8 & 0x80) val = -val;
                             tmpB[ikk] = val * scale;
                           }

          // Fused multiply-accumulate
#pragma unroll
                           for (int ikk = 0; ikk < TileK1; ikk += 1) {
                             tmpAcc[ikk % NumAcc] += T(tmpA[ikk]) * tmpB[ikk];
                           }

                           sptr += SGK1 / blocksize;
                           aptr += SGK1;
                           bptr += SGK1;
                         }

                         // Combine accumulators
                         T finalAcc = 0.f;
#pragma unroll
                         for (int i = 0; i < NumAcc; ++i) finalAcc += tmpAcc[i];

                         float sum = sycl::reduce_over_group(sg, static_cast<float>(finalAcc), sycl::plus<float>());
                         if (sg_id == 0) {
                           if (bias_ptr) {
                             sum += static_cast<float>(bias_ptr[g_n]);
                           }
                           *cptr = static_cast<T>(sum);
                         }
                       });
    });
    return ev;
  }
};

template <class PrologueT>
struct needs_fp8_lut : std::false_type {};

template <typename ScaleT, bool Flag>
struct needs_fp8_lut<WeightF8T<ScaleT, Flag>> : std::true_type {};
}  // namespace sycl_prologue_b
}  // namespace bestla
#endif
