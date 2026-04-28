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

namespace bestla {
namespace sycl_prologue_a {

template <typename SrcT>
struct ParamActivationBase {
  const SrcT* A;
  int lda;
};

template <typename T>
class ActivationBase {
 public:
  using Param = ParamActivationBase<T>;

  struct CfgQuantF32 {
    static int constexpr SgSize = 32;
    static int constexpr Unroll = 4;
    static int constexpr MaxK = 32;
    static int constexpr WG_Size = 1024;
  };

  struct CfgQuantF16 {
    static int constexpr SgSize = 32;
    static int constexpr TileK = 8;
    static int constexpr TileN = 1;
    static int constexpr Unroll = 4;
  };

  template <typename Cfg>
  static inline sycl::event quant_s8(int m, int k, int mask, const Param& in, int8_t* outptr, T* scaleptr,
                                     sycl::queue* q) {
    int constexpr SgSize = Cfg::SgSize;
    int constexpr Unroll = Cfg::Unroll;
    int constexpr MaxK = Cfg::MaxK;
    int constexpr WG_Size = Cfg::WG_Size;
    int constexpr SG_Num = WG_Size / SgSize;
    assert(k <= WG_Size * MaxK);

    int nsg_m = m;
    sycl::range<1> group{WG_Size};
    sycl::range<1> problem{static_cast<size_t>(nsg_m) * WG_Size};
    auto A_d = in.A;
    int lda = in.lda;
    auto ker = [&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> slm_maxabs(sycl::range(SG_Num), cgh);
      cgh.parallel_for(
          sycl::and_range<1>(problem, group), [=](sycl::and_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            auto wg_idx = it.get_local_id();
            int sg_id = sg.get_local_id()[0];
            int sg_group_id = sg.get_group_id()[0];
            int g_m = g_idx;
            auto aptr = A_d + g_m * lda;
            auto qptr = outptr + g_m * lda;
            float maxabs = float(0);
            T srcs[MaxK];
            auto pS8 = syclex::annotated_ptr{
                outptr, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                            syclintelex::cache_mode::write_back, syclex::cache_level::L1, syclex::cache_level::L3>>}};
#pragma unroll
            for (int ik = 0; ik < MaxK; ik += Unroll) {
              if (wg_idx * Unroll + ik * WG_Size >= k) break;
              *(sycl::vec<T, Unroll>*)&srcs[ik] = *(sycl::vec<T, Unroll>*)(aptr + wg_idx * Unroll + ik * WG_Size);
#pragma unroll
              for (int ir = 0; ir < Unroll; ir += 1) {
                maxabs = sycl::max(maxabs, std::abs((float)srcs[ik + ir]));
              }
            }
            maxabs = sycl::reduce_over_group(sg, maxabs, sycl::maximum<float>());
            if (sg_id == 0) {
              slm_maxabs[sg_group_id] = maxabs;
            }
            it.barrier(sycl::access::fence_space::local_space);
            float g_maxabs = maxabs;
#pragma unroll
            for (size_t i = 0; i < SG_Num; i++) {
              g_maxabs = sycl::max(g_maxabs, slm_maxabs[i]);
            }
            T scale = g_maxabs / 127.f;
            float ratio = 1 / scale;
#pragma unroll
            for (int ik = 0; ik < MaxK; ik += Unroll) {
              if (wg_idx * Unroll + ik * WG_Size >= k) break;
              int8_t dst[Unroll];
#pragma unroll
              for (int ir = 0; ir < Unroll; ir += 1) {
                auto tmp = sycl::round((float)srcs[ik + ir] * ratio);
                dst[ir] = static_cast<int8_t>(tmp);
                dst[ir] = sycl::min(dst[ir], int8_t(127));
                dst[ir] = sycl::max(dst[ir], int8_t(-128));
              }
              *(sycl::vec<int8_t, Unroll>*)(pS8.get() + g_m * lda + wg_idx * Unroll + ik * WG_Size) =
                  *(sycl::vec<int8_t, Unroll>*)dst;
            }
            if (wg_idx == 0) scaleptr[g_m] = scale;
          });
    };
    auto ev = q->submit(ker);
    if (mask == 0) return ev;
    auto ker2 = [&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> slm_maxabs(sycl::range(SG_Num), cgh);
      cgh.parallel_for(
          sycl::and_range<1>(problem, group), [=](sycl::and_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            auto wg_idx = it.get_local_id();
            int sg_id = sg.get_local_id()[0];
            int sg_group_id = sg.get_group_id()[0];
            int g_m = g_idx;
            auto aptr = outptr + g_m * lda;
            auto qptr = outptr + g_m * lda;
            float maxabs = float(0);
            for (int i = wg_idx; i < m; i += WG_Size) {
              maxabs = std::max(maxabs, (float)scaleptr[i]);
            }
            maxabs = sycl::reduce_over_group(sg, maxabs, sycl::maximum<float>());
            if (sg_id == 0) {
              slm_maxabs[sg_group_id] = maxabs;
            }
            it.barrier(sycl::access::fence_space::local_space);
            float g_maxabs = maxabs;
#pragma unroll
            for (size_t i = 0; i < SG_Num; i++) {
              g_maxabs = sycl::max(g_maxabs, slm_maxabs[i]);
            }
            float ratio = scaleptr[g_m] / g_maxabs;
            int8_t srcs[Unroll];
            float tmps[Unroll];
#pragma unroll
            for (int ik = 0; ik < MaxK; ik += Unroll) {
              if (wg_idx * Unroll + ik * WG_Size >= k) break;
              *(sycl::vec<int8_t, Unroll>*)srcs = *(sycl::vec<int8_t, Unroll>*)(aptr + wg_idx * Unroll + ik * WG_Size);
#pragma unroll
              for (int ir = 0; ir < Unroll; ir += 1) {
                tmps[ir] = sycl::round(float(srcs[ir]) * ratio);
                srcs[ir] = static_cast<int8_t>(tmps[ir]);
                srcs[ir] = sycl::min(srcs[ir], int8_t(127));
                srcs[ir] = sycl::max(srcs[ir], int8_t(-128));
              }
              *(sycl::vec<int8_t, Unroll>*)(qptr + wg_idx * Unroll + ik * WG_Size) = *(sycl::vec<int8_t, Unroll>*)srcs;
            }
            if (wg_idx == 0) scaleptr[g_m] = g_maxabs;
          });
    };
    return q->submit(ker2);
  }
};

}  // namespace sycl_prologue_a
}  // namespace bestla
#endif
