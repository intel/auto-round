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

#include "sycl_utils.h"

namespace bestla {
namespace sycl_epilogue {
template <typename DstT>
struct ParamOutputBase {
  DstT* C;
  int ldc;
};
template <class GemmCoreT, typename DstT>
class OutputBase {
 public:
  using CType = typename GemmCoreT::TACC;
  using DstType = DstT;
  using Param = ParamOutputBase<DstType>;
  static void store(const Param& _param, CType* tmpAcc, const sycl_utils::nd_item_helper<GemmCoreT>& helper) {
#pragma unroll
    for (int im = 0; im < GemmCoreT::TileM; im++) {
      *(sycl::vec<CType, GemmCoreT::TileN>*)&_param.C[(helper.item_g_m() + im) * _param.ldc + helper.item_g_n()] =
          *(sycl::vec<CType, GemmCoreT::TileN>*)&tmpAcc[(im)*GemmCoreT::TileN];
    }
  }

  static void store_tail(const Param& _param, CType* tmpAcc, const sycl_utils::nd_item_helper<GemmCoreT>& helper,
                         int m_tail) {
    if (m_tail) {
#pragma unroll(4)
      for (int im = 0; im < m_tail; im += 1) {
        *(sycl::vec<CType, GemmCoreT::TileN>*)&_param.C[(helper.item_g_m() + im) * _param.ldc + helper.item_g_n()] =
            *(sycl::vec<CType, GemmCoreT::TileN>*)&tmpAcc[(im)*GemmCoreT::TileN];
      }
    }
  }
};

}  // namespace sycl_epilogue
}  // namespace bestla
#endif
