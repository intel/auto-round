/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include "xe_sagev1_fwd_mainloop.hpp"

namespace cutlass::fmha::collective {

template <class DispatchPolicy_, bool CausalMask_, bool FullMask_, bool CachedKV_, bool PagedKV_, bool UseInt8PV_,
          bool WriteBackInt8PV_, bool ExecuteInt8PV_, class TiledMMAQK_, class TiledMMAPV_, int VTiles_,
          class TensorQ_, class TensorK_, class TensorV_, class TensorK_cache_, class TensorV_cache_,
          class TiledCopyQ_ = void, class TiledCopyK_ = void, class TiledCopyV_ = void,
          class TiledCopyK_cache_ = void, class TiledCopyV_cache_ = void>
struct SPARSESAGEV1FwdMainloop
    : public SAGEV1FwdMainloop<DispatchPolicy_, CausalMask_, FullMask_, CachedKV_, PagedKV_, UseInt8PV_,
                               WriteBackInt8PV_, ExecuteInt8PV_, TiledMMAQK_, TiledMMAPV_, VTiles_, TensorQ_,
                               TensorK_, TensorV_, TensorK_cache_, TensorV_cache_, TiledCopyQ_, TiledCopyK_,
                               TiledCopyV_, TiledCopyK_cache_, TiledCopyV_cache_> {
  using Base = SAGEV1FwdMainloop<DispatchPolicy_, CausalMask_, FullMask_, CachedKV_, PagedKV_, UseInt8PV_,
                                 WriteBackInt8PV_, ExecuteInt8PV_, TiledMMAQK_, TiledMMAPV_, VTiles_, TensorQ_,
                                 TensorK_, TensorV_, TensorK_cache_, TensorV_cache_, TiledCopyQ_, TiledCopyK_,
                                 TiledCopyV_, TiledCopyK_cache_, TiledCopyV_cache_>;

  using Base::Base;
  using typename Base::Arguments;
  using typename Base::Params;
  using typename Base::SharedStorage;

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const& args) { return Base::can_implement(args); }

  static constexpr Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return Base::to_underlying_arguments(args, workspace);
  }
};

}  // namespace cutlass::fmha::collective
