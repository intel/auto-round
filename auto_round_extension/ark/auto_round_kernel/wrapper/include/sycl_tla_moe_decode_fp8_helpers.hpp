#pragma once

#include "sycl_tla_common.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_decode_fp8_detail {

// Declaration-only view of the FP8 scalar-GEMV decode kernels, mirroring
// `sycl_tla_moe_prefill_fp8_helpers.hpp`. Each `dispatch_*` is defined in its
// own generated translation unit (`sycl_tla_moe_decode_fp8_<dtype>_<format>_
// <mode>.cpp`) and instantiates exactly four SYCL kernels: the plain
// `MoEDecodeKernelFP8` plus the three `MoEDecodeKernelFP8KSplit` column
// factors. Expanding all of them inside `sycl_tla_moe_decode_fp8.cpp` instead
// means 2 dtypes x 2 formats x 3 decode modes x 4 kernels = 48 kernels in one
// TU, which is what pushed that file's peak compiler RSS into the tens of GiB.
//
// The mode is chosen on the host by `moe_dequant::fp8_decode_mode()`
// (`ARK_FP8_DECODE_MODE`); every variant is numerically equivalent, so the
// split is invisible to callers. The K-split vs legacy choice stays inside
// each TU, exactly as `launch_fp8_by_mode` did.
void dispatch_f16_e4m3_word(const MoeDecodeParams& params);
void dispatch_f16_e4m3_lut(const MoeDecodeParams& params);
void dispatch_f16_e4m3_bits(const MoeDecodeParams& params);
void dispatch_f16_e5m2_word(const MoeDecodeParams& params);
void dispatch_f16_e5m2_lut(const MoeDecodeParams& params);
void dispatch_f16_e5m2_bits(const MoeDecodeParams& params);
void dispatch_bf16_e4m3_word(const MoeDecodeParams& params);
void dispatch_bf16_e4m3_lut(const MoeDecodeParams& params);
void dispatch_bf16_e4m3_bits(const MoeDecodeParams& params);
void dispatch_bf16_e5m2_word(const MoeDecodeParams& params);
void dispatch_bf16_e5m2_lut(const MoeDecodeParams& params);
void dispatch_bf16_e5m2_bits(const MoeDecodeParams& params);

}  // namespace moe_decode_fp8_detail
}  // namespace ark

#endif
