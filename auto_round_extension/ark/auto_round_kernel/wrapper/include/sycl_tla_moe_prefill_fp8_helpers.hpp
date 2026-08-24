#pragma once

#include "sycl_tla_common.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_prefill_fp8_detail {

bool enabled();
bool shape_ok(int N, int K, int group_size);
void dispatch_f16_e4m3_small(const MoePrefillParams& params);
void dispatch_f16_e4m3_mid(const MoePrefillParams& params);
void dispatch_f16_e4m3_large(const MoePrefillParams& params);
void dispatch_f16_e5m2_small(const MoePrefillParams& params);
void dispatch_f16_e5m2_mid(const MoePrefillParams& params);
void dispatch_f16_e5m2_large(const MoePrefillParams& params);
void dispatch_bf16_e4m3_small(const MoePrefillParams& params);
void dispatch_bf16_e4m3_mid(const MoePrefillParams& params);
void dispatch_bf16_e4m3_large(const MoePrefillParams& params);
void dispatch_bf16_e5m2_small(const MoePrefillParams& params);
void dispatch_bf16_e5m2_mid(const MoePrefillParams& params);
void dispatch_bf16_e5m2_large(const MoePrefillParams& params);

}  // namespace moe_prefill_fp8_detail
}  // namespace ark

#endif