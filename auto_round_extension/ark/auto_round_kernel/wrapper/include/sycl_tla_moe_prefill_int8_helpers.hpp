#pragma once

#include "sycl_tla_common.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_prefill_int8_detail {

bool enabled();
bool shape_ok(int N, int K, int group_size);
void dispatch_f16_small(const MoePrefillParams& params);
void dispatch_f16_mid(const MoePrefillParams& params);
void dispatch_f16_large(const MoePrefillParams& params);
void dispatch_bf16_small(const MoePrefillParams& params);
void dispatch_bf16_mid(const MoePrefillParams& params);
void dispatch_bf16_large(const MoePrefillParams& params);
void dispatch_lowbit_f16_small(const MoePrefillParams& params);
void dispatch_lowbit_f16_mid(const MoePrefillParams& params);
void dispatch_lowbit_f16_large(const MoePrefillParams& params);
void dispatch_lowbit_bf16_small(const MoePrefillParams& params);
void dispatch_lowbit_bf16_mid(const MoePrefillParams& params);
void dispatch_lowbit_bf16_large(const MoePrefillParams& params);

}  // namespace moe_prefill_int8_detail
}  // namespace ark

#endif