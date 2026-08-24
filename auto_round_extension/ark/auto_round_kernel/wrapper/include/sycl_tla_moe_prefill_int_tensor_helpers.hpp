#pragma once

#include "sycl_tla_common.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_prefill_int_tensor_detail {

void dispatch_f16_small(const MoePrefillParams& params);
void dispatch_f16_mid(const MoePrefillParams& params);
void dispatch_f16_large(const MoePrefillParams& params);
void dispatch_bf16_small(const MoePrefillParams& params);
void dispatch_bf16_mid(const MoePrefillParams& params);
void dispatch_bf16_large(const MoePrefillParams& params);

}  // namespace moe_prefill_int_tensor_detail
}  // namespace ark

#endif