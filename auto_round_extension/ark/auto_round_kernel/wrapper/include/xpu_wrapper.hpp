//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include "utils.hpp"
#include "dnnl_wrapper.hpp"
#include "sycl_tla_common.hpp"

namespace ark {

class XpuWrapper {
 public:
  static inline size_t get_weight_qbits(QuantParam* p) { return bestla::utils::bestla_dtype_bits(p->weight_type); }

  static inline size_t get_weight_qbytes(QuantParam* p) { return get_weight_qbits(p) * p->k * p->n / 8; }

  static inline float get_weight_fullrange(QuantParam* p) { return float(1u << (get_weight_qbits(p) - 1)); }

  static inline size_t get_packw_qsize(QuantParam* p) {
    size_t size = 0;
    if (p->weight_type == BTLA_DTYPE::S2 || p->weight_type == BTLA_DTYPE::S4) {
      size += get_weight_qbytes(p);
    }
    if (p->weight_type == BTLA_DTYPE::S8) {
      size += p->k * p->n;
    }
    if (p->weight_type == BTLA_DTYPE::F8_E4M3 || p->weight_type == BTLA_DTYPE::F8_E5M2) {
      size += p->k * p->n;  // FP8 is stored as uint8_t, same size as int8
    }
    return size;
  }

  static inline size_t get_scale_size_bytes(QuantParam* p) { return bestla::utils::bestla_dtype_bytes(p->scale_type); }

  static inline size_t get_scale_size(QuantParam* p) {
    using namespace bestla::utils;
    return p->blks() * p->n * bestla_dtype_bytes(p->scale_type);
  }

  static inline size_t get_zp_size(QuantParam* p) {
    using namespace bestla::utils;
    if (!p->asym) return 0;
    return p->blks() * p->n;
  }

  static inline size_t get_scalext_size(QuantParam* p) {
    using namespace bestla::utils;
    if (env_params::Instance()->auto_s8 == 0) return 0;
    size_t nblk = 1;
    if (env_params::Instance()->auto_s8 != -1) {
      nblk *= p->k / env_params::Instance()->auto_s8;
    }
    return nblk * p->n * bestla_dtype_bytes(p->scale_type);
  }

  static inline size_t get_packw_size(QuantParam* p) {
    size_t size = get_packw_qsize(p);
    size += get_scale_size(p);
    size += get_zp_size(p);
    size += get_scalext_size(p);
    return size;
  }

  static inline size_t get_scale_offset(QuantParam* p) {
    size_t size = get_packw_qsize(p);
    return size;
  }

  static inline size_t get_zp_offset(QuantParam* p) {
    size_t size = get_scale_offset(p);
    size += get_scale_size(p);
    return size;
  }

  static inline size_t get_scalext_offset(QuantParam* p) {
    size_t size = get_zp_offset(p);
    size += get_zp_size(p);
    return size;
  }

  static bool can_comps8(QuantParam* p) {
    if (p->asym && p->weight_type == BTLA_DTYPE::S8) return false;
    if (p->blocksize == -1 || p->blocksize == p->k) return true;
    if (rescale(p)) return true;
    if (p->blocksize % 64 == 0) return true;
    return false;
  }

  static inline void check_compute_type(QuantParam* p) {
    using namespace bestla;
    switch (p->compute_type) {
      case BTLA_DTYPE::EleBitsUndef:  // from low precision to high precision
        [[fallthrough]];
      case BTLA_DTYPE::S8:
        // TODO(Yu): add Xe Arch condition
        if (can_comps8(p)) {
          p->compute_type = BTLA_DTYPE::S8;
          return;
        }
        [[fallthrough]];
      case BTLA_DTYPE::F16:
        p->compute_type = BTLA_DTYPE::F16;
        return;
        [[fallthrough]];
      case BTLA_DTYPE::F32:
      default:
        p->compute_type = BTLA_DTYPE::F32;
        return;
    }
  }

  static void packq_int8(int8_t* raws8, int8_t* blob, QuantParam* p, sycl::queue* q) {
    size_t k = p->k;
    size_t n = p->n;
    auto psrc = raws8;
    constexpr int SG_SIZE = 16;
    auto ker = [&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<2>({k, n}, {1, SG_SIZE}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]]
                       {
                         auto g_0 = it.get_global_id()[0];
                         auto g_1 = it.get_global_id()[1];
                         int8_t src0 = psrc[g_0 * n + g_1];
                         *((int8_t*)blob + g_1 * k + g_0) = src0;
                       });
    };
    q->submit(ker);
  }

  static void packq_int4(int8_t* raws8, int8_t* blob, QuantParam* p, sycl::queue* q) {
    size_t k = p->k;
    size_t n = p->n;
    auto psrc = raws8;
    constexpr int SG_SIZE = 16;
    auto ker = [&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<2>({k / 2, n}, {1, SG_SIZE}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]]
                       {
                         auto g_0 = it.get_global_id()[0];
                         auto g_1 = it.get_global_id()[1];
                         int8_t src0 = psrc[(g_0 * 2 + 0) * n + g_1] + 8;
                         int8_t src1 = psrc[(g_0 * 2 + 1) * n + g_1] + 8;
                         uint8_t dst = src0 | (src1 << 4);
                         *((uint8_t*)blob + g_1 * k / 2 + g_0) = dst;
                       });
    };
    q->submit(ker);
  }

  static void packq_int2(int8_t* raws8, int8_t* blob, QuantParam* p, sycl::queue* q) {
    size_t k = p->k;
    size_t n = p->n;
    auto psrc = raws8;
    constexpr int SG_SIZE = 16;
    auto ker = [&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<2>({k / 4, n}, {1, SG_SIZE}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]]
                       {
                         auto g_0 = it.get_global_id()[0];
                         auto g_1 = it.get_global_id()[1];
                         uint8_t src0 = static_cast<uint8_t>(psrc[(g_0 * 4 + 0) * n + g_1] + 2);
                         uint8_t src1 = static_cast<uint8_t>(psrc[(g_0 * 4 + 1) * n + g_1] + 2);
                         uint8_t src2 = static_cast<uint8_t>(psrc[(g_0 * 4 + 2) * n + g_1] + 2);
                         uint8_t src3 = static_cast<uint8_t>(psrc[(g_0 * 4 + 3) * n + g_1] + 2);
                         uint8_t dst = src0 | (src1 << 2) | (src2 << 4) | (src3 << 6);
                         *((uint8_t*)blob + g_1 * k / 4 + g_0) = dst;
                       });
    };
    q->submit(ker);
  }

  static void packscale(void* scaleptr, int8_t* blobptr, QuantParam* p, sycl::queue* q) {
    size_t k = p->k;
    size_t n = p->n;
    int block = p->blocksize;
    size_t blks = p->blks();
    auto ssize = get_scale_size_bytes(p);
    constexpr int SG_SIZE = 16;
    auto dstptr = blobptr + get_scale_offset(p);
    if (p->scale_type == BTLA_DTYPE::F8_E8M0) {
      auto psrc_f32 = (const float*)scaleptr;
      auto pdst_s8 = (int8_t*)dstptr;
      auto ker = [&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<2>({blks, n}, {1, SG_SIZE}),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                           auto g_0 = it.get_global_id()[0];
                           auto g_1 = it.get_global_id()[1];
                           float v = psrc_f32[g_0 * n + g_1];
                           int exp = static_cast<int>(sycl::rint(v));
                           exp = (exp > 127) ? 127 : exp;
                           exp = (exp < -127) ? -127 : exp;
                           pdst_s8[g_1 * blks + g_0] = static_cast<int8_t>(exp);
                         });
      };
      q->submit(ker);
      return;
    }
    auto psrc = (int8_t*)scaleptr;
    auto ker = [&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<2>({blks, n}, {1, SG_SIZE}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]]
                       {
                         auto g_0 = it.get_global_id()[0];
                         auto g_1 = it.get_global_id()[1];
                         int8_t tmp[4];
                         for (size_t i = 0; i < ssize; i++) {
                           tmp[i] = psrc[g_0 * n * ssize + g_1 * ssize + i];
                         }
                         for (size_t i = 0; i < ssize; i++) {
                           *((int8_t*)dstptr + g_0 * ssize + g_1 * blks * ssize + i) = tmp[i];
                         }
                       });
    };
    q->submit(ker);
    if (rescale(p)) {
#ifdef ARK_RESCALE
      auto scalext_ptr = (int8_t*)blobptr + get_scalext_offset(p);
      int newblock_size = rescale_blocksize(p);
      auto newblks = p->k / newblock_size;
      auto st = p->scale_type;
      if (st == BTLA_DTYPE::F32) {
        using T = float;
        auto ker2 = [&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::nd_range<2>({1, n}, {1, SG_SIZE}),
                           [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]]
                           {
                             auto g_1 = it.get_global_id()[1];
                             auto sg = it.get_sub_group();
                             auto sptr = (T*)scaleptr;
                             auto sxtptr = (T*)scalext_ptr;
                             for (int j = 0; j < newblks; j++) {
                               float maxv = 0.f;
                               int start_blk = j * newblock_size / block;
                               int end_blk = (j + 1) * newblock_size / block;
                               for (int i = start_blk; i < end_blk; i += 1) {
                                 maxv = std::max(abs(sptr[i * n + g_1]), maxv);
                               }
                               sxtptr[g_1 * newblks + j] = maxv * get_weight_fullrange(p) / 127.f;
                             }
                           });
        };
        q->submit(ker2);
      }
      if (st == BTLA_DTYPE::F16) {
        using T = sycl::half;
        auto ker2 = [&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::nd_range<2>({1, n}, {1, SG_SIZE}),
                           [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]]
                           {
                             auto g_1 = it.get_global_id()[1];
                             auto sg = it.get_sub_group();
                             auto sptr = (T*)scaleptr;
                             auto sxtptr = (T*)scalext_ptr;
                             for (int j = 0; j < newblks; j++) {
                               float maxv = 0.f;
                               int start_blk = j * newblock_size / block;
                               int end_blk = (j + 1) * newblock_size / block;
                               for (int i = start_blk; i < end_blk; i += 1) {
                                 maxv = std::max(abs((float)sptr[i * n + g_1]), maxv);
                               }
                               sxtptr[g_1 * newblks + j] = maxv * get_weight_fullrange(p) / 127.f;
                             }
                           });
        };
        q->submit(ker2);
      }
#endif
    }
  }

  static void packzp(int8_t* zpptr, int8_t* blob, QuantParam* p, sycl::queue* q) {
    if (zpptr == nullptr || !p->asym) return;
    size_t k = p->k;
    size_t n = p->n;
    int block = p->blocksize;
    size_t blks = p->blks();
    auto dstptr = (int8_t*)blob + get_zp_offset(p);
    constexpr int SG_SIZE = 16;
    auto ker = [&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<2>({blks, n}, {1, SG_SIZE}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]]
                       {
                         auto g_0 = it.get_global_id()[0];
                         auto g_1 = it.get_global_id()[1];
                         *(dstptr + g_0 + g_1 * blks) = zpptr[g_0 * n + g_1];
                       });
    };
    q->submit(ker);
  }

  static void packq_fp8(int8_t* raws8, int8_t* blob, QuantParam* p, sycl::queue* q) {
    size_t k = p->k;
    size_t n = p->n;
    auto psrc = (uint8_t*)raws8;
    constexpr int SG_SIZE = 16;
    auto ker = [&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<2>({k, n}, {1, SG_SIZE}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]]
                       {
                         auto g_0 = it.get_global_id()[0];
                         auto g_1 = it.get_global_id()[1];
                         uint8_t src0 = psrc[g_0 * n + g_1];
                         *((uint8_t*)blob + g_1 * k + g_0) = src0;
                       });
    };
    q->submit(ker);
  }

  static void packq(int8_t* raws8, void* scaleptr, int8_t* zpptr, int8_t* blob, QuantParam* p, sycl::queue* q) {
    auto type = bestla::utils::bestla_dtype_type(p->weight_type);
    auto typeint = bestla::utils::bestla_dtype_type(BTLA_DTYPE::TypeInt);
    if (type == typeint) {
      // integer data is row-major NT layout
      if (p->weight_type == BTLA_DTYPE::S2) {
        packq_int2(raws8, blob, p, q);
      }
      if (p->weight_type == BTLA_DTYPE::S4) {
        packq_int4(raws8, blob, p, q);
      }
      if (p->weight_type == BTLA_DTYPE::S8) {
        packq_int8(raws8, blob, p, q);
      }
    }
    if (p->weight_type == BTLA_DTYPE::F8_E4M3 || p->weight_type == BTLA_DTYPE::F8_E5M2) {
      // FP8 data is row-major NT layout, same as int8
      packq_fp8(raws8, blob, p, q);
    }
    packscale(scaleptr, blob, p, q);
    packzp(zpptr, blob, p, q);
  }

  static void unpackq(BTLA_DTYPE outt, int8_t* blob, void* optr, QuantParam* p, sycl::queue* q) {
    using namespace bestla;
    using namespace bestla::sycl_prologue_b;
    auto qptr = (uint8_t*)blob;
    auto scale_ptr = blob + get_scale_offset(p);
    auto zp_ptr = p->asym ? blob + get_zp_offset(p) : nullptr;
    int blks = p->blks();
    if (p->weight_type == BTLA_DTYPE::S2) {
      if (outt == BTLA_DTYPE::S8) {
        bool _rescale = rescale(p);
        if (_rescale) {
#ifdef ARK_RESCALE
          auto scalext_ptr = blob + get_scalext_offset(p);
          int newblocksize = rescale_blocksize(p);
          if (p->scale_type == BTLA_DTYPE::F32) {
            using ProB = WeightS2T<float>;
            ProB::template dequantS8<typename ProB::CfgDequantS8Rescale>(
                p->n, p->k, p->blocksize, {qptr, (float*)scale_ptr, blks, nullptr, zp_ptr, (float*)scalext_ptr},
                (int8_t*)optr, q, newblocksize);
          }
          if (p->scale_type == BTLA_DTYPE::F16) {
            using ProB = WeightS2T<sycl::half>;
            ProB::template dequantS8<typename ProB::CfgDequantS8Rescale>(
                p->n, p->k, p->blocksize,
                {qptr, (sycl::half*)scale_ptr, blks, nullptr, zp_ptr, (sycl::half*)scalext_ptr}, (int8_t*)optr, q,
                newblocksize);
          }
#endif
        } else {
          using ProB = WeightS2T<float>;
          ProB::template dequantS8<typename ProB::CfgDequantS8>(
              p->n, p->k, p->blocksize, {qptr, nullptr, blks, nullptr, zp_ptr}, (int8_t*)optr, q);
        }
        return;
      }
      if (p->scale_type == BTLA_DTYPE::F32) {
        using ProB = WeightS2T<float>;
        ProB::template dequant<typename ProB::CfgDequantF32>(
            p->n, p->k, p->blocksize, {qptr, (float*)scale_ptr, blks, nullptr, zp_ptr}, (float*)optr, q);
      }
      if (p->scale_type == BTLA_DTYPE::F16) {
        using ProB = WeightS2T<sycl::half>;
        ProB::template dequant<typename ProB::CfgDequantF16>(
            p->n, p->k, p->blocksize, {qptr, (sycl::half*)scale_ptr, blks, nullptr, zp_ptr}, (sycl::half*)optr, q);
      }
      return;
    }
    if (p->weight_type == BTLA_DTYPE::S4) {
      if (outt == BTLA_DTYPE::S8) {
        bool _rescale = rescale(p);
        if (_rescale) {
#ifdef ARK_RESCALE
          auto scalext_ptr = blob + get_scalext_offset(p);
          int newblocksize = rescale_blocksize(p);
          if (p->scale_type == BTLA_DTYPE::F32) {
            using ProB = WeightS4T<float>;
            ProB::template dequantS8<typename ProB::CfgDequantS8Rescale>(
                p->n, p->k, p->blocksize, {qptr, (float*)scale_ptr, blks, nullptr, zp_ptr, (float*)scalext_ptr},
                (int8_t*)optr, q, newblocksize);
          }
          if (p->scale_type == BTLA_DTYPE::F16) {
            using ProB = WeightS4T<sycl::half>;
            ProB::template dequantS8<typename ProB::CfgDequantS8Rescale>(
                p->n, p->k, p->blocksize,
                {qptr, (sycl::half*)scale_ptr, blks, nullptr, zp_ptr, (sycl::half*)scalext_ptr}, (int8_t*)optr, q,
                newblocksize);
          }
#endif
        } else {
          using ProB = WeightS4T<float>;
          ProB::template dequantS8<typename ProB::CfgDequantS8>(
              p->n, p->k, p->blocksize, {qptr, nullptr, blks, nullptr, zp_ptr}, (int8_t*)optr, q);
        }
        return;
      }
      if (p->scale_type == BTLA_DTYPE::F32) {
        using ProB = WeightS4T<float>;
        ProB::template dequant<typename ProB::CfgDequantF32>(
            p->n, p->k, p->blocksize, {qptr, (float*)scale_ptr, blks, nullptr, zp_ptr}, (float*)optr, q);
      }
      if (p->scale_type == BTLA_DTYPE::F16) {
        using ProB = WeightS4T<sycl::half>;
        ProB::template dequant<typename ProB::CfgDequantF16>(
            p->n, p->k, p->blocksize, {qptr, (sycl::half*)scale_ptr, blks, nullptr, zp_ptr}, (sycl::half*)optr, q);
      }
      return;
    }
    if (p->weight_type == BTLA_DTYPE::S8) {
      if (p->scale_type == BTLA_DTYPE::F32) {
        using ProB = WeightS8T<float>;
        ProB::template dequant<typename ProB::CfgDequantF32>(
            p->n, p->k, p->blocksize, {(int8_t*)qptr, (float*)scale_ptr, blks, nullptr, zp_ptr}, (float*)optr, q);
      }
      if (p->scale_type == BTLA_DTYPE::F16) {
        using ProB = WeightS8T<sycl::half>;
        ProB::template dequant<typename ProB::CfgDequantF16>(
            p->n, p->k, p->blocksize, {(int8_t*)qptr, (sycl::half*)scale_ptr, blks, nullptr, zp_ptr}, (sycl::half*)optr,
            q);
      }
      return;
    }
    if (p->weight_type == BTLA_DTYPE::F8_E4M3 || p->weight_type == BTLA_DTYPE::F8_E5M2) {
      auto e4m3 = p->weight_type == BTLA_DTYPE::F8_E4M3;
      if (p->scale_type != BTLA_DTYPE::F8_E8M0) {
        if (p->scale_type == BTLA_DTYPE::F32) {
          using ST = float;
          if (e4m3) {
            using ProB = WeightF8T<ST, true>;
            ProB::dequant<ProB::Cfg>(p->n, p->k, p->blocksize, {(const uint8_t*)qptr, (ST*)scale_ptr, blks}, (ST*)optr,
                                     q);
          } else {
            using ProB = WeightF8T<ST, false>;
            ProB::dequant<ProB::Cfg>(p->n, p->k, p->blocksize, {(const uint8_t*)qptr, (ST*)scale_ptr, blks}, (ST*)optr,
                                     q);
          }
          return;
        }
        if (p->scale_type == BTLA_DTYPE::F16) {
          using ST = sycl::half;
          if (e4m3) {
            using ProB = WeightF8T<ST, true>;
            ProB::dequant<ProB::Cfg>(p->n, p->k, p->blocksize, {(const uint8_t*)qptr, (ST*)scale_ptr, blks}, (ST*)optr,
                                     q);
          } else {
            using ProB = WeightF8T<ST, false>;
            ProB::dequant<ProB::Cfg>(p->n, p->k, p->blocksize, {(const uint8_t*)qptr, (ST*)scale_ptr, blks}, (ST*)optr,
                                     q);
          }
          return;
        }
      } else {
        using ST = bestla::utils::f8;
        if (outt == BTLA_DTYPE::F32) {
          using T = float;
          if (e4m3) {
            using ProB = WeightF8T<ST, true>;
            ProB::dequant<ProB::Cfg>(p->n, p->k, p->blocksize, {(const uint8_t*)qptr, (ST*)scale_ptr, blks}, (T*)optr,
                                     q);
          } else {
            using ProB = WeightF8T<ST, false>;
            ProB::dequant<ProB::Cfg>(p->n, p->k, p->blocksize, {(const uint8_t*)qptr, (ST*)scale_ptr, blks}, (T*)optr,
                                     q);
          }
        } else if (outt == BTLA_DTYPE::F16) {
          using T = sycl::half;
          if (e4m3) {
            using ProB = WeightF8T<ST, true>;
            ProB::dequant<ProB::Cfg>(p->n, p->k, p->blocksize, {(const uint8_t*)qptr, (ST*)scale_ptr, blks}, (T*)optr,
                                     q);
          } else {
            using ProB = WeightF8T<ST, false>;
            ProB::dequant<ProB::Cfg>(p->n, p->k, p->blocksize, {(const uint8_t*)qptr, (ST*)scale_ptr, blks}, (T*)optr,
                                     q);
          }
        }

        return;
      }
      return;
    }
  }

  static int woq_gemv(sycl::queue* q, size_t m, QuantParam* p, const void* matA, const void* blobB, void* matC,
                      const void* bias, BTLA_DTYPE outt) {
    if (m > 1) return -2;
    using namespace bestla;
    using namespace bestla::sycl_prologue_b;
    auto qptr = (uint8_t*)blobB;
    auto scale_ptr = (int8_t*)blobB + get_scale_offset(p);
    auto zp_ptr = p->asym ? (int8_t*)blobB + get_zp_offset(p) : nullptr;
    int blks = p->blks();
    if (p->weight_type == BTLA_DTYPE::S2) {
      if (p->scale_type == BTLA_DTYPE::F32) {
        using ST = float;
        using T = float;
        using ProB = WeightS2T<ST>;
        ProB::gemv<ProB::CfgGemvF32>((const T*)matA,
                                     {(const uint8_t*)qptr, (ST*)scale_ptr, blks, (const T*)bias, zp_ptr}, (T*)matC,
                                     p->n, p->k, p->blocksize, q);
      }
      if (p->scale_type == BTLA_DTYPE::F16) {
        using ST = sycl::half;
        using ProB = WeightS2T<ST>;
        using T = sycl::half;
        ProB::gemv<ProB::CfgGemvF16>((const T*)matA,
                                     {(const uint8_t*)qptr, (ST*)scale_ptr, blks, (const T*)bias, zp_ptr}, (T*)matC,
                                     p->n, p->k, p->blocksize, q);
      }
      if (p->scale_type == BTLA_DTYPE::BF16) {
        return -1;
      }
      return 0;
    }
    if (p->weight_type == BTLA_DTYPE::S4) {
      if (p->scale_type == BTLA_DTYPE::F32) {
        using ST = float;
        using T = float;
        using ProB = WeightS4T<ST>;
        ProB::gemv<ProB::CfgGemvF32>((const T*)matA,
                                     {(const uint8_t*)qptr, (ST*)scale_ptr, blks, (const T*)bias, zp_ptr}, (T*)matC,
                                     p->n, p->k, p->blocksize, q);
      }
      if (p->scale_type == BTLA_DTYPE::F16) {
        using ST = sycl::half;
        using ProB = WeightS4T<ST>;
        using T = sycl::half;
        ProB::gemv<ProB::CfgGemvF16>((const T*)matA,
                                     {(const uint8_t*)qptr, (ST*)scale_ptr, blks, (const T*)bias, zp_ptr}, (T*)matC,
                                     p->n, p->k, p->blocksize, q);
      }
      if (p->scale_type == BTLA_DTYPE::BF16) {
        return -1;
      }
      return 0;
    }
    if (p->weight_type == BTLA_DTYPE::S8) {
      if (p->scale_type == BTLA_DTYPE::F32) {
        using ST = float;
        using T = float;
        using ProB = WeightS8T<ST>;
        ProB::gemv<ProB::CfgGemvF32>((const T*)matA,
                                     {(const int8_t*)qptr, (ST*)scale_ptr, blks, (const T*)bias, zp_ptr}, (T*)matC,
                                     p->n, p->k, p->blocksize, q);
      }
      if (p->scale_type == BTLA_DTYPE::F16) {
        using ST = sycl::half;
        using ProB = WeightS8T<ST>;
        using T = sycl::half;
        ProB::gemv<ProB::CfgGemvF16>((const T*)matA,
                                     {(const int8_t*)qptr, (ST*)scale_ptr, blks, (const T*)bias, zp_ptr}, (T*)matC,
                                     p->n, p->k, p->blocksize, q);
      }
      if (p->scale_type == BTLA_DTYPE::BF16) {
        return -1;
      }
      return 0;
    }
    if (p->weight_type == BTLA_DTYPE::F8_E4M3 || p->weight_type == BTLA_DTYPE::F8_E5M2) {
      auto e4m3 = p->weight_type == BTLA_DTYPE::F8_E4M3;
      if (p->scale_type != BTLA_DTYPE::F8_E8M0) {
        if (e4m3) {
          auto constexpr E4M3_T = true;
          if (p->scale_type == BTLA_DTYPE::F32) {
            using T = float;
            using ProB = WeightF8T<T, E4M3_T>;
            ProB::gemv((T*)matA, {(const uint8_t*)qptr, (T*)scale_ptr, blks, bias}, (T*)matC, p->n, p->k, p->blocksize,
                       q);
          }
          if (p->scale_type == BTLA_DTYPE::F16) {
            using T = sycl::half;
            using ProB = WeightF8T<T, E4M3_T>;
            ProB::gemv((T*)matA, {(const uint8_t*)qptr, (T*)scale_ptr, blks, bias}, (T*)matC, p->n, p->k, p->blocksize,
                       q);
          }
        } else {
          auto constexpr E4M3_T = false;
          if (p->scale_type == BTLA_DTYPE::F32) {
            using T = float;
            using ProB = WeightF8T<T, E4M3_T>;
            ProB::gemv((T*)matA, {(const uint8_t*)qptr, (T*)scale_ptr, blks, bias}, (T*)matC, p->n, p->k, p->blocksize,
                       q);
          }
          if (p->scale_type == BTLA_DTYPE::F16) {
            using T = sycl::half;
            using ProB = WeightF8T<T, E4M3_T>;
            ProB::gemv((T*)matA, {(const uint8_t*)qptr, (T*)scale_ptr, blks, bias}, (T*)matC, p->n, p->k, p->blocksize,
                       q);
          }
        }
      } else {
        if (e4m3) {
          auto constexpr E4M3_T = true;
          using ST = bestla::utils::f8;
          using ProB = WeightF8T<ST, E4M3_T>;
          if (outt == BTLA_DTYPE::F32) {
            using T = float;
            ProB::gemv((T*)matA, {(const uint8_t*)qptr, (ST*)scale_ptr, blks, bias}, (T*)matC, p->n, p->k, p->blocksize,
                       q);
          }
          if (outt == BTLA_DTYPE::F16) {
            using T = sycl::half;
            ProB::gemv((T*)matA, {(const uint8_t*)qptr, (ST*)scale_ptr, blks, bias}, (T*)matC, p->n, p->k, p->blocksize,
                       q);
          }
        } else {
          auto constexpr E4M3_T = false;
          using ST = bestla::utils::f8;
          using ProB = WeightF8T<ST, E4M3_T>;
          if (outt == BTLA_DTYPE::F32) {
            using T = float;
            ProB::gemv((T*)matA, {(const uint8_t*)qptr, (ST*)scale_ptr, blks, bias}, (T*)matC, p->n, p->k, p->blocksize,
                       q);
          }
          if (outt == BTLA_DTYPE::F16) {
            using T = sycl::half;
            ProB::gemv((T*)matA, {(const uint8_t*)qptr, (ST*)scale_ptr, blks, bias}, (T*)matC, p->n, p->k, p->blocksize,
                       q);
          }
        }
      }
      return 0;
    }
    return -1;
  }

  static inline bool rescale(QuantParam* p) {
#ifdef ARK_RESCALE
    if (env_params::Instance()->auto_s8 == -1) return true;
    if (env_params::Instance()->auto_s8 > p->blocksize && p->k % env_params::Instance()->auto_s8 == 0) return true;
#endif
    return false;
  }

  static inline int rescale_blocksize(QuantParam* p) {
#ifdef ARK_RESCALE
    if (env_params::Instance()->auto_s8 == 0) return p->blocksize;
    return env_params::Instance()->auto_s8 == -1 ? p->k : env_params::Instance()->auto_s8;
#else
    return p->blocksize;
#endif
  }

  static void woq_gemm(int m, const void* a, const void* b, void* c, const void* bias, BTLA_DTYPE acdt, QuantParam* p,
                       sycl::queue* q) {
    auto ret = woq_gemv(q, m, p, a, b, c, bias, acdt);
    if (ret) {
      auto dnnl_dt = to_dt(acdt);
      check_compute_type(p);
      if (p->compute_type != BTLA_DTYPE::S8) {
        size_t elesize = bestla::utils::bestla_dtype_bytes(acdt);
        size_t total_size = elesize * p->k * p->n;
        auto ptr = DnnlContext::Instance()->get_scratch_mem(total_size, 1, q);
        unpackq(acdt, (int8_t*)b, ptr, p, q);
        DnnlWrapper::gemm(q, m, p->n, p->k, a, dnnl_dt, ptr, dnnl_dt, true, c, dnnl_dt, bias);
      } else {
        auto bptr = (int8_t*)b;
        bool _rescale = rescale(p);
        auto scaleb_ptr = (int8_t*)b + get_scale_offset(p);
        if (p->weight_type == BTLA_DTYPE::S8) {
          scaleb_ptr = (int8_t*)b + get_scale_offset(p);
        } else {
          size_t elesize = 1;
          size_t total_size = elesize * p->k * p->n;
          bptr = (int8_t*)DnnlContext::Instance()->get_scratch_mem(total_size, 2, q);
          unpackq(BTLA_DTYPE::S8, (int8_t*)b, bptr, p, q);
          scaleb_ptr = _rescale ? (int8_t*)b + get_scalext_offset(p) : (int8_t*)b + get_scale_offset(p);
        }
        auto blocksize = rescale_blocksize(p);
        DnnlWrapper::woq_s8(q, m, p->n, p->k, a, bptr, true, c, dnnl_dt, scaleb_ptr, (void*)bias, blocksize);
      }
    }
  }

  template <typename T>
  static void compute_seq_mean_bias(sycl::queue* q, const T* in_ptr, T* bias_ptr, int num_rows, int seq, int head_dim) {
    constexpr int SG_SIZE = 32;
    constexpr int WG_SIZE = 512;
    constexpr int SG_PER_WG = WG_SIZE / SG_SIZE;
    constexpr int SeqUnroll = 4;
    size_t total = size_t(num_rows) * size_t(head_dim);
    size_t num_groups = (total + SG_PER_WG - 1) / SG_PER_WG;
    size_t global = num_groups * WG_SIZE;
    q->parallel_for(sycl::nd_range<1>(global, WG_SIZE),
                    [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                      size_t group_id = item.get_group(0);
                      int local_id = item.get_local_id(0);
                      int subgroup_id = local_id / SG_SIZE;
                      int lane_id = local_id % SG_SIZE;
                      size_t index = group_id * SG_PER_WG + subgroup_id;
                      if (index >= total) {
                        return;
                      }

                      int row_id = index / head_dim;
                      int dim_id = index % head_dim;
                      const T* row_ptr = in_ptr + size_t(row_id) * seq * head_dim + dim_id;
                      auto sg = item.get_sub_group();

                      float sum = 0.0f;
                      int token = lane_id;
                      for (; token + (SeqUnroll - 1) * SG_SIZE < seq; token += SeqUnroll * SG_SIZE) {
 #pragma unroll
                        for (int unroll_idx = 0; unroll_idx < SeqUnroll; ++unroll_idx) {
                          sum += static_cast<float>(row_ptr[size_t(token + unroll_idx * SG_SIZE) * head_dim]);
                        }
                      }
                      for (; token < seq; token += SG_SIZE) {
                        sum += static_cast<float>(row_ptr[size_t(token) * head_dim]);
                      }
                      sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>{});
                      if (lane_id == 0) {
                        bias_ptr[index] = static_cast<T>(sum / static_cast<float>(seq));
                      }
                    });
  }

  template <typename T>
  static void print_value_distribution(sycl::queue* q, const T* dev_ptr, size_t count, const char* name) {
    if (dev_ptr == nullptr || count == 0) {
      std::printf("[%s] empty\n", name);
      return;
    }

    std::vector<T> host_values(count);
    q->memcpy(host_values.data(), dev_ptr, count * sizeof(T)).wait();

    float min_value = std::numeric_limits<float>::max();
    float max_value = std::numeric_limits<float>::lowest();
    double sum = 0.0;

    for (size_t i = 0; i < count; ++i) {
      float value = static_cast<float>(host_values[i]);
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
      sum += value;
    }

    double mean = sum / static_cast<double>(count);
    std::printf("[%s] min=%f max=%f mean=%f\n", name, min_value, max_value, static_cast<float>(mean));
  }

  // input: num_rows x head_dim matrix, output: int8 quantized matrix + scale (per block)
  // scale: num_rows // block_size
  template <typename T>
  static void sage_dynamic_quant(sycl::queue* q, const T* in_ptr, int8_t* out_ptr, float* scale_ptr, int num_rows,
                                 int seq, int n_seq_blk, int head_dim, int block_size, const T* bias_ptr = nullptr) {
    int num_blocks = num_rows * n_seq_blk;
    int elems_per_block = block_size * head_dim;

    // Work-group size: use 256 threads (16 sub-groups × 16 lanes).
    // For block_size=1, head_dim=128: 128 elements / 256 threads < 1, so we use smaller WG.
    // For block_size=64, head_dim=128: 8192 elements / 256 threads = 32 elements per thread.
    constexpr int SG_SIZE = 32;
    constexpr int MAX_Reg = 64;
    constexpr int MAX_WG_SIZE = 256;
    constexpr int Unroll = 4;
    bool fast_vector_bias = bias_ptr && (head_dim % Unroll == 0);
    bool power_of_two_head_dim = fast_vector_bias && ((head_dim & (head_dim - 1)) == 0);
    int bias_mask = head_dim - 1;
    if (elems_per_block > MAX_Reg * MAX_WG_SIZE) {
      int wg_size = (elems_per_block <= 256) ? SG_SIZE : 256;
      // Ensure wg_size is a multiple of SG_SIZE
      wg_size = ((wg_size + SG_SIZE - 1) / SG_SIZE) * SG_SIZE;

      if (bias_ptr) {
        q->parallel_for(sycl::nd_range<1>(num_blocks * wg_size, wg_size),
                        [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                          int block_id = item.get_group(0);
                          int row_id = block_id / n_seq_blk;
                          int seq_id = block_id % n_seq_blk;
                          int tid = item.get_local_id(0);
                          auto wg = item.get_group();
                          auto* block_in = in_ptr + (size_t)row_id * seq * head_dim + seq_id * block_size * head_dim;
                          auto* block_out = out_ptr + (size_t)row_id * seq * head_dim + seq_id * block_size * head_dim;
                          auto* row_bias = bias_ptr + (size_t)row_id * head_dim;

                          int elems_this_wg = (seq_id + 1) * block_size <= seq ? elems_per_block
                                                                                 : (seq - seq_id * block_size) * head_dim;

                          float local_max = 0.0f;
                          sycl::vec<T, Unroll> local_data, local_max_vec;
                          local_max_vec = sycl::vec<T, Unroll>(0.0f);
                          for (int i = tid * Unroll; i < elems_this_wg; i += wg_size * Unroll) {
                            local_data = *(sycl::vec<T, Unroll>*)(&block_in[i]);
                            if (fast_vector_bias) {
                              int bias_offset = power_of_two_head_dim ? (i & bias_mask) : (i % head_dim);
                              local_data = local_data - *(sycl::vec<T, Unroll>*)(&row_bias[bias_offset]);
                            } else {
#pragma unroll
                              for (int j = 0; j < Unroll; ++j) {
                                local_data[j] = local_data[j] - row_bias[(i + j) % head_dim];
                              }
                            }
                            local_max_vec = sycl::fmax(local_max_vec, sycl::fabs(local_data));
                          }
                          for (int i = 0; i < Unroll; ++i) {
                            local_max = sycl::fmax(local_max, static_cast<float>(local_max_vec[i]));
                          }
                          float absmax = sycl::reduce_over_group(wg, local_max, sycl::maximum<float>{});
                          float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

                          if (tid == 0) {
                            scale_ptr[row_id * n_seq_blk + seq_id] = absmax / 127.0f;
                          }

                          for (int i = tid * Unroll; i < elems_this_wg; i += wg_size * Unroll) {
                            sycl::vec<T, Unroll> quant_input = *(sycl::vec<T, Unroll>*)(&block_in[i]);
                            if (fast_vector_bias) {
                              int bias_offset = power_of_two_head_dim ? (i & bias_mask) : (i % head_dim);
                              quant_input = quant_input - *(sycl::vec<T, Unroll>*)(&row_bias[bias_offset]);
                            } else {
#pragma unroll
                              for (int j = 0; j < Unroll; ++j) {
                                quant_input[j] = quant_input[j] - row_bias[(i + j) % head_dim];
                              }
                            }
                            sycl::vec<float, Unroll> val =
                                quant_input.template convert<float, sycl::rounding_mode::automatic>();
                            val = val * inv_scale;
                            val = sycl::round(val);
                            val = sycl::clamp(val, -127, 127);
                            sycl::vec<int8_t, Unroll> qv =
                                val.template convert<int8_t, sycl::rounding_mode::automatic>();
                            *(sycl::vec<int8_t, Unroll>*)(&block_out[i]) = qv;
                          }
                        });
      } else {
        q->parallel_for(sycl::nd_range<1>(num_blocks * wg_size, wg_size),
                        [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                          int block_id = item.get_group(0);
                          int row_id = block_id / n_seq_blk;
                          int seq_id = block_id % n_seq_blk;
                          int tid = item.get_local_id(0);
                          auto wg = item.get_group();
                          auto* block_in = in_ptr + (size_t)row_id * seq * head_dim + seq_id * block_size * head_dim;
                          auto* block_out = out_ptr + (size_t)row_id * seq * head_dim + seq_id * block_size * head_dim;

                          int elems_this_wg = (seq_id + 1) * block_size <= seq ? elems_per_block
                                                                                 : (seq - seq_id * block_size) * head_dim;

                          float local_max = 0.0f;
                          sycl::vec<T, Unroll> local_data, local_max_vec;
                          local_max_vec = sycl::vec<T, Unroll>(0.0f);
                          for (int i = tid * Unroll; i < elems_this_wg; i += wg_size * Unroll) {
                            local_data = *(sycl::vec<T, Unroll>*)(&block_in[i]);
                            local_max_vec = sycl::fmax(local_max_vec, sycl::fabs(local_data));
                          }
                          for (int i = 0; i < Unroll; ++i) {
                            local_max = sycl::fmax(local_max, static_cast<float>(local_max_vec[i]));
                          }
                          float absmax = sycl::reduce_over_group(wg, local_max, sycl::maximum<float>{});
                          float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

                          if (tid == 0) {
                            scale_ptr[row_id * n_seq_blk + seq_id] = absmax / 127.0f;
                          }

                          for (int i = tid * Unroll; i < elems_this_wg; i += wg_size * Unroll) {
                            sycl::vec<T, Unroll> quant_input = *(sycl::vec<T, Unroll>*)(&block_in[i]);
                            sycl::vec<float, Unroll> val =
                                quant_input.template convert<float, sycl::rounding_mode::automatic>();
                            val = val * inv_scale;
                            val = sycl::round(val);
                            val = sycl::clamp(val, -127, 127);
                            sycl::vec<int8_t, Unroll> qv =
                                val.template convert<int8_t, sycl::rounding_mode::automatic>();
                            *(sycl::vec<int8_t, Unroll>*)(&block_out[i]) = qv;
                          }
                        });
      }
    } else {
      int wg_size = MAX_WG_SIZE;
      if (bias_ptr) {
        if (fast_vector_bias) {
          q->parallel_for(sycl::nd_range<1>(num_blocks * wg_size, wg_size),
                          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                            int seq_id = item.get_group(0);
                            int row_id = seq_id / n_seq_blk;
                            seq_id = seq_id % n_seq_blk;
                            int tid = item.get_local_id(0);
                            auto wg = item.get_group();
                            auto* block_in = in_ptr + (size_t)row_id * seq * head_dim + seq_id * block_size * head_dim;
                            auto* block_out = out_ptr + (size_t)row_id * seq * head_dim + seq_id * block_size * head_dim;
                            auto* row_bias = bias_ptr + (size_t)row_id * head_dim;

                            float local_max = 0.0f;
                            sycl::vec<T, Unroll> local_data[MAX_Reg / Unroll], local_max_vec;
                            local_max_vec = sycl::vec<T, Unroll>(0.0f);
                            int local_i = 0;
                            int elems_this_wg = (seq_id + 1) * block_size <= seq ? block_size * head_dim
                                                                                 : (seq - seq_id * block_size) * head_dim;
                            for (int i = tid * Unroll; i < elems_this_wg; i += wg_size * Unroll, local_i++) {
                              int bias_offset = power_of_two_head_dim ? (i & bias_mask) : (i % head_dim);
                              local_data[local_i] = *(sycl::vec<T, Unroll>*)&block_in[i];
                              local_data[local_i] = local_data[local_i] - *(sycl::vec<T, Unroll>*)(&row_bias[bias_offset]);
                              local_max_vec = sycl::fmax(local_max_vec, sycl::fabs(local_data[local_i]));
                            }
#pragma unroll
                            for (int i = 0; i < Unroll; ++i) {
                              local_max = sycl::fmax(local_max, static_cast<float>(local_max_vec[i]));
                            }
                            float absmax = sycl::reduce_over_group(wg, local_max, sycl::maximum<float>{});
                            float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

                            if (tid == 0) {
                              scale_ptr[row_id * n_seq_blk + seq_id] = absmax / 127.0f;
                            }

                            local_i = 0;
                            for (int i = tid * Unroll; i < elems_this_wg; i += wg_size * Unroll, local_i++) {
                              sycl::vec<float, Unroll> val =
                                  local_data[local_i].template convert<float, sycl::rounding_mode::automatic>();
                              val = val * inv_scale;
                              val = sycl::round(val);
                              val = sycl::clamp(val, -127, 127);
                              sycl::vec<int8_t, Unroll> qv =
                                  val.template convert<int8_t, sycl::rounding_mode::automatic>();
                              *(sycl::vec<int8_t, Unroll>*)(&block_out[i]) = qv;
                            }
                          });
        } else {
          q->parallel_for(sycl::nd_range<1>(num_blocks * wg_size, wg_size),
                          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                            int seq_id = item.get_group(0);
                            int row_id = seq_id / n_seq_blk;
                            seq_id = seq_id % n_seq_blk;
                            int tid = item.get_local_id(0);
                            auto wg = item.get_group();
                            auto* block_in = in_ptr + (size_t)row_id * seq * head_dim + seq_id * block_size * head_dim;
                            auto* block_out = out_ptr + (size_t)row_id * seq * head_dim + seq_id * block_size * head_dim;
                            auto* row_bias = bias_ptr + (size_t)row_id * head_dim;

                            float local_max = 0.0f;
                            sycl::vec<T, Unroll> local_data[MAX_Reg / Unroll], local_max_vec;
                            local_max_vec = sycl::vec<T, Unroll>(0.0f);
                            int local_i = 0;
                            int elems_this_wg = (seq_id + 1) * block_size <= seq ? block_size * head_dim
                                                                                 : (seq - seq_id * block_size) * head_dim;
                            for (int i = tid * Unroll; i < elems_this_wg; i += wg_size * Unroll, local_i++) {
                              local_data[local_i] = *(sycl::vec<T, Unroll>*)&block_in[i];
#pragma unroll
                              for (int j = 0; j < Unroll; ++j) {
                                local_data[local_i][j] = local_data[local_i][j] - row_bias[(i + j) % head_dim];
                              }
                              local_max_vec = sycl::fmax(local_max_vec, sycl::fabs(local_data[local_i]));
                            }
#pragma unroll
                            for (int i = 0; i < Unroll; ++i) {
                              local_max = sycl::fmax(local_max, static_cast<float>(local_max_vec[i]));
                            }
                            float absmax = sycl::reduce_over_group(wg, local_max, sycl::maximum<float>{});
                            float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

                            if (tid == 0) {
                              scale_ptr[row_id * n_seq_blk + seq_id] = absmax / 127.0f;
                            }

                            local_i = 0;
                            for (int i = tid * Unroll; i < elems_this_wg; i += wg_size * Unroll, local_i++) {
                              sycl::vec<float, Unroll> val =
                                  local_data[local_i].template convert<float, sycl::rounding_mode::automatic>();
                              val = val * inv_scale;
                              val = sycl::round(val);
                              val = sycl::clamp(val, -127, 127);
                              sycl::vec<int8_t, Unroll> qv =
                                  val.template convert<int8_t, sycl::rounding_mode::automatic>();
                              *(sycl::vec<int8_t, Unroll>*)(&block_out[i]) = qv;
                            }
                          });
        }
      } else {
        q->parallel_for(sycl::nd_range<1>(num_blocks * wg_size, wg_size),
                        [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                          int seq_id = item.get_group(0);
                          int row_id = seq_id / n_seq_blk;
                          seq_id = seq_id % n_seq_blk;
                          int tid = item.get_local_id(0);
                          auto wg = item.get_group();
                          auto* block_in = in_ptr + (size_t)row_id * seq * head_dim + seq_id * block_size * head_dim;
                          auto* block_out = out_ptr + (size_t)row_id * seq * head_dim + seq_id * block_size * head_dim;

                          float local_max = 0.0f;
                          sycl::vec<T, Unroll> local_data[MAX_Reg / Unroll], local_max_vec;
                          local_max_vec = sycl::vec<T, Unroll>(0.0f);
                          int local_i = 0;
                          int elems_this_wg = (seq_id + 1) * block_size <= seq ? block_size * head_dim
                                                                               : (seq - seq_id * block_size) * head_dim;
                          for (int i = tid * Unroll; i < elems_this_wg; i += wg_size * Unroll, local_i++) {
                            local_data[local_i] = *(sycl::vec<T, Unroll>*)&block_in[i];
                            local_max_vec = sycl::fmax(local_max_vec, sycl::fabs(local_data[local_i]));
                          }
#pragma unroll
                          for (int i = 0; i < Unroll; ++i) {
                            local_max = sycl::fmax(local_max, static_cast<float>(local_max_vec[i]));
                          }
                          float absmax = sycl::reduce_over_group(wg, local_max, sycl::maximum<float>{});
                          float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

                          if (tid == 0) {
                            scale_ptr[row_id * n_seq_blk + seq_id] = absmax / 127.0f;
                          }

                          local_i = 0;
                          for (int i = tid * Unroll; i < elems_this_wg; i += wg_size * Unroll, local_i++) {
                            sycl::vec<float, Unroll> val =
                                local_data[local_i].template convert<float, sycl::rounding_mode::automatic>();
                            val = val * inv_scale;
                            val = sycl::round(val);
                            val = sycl::clamp(val, -127, 127);
                            sycl::vec<int8_t, Unroll> qv =
                                val.template convert<int8_t, sycl::rounding_mode::automatic>();
                            *(sycl::vec<int8_t, Unroll>*)(&block_out[i]) = qv;
                          }
                        });
      }
    }
  }

  static void sagev1(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                     int scale_block_size, int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
                     int head_dim, float softmax_scale, bool is_causal) {
    bool use_mean_bias = env_params::Instance()->sage_use_mean_bias != 0;
    size_t seq_q_blk = (seq_len_q + scale_block_size - 1) / scale_block_size;
    size_t seq_kv_blk = (seq_len_kv + scale_block_size - 1) / scale_block_size;
    size_t q_size = num_heads_q * seq_len_q * head_dim * batch;
    size_t q_scale_size = num_heads_q * seq_q_blk * batch;
    size_t k_size = num_heads_kv * seq_len_kv * head_dim * batch;
    size_t k_scale_size = num_heads_kv * seq_kv_blk * batch;
    size_t k_bias_size = use_mean_bias ? num_heads_kv * head_dim * batch * sizeof(sycl::half) : 0;
    size_t total_size = k_size + q_size + k_scale_size * sizeof(float) + q_scale_size * sizeof(float) + k_bias_size;
    auto ptr = DnnlContext::Instance()->get_scratch_mem(total_size, 1, q);
    auto q_out_ptr = (int8_t*)ptr;
    auto k_out_ptr = (int8_t*)ptr + q_size;
    auto qscale = (float*)((int8_t*)ptr + q_size + k_size);
    auto kscale = (float*)((int8_t*)ptr + q_size + k_size + q_scale_size * sizeof(float));
    auto kbias = use_mean_bias
                     ? (sycl::half*)((int8_t*)ptr + q_size + k_size + q_scale_size * sizeof(float) +
                                     k_scale_size * sizeof(float))
                     : nullptr;
    sage_dynamic_quant<sycl::half>(q, (sycl::half*)Q_ptr, (int8_t*)q_out_ptr, (float*)qscale, batch * num_heads_q,
                     seq_len_q, seq_q_blk, head_dim, scale_block_size);
    if (use_mean_bias) {
      compute_seq_mean_bias<sycl::half>(q, (sycl::half*)K_ptr, kbias, batch * num_heads_kv, seq_len_kv, head_dim);
      if (env_params::Instance()->sage_print_kbias != 0) {
        print_value_distribution(q, kbias, size_t(batch) * num_heads_kv * head_dim, "kbias");
      }
    }
    sage_dynamic_quant<sycl::half>(q, (sycl::half*)K_ptr, (int8_t*)k_out_ptr, (float*)kscale, batch * num_heads_kv,
                     seq_len_kv, seq_kv_blk, head_dim, scale_block_size, kbias);
    ark::sdpa_impl_qks8_pvhalf(q, q_out_ptr, k_out_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, batch,
                               num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  }
};

}  // namespace ark
