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
#include "bestla.h"
#include "bestla_utils.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <type_traits>
#include <vector>

namespace bestla {
namespace kernel {
namespace ref {

template <typename T>
static inline BTLA_CODE shuffle_activation(T* src, T* dst, int shuffle_m, int shuffle_k, int m_offset, int k_offset,
                                           int* indices, int src_stride, int dst_stride) {
  T* cur_src = src + m_offset * src_stride;
  for (int i = 0; i < shuffle_m; i++) {
    for (int j = 0; j < shuffle_k; j++) {
      dst[i * dst_stride + j] = cur_src[i * src_stride + indices[k_offset + j]];
    }
  }
  return BTLA_CODE::Success;
}

template <typename T_SRC, typename T_DST = T_SRC>
static inline BTLA_CODE padding_interleave(const T_SRC* src_ptr, T_DST* dst_ptr, int row, int col, int rowpad,
                                           int colpad, int src_step, int dst_step, int NTile, int RowPack) {
  const T_DST dst_0(0);
  static_assert(sizeof(T_SRC) == sizeof(T_DST), "SRC & DST size should be the same");
  for (int i = 0; i < rowpad; i += RowPack) {
    for (int j = 0; j < colpad; j += NTile) {
      for (int jj = 0; jj < NTile; jj++) {
        for (int ii = 0; ii < RowPack; ii++) {
          dst_ptr[i * NTile + j * dst_step + jj * RowPack + ii] =
              (i + ii) < row && (j + jj) < col  //
                  ? static_cast<T_DST>(src_ptr[(i + ii) * src_step + (j + jj)])
                  : dst_0;
        }
      }
    }
  }
  return BTLA_CODE::Success;
}

// revert padding and interleave
// row*col <= colpad/NTile*rowpad*NTile
template <typename T_SRC, typename T_DST = T_SRC>
static inline BTLA_CODE revert_padding_interleave(const T_SRC* src_ptr, T_DST* dst_ptr, int row, int col, int rowpad,
                                                  int colpad, int src_step, int dst_step, int NTile, int RowPack) {
  static_assert(sizeof(T_SRC) == sizeof(T_DST), "SRC & DST size should be the same");
  for (int i = 0; i < rowpad; i += RowPack) {
    for (int j = 0; j < colpad; j += NTile) {
      for (int jj = 0; jj < NTile; jj++) {
        if ((j + jj) < col) {
          for (int ii = 0; ii < RowPack; ii++) {
            if ((i + ii) < row) {
              dst_ptr[(i + ii) * dst_step + (j + jj)] =
                  static_cast<T_DST>(src_ptr[i * NTile + j * src_step + jj * RowPack + ii]);
            }
          }
        }
      }
    }
  }
  return BTLA_CODE::Success;
}

// M x N ===> M/MTile x N/colPack x MTile x colPack (leading dim stride = MTile
// * dst_stride)
template <typename T_SRC, typename T_DST = T_SRC>
static inline BTLA_CODE padding_trans_interleave(const T_SRC* src, T_DST* dst, int row, int col, int rowpad, int colpad,
                                                 int src_step, int dst_step, int MTile, int ColPack) {
  // Note: rows/cols and i/j are in terms of src
  static_assert(sizeof(T_SRC) == sizeof(T_DST), "SRC & DST size should be the same");
  const T_DST dst_0(0);
  for (int i = 0; i < rowpad; i += MTile) {
    for (int j = 0; j < colpad; j += ColPack) {
      for (int ii = 0; ii < MTile; ii++) {
        for (int jj = 0; jj < ColPack; jj++) {
          dst[i * dst_step + j * MTile + ii * ColPack + jj] =
              (i + ii) < row && (j + jj) < col  //
                  ? static_cast<T_DST>(src[(i + ii) * src_step + (j + jj)])
                  : dst_0;
        }
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename SRC_DT, typename DST_DT>
static inline BTLA_CODE dt_cvt_2D_write_back(const void* raw_srcptr, void* raw_dstptr, int row, int col, int srcstride,
                                             int dststride, bool zeropadding) {
  for (int i = 0; i < row; i++) {
    int j = 0;
    for (; j < col; j++) {
      const auto src = reinterpret_cast<const SRC_DT*>(reinterpret_cast<const char*>(raw_srcptr) + i * srcstride);
      const auto dst = reinterpret_cast<DST_DT*>(reinterpret_cast<char*>(raw_dstptr) + i * dststride);
      dst[j] = static_cast<DST_DT>(src[j]);
    }
    if (zeropadding) {
      for (int bj = j * sizeof(DST_DT); bj < dststride; bj++) {
        (reinterpret_cast<char*>(raw_dstptr) + i * dststride)[bj] = 0;
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename _DST_T>
static inline BTLA_CODE dequan_s8_fp(int8_t* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                     float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dstptr[i * ld_dst + j] = static_cast<float>(srcptr[i * ld_src + j]) * scales[j];
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE dequan_s8_bf16(int8_t* srcptr, uint16_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                       float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dstptr[i * ld_dst + j] =
          utils::cast<float, utils::bf16>(static_cast<float>(srcptr[i * ld_src + j]) * scales[j]).x;
    }
  }
  return BTLA_CODE::Success;
}

template <typename _T>
static inline BTLA_CODE transpose2d(const _T* srcptr, _T* dstptr, int row, int col, int ld_src, int ld_dst) {
  for (int i = 0; i < col; i++) {
    for (size_t j = 0; j < row; j++) {
      dstptr[j + i * ld_dst] = srcptr[j * ld_src + i];
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE compress_s8_s4(const int8_t* srcptr, utils::int4x2* dstptr, size_t size) {
  int8_t constexpr FullRange = 1 << (4 - 1);
  assert(size % 2 == 0);
  for (int ii = 0; ii < size; ii += 2) {
    utils::int4x2 tmp;
    tmp.x = srcptr[ii + 0] + FullRange;
    tmp.y = srcptr[ii + 1] + FullRange;
    dstptr[ii / 2] = tmp;
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE compress_f4(const int8_t* srcptr, utils::f4x2* dstptr, size_t size) {
  for (int ii = 0; ii < size; ii += 2) {
    assert(size % 2 == 0);
    utils::f4x2 tmp;
    tmp.x = srcptr[ii + 0];
    tmp.y = srcptr[ii + 1];
    dstptr[ii / 2] = tmp;
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE compress_7bit(const int8_t* srcptr, bestla::utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr,
                                      utils::bit1x8* bit1ptr, size_t size) {
  assert(size % 8 == 0);
  int8_t constexpr FullRange = 1 << (7 - 1);
  for (int j = 0; j < size; j += 8) {
    auto tmp = srcptr[j + 0] + FullRange;
    bit4ptr[j / 2 + 0].x = tmp & 0xf;
    tmp = tmp >> 4;
    bit2ptr[j / 4].a = tmp & 0x3;
    tmp = tmp >> 2;
    bit1ptr[j / 8].a = tmp;
    tmp = srcptr[j + 1] + FullRange;
    bit4ptr[j / 2 + 0].y = tmp & 0xf;
    tmp = tmp >> 4;
    bit2ptr[j / 4].b = tmp & 0x3;
    tmp = tmp >> 2;
    bit1ptr[j / 8].b = tmp;
    tmp = srcptr[j + 2] + FullRange;
    bit4ptr[j / 2 + 1].x = tmp & 0xf;
    tmp = tmp >> 4;
    bit2ptr[j / 4].c = tmp & 0x3;
    tmp = tmp >> 2;
    bit1ptr[j / 8].c = tmp;
    tmp = srcptr[j + 3] + FullRange;
    bit4ptr[j / 2 + 1].y = tmp & 0xf;
    tmp = tmp >> 4;
    bit2ptr[j / 4].d = tmp & 0x3;
    tmp = tmp >> 2;
    bit1ptr[j / 8].d = tmp;

    tmp = srcptr[j + 4] + FullRange;
    bit4ptr[j / 2 + 2].x = tmp & 0xf;
    tmp = tmp >> 4;
    bit2ptr[j / 4 + 1].a = tmp & 0x3;
    tmp = tmp >> 2;
    bit1ptr[j / 8].e = tmp;
    tmp = srcptr[j + 5] + FullRange;
    bit4ptr[j / 2 + 2].y = tmp & 0xf;
    tmp = tmp >> 4;
    bit2ptr[j / 4 + 1].b = tmp & 0x3;
    tmp = tmp >> 2;
    bit1ptr[j / 8].f = tmp;
    tmp = srcptr[j + 6] + FullRange;
    bit4ptr[j / 2 + 3].x = tmp & 0xf;
    tmp = tmp >> 4;
    bit2ptr[j / 4 + 1].c = tmp & 0x3;
    tmp = tmp >> 2;
    bit1ptr[j / 8].g = tmp;
    tmp = srcptr[j + 7] + FullRange;
    bit4ptr[j / 2 + 3].y = tmp & 0xf;
    tmp = tmp >> 4;
    bit2ptr[j / 4 + 1].d = tmp & 0x3;
    tmp = tmp >> 2;
    bit1ptr[j / 8].h = tmp;
  }

  return BTLA_CODE::Success;
}

static inline BTLA_CODE compress_6bit(const int8_t* srcptr, bestla::utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr,
                                      size_t size) {
  assert(size % 4 == 0);
  int8_t constexpr FullRange = 1 << (6 - 1);
  for (int j = 0; j < size; j += 4) {
    auto tmp = srcptr[j + 0] + FullRange;
    bit4ptr[j / 2 + 0].x = tmp & 0xf;
    bit2ptr[j / 4].a = tmp >> 4;
    tmp = srcptr[j + 1] + FullRange;
    bit4ptr[j / 2 + 0].y = tmp & 0xf;
    bit2ptr[j / 4].b = tmp >> 4;
    tmp = srcptr[j + 2] + FullRange;
    bit4ptr[j / 2 + 1].x = tmp & 0xf;
    bit2ptr[j / 4].c = tmp >> 4;
    tmp = srcptr[j + 3] + FullRange;
    bit4ptr[j / 2 + 1].y = tmp & 0xf;
    bit2ptr[j / 4].d = tmp >> 4;
  }

  return BTLA_CODE::Success;
}

static inline BTLA_CODE compress_5bit(const int8_t* srcptr, bestla::utils::bit4x2* bit4ptr, utils::bit1x8* bit1ptr,
                                      size_t size) {
  assert(size % 8 == 0);
  int8_t constexpr FullRange = 1 << (5 - 1);
  for (int j = 0; j < size; j += 8) {
    auto tmp = srcptr[j + 0] + FullRange;
    bit4ptr[j / 2 + 0].x = tmp & 0xf;
    bit1ptr[j / 8].a = tmp >> 4;
    tmp = srcptr[j + 1] + FullRange;
    bit4ptr[j / 2 + 0].y = tmp & 0xf;
    bit1ptr[j / 8].b = tmp >> 4;
    tmp = srcptr[j + 2] + FullRange;
    bit4ptr[j / 2 + 1].x = tmp & 0xf;
    bit1ptr[j / 8].c = tmp >> 4;
    tmp = srcptr[j + 3] + FullRange;
    bit4ptr[j / 2 + 1].y = tmp & 0xf;
    bit1ptr[j / 8].d = tmp >> 4;

    tmp = srcptr[j + 4] + FullRange;
    bit4ptr[j / 2 + 2].x = tmp & 0xf;
    bit1ptr[j / 8].e = tmp >> 4;
    tmp = srcptr[j + 5] + FullRange;
    bit4ptr[j / 2 + 2].y = tmp & 0xf;
    bit1ptr[j / 8].f = tmp >> 4;
    tmp = srcptr[j + 6] + FullRange;
    bit4ptr[j / 2 + 3].x = tmp & 0xf;
    bit1ptr[j / 8].g = tmp >> 4;
    tmp = srcptr[j + 7] + FullRange;
    bit4ptr[j / 2 + 3].y = tmp & 0xf;
    bit1ptr[j / 8].h = tmp >> 4;
  }

  return BTLA_CODE::Success;
}

static inline BTLA_CODE compress_3bit(const int8_t* srcptr, bestla::utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr,
                                      size_t size) {
  int8_t constexpr FullRange = 1 << (3 - 1);
  assert(size % 8 == 0);

  for (int j = 0; j < size; j += 8) {
    auto tmp = srcptr[j + 0] + FullRange;
    bit2ptr[j / 4 + 0].a = tmp & 0x3;
    bit1ptr[j / 8].a = tmp >> 2;
    tmp = srcptr[j + 1] + FullRange;
    bit2ptr[j / 4 + 0].b = tmp & 0x3;
    bit1ptr[j / 8].b = tmp >> 2;
    tmp = srcptr[j + 2] + FullRange;
    bit2ptr[j / 4 + 0].c = tmp & 0x3;
    bit1ptr[j / 8].c = tmp >> 2;
    tmp = srcptr[j + 3] + FullRange;
    bit2ptr[j / 4 + 0].d = tmp & 0x3;
    bit1ptr[j / 8].d = tmp >> 2;

    tmp = srcptr[j + FullRange] + FullRange;
    bit2ptr[j / 4 + 1].a = tmp & 0x3;
    bit1ptr[j / 8].e = tmp >> 2;
    tmp = srcptr[j + 5] + FullRange;
    bit2ptr[j / 4 + 1].b = tmp & 0x3;
    bit1ptr[j / 8].f = tmp >> 2;
    tmp = srcptr[j + 6] + FullRange;
    bit2ptr[j / 4 + 1].c = tmp & 0x3;
    bit1ptr[j / 8].g = tmp >> 2;
    tmp = srcptr[j + 7] + FullRange;
    bit2ptr[j / 4 + 1].d = tmp & 0x3;
    bit1ptr[j / 8].h = tmp >> 2;
  }

  return BTLA_CODE::Success;
}

static inline BTLA_CODE compress_2bit(const int8_t* srcptr, bestla::utils::bit2x4* bit2ptr, size_t size) {
  int8_t constexpr FullRange = 1 << (2 - 1);
  assert(size % 4 == 0);
  for (size_t i = 0; i < size; i += 4) {
    bit2ptr[i / 4].a = *(srcptr + i) + FullRange;
    bit2ptr[i / 4].b = *(srcptr + i + 1) + FullRange;
    bit2ptr[i / 4].c = *(srcptr + i + 2) + FullRange;
    bit2ptr[i / 4].d = *(srcptr + i + 3) + FullRange;
  }

  return BTLA_CODE::Success;
}

static inline BTLA_CODE compress_1bit(const int8_t* srcptr, bestla::utils::bit1x8* bit1ptr, size_t size) {
  int8_t constexpr FullRange = 1 << (1 - 1);
  assert(size % 8 == 0);
  for (int j = 0; j < size; j += 8) {
    auto tmp = srcptr[j + 0] + FullRange;
    bit1ptr[j / 8].a = tmp;
    tmp = srcptr[j + 1] + FullRange;
    bit1ptr[j / 8].b = tmp;
    tmp = srcptr[j + 2] + FullRange;
    bit1ptr[j / 8].c = tmp;
    tmp = srcptr[j + 3] + FullRange;
    bit1ptr[j / 8].d = tmp;
    tmp = srcptr[j + FullRange] + FullRange;
    bit1ptr[j / 8].e = tmp;
    tmp = srcptr[j + 5] + FullRange;
    bit1ptr[j / 8].f = tmp;
    tmp = srcptr[j + 6] + FullRange;
    bit1ptr[j / 8].g = tmp;
    tmp = srcptr[j + 7] + FullRange;
    bit1ptr[j / 8].h = tmp;
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE Q4T>
static inline void convert_s4_s8_8(int8_t* dstptr, int8_t* srcptr) {
  auto src32 = *reinterpret_cast<uint32_t*>(srcptr);
  auto tmp = static_cast<int>(src32 & 0xf);
  if constexpr (Q4T == BTLA_DTYPE::S4_CLIP) {
    tmp -= 8;
  }
  dstptr[0] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>(src32 & 0xf0) >> 4;
  if constexpr (Q4T == BTLA_DTYPE::S4_CLIP) {
    tmp -= 8;
  }
  dstptr[1] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>((src32 & 0xf00) >> 8);
  if constexpr (Q4T == BTLA_DTYPE::S4_CLIP) {
    tmp -= 8;
  }
  dstptr[2] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>((src32 & 0xf000) >> 12);
  if constexpr (Q4T == BTLA_DTYPE::S4_CLIP) {
    tmp -= 8;
  }
  dstptr[3] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>((src32 & 0xf0000) >> 16);
  if constexpr (Q4T == BTLA_DTYPE::S4_CLIP) {
    tmp -= 8;
  }
  dstptr[4] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>((src32 & 0xf00000) >> 20);
  if constexpr (Q4T == BTLA_DTYPE::S4_CLIP) {
    tmp -= 8;
  }
  dstptr[5] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>((src32 & 0xf000000) >> 24);
  if constexpr (Q4T == BTLA_DTYPE::S4_CLIP) {
    tmp -= 8;
  }
  dstptr[6] = static_cast<int8_t>(tmp);
  tmp = static_cast<int>((src32 & 0xf0000000) >> 28);
  if constexpr (Q4T == BTLA_DTYPE::S4_CLIP) {
    tmp -= 8;
  }
  dstptr[7] = static_cast<int8_t>(tmp);
}

static inline BTLA_CODE decompress_s6_s8(utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, int8_t* dstptr,
                                         size_t unpack_elt, int8_t* tmp, size_t tmpsize) {
  int constexpr FullRange = 1 << (6 - 1);
  for (size_t i = 0; i < unpack_elt; i += 4) {
    auto bit2 = bit2ptr[i / 4];
    auto tmp = bit4ptr[i / 2];
    dstptr[i + 0] = (tmp.x | (bit2.a << 4)) - FullRange;
    dstptr[i + 1] = (tmp.y | (bit2.b << 4)) - FullRange;
    tmp = bit4ptr[i / 2 + 1];
    dstptr[i + 2] = (tmp.x | (bit2.c << 4)) - FullRange;
    dstptr[i + 3] = (tmp.y | (bit2.d << 4)) - FullRange;
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s5_s8(utils::bit4x2* bit4ptr, utils::bit1x8* bit1ptr, int8_t* dstptr,
                                         size_t unpack_elt, int8_t* tmp, size_t tmpsize) {
  int constexpr FullRange = 1 << (5 - 1);
  for (size_t i = 0; i < unpack_elt; i += 8) {
    auto bit1 = bit1ptr[i / 8];
    auto tmp = bit4ptr[i / 2];
    dstptr[i + 0] = (tmp.x | (bit1.a << 4)) - FullRange;
    dstptr[i + 1] = (tmp.y | (bit1.b << 4)) - FullRange;
    tmp = bit4ptr[i / 2 + 1];
    dstptr[i + 2] = (tmp.x | (bit1.c << 4)) - FullRange;
    dstptr[i + 3] = (tmp.y | (bit1.d << 4)) - FullRange;
    tmp = bit4ptr[i / 2 + 2];
    dstptr[i + 4] = (tmp.x | (bit1.e << 4)) - FullRange;
    dstptr[i + 5] = (tmp.y | (bit1.f << 4)) - FullRange;
    tmp = bit4ptr[i / 2 + 3];
    dstptr[i + 6] = (tmp.x | (bit1.g << 4)) - FullRange;
    dstptr[i + 7] = (tmp.y | (bit1.h << 4)) - FullRange;
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s7_s8(utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr,
                                         int8_t* dstptr, size_t unpack_elt, int8_t* tmp, size_t tmpsize) {
  int constexpr FullRange = 1 << (7 - 1);
  for (size_t i = 0; i < unpack_elt; i += 8) {
    auto bit2 = bit2ptr[i / 4];
    auto bit1 = bit1ptr[i / 8];
    auto tmp = bit4ptr[i / 2];
    dstptr[i + 0] = (tmp.x | (bit2.a << 4) | (bit1.a << 6)) - FullRange;
    dstptr[i + 1] = (tmp.y | (bit2.b << 4) | (bit1.b << 6)) - FullRange;
    tmp = bit4ptr[i / 2 + 1];
    dstptr[i + 2] = (tmp.x | (bit2.c << 4) | (bit1.c << 6)) - FullRange;
    dstptr[i + 3] = (tmp.y | (bit2.d << 4) | (bit1.d << 6)) - FullRange;
    bit2 = bit2ptr[i / 4 + 1];
    tmp = bit4ptr[i / 2 + 2];
    dstptr[i + 4] = (tmp.x | (bit2.a << 4) | (bit1.e << 6)) - FullRange;
    dstptr[i + 5] = (tmp.y | (bit2.b << 4) | (bit1.f << 6)) - FullRange;
    tmp = bit4ptr[i / 2 + 3];
    dstptr[i + 6] = (tmp.x | (bit2.c << 4) | (bit1.g << 6)) - FullRange;
    dstptr[i + 7] = (tmp.y | (bit2.d << 4) | (bit1.h << 6)) - FullRange;
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s4_s8(utils::int4x2* srcptr, int8_t* dstptr, size_t unpackelt, int8_t* tmp,
                                         size_t tmpsize) {
  for (int j = 0; j < unpackelt; j += 2) {
    auto tmp = srcptr[j / 2];
    dstptr[j + 0] = tmp.x - 8;
    dstptr[j + 1] = tmp.y - 8;
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s3_s8(utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr, int8_t* dstptr,
                                         size_t unpack_elt, int8_t* tmp, size_t tmpsize) {
  for (size_t i = 0; i < unpack_elt; i += 8) {
    auto bit1 = bit1ptr[i / 8];
    auto tmp = bit2ptr[i / 4];
    dstptr[i + 0] = (tmp.a | (bit1.a << 2)) - 4;
    dstptr[i + 1] = (tmp.b | (bit1.b << 2)) - 4;
    dstptr[i + 2] = (tmp.c | (bit1.c << 2)) - 4;
    dstptr[i + 3] = (tmp.d | (bit1.d << 2)) - 4;
    tmp = bit2ptr[i / 4 + 1];
    dstptr[i + 4] = (tmp.a | (bit1.e << 2)) - 4;
    dstptr[i + 5] = (tmp.b | (bit1.f << 2)) - 4;
    dstptr[i + 6] = (tmp.c | (bit1.g << 2)) - 4;
    dstptr[i + 7] = (tmp.d | (bit1.h << 2)) - 4;
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s2_s8(utils::bit2x4* srcptr, int8_t* dstptr, size_t unpackelt, int8_t* tmp,
                                         size_t tmpsize) {
  for (int j = 0; j < unpackelt; j += 4) {
    auto tmp = srcptr[j / 4];
    dstptr[j + 0] = tmp.a - 2;
    dstptr[j + 1] = tmp.b - 2;
    dstptr[j + 2] = tmp.c - 2;
    dstptr[j + 3] = tmp.d - 2;
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s1_s8(utils::bit1x8* bit1ptr, int8_t* dstptr, size_t unpack_elt, int8_t* tmp,
                                         size_t tmpsize) {
  int8_t constexpr FullRange = 1 << (1 - 1);
  for (size_t i = 0; i < unpack_elt; i += 8) {
    auto bit1 = bit1ptr[i / 8];
    dstptr[i + 0] = bit1.a - FullRange;
    dstptr[i + 1] = bit1.b - FullRange;
    dstptr[i + 2] = bit1.c - FullRange;
    dstptr[i + 3] = bit1.d - FullRange;
    dstptr[i + 4] = bit1.e - FullRange;
    dstptr[i + 5] = bit1.f - FullRange;
    dstptr[i + 6] = bit1.g - FullRange;
    dstptr[i + 7] = bit1.h - FullRange;
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s7_s8(utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr,
                                                int8_t* zpptr, int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                int k_offset, int row, int col, int8_t* tmp, size_t tmpsize) {
  int constexpr FullRange = 1 << (7 - 1);
  static_assert(NTILE % 8 == 0);
  assert(((col * PackRow) % 8) == 0);
  if (zpptr) {
    if constexpr (PackRow == 4) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 2) {
          auto zp = zptr[j] + FullRange;
          auto bit1 = bit1ptr[(i * col + j * PackRow) / 8];
          auto bit2 = bit2ptr[(i * col + j * PackRow) / 4];
          auto tmp = bit4ptr[(i * col + j * PackRow) / 2];
          dstptr[i * col + j * PackRow + 0] = (tmp.x | (bit2.a << 4) | (bit1.a << 6)) - zp;
          dstptr[i * col + j * PackRow + 1] = (tmp.y | (bit2.b << 4) | (bit1.b << 6)) - zp;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 1];
          dstptr[i * col + j * PackRow + 2] = (tmp.x | (bit2.c << 4) | (bit1.c << 6)) - zp;
          dstptr[i * col + j * PackRow + 3] = (tmp.y | (bit2.d << 4) | (bit1.d << 6)) - zp;
          zp = zptr[j + 1] + FullRange;
          bit2 = bit2ptr[(i * col + j * PackRow) / 4 + 1];
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 2];
          dstptr[i * col + j * PackRow + 4] = (tmp.x | (bit2.a << 4) | (bit1.e << 6)) - zp;
          dstptr[i * col + j * PackRow + 5] = (tmp.y | (bit2.b << 4) | (bit1.f << 6)) - zp;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 3];
          dstptr[i * col + j * PackRow + 6] = (tmp.x | (bit2.c << 4) | (bit1.g << 6)) - zp;
          dstptr[i * col + j * PackRow + 7] = (tmp.y | (bit2.d << 4) | (bit1.h << 6)) - zp;
        }
      }
    } else if constexpr (PackRow == 1) {
      for (int i = 0; i < row; i += 1) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 8) {
          auto bit2 = bit2ptr[(i * col + j * PackRow) / 4];
          auto bit1 = bit1ptr[(i * col + j * PackRow) / 8];
          auto tmp = bit4ptr[(i * col + j * PackRow) / 2];
          dstptr[i * col + j * PackRow + 0] = (tmp.x | (bit2.a << 4) | (bit1.a << 6)) - FullRange - zptr[j + 0];
          dstptr[i * col + j * PackRow + 1] = (tmp.y | (bit2.b << 4) | (bit1.b << 6)) - FullRange - zptr[j + 1];
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 1];
          dstptr[i * col + j * PackRow + 2] = (tmp.x | (bit2.c << 4) | (bit1.c << 6)) - FullRange - zptr[j + 2];
          dstptr[i * col + j * PackRow + 3] = (tmp.y | (bit2.d << 4) | (bit1.d << 6)) - FullRange - zptr[j + 3];
          bit2 = bit2ptr[(i * col + j * PackRow) / 4 + 1];
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 2];
          dstptr[i * col + j * PackRow + 4] = (tmp.x | (bit2.a << 4) | (bit1.e << 6)) - FullRange - zptr[j + 4];
          dstptr[i * col + j * PackRow + 5] = (tmp.y | (bit2.b << 4) | (bit1.f << 6)) - FullRange - zptr[j + 5];
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 3];
          dstptr[i * col + j * PackRow + 6] = (tmp.x | (bit2.c << 4) | (bit1.g << 6)) - FullRange - zptr[j + 6];
          dstptr[i * col + j * PackRow + 7] = (tmp.y | (bit2.d << 4) | (bit1.h << 6)) - FullRange - zptr[j + 7];
        }
      }
    } else if constexpr (PackRow == 2) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 4) {
          auto bit2 = bit2ptr[(i * col + j * PackRow) / 4];
          auto bit1 = bit1ptr[(i * col + j * PackRow) / 8];
          auto tmp = bit4ptr[(i * col + j * PackRow) / 2];
          auto zp = zptr[j] + FullRange;
          dstptr[i * col + j * PackRow + 0] = (tmp.x | (bit2.a << 4) | (bit1.a << 6)) - zp;
          dstptr[i * col + j * PackRow + 1] = (tmp.y | (bit2.b << 4) | (bit1.b << 6)) - zp;
          zp = zptr[j + 1] + FullRange;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 1];
          dstptr[i * col + j * PackRow + 2] = (tmp.x | (bit2.c << 4) | (bit1.c << 6)) - zp;
          dstptr[i * col + j * PackRow + 3] = (tmp.y | (bit2.d << 4) | (bit1.d << 6)) - zp;
          bit2 = bit2ptr[(i * col + j * PackRow) / 4 + 1];
          zp = zptr[j + 2] + FullRange;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 2];
          dstptr[i * col + j * PackRow + 4] = (tmp.x | (bit2.a << 4) | (bit1.e << 6)) - zp;
          dstptr[i * col + j * PackRow + 5] = (tmp.y | (bit2.b << 4) | (bit1.f << 6)) - zp;
          zp = zptr[j + 3] + FullRange;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 3];
          dstptr[i * col + j * PackRow + 6] = (tmp.x | (bit2.c << 4) | (bit1.g << 6)) - zp;
          dstptr[i * col + j * PackRow + 7] = (tmp.y | (bit2.d << 4) | (bit1.h << 6)) - zp;
        }
      }
    } else {
      static_assert(PackRow == 1 || PackRow == 2 || PackRow == 4);
    }
  } else {
    return decompress_s7_s8(bit4ptr, bit2ptr, bit1ptr, dstptr, size_t(row) * col, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s6_s8(utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, int8_t* zpptr,
                                                int8_t* dstptr, int blocksize, int ldzp, int n_offset, int k_offset,
                                                int row, int col, int8_t* tmp, size_t tmpsize) {
  int constexpr FullRange = 1 << (6 - 1);
  static_assert(NTILE % 4 == 0);
  assert(((col * PackRow) % 4) == 0);
  if (zpptr) {
    if constexpr (PackRow == 4) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 1) {
          auto zp = zptr[j] + FullRange;
          auto bit2 = bit2ptr[(i * col + j * PackRow) / 4];
          auto tmp = bit4ptr[(i * col + j * PackRow) / 2];
          dstptr[i * col + j * PackRow + 0] = (tmp.x | (bit2.a << 4)) - zp;
          dstptr[i * col + j * PackRow + 1] = (tmp.y | (bit2.b << 4)) - zp;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 1];
          dstptr[i * col + j * PackRow + 2] = (tmp.x | (bit2.c << 4)) - zp;
          dstptr[i * col + j * PackRow + 3] = (tmp.y | (bit2.d << 4)) - zp;
        }
      }
    } else if constexpr (PackRow == 1) {
      for (int i = 0; i < row; i += 1) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 4) {
          auto bit2 = bit2ptr[(i * col + j * PackRow) / 4];
          auto tmp = bit4ptr[(i * col + j * PackRow) / 2];
          dstptr[i * col + j * PackRow + 0] = (tmp.x | (bit2.a << 4)) - FullRange - zptr[j + 0];
          dstptr[i * col + j * PackRow + 1] = (tmp.y | (bit2.b << 4)) - FullRange - zptr[j + 1];
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 1];
          dstptr[i * col + j * PackRow + 2] = (tmp.x | (bit2.c << 4)) - FullRange - zptr[j + 2];
          dstptr[i * col + j * PackRow + 3] = (tmp.y | (bit2.d << 4)) - FullRange - zptr[j + 3];
        }
      }
    } else if constexpr (PackRow == 2) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 2) {
          auto bit2 = bit2ptr[(i * col + j * PackRow) / 4];
          auto tmp = bit4ptr[(i * col + j * PackRow) / 2];
          auto zp = zptr[j] + FullRange;
          dstptr[i * col + j * PackRow + 0] = (tmp.x | (bit2.a << 4)) - zp;
          dstptr[i * col + j * PackRow + 1] = (tmp.y | (bit2.b << 4)) - zp;
          zp = zptr[j + 1] + FullRange;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 1];
          dstptr[i * col + j * PackRow + 2] = (tmp.x | (bit2.c << 4)) - zp;
          dstptr[i * col + j * PackRow + 3] = (tmp.y | (bit2.d << 4)) - zp;
        }
      }
    } else {
      static_assert(PackRow == 1 || PackRow == 2 || PackRow == 4);
    }
  } else {
    return decompress_s6_s8(bit4ptr, bit2ptr, dstptr, size_t(row) * col, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s5_s8(utils::bit4x2* bit4ptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                int8_t* dstptr, int blocksize, int ldzp, int n_offset, int k_offset,
                                                int row, int col, int8_t* tmp, size_t tmpsize) {
  int constexpr FullRange = 1 << (5 - 1);
  static_assert(NTILE % 8 == 0);
  assert(((col * PackRow) % 8) == 0);
  if (zpptr) {
    if constexpr (PackRow == 4) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 2) {
          auto zp = zptr[j] + FullRange;
          auto bit1 = bit1ptr[(i * col + j * PackRow) / 8];
          auto tmp = bit4ptr[(i * col + j * PackRow) / 2];
          dstptr[i * col + j * PackRow + 0] = (tmp.x | (bit1.a << 4)) - zp;
          dstptr[i * col + j * PackRow + 1] = (tmp.y | (bit1.b << 4)) - zp;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 1];
          dstptr[i * col + j * PackRow + 2] = (tmp.x | (bit1.c << 4)) - zp;
          dstptr[i * col + j * PackRow + 3] = (tmp.y | (bit1.d << 4)) - zp;
          zp = zptr[j + 1] + FullRange;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 2];
          dstptr[i * col + j * PackRow + 4] = (tmp.x | (bit1.e << 4)) - zp;
          dstptr[i * col + j * PackRow + 5] = (tmp.y | (bit1.f << 4)) - zp;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 3];
          dstptr[i * col + j * PackRow + 6] = (tmp.x | (bit1.g << 4)) - zp;
          dstptr[i * col + j * PackRow + 7] = (tmp.y | (bit1.h << 4)) - zp;
        }
      }
    } else if constexpr (PackRow == 1) {
      for (int i = 0; i < row; i += 1) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 8) {
          auto bit1 = bit1ptr[(i * col + j * PackRow) / 8];
          auto tmp = bit4ptr[(i * col + j * PackRow) / 2];
          dstptr[i * col + j * PackRow + 0] = (tmp.x | (bit1.a << 4)) - FullRange - zptr[j + 0];
          dstptr[i * col + j * PackRow + 1] = (tmp.y | (bit1.b << 4)) - FullRange - zptr[j + 1];
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 1];
          dstptr[i * col + j * PackRow + 2] = (tmp.x | (bit1.c << 4)) - FullRange - zptr[j + 2];
          dstptr[i * col + j * PackRow + 3] = (tmp.y | (bit1.d << 4)) - FullRange - zptr[j + 3];
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 2];
          dstptr[i * col + j * PackRow + 4] = (tmp.x | (bit1.e << 4)) - FullRange - zptr[j + 4];
          dstptr[i * col + j * PackRow + 5] = (tmp.y | (bit1.f << 4)) - FullRange - zptr[j + 5];
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 3];
          dstptr[i * col + j * PackRow + 6] = (tmp.x | (bit1.g << 4)) - FullRange - zptr[j + 6];
          dstptr[i * col + j * PackRow + 7] = (tmp.y | (bit1.h << 4)) - FullRange - zptr[j + 7];
        }
      }
    } else if constexpr (PackRow == 2) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 4) {
          auto bit1 = bit1ptr[(i * col + j * PackRow) / 8];
          auto tmp = bit4ptr[(i * col + j * PackRow) / 2];
          auto zp = zptr[j] + FullRange;
          dstptr[i * col + j * PackRow + 0] = (tmp.x | (bit1.a << 4)) - zp;
          dstptr[i * col + j * PackRow + 1] = (tmp.y | (bit1.b << 4)) - zp;
          zp = zptr[j + 1] + FullRange;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 1];
          dstptr[i * col + j * PackRow + 2] = (tmp.x | (bit1.c << 4)) - zp;
          dstptr[i * col + j * PackRow + 3] = (tmp.y | (bit1.d << 4)) - zp;
          zp = zptr[j + 2] + FullRange;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 2];
          dstptr[i * col + j * PackRow + 4] = (tmp.x | (bit1.e << 4)) - zp;
          dstptr[i * col + j * PackRow + 5] = (tmp.y | (bit1.f << 4)) - zp;
          zp = zptr[j + 3] + FullRange;
          tmp = bit4ptr[(i * col + j * PackRow) / 2 + 3];
          dstptr[i * col + j * PackRow + 6] = (tmp.x | (bit1.g << 4)) - zp;
          dstptr[i * col + j * PackRow + 7] = (tmp.y | (bit1.h << 4)) - zp;
        }
      }
    } else {
      static_assert(PackRow == 1 || PackRow == 2 || PackRow == 4);
    }
  } else {
    return decompress_s5_s8(bit4ptr, bit1ptr, dstptr, size_t(row) * col, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s4_s8(utils::int4x2* srcptr, int8_t* zpptr, int8_t* dstptr, int blocksize,
                                                int ldzp, int n_offset, int k_offset, int row, int col, int8_t* tmp,
                                                size_t tmpsize) {
  int8_t constexpr FullRange = 1 << (4 - 1);
  if (zpptr) {
    if constexpr (PackRow == 4 || PackRow == 2) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 1) {
          auto zp = zptr[j] + FullRange;
          for (int ir = 0; ir < PackRow; ir += 2) {
            auto tmp = srcptr[i * col / 2 + j * PackRow / 2 + ir / 2];
            dstptr[i * col + j * PackRow + ir + 0] = tmp.x - zp;
            dstptr[i * col + j * PackRow + ir + 1] = tmp.y - zp;
          }
        }
      }
    } else if constexpr (PackRow == 1) {
      for (int i = 0; i < row; i += 1) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 2) {
          auto tmp = srcptr[i * col / 2 + j / 2];
          dstptr[i * col + j + 0] = tmp.x - FullRange - zptr[j + 0];
          dstptr[i * col + j + 1] = tmp.y - FullRange - zptr[j + 1];
        }
      }
    } else {
      static_assert(PackRow == 1 || PackRow == 2 || PackRow == 4);
    }
  } else {
    return decompress_s4_s8(srcptr, dstptr, size_t(row) * col, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s3_s8(utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                int8_t* dstptr, int blocksize, int ldzp, int n_offset, int k_offset,
                                                int row, int col, int8_t* tmp, size_t tmpsize) {
  static_assert(NTILE % 8 == 0);
  int8_t constexpr FullRange = 1 << (3 - 1);
  assert(((col * PackRow) % 8) == 0);
  if (zpptr) {
    if constexpr (PackRow == 4) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 2) {
          auto zp = zptr[j] + FullRange;
          auto bit1 = bit1ptr[(i * col + j * PackRow) / 8];
          auto tmp = bit2ptr[(i * col + j * PackRow) / 4];
          dstptr[i * col + j * PackRow + 0] = (tmp.a | (bit1.a << 2)) - zp;
          dstptr[i * col + j * PackRow + 1] = (tmp.b | (bit1.b << 2)) - zp;
          dstptr[i * col + j * PackRow + 2] = (tmp.c | (bit1.c << 2)) - zp;
          dstptr[i * col + j * PackRow + 3] = (tmp.d | (bit1.d << 2)) - zp;
          zp = zptr[j + 1] + FullRange;
          tmp = bit2ptr[(i * col + j * PackRow) / 4 + 1];
          dstptr[i * col + j * PackRow + 4] = (tmp.a | (bit1.e << 2)) - zp;
          dstptr[i * col + j * PackRow + 5] = (tmp.b | (bit1.f << 2)) - zp;
          dstptr[i * col + j * PackRow + 6] = (tmp.c | (bit1.g << 2)) - zp;
          dstptr[i * col + j * PackRow + 7] = (tmp.d | (bit1.h << 2)) - zp;
        }
      }
    } else if constexpr (PackRow == 1) {
      for (int i = 0; i < row; i += 1) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 8) {
          auto bit1 = bit1ptr[(i * col + j * PackRow) / 8];
          auto tmp = bit2ptr[(i * col + j * PackRow) / 4];
          dstptr[i * col + j * PackRow + 0] = (tmp.a | (bit1.a << 2)) - FullRange - zptr[j + 0];
          dstptr[i * col + j * PackRow + 1] = (tmp.b | (bit1.b << 2)) - FullRange - zptr[j + 1];
          dstptr[i * col + j * PackRow + 2] = (tmp.c | (bit1.c << 2)) - FullRange - zptr[j + 2];
          dstptr[i * col + j * PackRow + 3] = (tmp.d | (bit1.d << 2)) - FullRange - zptr[j + 3];
          tmp = bit2ptr[(i * col + j * PackRow) / 4 + 1];
          dstptr[i * col + j * PackRow + 4] = (tmp.a | (bit1.e << 2)) - FullRange - zptr[j + 4];
          dstptr[i * col + j * PackRow + 5] = (tmp.b | (bit1.f << 2)) - FullRange - zptr[j + 5];
          dstptr[i * col + j * PackRow + 6] = (tmp.c | (bit1.g << 2)) - FullRange - zptr[j + 6];
          dstptr[i * col + j * PackRow + 7] = (tmp.d | (bit1.h << 2)) - FullRange - zptr[j + 7];
        }
      }
    } else if constexpr (PackRow == 2) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 4) {
          auto bit1 = bit1ptr[(i * col + j * PackRow) / 8];
          auto tmp = bit2ptr[(i * col + j * PackRow) / 4];
          auto zp = zptr[j] + FullRange;
          dstptr[i * col + j * PackRow + 0] = (tmp.a | (bit1.a << 2)) - zp;
          dstptr[i * col + j * PackRow + 1] = (tmp.b | (bit1.b << 2)) - zp;
          zp = zptr[j + 1] + FullRange;
          dstptr[i * col + j * PackRow + 2] = (tmp.c | (bit1.c << 2)) - zp;
          dstptr[i * col + j * PackRow + 3] = (tmp.d | (bit1.d << 2)) - zp;
          tmp = bit2ptr[(i * col + j * PackRow) / 4 + 1];
          zp = zptr[j + 2] + FullRange;
          dstptr[i * col + j * PackRow + 4] = (tmp.a | (bit1.e << 2)) - zp;
          dstptr[i * col + j * PackRow + 5] = (tmp.b | (bit1.f << 2)) - zp;
          zp = zptr[j + 3] + FullRange;
          dstptr[i * col + j * PackRow + 6] = (tmp.c | (bit1.g << 2)) - zp;
          dstptr[i * col + j * PackRow + 7] = (tmp.d | (bit1.h << 2)) - zp;
        }
      }
    } else {
      static_assert(PackRow == 1 || PackRow == 2 || PackRow == 4);
    }
  } else {
    return decompress_s3_s8(bit2ptr, bit1ptr, dstptr, size_t(row) * col, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s2_s8(utils::bit2x4* bit2ptr, int8_t* zpptr, int8_t* dstptr, int blocksize,
                                                int ldzp, int n_offset, int k_offset, int row, int col, int8_t* tmp,
                                                size_t tmpsize) {
  static_assert(NTILE % 4 == 0);
  int8_t constexpr FullRange = 1 << (2 - 1);
  assert(((col * PackRow) % 4) == 0);
  if (zpptr) {
    if constexpr (PackRow == 4) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 1) {
          auto zp = zptr[j] + FullRange;
          auto tmp = bit2ptr[(i * col + j * PackRow) / 4];
          dstptr[i * col + j * PackRow + 0] = (tmp.a) - zp;
          dstptr[i * col + j * PackRow + 1] = (tmp.b) - zp;
          dstptr[i * col + j * PackRow + 2] = (tmp.c) - zp;
          dstptr[i * col + j * PackRow + 3] = (tmp.d) - zp;
        }
      }
    } else if constexpr (PackRow == 1) {
      for (int i = 0; i < row; i += 1) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 4) {
          auto tmp = bit2ptr[(i * col + j * PackRow) / 4];
          dstptr[i * col + j * PackRow + 0] = (tmp.a) - FullRange - zptr[j + 0];
          dstptr[i * col + j * PackRow + 1] = (tmp.b) - FullRange - zptr[j + 1];
          dstptr[i * col + j * PackRow + 2] = (tmp.c) - FullRange - zptr[j + 2];
          dstptr[i * col + j * PackRow + 3] = (tmp.d) - FullRange - zptr[j + 3];
        }
      }
    } else if constexpr (PackRow == 2) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 2) {
          auto tmp = bit2ptr[(i * col + j * PackRow) / 4];
          auto zp = zptr[j] + FullRange;
          dstptr[i * col + j * PackRow + 0] = (tmp.a) - zp;
          dstptr[i * col + j * PackRow + 1] = (tmp.b) - zp;
          zp = zptr[j + 1] + FullRange;
          dstptr[i * col + j * PackRow + 2] = (tmp.c) - zp;
          dstptr[i * col + j * PackRow + 3] = (tmp.d) - zp;
        }
      }
    } else {
      static_assert(PackRow == 1 || PackRow == 2 || PackRow == 4);
    }
  } else {
    return decompress_s2_s8(bit2ptr, dstptr, size_t(row) * col, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s1_s8(utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr, int blocksize,
                                                int ldzp, int n_offset, int k_offset, int row, int col, int8_t* tmp,
                                                size_t tmpsize) {
  static_assert(NTILE % 8 == 0);
  int8_t constexpr FullRange = 1 << (1 - 1);
  assert(((col * PackRow) % 8) == 0);
  if (zpptr) {
    if constexpr (PackRow == 4) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 2) {
          auto zp = zptr[j] + FullRange;
          auto bit1 = bit1ptr[(i * col + j * PackRow) / 8];
          dstptr[i * col + j * PackRow + 0] = bit1.a - zp;
          dstptr[i * col + j * PackRow + 1] = bit1.b - zp;
          dstptr[i * col + j * PackRow + 2] = bit1.c - zp;
          dstptr[i * col + j * PackRow + 3] = bit1.d - zp;
          zp = zptr[j + 1] + FullRange;
          dstptr[i * col + j * PackRow + 4] = bit1.e - zp;
          dstptr[i * col + j * PackRow + 5] = bit1.f - zp;
          dstptr[i * col + j * PackRow + 6] = bit1.g - zp;
          dstptr[i * col + j * PackRow + 7] = bit1.h - zp;
        }
      }
    } else if constexpr (PackRow == 1) {
      for (int i = 0; i < row; i += 1) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 8) {
          auto bit1 = bit1ptr[(i * col + j * PackRow) / 8];
          dstptr[i * col + j * PackRow + 0] = bit1.a - FullRange - zptr[j + 0];
          dstptr[i * col + j * PackRow + 1] = bit1.b - FullRange - zptr[j + 1];
          dstptr[i * col + j * PackRow + 2] = bit1.c - FullRange - zptr[j + 2];
          dstptr[i * col + j * PackRow + 3] = bit1.d - FullRange - zptr[j + 3];
          dstptr[i * col + j * PackRow + 4] = bit1.e - FullRange - zptr[j + 4];
          dstptr[i * col + j * PackRow + 5] = bit1.f - FullRange - zptr[j + 5];
          dstptr[i * col + j * PackRow + 6] = bit1.g - FullRange - zptr[j + 6];
          dstptr[i * col + j * PackRow + 7] = bit1.h - FullRange - zptr[j + 7];
        }
      }
    } else if constexpr (PackRow == 2) {
      for (int i = 0; i < row; i += PackRow) {
        auto zptr = zpptr + (i + k_offset) / blocksize * ldzp + n_offset;
        for (int j = 0; j < col; j += 4) {
          auto bit1 = bit1ptr[(i * col + j * PackRow) / 8];
          auto zp = zptr[j] + FullRange;
          dstptr[i * col + j * PackRow + 0] = bit1.a - zp;
          dstptr[i * col + j * PackRow + 1] = bit1.b - zp;
          zp = zptr[j + 1] + FullRange;
          dstptr[i * col + j * PackRow + 2] = bit1.c - zp;
          dstptr[i * col + j * PackRow + 3] = bit1.d - zp;
          zp = zptr[j + 2] + FullRange;
          dstptr[i * col + j * PackRow + 4] = bit1.e - zp;
          dstptr[i * col + j * PackRow + 5] = bit1.f - zp;
          zp = zptr[j + 3] + FullRange;
          dstptr[i * col + j * PackRow + 6] = bit1.g - zp;
          dstptr[i * col + j * PackRow + 7] = bit1.h - zp;
        }
      }
    } else {
      static_assert(PackRow == 1 || PackRow == 2 || PackRow == 4);
    }
  } else {
    return decompress_s1_s8(bit1ptr, dstptr, size_t(row) * col, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

static inline float f8_to_fp32(utils::f8 v, BTLA_DTYPE f8_t) {
  uint32_t sign_revert = v.x;
  uint32_t e_revert = v.x;
  uint32_t mantissa_revert = v.x;
  sign_revert <<= 24;
  sign_revert &= 0x80000000;
  auto ebits = utils::bestla_dtype_get_f8_ebits(f8_t);
  auto mantissabit = 7 - ebits;
  e_revert &= 0x7f;
  e_revert >>= mantissabit;
  e_revert = e_revert - static_cast<uint32_t>(std::pow(2, ebits - 1)) + 1 + 127;
  e_revert <<= 23;
  mantissa_revert <<= (23 - mantissabit);
  mantissa_revert &= 0x007fffff;
  uint32_t revert = sign_revert | e_revert | mantissa_revert;
  float* fp_v = reinterpret_cast<float*>(&revert);
  return *fp_v;
}

template <typename _DST_T, int _PACK_ROW, typename _S_T>
static inline BTLA_CODE decompress_kblock_f8_fp(utils::f8* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                int ld_dst, _S_T* scales, int k_offset, int kblock, int NPad,
                                                BTLA_DTYPE src_f8_type) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j++) {
      auto fp_v = f8_to_fp32(srcptr[i * ld_src + j], src_f8_type);
      float scale;
      if constexpr (std::is_same_v<_S_T, utils::f8>) {
        int shared_exp = sptr[j / _PACK_ROW].x;
        scale = static_cast<float>(std::pow(2, shared_exp));
      } else if constexpr (std::is_same_v<_S_T, float>) {
        scale = scales[j / _PACK_ROW];
      } else {
        assert(0);
      }
      dstptr[i * ld_dst + j] = fp_v * scale;
    }
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s8_fp(int8_t* srcptr, DST_T* dstptr, int row, int col, void* scales_,
                                         BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                         int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  for (int i = 0; i < row; i += PackRow) {
    int kpos = (k_offset + i) / blocksize * ldzp + n_offset;
    if (sdtype == BTLA_DTYPE::F32) {
      auto sptr = (float*)scales_ + kpos;
      for (int j = 0; j < col; j += 1) {
        auto scale = float(sptr[j]);
        auto zp = zero_points ? zero_points[kpos + j] : 0;
        for (int ir = 0; ir < PackRow; ir++) {
          float tmp = static_cast<float>(srcptr[i * col + j * PackRow + ir] - zp) * scale;
          dstptr[i * col + j * PackRow + ir] = static_cast<DST_T>(tmp);
        }
      }
    } else if (sdtype == BTLA_DTYPE::BF16) {
      auto sptr = (utils::bf16*)scales_ + kpos;
      for (int j = 0; j < col; j += 1) {
        auto scale = float(sptr[j]);
        auto zp = zero_points ? zero_points[kpos + j] : 0;
        for (int ir = 0; ir < PackRow; ir++) {
          float tmp = static_cast<float>(srcptr[i * col + j * PackRow + ir] - zp) * scale;
          dstptr[i * col + j * PackRow + ir] = static_cast<DST_T>(tmp);
        }
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE, typename DST_T>
static inline BTLA_CODE decompress_kblock_s7_fp(utils::bit4x2* b4ptr, utils::bit2x4* b2ptr, utils::bit1x8* b1ptr,
                                                DST_T* dstptr, int row, int col, void* scales_, BTLA_DTYPE sdtype,
                                                int8_t* zero_points, int k_offset, int n_offset, int blocksize,
                                                int ldzp, int8_t* tmp, size_t tmpsize) {
  assert(tmpsize >= PackRow * NTILE);
  assert(NTILE == col);
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  decompress_kblock_s7_s8<PackRow, NTILE>(b4ptr, b2ptr, b1ptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset,
                                          k_offset, row, col, tmp, tmpsize);
  decompress_kblock_s8_fp<PackRow, NTILE>(tmps8ptr, dstptr, row, col, scales_, sdtype, nullptr, k_offset, n_offset,
                                          blocksize, ldzp, tmp, tmpsize);
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE, typename DST_T>
static inline BTLA_CODE decompress_kblock_s6_fp(utils::bit4x2* b4ptr, utils::bit2x4* b2ptr, DST_T* dstptr, int row,
                                                int col, void* scales_, BTLA_DTYPE sdtype, int8_t* zero_points,
                                                int k_offset, int n_offset, int blocksize, int ldzp, int8_t* tmp,
                                                size_t tmpsize) {
  assert(tmpsize >= PackRow * NTILE);
  assert(NTILE == col);
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  decompress_kblock_s6_s8<PackRow, NTILE>(b4ptr, b2ptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset, k_offset, row,
                                          col, tmp, tmpsize);
  decompress_kblock_s8_fp<PackRow, NTILE>(tmps8ptr, dstptr, row, col, scales_, sdtype, nullptr, k_offset, n_offset,
                                          blocksize, ldzp, tmp, tmpsize);
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE, typename DST_T>
static inline BTLA_CODE decompress_kblock_s5_fp(utils::bit4x2* b4ptr, utils::bit1x8* b1ptr, DST_T* dstptr, int row,
                                                int col, void* scales_, BTLA_DTYPE sdtype, int8_t* zero_points,
                                                int k_offset, int n_offset, int blocksize, int ldzp, int8_t* tmp,
                                                size_t tmpsize) {
  assert(tmpsize >= PackRow * NTILE);
  assert(NTILE == col);
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  decompress_kblock_s5_s8<PackRow, NTILE>(b4ptr, b1ptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset, k_offset, row,
                                          col, tmp, tmpsize);
  decompress_kblock_s8_fp<PackRow, NTILE>(tmps8ptr, dstptr, row, col, scales_, sdtype, nullptr, k_offset, n_offset,
                                          blocksize, ldzp, tmp, tmpsize);
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE, typename DST_T>
static inline BTLA_CODE decompress_kblock_s4_fp(utils::int4x2* srcptr, DST_T* dstptr, int row, int col, void* scales_,
                                                BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                                int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  assert(tmpsize >= PackRow * NTILE);
  assert(NTILE == col);
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  decompress_kblock_s4_s8<PackRow, NTILE>(srcptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset, k_offset, row, col,
                                          tmp, tmpsize);
  decompress_kblock_s8_fp<PackRow, NTILE>(tmps8ptr, dstptr, row, col, scales_, sdtype, nullptr, k_offset, n_offset,
                                          blocksize, ldzp, tmp, tmpsize);
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE, typename DST_T>
static inline BTLA_CODE decompress_kblock_s3_fp(utils::bit2x4* b2ptr, utils::bit1x8* b1ptr, DST_T* dstptr, int row,
                                                int col, void* scales_, BTLA_DTYPE sdtype, int8_t* zero_points,
                                                int k_offset, int n_offset, int blocksize, int ldzp, int8_t* tmp,
                                                size_t tmpsize) {
  assert(tmpsize >= PackRow * NTILE);
  assert(NTILE == col);
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  decompress_kblock_s3_s8<PackRow, NTILE>(b2ptr, b1ptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset, k_offset, row,
                                          col, tmp, tmpsize);
  decompress_kblock_s8_fp<PackRow, NTILE>(tmps8ptr, dstptr, row, col, scales_, sdtype, nullptr, k_offset, n_offset,
                                          blocksize, ldzp, tmp, tmpsize);
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE, typename DST_T>
static inline BTLA_CODE decompress_kblock_s2_fp(utils::bit2x4* b2ptr, DST_T* dstptr, int row, int col, void* scales_,
                                                BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                                int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  assert(tmpsize >= PackRow * NTILE);
  assert(NTILE == col);
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  decompress_kblock_s2_s8<PackRow, NTILE>(b2ptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset, k_offset, row, col,
                                          tmp, tmpsize);
  decompress_kblock_s8_fp<PackRow, NTILE>(tmps8ptr, dstptr, row, col, scales_, sdtype, nullptr, k_offset, n_offset,
                                          blocksize, ldzp, tmp, tmpsize);
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE, typename DST_T>
static inline BTLA_CODE decompress_kblock_s1_fp(utils::bit1x8* b1ptr, DST_T* dstptr, int row, int col, void* scales_,
                                                BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                                int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  assert(tmpsize >= PackRow * NTILE);
  assert(NTILE == col);
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  decompress_kblock_s1_s8<PackRow, NTILE>(b1ptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset, k_offset, row, col,
                                          tmp, tmpsize);
  decompress_kblock_s8_fp<PackRow, NTILE>(tmps8ptr, dstptr, row, col, scales_, sdtype, nullptr, k_offset, n_offset,
                                          blocksize, ldzp, tmp, tmpsize);
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE S4_T, typename _DST_T, int _PACK_ROW>
static inline BTLA_CODE decompress_dq_kblock_s4_fp(utils::int4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                   int ld_dst, uint8_t* scales, float* dq_scale, int k_offset,
                                                   int n_offset, int kblock, int dq_blk, int dq_offset_idx, int NPad,
                                                   int N, void* tmp, size_t tmpsize) {
  auto sptr_base = scales + n_offset;
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = sptr_base + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      float scale0, scale1, dst0, dst1;
      int s0_idx, s1_idx;
      s0_idx = j / _PACK_ROW;
      s1_idx = (j + 1) / _PACK_ROW;
      auto dq_s0_idx = (n_offset + kpos * N + s0_idx) / dq_blk;
      auto dq_s1_idx = (n_offset + kpos * N + s1_idx) / dq_blk;
      scale0 = dq8_bnb_LUT[sptr[s0_idx]] * dq_scale[dq_s0_idx] + dq_scale[dq_offset_idx];
      scale1 = dq8_bnb_LUT[sptr[s1_idx]] * dq_scale[dq_s1_idx] + dq_scale[dq_offset_idx];
      dst0 = static_cast<float>(tmp.x - 8) * scale0;
      dst1 = static_cast<float>(tmp.y - 8) * scale1;
      dstptr[i * ld_dst + j + 0] = static_cast<_DST_T>(dst0);
      dstptr[i * ld_dst + j + 1] = static_cast<_DST_T>(dst1);
    }
  }
  return BTLA_CODE::Success;
}

static inline float fp4_bnb_unpack(uint8_t val) {
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if ((val & 0b0100) == 4)          // 0
    if ((val & 0b0010) == 2)        // 01
      if ((val & 0b0001) == 1)      // 111
        return 0.25000000f * sign;  // 1111
      else
        return 0.16666667f * sign;  // 1110
    else if ((val & 0b0001) == 1)   // 110
      return 0.50000000f * sign;    // 1101
    else
      return 0.33333333f * sign;  // 1100
  else if ((val & 0b0010) == 2)   // 10
    if ((val & 0b0001) == 1)      // 101
      return 1.00000000f * sign;  // 1011
    else
      return 0.66666667f * sign;     // 1010
  else if ((val & 0b0001) == 1)      // 100
    return 5.208333333e-03f * sign;  // 1001
  else
    return 0.00000000f * sign;  // 1000
}

static inline float fp4_bnb_dequantize(uint8_t val, float absmax) { return fp4_bnb_unpack(val) * absmax; }

static inline int8_t fp4_bnb_quantize(float x) {
  int sign = x < 0 ? 0b1000 : 0b0000;
  x = fabs(x);
  if (x > 0.29166667f)
    if (x > 0.583333f)
      if (x > 0.8333333f)
        return static_cast<int8_t>(0b0011 + sign);
      else
        return static_cast<int8_t>(0b0010 + sign);
    else if (x > 0.4166667f)
      return static_cast<int8_t>(0b101 + sign);
    else
      return static_cast<int8_t>(0b100 + sign);
  else if (x > 0.0859375f)
    if (x > 0.20833333f)
      return static_cast<int8_t>(0b0111 + sign);
    else
      return static_cast<int8_t>(0b0110 + sign);
  else if (x > 0.00260417f)
    return static_cast<int8_t>(0b0001 + sign);
  else
    return static_cast<int8_t>(0b0000 + sign);
}

static inline int8_t fp4_e2m1_quantize(float x) {
  // FP4 with bias of 1
  // first bit is a sign
  // subnormals
  // 0b000 = 0
  // 0b001 = 0.0625
  // 0b010 = 1
  // 0b011 = 1.5
  // 0b100 = 2
  // 0b101 = 3
  // 0b110 = 4
  // 0b111 = 6

  int sign = x < 0 ? 0b1000 : 0b0000;
  x = fabs(x);
  if (x > 1.75f / 6) {
    if (x > 3.5f / 6) {
      if (x > 5.f / 6)
        return static_cast<int8_t>(0b111 + sign);  // 6
      else
        return static_cast<int8_t>(0b110 + sign);  // 4
    } else {
      if (x > 2.5f / 6)
        return static_cast<int8_t>(0b101 + sign);  // 3
      else
        return static_cast<int8_t>(0b100 + sign);  // 2
    }
  } else {
    if (x > 0.53125f / 6) {
      if (x > 1.25f / 6)
        return static_cast<int8_t>(0b011 + sign);  // 1.5
      else
        return static_cast<int8_t>(0b010 + sign);  // 1
    } else {
      if (x > 0.03125f / 6)
        return static_cast<int8_t>(0b0001 + sign);  // 0.0625
      else
        return static_cast<int8_t>(0b0000 + sign);  // 0
    }
  }
}

static inline float fp4_e2m1_unpack(uint8_t val) {
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if ((val & 0b0100) == 4)      // 0
    if ((val & 0b0010) == 2)    // 01
      if ((val & 0b0001) == 1)  // 111
        return 1.f * sign;      // 1111
      else
        return 0.6666666666666666f * sign;  // 1110
    else if ((val & 0b0001) == 1)           // 110
      return 0.5f * sign;                   // 1101
    else
      return 0.3333333333333333f * sign;  // 1100
  else if ((val & 0b0010) == 2)           // 10
    if ((val & 0b0001) == 1)              // 101
      return 0.25f * sign;                // 1011
    else
      return 0.16666666666666666f * sign;  // 1010
  else if ((val & 0b0001) == 1)            // 100
    return 0.010416666666666666f * sign;   // 1001
  else
    return 0.00000000f * sign;  // 1000
}

static inline float fp4_e2m1_dequantize(uint8_t val, float absmax) { return fp4_e2m1_unpack(val) * absmax; }

static inline float nf4_unpack(int8_t val) {
  if ((val & 0b1000) == 8)
    if ((val & 0b0100) == 4)      // 1
      if ((val & 0b0010) == 2)    // 11
        if ((val & 0b0001) == 1)  // 111
          return 1.0f;
        else
          return 0.7229568362236023f;
      else if ((val & 0b0001) == 1)  // 110
        return 0.5626170039176941f;
      else
        return 0.44070982933044434f;
    else if ((val & 0b0010) == 2)  // 10
      if ((val & 0b0001) == 1)     // 101
        return 0.33791524171829224f;
      else
        return 0.24611230194568634f;
    else if ((val & 0b0001) == 1)  // 100
      return 0.16093020141124725f;
    else
      return 0.07958029955625534f;

  else if ((val & 0b0100) == 4)  // 0
    if ((val & 0b0010) == 2)     // 01
      if ((val & 0b0001) == 1)   // 011
        return -1.f;
      else
        return -0.09105003625154495f;
    else if ((val & 0b0001) == 1)  // 010
      return -0.18477343022823334f;
    else
      return -0.28444138169288635f;
  else if ((val & 0b0010) == 2)  // 00
    if ((val & 0b0001) == 1)     // 001
      return -0.39491748809814453f;
    else
      return -0.5250730514526367f;
  else if ((val & 0b0001) == 1)  // 000
    return -0.6961928009986877f;
  else
    return 0.f;
}

static inline float nf4_dequantize(int8_t val, float absmax) { return nf4_unpack(val) * absmax; }

// Note: In the BNB Nf4 definition, 0 has a non-zero value after dequantization,
// but BTLA uses 0 for padding, which leads to calculation errors. We ultimately
// choose to swap the binary bits of -1 and 0 in Nf4 to avoid this conflict.
static inline int8_t nf4_quantize(float x) {
  if (x > 0.03979014977812767f)
    if (x > 0.3893125355243683f)      // 1
      if (x > 0.6427869200706482f)    // 11
        if (x > 0.8614784181118011f)  // 111
          return 0b1111;
        else
          return 0b1110;
      else if (x > 0.5016634166240692f)  // 110
        return 0b1101;
      else
        return 0b1100;
    else if (x > 0.2035212516784668f)  // 10
      if (x > 0.2920137718319893f)     // 101
        return 0b1011;
      else
        return 0b1010;
    else if (x > 0.1202552504837513f)  // 100
      return 0b1001;
    else
      return 0b1000;
  else if (x > -0.33967943489551544f)  // 0
    if (x > -0.13791173323988914f)     // 01
      if (x > -0.045525018125772476f)  // 011
        return 0b0000;
      else
        return 0b0110;
    else if (x > -0.23460740596055984f)  // 010
      return 0b0101;
    else
      return 0b0100;
  else if (x > -0.6106329262256622f)  // 00
    if (x > -0.4599952697753906f)     // 001
      return 0b0011;
    else
      return 0b0010;
  else if (x > -0.8480964004993439f)  // 000
    return 0b0001;
  else
    return 0b0111;
}

template <BTLA_DTYPE F4_T>
static inline float f4_unpack(int8_t v) {
  static_assert(F4_T == BTLA_DTYPE::F4_BNB || F4_T == BTLA_DTYPE::F4_NF4 || F4_T == BTLA_DTYPE::F4_E2M1,
                "Unsupported F4 type");
  switch (F4_T) {
    case BTLA_DTYPE::F4_BNB:
      return fp4_bnb_unpack(v);
    case BTLA_DTYPE::F4_NF4:
      return nf4_unpack(v);
    case BTLA_DTYPE::F4_E2M1:
      return fp4_e2m1_unpack(v);
    default:
      break;
  }
  return std::numeric_limits<float>::quiet_NaN();
}

template <BTLA_DTYPE F4_T>
static inline float f4_dequantize(int8_t v, float scale) {
  static_assert(F4_T == BTLA_DTYPE::F4_BNB || F4_T == BTLA_DTYPE::F4_NF4 || F4_T == BTLA_DTYPE::F4_E2M1,
                "Unsupported F4 type");
  return f4_unpack<F4_T>(v) * scale;
}

template <BTLA_DTYPE F4_T>
static inline int8_t f4_quantize(float x) {
  static_assert(F4_T == BTLA_DTYPE::F4_BNB || F4_T == BTLA_DTYPE::F4_NF4 || F4_T == BTLA_DTYPE::F4_E2M1,
                "Unsupported F4 type");
  switch (F4_T) {
    case BTLA_DTYPE::F4_BNB:
      return fp4_bnb_quantize(x);
    case BTLA_DTYPE::F4_NF4:
      return nf4_quantize(x);
    case BTLA_DTYPE::F4_E2M1:
      return fp4_e2m1_quantize(x);
    default:
      break;
  }
  return static_cast<int8_t>(0);
}

template <BTLA_DTYPE F4_T, typename _DST_T, int _PACK_ROW, typename _S_T>
static inline BTLA_CODE decompress_kblock_f4_fp(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                int ld_dst, _S_T* scales, int k_offset, int kblock, int NPad,
                                                int8_t* tmp, size_t tmpsize) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      float scale0, scale1, dst0, dst1;
      int s0_idx, s1_idx;
      s0_idx = j / _PACK_ROW;
      s1_idx = (j + 1) / _PACK_ROW;
      scale0 = static_cast<float>(sptr[s0_idx]);
      scale1 = static_cast<float>(sptr[s1_idx]);
      dst0 = f4_dequantize<F4_T>(tmp.x, scale0);
      dst1 = f4_dequantize<F4_T>(tmp.y, scale1);
      dstptr[i * ld_dst + j + 0] = static_cast<_DST_T>(dst0);
      dstptr[i * ld_dst + j + 1] = static_cast<_DST_T>(dst1);
    }
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE F4_T, int _PACK_ROW, typename _DST_T, typename SCA_T>
static inline BTLA_CODE decompress_dq_kblock_f4_fp(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                   int ld_dst, SCA_T* scales, float* dq_scale, int k_offset,
                                                   int n_offset, int kblock, int dq_blk, int dq_offset_idx, int NPad,
                                                   int N, void* tmp, size_t tmpsize) {
  auto sptr_base = scales + n_offset;
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = sptr_base + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      float scale0, scale1, dst0, dst1;
      int s0_idx, s1_idx;
      s0_idx = j / _PACK_ROW;
      s1_idx = (j + 1) / _PACK_ROW;
      auto dq_s0_idx = (n_offset + kpos * N + s0_idx) / dq_blk;
      auto dq_s1_idx = (n_offset + kpos * N + s1_idx) / dq_blk;
      scale0 = dq8_bnb_LUT[sptr[s0_idx]] * dq_scale[dq_s0_idx] + dq_scale[dq_offset_idx];
      scale1 = dq8_bnb_LUT[sptr[s1_idx]] * dq_scale[dq_s1_idx] + dq_scale[dq_offset_idx];
      dst0 = ref::f4_dequantize<F4_T>(tmp.x, scale0);
      dst1 = ref::f4_dequantize<F4_T>(tmp.y, scale1);
      dstptr[i * ld_dst + j + 0] = static_cast<_DST_T>(dst0);
      dstptr[i * ld_dst + j + 1] = static_cast<_DST_T>(dst1);
    }
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE F4_T, typename _DST_T>
static inline BTLA_CODE decompress_kblock_f4_fp_noscale(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col,
                                                        int ld_src, int ld_dst, int8_t* tmp, size_t tmpsize) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      dstptr[i * ld_dst + j + 0] = static_cast<_DST_T>(f4_unpack<F4_T>(tmp.x));
      dstptr[i * ld_dst + j + 1] = static_cast<_DST_T>(f4_unpack<F4_T>(tmp.y));
    }
  }
  return BTLA_CODE::Success;
}

template <typename _DST_T>
static inline BTLA_CODE decompress_kblock_f8_fp_noscale(utils::f8* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                        int ld_dst, BTLA_DTYPE src_f8_t) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dstptr[i * ld_dst + j] = f8_to_fp32(srcptr[i * ld_src + j], src_f8_t);
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE memcpy2d_dw2highw(const void* srcptr, void* dstptr, int row, int col, int srcstride,
                                          int dststride) {
  auto bsrcptr = (char*)srcptr;
  auto bdstptr = (char*)dstptr;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::memcpy(bdstptr + i * dststride + j * sizeof(utils::bf16), bsrcptr + i * srcstride + j * sizeof(float) + 2,
                  sizeof(utils::bf16));
    }
  }
  return BTLA_CODE::Success;
}

template <typename _SRC_T, typename _DST_T>
static inline BTLA_CODE memcpy2d(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstride, int dststride) {
  auto bsrcptr = (const char*)srcptr;
  auto bdstptr = (char*)dstptr;
  for (int i = 0; i < row; i++) {
    if constexpr (std::is_same_v<_SRC_T, _DST_T>) {
      std::memcpy(bdstptr + i * dststride, bsrcptr + i * srcstride, col);
    } else if constexpr (std::is_same_v<_SRC_T, float> &&
                         (std::is_same_v<_DST_T, utils::bf16> || std::is_same_v<_DST_T, utils::fp16>)) {
      for (int j = 0; j < col; j += sizeof(_SRC_T))
        dstptr[(i * dststride + j / 2) / sizeof(_DST_T)] =
            static_cast<_DST_T>(srcptr[(i * srcstride + j) / sizeof(_SRC_T)]);
    } else if constexpr ((std::is_same_v<_SRC_T, utils::bf16> || std::is_same_v<_SRC_T, utils::fp16>) &&
                         std::is_same_v<_DST_T, float>) {
      for (int j = 0; j < col; j += sizeof(_SRC_T))
        dstptr[(i * dststride + j * 2) / sizeof(_DST_T)] =
            static_cast<_DST_T>(srcptr[(i * srcstride + j) / sizeof(_SRC_T)]);
    } else {
      assert(0);
    }
  }
  return BTLA_CODE::Success;
}

template <BTLA_ELTWISEOP OP_T>
static inline float postop(float x, void* const_elt_v) {
  if constexpr (OP_T == BTLA_ELTWISEOP::GELU) {
    return 0.5f * x * (1.f + tanhf(0.7978845834732056f * (x + 0.044714998453855515f * x * x * x)));
  }
  if constexpr (OP_T == BTLA_ELTWISEOP::SWISH) {
    return x / (1 + exp(-x));
  }
  assert(0);
  return std::numeric_limits<float>::infinity();
}

template <typename _SRC_T, typename _DST_T, BTLA_ELTWISEOP OP_T>
static inline BTLA_CODE memcpy2d_withop(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstep,
                                        int dststep, void* const_elt_v) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 1) {
      float v = srcptr[i * srcstep + j];
      v = postop<OP_T>(v, const_elt_v);
      dstptr[i * dststep + j] = v;
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE get2d_e8m0_scale(const void* srcptr, void* dstptr, int row, int col, int srcstride,
                                         int dststride) {
  auto f8_v = (const utils::f8*)srcptr;
  auto f32_v = (float*)dstptr;
  auto f8_stride = srcstride / sizeof(utils::f8);
  auto f32_stride = dststride / sizeof(float);
  auto col_elt = col / sizeof(utils::f8);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col_elt; j++) {
      f32_v[i * f32_stride + j] = static_cast<float>(std::pow(2, f8_v[i * f8_stride + j].x));
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE quantize_f32_sign_int_rowblock(const float* srcptr, int8_t* dstptr, int row, int col,
                                                       int ld_src, int ld_dst, float* scales, int8_t* zero_points,
                                                       int blocksize, BTLA_DTYPE qtype) {
  int raw_blocksize = blocksize;
  auto NBits = utils::bestla_dtype_bits(qtype);
  int FullValue = 1 << (NBits - 1);
  int SymValue = FullValue - 1;
  auto clip = [&](int s) {
    s = std::max(s, -FullValue);
    s = std::min(s, SymValue);
    return s;
  };
  for (int i = 0; i < col; i++) {
    int align_row_loop = row / blocksize * blocksize;
    int j = 0;
    auto s8_calc_store_scale_and_quantv_sym = [&](int blocksize) {
      float maxval = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < blocksize; ij++) {
        maxval = std::max(maxval, std::abs(srcptr[(j + ij) * ld_src + i]));
      }
      float scale = maxval / 127;
      float rscale = 1.f / scale;
      scales[j / raw_blocksize * ld_dst + i] = scale;
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = utils::cast<float, int8_t>(srcptr[(j + ij) * ld_src + i] * rscale);
      }
    };
    auto s8_calc_store_scale_and_quantv_asym = [&](int blocksize) {
      float maxval = 0.f;
      float minval = 0.f;
      for (size_t ij = 0; ij < blocksize; ij++) {
        maxval = std::max(maxval, srcptr[(j + ij) * ld_src + i]);
        minval = std::min(minval, srcptr[(j + ij) * ld_src + i]);
      }
      float scale = (maxval - minval) / 255;
      float rscale = 1.f / scale;
      scales[j / raw_blocksize * ld_dst + i] = scale;
      int8_t bzp = clip(utils::cast<float, int>((0 - minval) * rscale) - 128);
      zero_points[j / raw_blocksize * ld_dst + i] = bzp;
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = clip(utils::cast<float, int>((srcptr[(j + ij) * ld_src + i]) * rscale) + bzp);
      }
    };
    auto sNauto_calc_store_scale_and_quantv_sym = [&](int blocksize) {
      float maxval = std::numeric_limits<float>::min();
      float minval = std::numeric_limits<float>::max();
      float absmax = 0;
      for (size_t ij = 0; ij < blocksize; ij++) {
        maxval = std::max(maxval, srcptr[(j + ij) * ld_src + i]);
        minval = std::min(minval, srcptr[(j + ij) * ld_src + i]);
        absmax = std::max(absmax, std::abs(srcptr[(j + ij) * ld_src + i]));
      }
      float NVal = SymValue + 0.5f;
      auto sum = maxval + minval;
      if (abs(sum) >= absmax / FullValue) {
        NVal = sum > 0.f ? static_cast<float>(-FullValue) : static_cast<float>(FullValue);
      }
      float scale = absmax / NVal;
      float rscale = 1.f / scale;
      scales[j / raw_blocksize * ld_dst + i] = scale;
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = clip(utils::cast<float, int8_t>(srcptr[(j + ij) * ld_src + i] * rscale));
      }
    };

    auto sNauto_calc_store_scale_and_quantv_asym = [&](int blocksize) {
      float maxval = 0.f;
      float minval = 0.f;
      for (size_t ij = 0; ij < blocksize; ij++) {
        maxval = std::max(maxval, srcptr[(j + ij) * ld_src + i]);
        minval = std::min(minval, srcptr[(j + ij) * ld_src + i]);
      }
      float scale = (maxval - minval) / ((1 << NBits) - 1);
      float rscale = 1.f / scale;
      scales[j / raw_blocksize * ld_dst + i] = scale;
      int bzp = utils::cast<float, int>((0 - minval) * rscale) - FullValue;

      bzp = clip(bzp);
      zero_points[j / raw_blocksize * ld_dst + i] = static_cast<int8_t>(bzp);
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto tmp = utils::cast<float, int>((srcptr[(j + ij) * ld_src + i]) * rscale) + bzp;
        tmp = clip(tmp);
        dstptr[(j + ij) * ld_dst + i] = tmp;
      }
    };
    auto dispatch_calc = [&](int blocksize) {
      switch (qtype) {
        case BTLA_DTYPE::S8:
        case BTLA_DTYPE::S2_CLIP:
        case BTLA_DTYPE::S1_CLIP:
        case BTLA_DTYPE::S3_CLIP:
        case BTLA_DTYPE::S4_CLIP:
        case BTLA_DTYPE::S5_CLIP:
        case BTLA_DTYPE::S6_CLIP:
        case BTLA_DTYPE::S7_CLIP:
          if (zero_points == nullptr) {
            sNauto_calc_store_scale_and_quantv_sym(blocksize);
          } else {
            sNauto_calc_store_scale_and_quantv_asym(blocksize);
          }
          break;
        default:
          assert(false);
          break;
      }
    };

    for (; j < align_row_loop; j += blocksize) dispatch_calc(blocksize);
    if (j < row) dispatch_calc(row - align_row_loop);
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE F8_T>
static inline int8_t f8_mx_quantize(float v, float scale, BTLA_DTYPE scale_dtype) {
  if (scale_dtype == BTLA_DTYPE::F8_E8M0) {
    v /= static_cast<float>(std::pow(2, scale));
  } else {
    v /= scale;
  }
  auto ebits = utils::bestla_dtype_get_f8_ebits(F8_T);
  auto quant_mantissa = utils::bestla_dtype_get_f8_quant_mbits(F8_T);
  auto store_mantissa = 7 - ebits;
  auto private_exp = std::floor(std::log2(std::abs(v == 0 ? v + 1 : v)));
  auto min_exp = static_cast<float>(-1 * (std::pow(2, ebits - 1)) + 2);
  private_exp = private_exp < min_exp ? min_exp : private_exp;

  // Scale up so appropriate number of bits are in the integer portion of the
  // number
  v = static_cast<float>(v / std::pow(2, private_exp) * std::pow(2, quant_mantissa - 2));
  auto sign = v > 0 ? 1 : -1;
  v = sign * static_cast<float>(std::floor(std::abs(v) + 0.5));
  // Undo scaling
  v = static_cast<float>(v / std::pow(2, quant_mantissa - 2) * std::pow(2, private_exp));

  // saturate normals.
  auto max_norm = utils::get_mxfp_maxnorm(F8_T, ebits, quant_mantissa);
  v = std::clamp(v, -1 * max_norm, max_norm);
  uint32_t* shift_v = reinterpret_cast<uint32_t*>(&v);
  // get sign;
  char* p = reinterpret_cast<char*>(&v);
  uint8_t store_signbit = (*(p + 3) & 0x80);
  *shift_v <<= 1;
  uint8_t store_ebit = (*(p + 3) & 0xFF);
  store_ebit = store_ebit - 127 + static_cast<uint8_t>(std::pow(2, ebits - 1)) - 1;
  if (store_ebit > 15 && F8_T == BTLA_DTYPE::F8_E4M3) store_ebit = 0;
  if (store_ebit > 31 && F8_T == BTLA_DTYPE::F8_E5M2) store_ebit = 0;
  store_ebit <<= store_mantissa;
  *shift_v <<= 8;
  int8_t ox80_shift = -128 >> (store_mantissa - 1);
  uint8_t store_mantissabit = (*(p + 3) & ox80_shift);
  store_mantissabit >>= (1 + ebits);
  auto ret = store_signbit | store_ebit | store_mantissabit;
  return ret;
}

template <BTLA_DTYPE F8_T>
static inline BTLA_CODE quantize_f32_f8_rowblock_mxscale(const float* srcptr, int8_t* dstptr, int row, int col,
                                                         int ld_src, int ld_dst, float* scales, int blocksize,
                                                         BTLA_DTYPE scale_dtype) {
  for (int i = 0; i < col; i++) {
    int align_row_loop = row / blocksize * blocksize;
    int j = 0;
    auto f8_blk_quant = [&](int blksize) {
      float scale = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < blksize; ij++) {
        scale = std::max(scale, std::abs(srcptr[(j + ij) * ld_src + i]));
      }
      if (scale_dtype == BTLA_DTYPE::F8_E8M0) {
        if (scale == 0) scale += std::abs(std::numeric_limits<float>::min());
        scale = std::floor(std::log2(scale));
        auto ebits = utils::bestla_dtype_get_f8_ebits(F8_T);
        auto emax = static_cast<float>(std::pow(2, ebits - 1));
        if (F8_T == BTLA_DTYPE::F8_E5M2) emax -= 1;
        scale -= emax;
        auto scale_max = static_cast<float>(std::pow(2, 7)) - 1;  // e8m0 scale type.
        scale = scale < (-1 * scale_max) ? (-1 * scale_max) : scale;
      } else if (scale_dtype == BTLA_DTYPE::F32) {
        scale /= utils::get_mxfp_maxnorm(F8_T, utils::bestla_dtype_get_f8_ebits(F8_T),
                                         utils::bestla_dtype_get_f8_quant_mbits(F8_T));
      } else {
        assert(0);
      }
      scales[j / blocksize * ld_dst + i] = scale;
      for (size_t ij = 0; ij < blksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = f8_mx_quantize<F8_T>(srcptr[(j + ij) * ld_src + i], scale, scale_dtype);
      }
    };
    for (; j < align_row_loop; j += blocksize) f8_blk_quant(blocksize);
    if (j < row) f8_blk_quant(row - align_row_loop);
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE F4_T>
static inline BTLA_CODE quantize_f32_f4_rowblock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                                 int ld_dst, float* scales, int blocksize) {
  int raw_blocksize = blocksize;
  for (int i = 0; i < col; i++) {
    int align_row_loop = row / blocksize * blocksize;
    int j = 0;
    auto calc_store_scale_and_quantv_sym = [&](int blocksize) {
      float absmax = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < blocksize; ij++) {
        absmax = std::max(absmax, std::abs(srcptr[(j + ij) * ld_src + i]));
      }
      scales[j / raw_blocksize * ld_dst + i] = absmax;
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = f4_quantize<F4_T>(srcptr[(j + ij) * ld_src + i] * (1.f / absmax));
      }
    };
    for (; j < align_row_loop; j += blocksize) calc_store_scale_and_quantv_sym(blocksize);
    if (j < row) calc_store_scale_and_quantv_sym(row - align_row_loop);
  }
  return BTLA_CODE::Success;
}

template <typename SRC_T>
static inline BTLA_CODE quantize_fp_u8_colblock(int row, int col, const SRC_T* srcptr, int ld_src, uint8_t* dstptr,
                                                int ld_dst, float* scales, int ld_scale, uint8_t* zps, int blocksize,
                                                float* blkreduce) {
  int colblk = utils::padto_le(col, blocksize);
  for (int i = 0; i < row; i++) {
    size_t j = 0;
    for (; j < colblk; j += blocksize) {
      float maxval = std::numeric_limits<float>::min();
      float minval = 0.f;
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto fsrc = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
        maxval = std::max(fsrc, maxval);
        minval = std::min(fsrc, minval);
      }
      float scale = (maxval - minval) / 255;
      uint8_t zp = utils::cast<float, uint8_t>((0 - minval) / scale);
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      zps[j / blocksize + i * ld_scale] = zp;
      int sum = 0;
      auto zpf = static_cast<float>(zp);
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto fsrc = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
        auto qtmp = utils::cast<float, int>(fsrc * rscale);
        sum += qtmp;
        dstptr[(j + ij) + i * ld_dst] = utils::cast<float, uint8_t>(zpf + qtmp);
      }
      if (blkreduce) {
        blkreduce[j / blocksize + i * ld_scale] = sum * scale;
      }
    }
    if (j < col) {
      float maxval = 0.f;
      float minval = 0.f;
      for (size_t ij = j; ij < col; ij++) {
        auto fsrc = static_cast<float>(srcptr[(ij) + i * ld_src]);
        maxval = std::max(fsrc, maxval);
        minval = std::min(fsrc, minval);
      }
      float scale = (maxval - minval) / 255;
      uint8_t zp = utils::cast<float, uint8_t>((0 - minval) / scale);
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      zps[j / blocksize + i * ld_scale] = zp;
      int sum = 0;
      auto zpf = float(zp);
      for (size_t ij = j; ij < col; ij++) {
        auto fsrc = static_cast<float>(srcptr[(ij) + i * ld_src]);
        auto qtmp = utils::cast<float, int>(fsrc * rscale);
        sum += qtmp;
        dstptr[(ij) + i * ld_dst] = utils::cast<float, uint8_t>(zpf + qtmp);
      }
      if (blkreduce) {
        blkreduce[j / blocksize + i * ld_scale] = sum * scale;
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename SRC_T>
static inline BTLA_CODE quantize_fp_s8_colblock(int row, int col, const SRC_T* srcptr, int ld_src, int8_t* dstptr,
                                                int ld_dst, float* scales, int ld_scale, int blocksize, float* reduce) {
  int colblk = utils::padto_le(col, blocksize);
  for (int i = 0; i < row; i++) {
    size_t j = 0;
    for (; j < colblk; j += blocksize) {
      float absmaxval = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto fsrc = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
        absmaxval = std::max(std::abs(fsrc), absmaxval);
      }
      float scale = absmaxval / 127;
      float rscale = 1.f / scale;
      int sum = 0;
      scales[j / blocksize + i * ld_scale] = scale;
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto fsrc = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
        auto tmp = utils::cast<float, int8_t>(fsrc * rscale);
        dstptr[(j + ij) + i * ld_dst] = tmp;
        sum += tmp;
      }
      if (reduce) reduce[j / blocksize + i * ld_scale] = sum * scale;
    }
    if (j < col) {
      float absmaxval = std::numeric_limits<float>::min();
      for (size_t ij = j; ij < col; ij++) {
        auto fsrc = static_cast<float>(srcptr[(ij) + i * ld_src]);
        absmaxval = std::max(std::abs(fsrc), absmaxval);
      }
      float scale = absmaxval / 127;
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      int sum = 0;
      for (size_t ij = j; ij < col; ij++) {
        auto fsrc = static_cast<float>(srcptr[(ij) + i * ld_src]);
        dstptr[(ij) + i * ld_dst] = utils::cast<float, int8_t>(fsrc * rscale);
        sum += dstptr[(ij) + i * ld_dst];
      }
      if (reduce) reduce[j / blocksize + i * ld_scale] = sum * scale;
    }
  }
  return BTLA_CODE::Success;
}

static inline uint8_t get_dq8_bnb(float v) {
  int left = 0;
  int right = 255;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (dq8_bnb_LUT[mid] == v) {
      return mid;
    } else if (dq8_bnb_LUT[mid] < v) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  if (right < 0) {
    return 0;
  } else if (left >= 256) {
    return 255;
  } else {
    return (v - dq8_bnb_LUT[right] < dq8_bnb_LUT[left] - v) ? right : left;
  }
}
template <bool QDQ_SCALE>
static inline BTLA_CODE dq8_bnb_double_quant(float* scale, size_t scale_size, int dq_blocksize, float* dq_buf) {
  float offset = 0.f;
  for (size_t i = 0; i < scale_size; i++) offset += scale[i];
  offset /= scale_size;
  dq_buf[utils::updiv(scale_size, dq_blocksize)] = offset;  // store zp.
  size_t align_blk_size = scale_size / dq_blocksize * dq_blocksize;
  size_t i = 0;
  auto calc_scale = [&](size_t blksize) {
    float absmax = std::numeric_limits<float>::min();
    for (size_t j = 0; j < blksize; j++) {
      scale[i + j] -= offset;
      absmax = std::max(absmax, std::abs(scale[i + j]));
    }
    for (size_t j = 0; j < blksize; j++) {
      scale[i + j] /= absmax;
      scale[i + j] = get_dq8_bnb(scale[i + j]);
      if constexpr (QDQ_SCALE) {
        scale[i + j] = dq8_bnb_LUT[static_cast<int>(scale[i + j])];
        scale[i + j] *= absmax;
        scale[i + j] += offset;
      }
    }
    return absmax;
  };
  for (; i < align_blk_size; i += dq_blocksize) dq_buf[i / dq_blocksize] = calc_scale(dq_blocksize);
  if (i < scale_size) dq_buf[i / dq_blocksize + 1] = calc_scale(scale_size - i);
  return BTLA_CODE::Success;
}

static inline BTLA_CODE dq8_get_fp_scale(uint8_t* src, float* dst, int row, int col, int scale_offset, int dq_blk,
                                         int dq_offset_idx, float* dq_scale, int src_stride, int dst_stride,
                                         bool zeropadding, int mN) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      auto dq_s_idx = (i * mN + scale_offset + j) / dq_blk;
      dst[i * dst_stride + j] = dq8_bnb_LUT[src[i * src_stride + j]] * dq_scale[dq_s_idx] + dq_scale[dq_offset_idx];
    }
  }
  if (zeropadding) assert(0);
  return BTLA_CODE::Success;
}

static inline BTLA_CODE alphabeta_f32_f32(const float alpha, const float* srcptr, const int srcstep, const float beta,
                                          const float* src1ptr, const int src1step, float* dstptr, const int dststep,
                                          const int M, const int N) {
  if (beta != 0.f) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j] + beta * src1ptr[i * src1step + j];
      }
    }
    return BTLA_CODE::Success;
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j];
    }
  }
  return BTLA_CODE::Success;
}
template <typename SCA_T>
static inline BTLA_CODE accum_alphaN_f32_f32(const SCA_T* alpha, const float* srcptr, const int srcstep, float* dstptr,
                                             const int dststep, const int M, const int N) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      if constexpr (!std::is_same_v<SCA_T, utils::f8>) {
        dstptr[i * dststep + j] = static_cast<float>(alpha[j]) * srcptr[i * srcstep + j] + dstptr[i * dststep + j];
      } else {
        dstptr[i * dststep + j] =
            std::pow(2, alpha[j].x) * srcptr[i * srcstep + j] + dstptr[i * dststep + j];  // e8m0 scale.
      }
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE accum_f32_f32(const float* srcptr, const int srcstep, float* dstptr, const int dststep,
                                      const int M, const int N) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      dstptr[i * dststep + j] = srcptr[i * srcstep + j] + dstptr[i * dststep + j];
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE quanout_s32_u32(const float alpha, const int32_t* srcptr, const int srcstep, uint8_t* dstptr,
                                        const int dststep, const int M, const int N, float scaleSrc, float scaleDst,
                                        int zpDst) {
  float factor = alpha * scaleSrc / scaleDst;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float fsrc = static_cast<float>(srcptr[i * srcstep + j]) * factor;
      dstptr[i * dststep + j] = utils::cast<float, uint8_t>(fsrc + static_cast<float>(zpDst));
    }
  }
  return BTLA_CODE::Success;
}

template <typename SCAB_T>
static inline BTLA_CODE dequant_s32_fp32(const int32_t* srcptr, const int srcstep, float* dstptr, const int dststep,
                                         const int M, const int N, const float* scaleA, const int ldsa,
                                         const SCAB_T* scaleB) {
  for (int i = 0; i < M; i++) {
    float scale = scaleA[i * ldsa];
    for (int j = 0; j < N; j++) {
      float fsrc = static_cast<float>(srcptr[i * srcstep + j]) * static_cast<float>(scaleB[j]) * scale;
      dstptr[i * dststep + j] = fsrc;
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE minmax_f32_kblock(const float* srcptr, int row, int col, int ld_src, float* minmaxptr,
                                          int ld_minmax, int fsize_minmax, int blocksize) {
  for (int i = 0; i < row; i++) {
    if (col >= blocksize) {
      for (int icol = 0; icol < col; icol += blocksize) {
        float maxval = std::numeric_limits<float>::min();
        float minval = std::numeric_limits<float>::max();
        for (int ii = 0; ii < blocksize; ii++) {
          maxval = std::max(srcptr[i * ld_src + icol + ii], maxval);
          minval = std::min(srcptr[i * ld_src + icol + ii], minval);
        }
        auto colptr = &minmaxptr[i * ld_minmax + icol / blocksize * fsize_minmax];
        colptr[0] = minval;
        colptr[1] = maxval;
      }
    } else {
      float maxval = std::numeric_limits<float>::min();
      float minval = std::numeric_limits<float>::max();
      for (int icol = 0; icol < col; icol++) {
        maxval = std::max(srcptr[i * ld_src + icol], maxval);
        minval = std::min(srcptr[i * ld_src + icol], minval);
      }
      minmaxptr[i * ld_minmax + 0] = minval;
      minmaxptr[i * ld_minmax + 1] = maxval;
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE accumulate_dequantize_s32_f32(const int32_t* srcptr, float* dstptr, float alpha, float beta,
                                                      int row, int col, int ld_src, int ld_dst, float* ascales,
                                                      int ldas, float* wscales) {
  for (int irow = 0; irow < row; irow++) {
    for (int icol = 0; icol < col; icol++) {
      float scale = ascales[irow * ldas] * wscales[icol] * alpha;
      dstptr[irow * ld_dst + icol] = scale * srcptr[irow * ld_src + icol] + beta * dstptr[irow * ld_dst + icol];
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE broadcast_u8(int num, const uint8_t& srcval, uint8_t* dstptr) {
  int i = 0;
  for (; i < num; i++) {
    dstptr[i] = srcval;
  }
  return BTLA_CODE::Success;
}

template <typename _RT>
static inline BTLA_CODE quant_s8_row_reduce_sum(const int8_t* srcptr, int ldsrc, const float* scales,
                                                const int8_t* zero_points, int row, int col, _RT* reduce) {
  std::memset(reduce, 0, sizeof(reduce[0]) * col);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (zero_points != nullptr) {
        reduce[j] += static_cast<_RT>((static_cast<float>(srcptr[i * ldsrc + j]) - static_cast<float>(zero_points[j])) *
                                      static_cast<float>(scales[j]));
      } else {
        reduce[j] += static_cast<_RT>(srcptr[i * ldsrc + j] * scales[j]);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename _SRCT, typename _DSTT>
static inline BTLA_CODE row_reduce_sum(const _SRCT* srcptr, int ldsrc, int row, int col, _DSTT* reduce) {
  for (int j = 0; j < col; j++) {
    _DSTT tmp = 0;
    for (int i = 0; i < row; i++) {
      tmp += static_cast<_DSTT>(srcptr[i * ldsrc + j]);
    }
    reduce[j] = tmp;
  }
  return BTLA_CODE::Success;
}

template <typename SRC_T>
static inline BTLA_CODE col_block_reduce_sum(const SRC_T* srcptr, int ldsrc, int row, int col, int blocksize,
                                             float* reduce, int ldr) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += blocksize) {
      auto tmp = 0.f;
      for (size_t jj = 0; jj < blocksize; jj++) {
        if (j + jj < col) {
          tmp += srcptr[i * ldsrc + j + jj];
        }
      }
      reduce[i * ldr + j / blocksize] = tmp;
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE remove_act_zeropoint_bias(float* accptr, int ldacc, int row, int col, uint8_t* zps,
                                                  float* scales, int lds, const float* reduce) {
  for (int i = 0; i < row; i++) {
    auto zpf = static_cast<float>(zps[i * lds]) * scales[i * lds];
    for (int j = 0; j < col; j++) {
      accptr[i * ldacc + j] -= zpf * reduce[j];
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE remove_wei_zeropoint_bias(float* accptr, int ldacc, int row, int col, int8_t* zps,
                                                  float* scales, int lds, const float* reduce) {
  for (int i = 0; i < row; i++) {
    auto reducef = reduce[i * lds];
    for (int j = 0; j < col; j++) {
      accptr[i * ldacc + j] -= static_cast<float>(zps[j]) * scales[j] * reducef;
    }
  }
  return BTLA_CODE::Success;
}

template <bool ZPA, bool ZPB>
static inline BTLA_CODE remove_zeropoint_bias(int32_t* accptr, int ldacc, int row, int col, uint8_t* zpa, int8_t* zpb,
                                              int lda, int k, const int32_t* reducea, const int32_t* reduceb) {
  for (int i = 0; i < row; i++) {
    int32_t vreducea = ZPB ? reducea[i * lda] : 0;
    uint8_t vzpa = ZPA ? zpa[i * lda] : 0;
    for (int j = 0; j < col; j++) {
      auto vzpb = ZPB ? zpb[j] : 0;
      accptr[i * ldacc + j] -= ZPB ? vzpb * vreducea : 0;
      accptr[i * ldacc + j] -= ZPA ? vzpa * reduceb[j] : 0;
      accptr[i * ldacc + j] -= ZPB && ZPA ? vzpa * vzpb * k : 0;
    }
  }
  return BTLA_CODE::Success;
}

template <typename T>
static inline BTLA_CODE layernorm(const T* srcptr, const T* scaleptr, const T* biasptr, T epsilon, int norm_size,
                                  T* dstptr, T* mean_out, T* mean_square_out, bool simplified) {
  T mean = 0;
  T mean_square = 0;

  for (int h = 0; h < norm_size; h++) {
    mean += srcptr[h];
    mean_square += srcptr[h] * srcptr[h];
  }

  mean = mean / norm_size;
  if (simplified) {
    mean_square = std::sqrt(mean_square / norm_size + epsilon);
  } else {
    mean_square = std::sqrt(mean_square / norm_size - mean * mean + epsilon);
  }
  float inv_mean_square = 1.f / mean_square;
  if (simplified) {
    if (scaleptr) {
      for (int h = 0; h < norm_size; h++) {
        dstptr[h] = srcptr[h] * inv_mean_square * scaleptr[h];
      }
    } else {
      for (int h = 0; h < norm_size; h++) {
        dstptr[h] = srcptr[h] * inv_mean_square;
      }
    }
  } else {
    if (scaleptr) {
      if (biasptr == nullptr) {
        for (int h = 0; h < norm_size; h++) {
          dstptr[h] = (srcptr[h] - mean) * inv_mean_square * scaleptr[h];
        }
      } else {
        for (int h = 0; h < norm_size; h++) {
          dstptr[h] = (srcptr[h] - mean) * inv_mean_square * scaleptr[h] + biasptr[h];
        }
      }
    } else {
      for (int h = 0; h < norm_size; h++) {
        dstptr[h] = (srcptr[h] - mean) * inv_mean_square;
      }
    }
  }

  if (mean_out) {
    *mean_out = mean;
  }
  if (mean_square_out) {
    *mean_square_out = mean_square;
  }
  return BTLA_CODE::Success;
}

inline float exp_ps_0_1(float x) {
  static const float log2e = std::log2(std::exp(1.f));
  static const float ln2 = std::log(2.f);
  const float x1 = x * log2e + .5f;
  const float z = std::floor(x1);
  const float f = x1 - z;
  constexpr std::array<float, 3> coeff{0.240226507f, 0.452920674f, 0.713483036f};
  // same as a * std::pow(2, z) but more precise
  return ldexpf(coeff[0] * f * f + coeff[1] * f + coeff[2], static_cast<int>(z));
}

template <BTLA_DTYPE S3_T, typename _DST_T>
static inline BTLA_CODE decompress_kblock_s3_s8fp(utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr, _DST_T* dstptr,
                                                  int interleave_n_offset, int unpack_elt, int8_t* tmp,
                                                  size_t tmpsize) {
  auto head_ignore_num = interleave_n_offset % 128;
  auto bit3_interleave_decompress_pack128 = [&](utils::bit2x4* src1, utils::bit1x8* src2, int8_t* dst) {
    auto b2ptr = reinterpret_cast<uint8_t*>(src1);
    for (size_t i = 0; i < 128; i += 8) {
      auto bit1off = i >> 3;
      auto bit2off = (i >> 5) << 1;
      auto byteoff = i % 32;
      uint8_t bit1 = *(uint8_t*)(src2 + bit1off);
      for (size_t j = 0; j < 8; j++) {
        uint8_t bit2 = *(b2ptr + byteoff + j);
        bit2 >>= bit2off;
        uint8_t dst8 = ((bit2 & 0x3)) | ((bit1 & 0x1) << 2);
        bit1 >>= 1;
        dst[i + j] = (*(int8_t*)&dst8) - 4;
      }
    }
  };

  assert(head_ignore_num % 8 == 0);

  auto base_bit2ptr = bit2ptr - head_ignore_num / 4;
  auto base_bit1ptr = bit1ptr - head_ignore_num / 8;
  int compress_wei_ptr_offset = 0;
  int8_t* s8_ptr = reinterpret_cast<int8_t*>(tmp);
  auto head_write_num = 128 - head_ignore_num;
  if (head_ignore_num != 0) {
    bit3_interleave_decompress_pack128(base_bit2ptr, base_bit1ptr, tmp);
    for (int i = 0; i < head_write_num; i++) dstptr[i] = s8_ptr[head_ignore_num + i];
    compress_wei_ptr_offset += head_write_num;
  }

  auto body_loop = (unpack_elt - head_write_num % 128) / 128;
  auto tail_proc_num = (unpack_elt - head_write_num % 128) % 128;
  for (size_t i = 0; i < body_loop; i++) {
    bit3_interleave_decompress_pack128(bit2ptr + compress_wei_ptr_offset / 4 + i * 32,
                                       bit1ptr + compress_wei_ptr_offset / 8 + i * 16, tmp);
    for (int j = 0; j < 128; j++) dstptr[compress_wei_ptr_offset + i * 128 + j] = tmp[j];
  }
  compress_wei_ptr_offset += body_loop * 128;
  if (tail_proc_num > 0) {
    bit3_interleave_decompress_pack128(base_bit2ptr, base_bit1ptr, tmp);
    bit3_interleave_decompress_pack128(bit2ptr + compress_wei_ptr_offset / 4, bit1ptr + compress_wei_ptr_offset / 8,
                                       tmp);
    for (int i = 0; i < tail_proc_num; i++) dstptr[compress_wei_ptr_offset + i] = s8_ptr[i];
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE _S3_T, typename _DST_T, int _PACK_ROW, typename _ST>
static inline BTLA_CODE decompress_kblock_bit3_packrow_fp(utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr,
                                                          _DST_T* dstptr, int interleave_n_offset, int row, int col,
                                                          _ST* scales, int8_t* zero_points, int k_offset, int kblock,
                                                          int NPad, void* tmp, size_t tmpsize) {
  auto unpack_elt = row * col;
  decompress_kblock_s3_s8fp<_S3_T>(bit2ptr, bit1ptr, dstptr, interleave_n_offset, unpack_elt,
                                   reinterpret_cast<int8_t*>(tmp), tmpsize);
  // TODO(zhe): simd version
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j++) {
      float tmp = static_cast<float>(dstptr[i * col + j]);
      if (zero_points != nullptr) tmp -= static_cast<float>(zero_points[kpos * NPad + j / _PACK_ROW]);
      dstptr[i * col + j] = static_cast<_DST_T>(tmp * sptr[j / _PACK_ROW]);
    }
  }

  return BTLA_CODE::Success;
}

template <BTLA_DTYPE S2_T, typename _DST_T>
static inline BTLA_CODE decompress_kblock_s2_s8fp(utils::bit2x4* bit2ptr, _DST_T* dstptr, size_t unpack_elt,
                                                  int8_t* tmp, size_t tmpsize) {
  for (size_t i = 0; i < unpack_elt; i += 4) {
    auto tmp = bit2ptr[i / 4];
    dstptr[i + 0] = _DST_T(tmp.a - 2);
    dstptr[i + 1] = _DST_T(tmp.b - 2);
    dstptr[i + 2] = _DST_T(tmp.c - 2);
    dstptr[i + 3] = _DST_T(tmp.d - 2);
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE _S2_T, typename _DST_T, int _PACK_ROW, typename _ST>
static inline BTLA_CODE decompress_kblock_bit2_packrow_fp(utils::bit2x4* bit2ptr, _DST_T* dstptr, int row, int col,
                                                          _ST* scales, int8_t* zero_points, int k_offset, int kblock,
                                                          int NPad, void* tmp, size_t tmpsize) {
  auto unpack_elt = row * col;
  decompress_kblock_s2_s8fp<_S2_T>(bit2ptr, dstptr, unpack_elt, reinterpret_cast<int8_t*>(tmp), tmpsize);
  // TODO(zhe): simd version
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j++) {
      float tmp = static_cast<float>(dstptr[i * col + j]);
      if (zero_points != nullptr) tmp -= static_cast<float>(zero_points[kpos * NPad + j / _PACK_ROW]);
      dstptr[i * col + j] = static_cast<_DST_T>(tmp * sptr[j / _PACK_ROW]);
    }
  }

  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_4bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = A.aptr;
  auto b4ptr = B.b4ptr;
  auto asptr = A.sptr;
  auto azptr = A.zpptr;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    if (B.zpptr) {
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int ik = 0; ik < blocksize; ik += 4) {
        for (int im = 0; im < MTILE; im++) {
          int azp = azptr[ib + im * A.ldzp];
          float ascale = asptr[ib + im * A.ldzp];
          for (int in = 0; in < NTILE; in++) {
            auto bv0 = *(utils::int4x2*)(b4ptr + in * 2);
            auto bv1 = *(utils::int4x2*)(b4ptr + in * 2 + 1);
            auto vscale = ascale * (float)bsptr[in];
            int bzp = bzptr[in] + 8;
            accf[im * NTILE + in] += int(a8ptr[0 + im * A.lda] - azp) * (bv0.x - bzp) * vscale;
            accf[im * NTILE + in] += int(a8ptr[1 + im * A.lda] - azp) * (bv0.y - bzp) * vscale;
            accf[im * NTILE + in] += int(a8ptr[2 + im * A.lda] - azp) * (bv1.x - bzp) * vscale;
            accf[im * NTILE + in] += int(a8ptr[3 + im * A.lda] - azp) * (bv1.y - bzp) * vscale;
          }
        }
        a8ptr += 4;
        b4ptr += NTILE * 2;
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += 4) {
        for (int im = 0; im < MTILE; im++) {
          int azp = azptr[ib + im * A.ldzp];
          float ascale = asptr[ib + im * A.ldzp];
          for (int in = 0; in < NTILE; in++) {
            auto bv0 = *(utils::int4x2*)(b4ptr + in * 2);
            auto bv1 = *(utils::int4x2*)(b4ptr + in * 2 + 1);
            auto vscale = ascale * (float)bsptr[in];
            accf[im * NTILE + in] += int(a8ptr[0 + im * A.lda] - azp) * (bv0.x - 8) * vscale;
            accf[im * NTILE + in] += int(a8ptr[1 + im * A.lda] - azp) * (bv0.y - 8) * vscale;
            accf[im * NTILE + in] += int(a8ptr[2 + im * A.lda] - azp) * (bv1.x - 8) * vscale;
            accf[im * NTILE + in] += int(a8ptr[3 + im * A.lda] - azp) * (bv1.y - 8) * vscale;
          }
        }
        a8ptr += 4;
        b4ptr += NTILE * 2;
      }
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_4bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = reinterpret_cast<int8_t*>(A.aptr);
  auto b4ptr = B.b4ptr;
  auto asptr = A.sptr;
  auto azptr = A.zpptr;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    if (B.zpptr) {
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int ik = 0; ik < blocksize; ik += 4) {
        for (int im = 0; im < MTILE; im++) {
          float ascale = asptr[ib + im * A.ldzp];
          for (int in = 0; in < NTILE; in++) {
            auto bv0 = *(utils::int4x2*)(b4ptr + in * 2);
            auto bv1 = *(utils::int4x2*)(b4ptr + in * 2 + 1);
            auto vscale = ascale * (float)bsptr[in];
            auto bzp = bzptr[in] + 8;
            accf[im * NTILE + in] += int(a8ptr[0 + im * A.lda]) * (bv0.x - bzp) * vscale;
            accf[im * NTILE + in] += int(a8ptr[1 + im * A.lda]) * (bv0.y - bzp) * vscale;
            accf[im * NTILE + in] += int(a8ptr[2 + im * A.lda]) * (bv1.x - bzp) * vscale;
            accf[im * NTILE + in] += int(a8ptr[3 + im * A.lda]) * (bv1.y - bzp) * vscale;
          }
        }
        a8ptr += 4;
        b4ptr += NTILE * 2;
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += 4) {
        for (int im = 0; im < MTILE; im++) {
          float ascale = asptr[ib + im * A.ldzp];
          for (int in = 0; in < NTILE; in++) {
            auto bv0 = *(utils::int4x2*)(b4ptr + in * 2);
            auto bv1 = *(utils::int4x2*)(b4ptr + in * 2 + 1);
            auto vscale = ascale * (float)bsptr[in];
            accf[im * NTILE + in] += int(a8ptr[0 + im * A.lda]) * (bv0.x - 8) * vscale;
            accf[im * NTILE + in] += int(a8ptr[1 + im * A.lda]) * (bv0.y - 8) * vscale;
            accf[im * NTILE + in] += int(a8ptr[2 + im * A.lda]) * (bv1.x - 8) * vscale;
            accf[im * NTILE + in] += int(a8ptr[3 + im * A.lda]) * (bv1.y - 8) * vscale;
          }
        }
        a8ptr += 4;
        b4ptr += NTILE * 2;
      }
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_4bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto b4ptr = B.b4ptr;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    if (B.zpptr) {
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int ik = 0; ik < blocksize; ik += 1) {
        for (int im = 0; im < MTILE; im++) {
          auto aval = A[ib * blocksize + ik + im * lda];
          for (int in = 0; in < NTILE; in += 2) {
            auto bv0 = *(utils::int4x2*)(b4ptr + in / 2);
            accf[im * NTILE + in + 0] += aval * (bv0.x - 8 - bzptr[in + 0]) * (float)bsptr[in + 0];
            accf[im * NTILE + in + 1] += aval * (bv0.y - 8 - bzptr[in + 1]) * (float)bsptr[in + 1];
          }
        }
        b4ptr += NTILE / 2;
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += 1) {
        for (int im = 0; im < MTILE; im++) {
          auto aval = A[ib * blocksize + ik + im * lda];
          for (int in = 0; in < NTILE; in += 2) {
            auto bv0 = *(utils::int4x2*)(b4ptr + in / 2);
            accf[im * NTILE + in + 0] += aval * (bv0.x - 8) * (float)bsptr[in + 0];
            accf[im * NTILE + in + 1] += aval * (bv0.y - 8) * (float)bsptr[in + 1];
          }
        }
        b4ptr += NTILE / 2;
      }
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE>
static inline BTLA_CODE gemv_3bit_u8s8_fp32_align128(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B,
                                                     float* C, int k, int ld_scaleb, int blocksize, int8_t* tmp,
                                                     size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = A.aptr;
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  auto asptr = A.sptr;
  auto azptr = A.zpptr;
  int constexpr EltPadding = 128;
  static_assert(NTILE % 8 == 0);
  int constexpr KTILE = 4;
  int constexpr UnpackElt = EltPadding / 8 / KTILE;
  int8_t UnpackBuf[UnpackElt * NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * ld_scaleb;
    int acci[NTILE];
    std::memset(acci, 0, sizeof(acci));
    int wacc[NTILE];
    std::memset(wacc, 0, sizeof(wacc));
    for (int ik = 0; ik < blocksize; ik += KTILE * UnpackElt) {
      decompress_kblock_s3_s8fp<BTLA_DTYPE::S3_CLIP, int8_t>(b2ptr, b1ptr, UnpackBuf, ik * NTILE,
                                                             NTILE * KTILE * UnpackElt, tmp, tmpsize);
      for (int iu = 0; iu < UnpackElt; iu++) {
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = UnpackBuf[iu * NTILE * KTILE + in * KTILE + ikt];
            acci[in] += int(a8ptr[iu * KTILE + ikt]) * bval;
            wacc[in] += bval;
          }
        }
      }

      b2ptr += KTILE * UnpackElt * NTILE / 4;
      b1ptr += KTILE * UnpackElt * NTILE / 8;
      a8ptr += KTILE * UnpackElt;
    }
    float scale = asptr[ib];
    int zp = azptr[ib];
    for (int in = 0; in < NTILE; in++) {
      auto tmp = float(acci[in] - zp * wacc[in]);
      tmp = tmp * (scale * (float)bsptr[in]);
      accf[in] += tmp;
    }
  }
  for (int in = 0; in < NTILE; in++) {
    C[in] = accf[in];
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE>
static inline BTLA_CODE gemv_3bit_s8s8_fp32_align128(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B,
                                                     float* C, int k, int ld_scaleb, int blocksize, int8_t* tmp,
                                                     size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = reinterpret_cast<int8_t*>(A.aptr);
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  auto asptr = A.sptr;
  int constexpr EltPadding = 128;
  static_assert(NTILE % 8 == 0);
  int constexpr KTILE = 4;
  int constexpr UnpackElt = EltPadding / 8 / KTILE;
  int8_t UnpackBuf[UnpackElt * NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * ld_scaleb;
    int acci[NTILE];
    std::memset(acci, 0, sizeof(acci));
    for (int ik = 0; ik < blocksize; ik += KTILE * UnpackElt) {
      decompress_kblock_s3_s8fp<BTLA_DTYPE::S3_CLIP, int8_t>(b2ptr, b1ptr, UnpackBuf, ik * NTILE,
                                                             NTILE * KTILE * UnpackElt, tmp, tmpsize);
      for (int iu = 0; iu < UnpackElt; iu++) {
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = UnpackBuf[iu * NTILE * KTILE + in * KTILE + ikt];
            acci[in] += int(a8ptr[iu * KTILE + ikt]) * bval;
          }
        }
      }
      b2ptr += KTILE * UnpackElt * NTILE / 4;
      b1ptr += KTILE * UnpackElt * NTILE / 8;
      a8ptr += KTILE * UnpackElt;
    }

    float scale = asptr[ib];
    for (int in = 0; in < NTILE; in++) {
      auto tmp = float(acci[in]);
      tmp = tmp * (scale * (float)bsptr[in]);
      accf[in] += tmp;
    }
  }
  for (int in = 0; in < NTILE; in++) {
    C[in] = accf[in];
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_2bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = A.aptr;
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  int constexpr KTILE = 4;
  int8_t UnpackBuf[NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += KTILE) {
      decompress_kblock_s2_s8<4, NTILE>(b2ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0, KTILE,
                                        NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        float ascale = A.sptr[ib + im * A.ldzp];
        auto azp = A.zpptr[ib + im * A.ldzp];
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = (UnpackBuf[in * KTILE + ikt]) * (float)bsptr[in];
            auto aval = int(a8ptr[ikt + im * A.lda] - azp) * ascale;
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b2ptr += KTILE * NTILE / 4;
      a8ptr += KTILE;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}
template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_2bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = (int8_t*)A.aptr;
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  int constexpr KTILE = 4;
  int8_t UnpackBuf[NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += KTILE) {
      decompress_kblock_s2_s8<4, NTILE>(b2ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0, KTILE,
                                        NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        float ascale = A.sptr[ib + im * A.ldzp];
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = (UnpackBuf[in * KTILE + ikt]) * (float)bsptr[in];
            auto aval = int(a8ptr[ikt + im * A.lda]) * ascale;
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b2ptr += KTILE * NTILE / 4;
      a8ptr += KTILE;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_2bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  int constexpr KTILE = 1;
  int constexpr Unroll = 4;
  assert((blocksize % 4) == 0);
  assert(tmpsize >= NTILE * Unroll);
  int8_t UnpackBuf[NTILE * Unroll];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += Unroll) {
      decompress_kblock_s2_s8fp<BTLA_DTYPE::S2_CLIP, int8_t>(b2ptr, UnpackBuf, NTILE * Unroll, tmp, tmpsize);
      if (B.zpptr) {
        for (int ikt = 0; ikt < Unroll; ikt++) {
          for (int im = 0; im < MTILE; im++) {
            for (int in = 0; in < NTILE; in++) {
              auto bval = (UnpackBuf[in + ikt * NTILE] - bzptr[in]) * (float)bsptr[in];
              auto aval = A[ikt + im * lda];
              accf[im * NTILE + in] += aval * bval;
            }
          }
        }
      } else {
        for (int im = 0; im < MTILE; im++) {
          for (int in = 0; in < NTILE; in++) {
            for (int ikt = 0; ikt < Unroll; ikt++) {
              auto bval = (UnpackBuf[in + ikt * NTILE]) * (float)bsptr[in];
              auto aval = A[ikt + im * lda];
              accf[im * NTILE + in] += aval * bval;
            }
          }
        }
      }
      b2ptr += Unroll * NTILE / 4;
      A += Unroll;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_1bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  int constexpr KTILE = 1;
  int constexpr Unroll = 4;
  assert((blocksize % 4) == 0);
  assert(tmpsize >= NTILE * Unroll);
  int8_t UnpackBuf[NTILE * Unroll];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += Unroll) {
      decompress_kblock_s1_s8<1, NTILE>(b1ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0, Unroll,
                                        NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < Unroll; ikt++) {
            auto bval = (UnpackBuf[in + ikt * NTILE]) * (float)bsptr[in];
            auto aval = A[ikt + im * lda];
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b1ptr += Unroll * NTILE / 8;
      A += Unroll;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_1bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = A.aptr;
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  int constexpr KTILE = 4;
  int8_t UnpackBuf[NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += KTILE) {
      decompress_kblock_s1_s8<4, NTILE>(b1ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0, KTILE,
                                        NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        float ascale = A.sptr[ib + im * A.ldzp];
        auto azp = A.zpptr[ib + im * A.ldzp];
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = (UnpackBuf[in * KTILE + ikt]) * (float)bsptr[in];
            auto aval = int(a8ptr[ikt + im * A.lda] - azp) * ascale;
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b1ptr += KTILE * NTILE / 8;
      a8ptr += KTILE;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_1bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = (int8_t*)A.aptr;
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  int constexpr KTILE = 4;
  int8_t UnpackBuf[NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += KTILE) {
      decompress_kblock_s1_s8<4, NTILE>(b1ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0, KTILE,
                                        NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        float ascale = A.sptr[ib + im * A.ldzp];
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = (UnpackBuf[in * KTILE + ikt]) * (float)bsptr[in];
            auto aval = int(a8ptr[ikt + im * A.lda]) * ascale;
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b1ptr += KTILE * NTILE / 8;
      a8ptr += KTILE;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_3bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  int constexpr KTILE = 1;
  int constexpr Unroll = 4;
  assert((blocksize % 4) == 0);
  assert(tmpsize >= NTILE * Unroll);
  int8_t UnpackBuf[NTILE * Unroll];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += Unroll) {
      decompress_kblock_s3_s8<1, NTILE>(b2ptr, b1ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0,
                                        Unroll, NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < Unroll; ikt++) {
            auto bval = (UnpackBuf[in + ikt * NTILE]) * (float)bsptr[in];
            auto aval = A[ikt + im * lda];
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b2ptr += Unroll * NTILE / 4;
      b1ptr += Unroll * NTILE / 8;
      A += Unroll;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_3bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = A.aptr;
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  int constexpr KTILE = 4;
  int8_t UnpackBuf[NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += KTILE) {
      decompress_kblock_s3_s8<4, NTILE>(b2ptr, b1ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0,
                                        KTILE, NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        float ascale = A.sptr[ib + im * A.ldzp];
        auto azp = A.zpptr[ib + im * A.ldzp];
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = (UnpackBuf[in * KTILE + ikt]) * (float)bsptr[in];
            auto aval = int(a8ptr[ikt + im * A.lda] - azp) * ascale;
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b2ptr += KTILE * NTILE / 4;
      b1ptr += KTILE * NTILE / 8;
      a8ptr += KTILE;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_3bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = (int8_t*)A.aptr;
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  int constexpr KTILE = 4;
  int8_t UnpackBuf[NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += KTILE) {
      decompress_kblock_s3_s8<4, NTILE>(b2ptr, b1ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0,
                                        KTILE, NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        float ascale = A.sptr[ib + im * A.ldzp];
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = (UnpackBuf[in * KTILE + ikt]) * (float)bsptr[in];
            auto aval = int(a8ptr[ikt + im * A.lda]) * ascale;
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b2ptr += KTILE * NTILE / 4;
      b1ptr += KTILE * NTILE / 8;
      a8ptr += KTILE;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_6bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto b4ptr = reinterpret_cast<utils::bit4x2*>(B.b4ptr);
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  int constexpr KTILE = 1;
  int constexpr Unroll = 4;
  assert((blocksize % 4) == 0);
  assert(tmpsize >= NTILE * Unroll);
  int8_t UnpackBuf[NTILE * Unroll];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += Unroll) {
      decompress_kblock_s6_s8<1, NTILE>(b4ptr, b2ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0,
                                        Unroll, NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < Unroll; ikt++) {
            auto bval = (UnpackBuf[in + ikt * NTILE]) * (float)bsptr[in];
            auto aval = A[ikt + im * lda];
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b4ptr += Unroll * NTILE / 2;
      b2ptr += Unroll * NTILE / 4;
      A += Unroll;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_6bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = A.aptr;
  auto b4ptr = reinterpret_cast<utils::bit4x2*>(B.b4ptr);
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  int constexpr KTILE = 4;
  int8_t UnpackBuf[NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += KTILE) {
      decompress_kblock_s6_s8<4, NTILE>(b4ptr, b2ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0,
                                        KTILE, NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        float ascale = A.sptr[ib + im * A.ldzp];
        auto azp = A.zpptr[ib + im * A.ldzp];
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = (UnpackBuf[in * KTILE + ikt]) * (float)bsptr[in];
            auto aval = int(a8ptr[ikt + im * A.lda] - azp) * ascale;
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b4ptr += KTILE * NTILE / 2;
      b2ptr += KTILE * NTILE / 4;
      a8ptr += KTILE;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_6bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = (int8_t*)A.aptr;
  auto b4ptr = reinterpret_cast<utils::bit4x2*>(B.b4ptr);
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  int constexpr KTILE = 4;
  int8_t UnpackBuf[NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += KTILE) {
      decompress_kblock_s6_s8<4, NTILE>(b4ptr, b2ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0,
                                        KTILE, NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        float ascale = A.sptr[ib + im * A.ldzp];
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = (UnpackBuf[in * KTILE + ikt]) * (float)bsptr[in];
            auto aval = int(a8ptr[ikt + im * A.lda]) * ascale;
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b4ptr += KTILE * NTILE / 2;
      b2ptr += KTILE * NTILE / 4;
      a8ptr += KTILE;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_5bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto b4ptr = reinterpret_cast<utils::bit4x2*>(B.b4ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  int constexpr KTILE = 1;
  int constexpr Unroll = 4;
  assert((blocksize % 4) == 0);
  assert(tmpsize >= NTILE * Unroll);
  int8_t UnpackBuf[NTILE * Unroll];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += Unroll) {
      decompress_kblock_s5_s8<1, NTILE>(b4ptr, b1ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0,
                                        Unroll, NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < Unroll; ikt++) {
            auto bval = (UnpackBuf[in + ikt * NTILE]) * (float)bsptr[in];
            auto aval = A[ikt + im * lda];
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b4ptr += Unroll * NTILE / 2;
      b1ptr += Unroll * NTILE / 8;
      A += Unroll;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_5bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = A.aptr;
  auto b4ptr = reinterpret_cast<utils::bit4x2*>(B.b4ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  int constexpr KTILE = 4;
  int8_t UnpackBuf[NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += KTILE) {
      decompress_kblock_s5_s8<4, NTILE>(b4ptr, b1ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0,
                                        KTILE, NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        float ascale = A.sptr[ib + im * A.ldzp];
        auto azp = A.zpptr[ib + im * A.ldzp];
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = (UnpackBuf[in * KTILE + ikt]) * (float)bsptr[in];
            auto aval = int(a8ptr[ikt + im * A.lda] - azp) * ascale;
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b4ptr += KTILE * NTILE / 2;
      b1ptr += KTILE * NTILE / 8;
      a8ptr += KTILE;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_5bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = (int8_t*)A.aptr;
  auto b4ptr = reinterpret_cast<utils::bit4x2*>(B.b4ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  int constexpr KTILE = 4;
  int8_t UnpackBuf[NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += KTILE) {
      decompress_kblock_s5_s8<4, NTILE>(b4ptr, b1ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0, 0,
                                        KTILE, NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        float ascale = A.sptr[ib + im * A.ldzp];
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = (UnpackBuf[in * KTILE + ikt]) * (float)bsptr[in];
            auto aval = int(a8ptr[ikt + im * A.lda]) * ascale;
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b4ptr += KTILE * NTILE / 2;
      b1ptr += KTILE * NTILE / 8;
      a8ptr += KTILE;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_7bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto b4ptr = reinterpret_cast<utils::bit4x2*>(B.b4ptr);
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  int constexpr KTILE = 1;
  int constexpr Unroll = 4;
  assert((blocksize % 4) == 0);
  assert(tmpsize >= NTILE * Unroll);
  int8_t UnpackBuf[NTILE * Unroll];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += Unroll) {
      decompress_kblock_s7_s8<1, NTILE>(b4ptr, b2ptr, b1ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0,
                                        0, Unroll, NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < Unroll; ikt++) {
            auto bval = (UnpackBuf[in + ikt * NTILE]) * (float)bsptr[in];
            auto aval = A[ikt + im * lda];
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b4ptr += Unroll * NTILE / 2;
      b2ptr += Unroll * NTILE / 4;
      b1ptr += Unroll * NTILE / 8;
      A += Unroll;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_7bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = A.aptr;
  auto b4ptr = reinterpret_cast<utils::bit4x2*>(B.b4ptr);
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  int constexpr KTILE = 4;
  int8_t UnpackBuf[NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += KTILE) {
      decompress_kblock_s7_s8<4, NTILE>(b4ptr, b2ptr, b1ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0,
                                        0, KTILE, NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        float ascale = A.sptr[ib + im * A.ldzp];
        auto azp = A.zpptr[ib + im * A.ldzp];
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = (UnpackBuf[in * KTILE + ikt]) * (float)bsptr[in];
            auto aval = int(a8ptr[ikt + im * A.lda] - azp) * ascale;
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b4ptr += KTILE * NTILE / 2;
      b2ptr += KTILE * NTILE / 4;
      b1ptr += KTILE * NTILE / 8;
      a8ptr += KTILE;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_7bit_s8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  int blks = k / blocksize;
  float accf[NTILE * MTILE];
  std::memset(accf, 0, sizeof(accf));
  auto a8ptr = (int8_t*)A.aptr;
  auto b4ptr = reinterpret_cast<utils::bit4x2*>(B.b4ptr);
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);
  int constexpr KTILE = 4;
  int8_t UnpackBuf[NTILE * KTILE];
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    auto bzptr = B.zpptr + ib * B.ldzp;
    for (int ik = 0; ik < blocksize; ik += KTILE) {
      decompress_kblock_s7_s8<4, NTILE>(b4ptr, b2ptr, b1ptr, B.zpptr ? bzptr : nullptr, UnpackBuf, blocksize, B.ldzp, 0,
                                        0, KTILE, NTILE, tmp, tmpsize);
      for (int im = 0; im < MTILE; im++) {
        float ascale = A.sptr[ib + im * A.ldzp];
        for (int in = 0; in < NTILE; in++) {
          for (int ikt = 0; ikt < KTILE; ikt++) {
            auto bval = (UnpackBuf[in * KTILE + ikt]) * (float)bsptr[in];
            auto aval = int(a8ptr[ikt + im * A.lda]) * ascale;
            accf[im * NTILE + in] += aval * bval;
          }
        }
      }
      b4ptr += KTILE * NTILE / 2;
      b2ptr += KTILE * NTILE / 4;
      b1ptr += KTILE * NTILE / 8;
      a8ptr += KTILE;
    }
  }
  for (int im = 0; im < MTILE; im++) {
    for (int in = 0; in < NTILE; in++) {
      C[in + im * ldc] = accf[im * NTILE + in];
    }
  }
  return BTLA_CODE::Success;
}

template <typename T>
static inline BTLA_CODE mul(const T* src0ptr, const T* src1ptr, T* dstptr, size_t size) {
  for (size_t i = 0; i < size; i++) {
    float tmp = float(src0ptr[i]) * float(src1ptr[i]);
    dstptr[i] = tmp;
  }
  return BTLA_CODE::Success;
}

template <typename T>
static inline BTLA_CODE inv(const T* src0ptr, T* dstptr, size_t size) {
  for (size_t i = 0; i < size; i++) {
    float tmp = 1.f / float(src0ptr[i]);
    dstptr[i] = tmp;
  }
  return BTLA_CODE::Success;
}

template <typename T>
static inline BTLA_CODE add(const T* src0ptr, const T* src1ptr, T* dstptr, size_t size) {
  for (size_t i = 0; i < size; i++) {
    float tmp = float(src0ptr[i]) + float(src1ptr[i]);
    dstptr[i] = tmp;
  }
  return BTLA_CODE::Success;
}

template <typename T_DST>
static inline BTLA_CODE scale_exp_acc_sum_fp32(const float* src, const int src_step, T_DST* dst, int ld_dst,
                                               float* dst_sum, const int M_offset, const int N_offset, const int M,
                                               const int N, float scale, int causal_offset, void* /* tmpcache */,
                                               size_t /* cachesize */) {
  for (int i = 0; i < M; ++i) {
    const auto N_unmasked = std::min(N, causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + causal_offset + 1);
    for (int j = 0; j < N_unmasked; ++j) {
      const auto exp_ = expf(src[i * src_step + j] * scale);
      dst[i * ld_dst + j] = static_cast<T_DST>(exp_);
      dst_sum[i] += exp_;
    }
    if (N_unmasked < utils::padto(N, 64))
      memset(dst + i * ld_dst + N_unmasked, 0, sizeof(*dst) * (utils::padto(N, 64) - N_unmasked));
  }
  return BTLA_CODE::Success;
}

template <typename T_SRC, typename T_DST>
static inline BTLA_CODE scale_track_max(const T_SRC* src, const int src_step, T_DST* dst, T_DST* dst_max, int ld_dst,
                                        const int M_offset, const int N_offset, const int M, const int N, float scale,
                                        int causal_offset, float alibi_slope, float tanh_scale, int padding_type,
                                        void* tmpcache, size_t cachesize) {
  for (int i = 0; i < M; ++i) {
    const auto N_unmasked = std::min(N, padding_type == 0   ? INT32_MAX
                                        : padding_type == 1 ? i + M_offset - N_offset + causal_offset + 1
                                                            : causal_offset - N_offset);
    for (int j = 0; j < N_unmasked; ++j) {
      const auto val_ = float(src[i * src_step + j]) * scale;
      dst[i * ld_dst + j] = static_cast<T_DST>(val_);
      dst_max[i] = std::max(dst_max[i], val_);
    }
  }
  return BTLA_CODE::Success;
}
}  // namespace ref
}  // namespace kernel
}  // namespace bestla
