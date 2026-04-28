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
#include <stdint.h>
enum class BTLA_CODE {
  Success = 0,
  InvalidParam = 1,
  InvalidISA = 2,
  RuntimeError = 4,
  NotSupport = 8,
};
enum class BTLA_ISA : uint8_t {
  NoSIMD = 0,
  AVX,
  AVX2,
  AVX_VNNI,
  AVX512F,
  AVX512BW,
  AVX512_VNNI,
  AVX512_BF16,
  AVX512_FP16,
  AMX_BF16,
  AMX_INT8,
  AMX_FP16,
  ISA_COUNT,
};
enum class BTLA_DTYPE : uint32_t {
  EleBitsMask = 0xff,
  EleBitsShift = 0,
  EleBitsUndef = 0,
  EleBits1 = 1,
  EleBits2 = 2,
  EleBits3 = 3,
  EleBits4 = 4,
  EleBits5 = 5,
  EleBits6 = 6,
  EleBits7 = 7,
  EleBits8 = 8,
  EleBits16 = 16,
  EleBits32 = 32,
  EleBits64 = 64,
  TypeMask = 0xff00,
  TypeShift = 8,
  TypeFloat = 0 << TypeShift,
  TypeInt = 1 << TypeShift,
  SubTypeMask = 0xff0000,
  SubTypeShift = 16,
  SubType0 = 0 << SubTypeShift,
  SubType1 = 1 << SubTypeShift,
  SubType2 = 2 << SubTypeShift,
  SubType3 = 3 << SubTypeShift,
  SubType4 = 4 << SubTypeShift,
  F64 = EleBits64 | TypeFloat,
  F32 = EleBits32 | TypeFloat,
  F16 = EleBits16 | TypeFloat,
  BF16 = EleBits16 | TypeFloat | SubType1,
  F8_E4M3 = EleBits8 | TypeFloat,
  F8_E5M2 = EleBits8 | TypeFloat | SubType1,
  F8_E3M4 = EleBits8 | TypeFloat | SubType2,
  F8_E8M0 = EleBits8 | TypeFloat | SubType3,
  DQ8_BNB = EleBits8 | TypeFloat | SubType4,
  S8 = EleBits8 | TypeInt,
  U8 = EleBits8 | TypeInt | SubType1,
  S1_CLIP = EleBits1 | TypeInt,
  S1 = S1_CLIP,
  S2_CLIP = EleBits2 | TypeInt,
  S2 = S2_CLIP,
  S3_CLIP = EleBits3 | TypeInt,
  S3 = S3_CLIP,
  S4_CLIP = EleBits4 | TypeInt,
  S4 = S4_CLIP,
  S5_CLIP = EleBits5 | TypeInt,
  S5 = S5_CLIP,
  S6_CLIP = EleBits6 | TypeInt,
  S6 = S6_CLIP,
  S7_CLIP = EleBits7 | TypeInt,
  S7 = S7_CLIP,
  F4_E2M1 = EleBits4 | TypeFloat,
  F4_BNB = EleBits4 | TypeFloat | SubType1,
  F4_NF4 = EleBits4 | TypeFloat | SubType2,
  S32 = EleBits32 | TypeInt,
  U32 = EleBits32 | TypeInt | SubType1,
};

enum class BTLA_ELTWISEOP { GELU, SWISH, TANH, EXP, LOW_PRECISION_EXP, RELU, LINEAR };

enum class BTLA_PROLOGUEB_IDS : uint32_t {
  Undef = (uint32_t)-1,
  Begin = 0,
  NormalBegin = Begin,
  WeightPack = NormalBegin,
  NormalEnd,
  KBlockBegin = NormalEnd,
  WeightKBlockNInteger = KBlockBegin,
  WeightKBlockNFloat,
  KBlockEnd,
  End,
};
