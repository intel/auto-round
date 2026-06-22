//  Copyright (c) 2026 Intel Corporation
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

#include "ark/cpu/mha_dense.h"
#include "ark/cpu/mha_dense_wrapper.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ark::cpu {
namespace {

float fp16_to_float(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000U)) << 16;
  uint32_t exp = (h >> 10) & 0x1FU;
  uint32_t mant = h & 0x03FFU;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400U) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x03FFU;
      bits = sign | ((exp + 112U) << 23) | (mant << 13);
    }
  } else if (exp == 0x1FU) {
    bits = sign | 0x7F800000U | (mant << 13);
  } else {
    bits = sign | ((exp + 112U) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint16_t float_to_fp16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000U;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
  uint32_t mant = bits & 0x7FFFFFU;
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant = (mant | 0x800000U) >> (1 - exp);
    return static_cast<uint16_t>(sign | ((mant + 0x1000U) >> 13));
  }
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00U);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000U) >> 13));
}

float bf16_to_float(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint16_t float_to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1U;
  bits += 0x7FFFU + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

size_t qko_offset(const AttentionStrides& strides, int b, int h, int s, int d) {
  return static_cast<size_t>(b) * strides.batch + static_cast<size_t>(h) * strides.head +
         static_cast<size_t>(s) * strides.seq + static_cast<size_t>(d) * strides.dim;
}

size_t value_offset(const ValueStrides& strides, int b, int h, int s, int d) {
  return static_cast<size_t>(b) * strides.batch + static_cast<size_t>(h) * strides.head +
         static_cast<size_t>(s) * strides.seq + static_cast<size_t>(d) * strides.dim;
}

void validate_args(const MhaDenseArgs& args) {
  if (!args.query || !args.key || !args.value || !args.output) {
    throw std::invalid_argument("ark::cpu::sdpa: Q/K/V/O pointers must be non-null");
  }
  if (args.batch <= 0 || args.num_heads_q <= 0 || args.num_heads_kv <= 0 || args.seq_len_q <= 0 ||
      args.seq_len_kv <= 0 || args.head_dim <= 0) {
    throw std::invalid_argument("ark::cpu::sdpa: dimensions must be positive");
  }
  if (args.num_heads_q % args.num_heads_kv != 0) {
    throw std::invalid_argument("ark::cpu::sdpa: num_heads_q must be divisible by num_heads_kv for GQA");
  }
  if (args.q_strides.dim != 1 || args.k_strides.dim != 1 || args.v_strides.dim != 1 || args.o_strides.dim != 1) {
    throw std::invalid_argument("ark::cpu::sdpa: head-dim stride must be 1 for Q/K/V/O");
  }
  (void)element_size(args.dtype);
}

int effective_kv_block(const MhaDenseArgs& args) {
  const int block = args.kv_block_size > 0 ? args.kv_block_size : kDefaultKvBlock;
  return std::min(block, args.seq_len_kv);
}

int max_threads() {
#ifdef _OPENMP
  return std::max(1, omp_get_max_threads());
#else
  return 1;
#endif
}

int current_thread() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

}  // namespace

size_t mha_dense_workspace_size(const MhaDenseArgs& args) {
  // Per thread we keep an output accumulator (head_dim) plus a score tile
  // (kv_block_size) of FP32 scratch.
  const size_t per_thread =
      static_cast<size_t>(args.head_dim) + static_cast<size_t>(effective_kv_block(args));
  return per_thread * static_cast<size_t>(max_threads());
}

size_t element_size(BTLA_DTYPE dtype) {
  switch (dtype) {
    case BTLA_DTYPE::F32:
      return sizeof(float);
    case BTLA_DTYPE::BF16:
    case BTLA_DTYPE::F16:
      return sizeof(uint16_t);
    default:
      throw std::invalid_argument("ark::cpu::sdpa: only FP32, BF16, and FP16 tensors are supported");
  }
}

float load_scalar(const void* base, size_t element_offset, BTLA_DTYPE dtype) {
  switch (dtype) {
    case BTLA_DTYPE::F32:
      return static_cast<const float*>(base)[element_offset];
    case BTLA_DTYPE::BF16:
      return bf16_to_float(static_cast<const uint16_t*>(base)[element_offset]);
    case BTLA_DTYPE::F16:
      return fp16_to_float(static_cast<const uint16_t*>(base)[element_offset]);
    default:
      throw std::invalid_argument("ark::cpu::sdpa: unsupported dtype");
  }
}

void store_scalar(void* base, size_t element_offset, BTLA_DTYPE dtype, float value) {
  switch (dtype) {
    case BTLA_DTYPE::F32:
      static_cast<float*>(base)[element_offset] = value;
      return;
    case BTLA_DTYPE::BF16:
      static_cast<uint16_t*>(base)[element_offset] = float_to_bf16(value);
      return;
    case BTLA_DTYPE::F16:
      static_cast<uint16_t*>(base)[element_offset] = float_to_fp16(value);
      return;
    default:
      throw std::invalid_argument("ark::cpu::sdpa: unsupported dtype");
  }
}

void mha_dense_forward(const MhaDenseArgs& args) {
  validate_args(args);
  const int group_size = args.num_heads_q / args.num_heads_kv;
  const int causal_shift = args.seq_len_kv - args.seq_len_q;
  const int head_dim = args.head_dim;
  const int kv_block = effective_kv_block(args);
  const size_t per_thread = static_cast<size_t>(head_dim) + static_cast<size_t>(kv_block);

  // Use the caller-provided workspace when available, otherwise fall back to a
  // self-managed buffer so the kernel stays usable in isolation.
  std::vector<float> local_workspace;
  float* workspace = args.workspace;
  if (workspace == nullptr) {
    local_workspace.resize(per_thread * static_cast<size_t>(max_threads()));
    workspace = local_workspace.data();
  }
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();

#pragma omp parallel for collapse(3) schedule(static)
  for (int b = 0; b < args.batch; ++b) {
    for (int hq = 0; hq < args.num_heads_q; ++hq) {
      for (int sq = 0; sq < args.seq_len_q; ++sq) {
        const int hkv = hq / group_size;
        float* scratch = workspace + static_cast<size_t>(current_thread()) * per_thread;
        float* acc = scratch;                 // [head_dim] output accumulator
        float* tile_scores = scratch + head_dim;  // [kv_block] score tile

        for (int d = 0; d < head_dim; ++d) {
          acc[d] = 0.0f;
        }
        float running_max = kNegInf;  // m_i
        float running_sum = 0.0f;     // l_i

        for (int kv_start = 0; kv_start < args.seq_len_kv; kv_start += kv_block) {
          const int kv_end = std::min(kv_start + kv_block, args.seq_len_kv);
          float tile_max = kNegInf;

          // Stage 1: compute the raw scores for this K tile and its max.
          for (int sk = kv_start; sk < kv_end; ++sk) {
            float score = kNegInf;
            if (!(args.is_causal && sk > sq + causal_shift)) {
              score = 0.0f;
              for (int d = 0; d < head_dim; ++d) {
                const float q = load_scalar(args.query, qko_offset(args.q_strides, b, hq, sq, d), args.dtype);
                const float k = load_scalar(args.key, qko_offset(args.k_strides, b, hkv, sk, d), args.dtype);
                score += q * k;
              }
              score *= args.softmax_scale;
              if (args.attn_mask) {
                score += args.attn_mask[(static_cast<size_t>(b) * args.seq_len_q + sq) * args.seq_len_kv + sk];
              }
            }
            tile_scores[sk - kv_start] = score;
            tile_max = std::max(tile_max, score);
          }

          // Fully masked tile contributes nothing.
          if (!std::isfinite(tile_max)) {
            continue;
          }

          // Stage 2: online softmax rescaling against the new running max.
          const float new_max = std::max(running_max, tile_max);
          const float alpha = std::isfinite(running_max) ? std::exp(running_max - new_max) : 0.0f;
          if (alpha != 1.0f) {
            running_sum *= alpha;
            for (int d = 0; d < head_dim; ++d) {
              acc[d] *= alpha;
            }
          }

          // Stage 3: accumulate the rescaled probabilities and weighted values.
          for (int sk = kv_start; sk < kv_end; ++sk) {
            const float score = tile_scores[sk - kv_start];
            if (!std::isfinite(score)) {
              continue;
            }
            const float p = std::exp(score - new_max);
            running_sum += p;
            for (int d = 0; d < head_dim; ++d) {
              const float v = load_scalar(args.value, value_offset(args.v_strides, b, hkv, sk, d), args.dtype);
              acc[d] += p * v;
            }
          }
          running_max = new_max;
        }

        const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          store_scalar(args.output, qko_offset(args.o_strides, b, hq, sq, d), args.dtype, acc[d] * inv_sum);
        }
      }
    }
  }
}

}  // namespace ark::cpu
