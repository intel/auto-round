// Temporary BF16-only sparse SDPA binding for fast kernel iteration.

#if defined(ARK_SPARSE_BF16_ONLY)

#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include "flash_attn_dtype.hpp"
#include "sparse_execution_schedule.hpp"
#include "sycl_tla_common.hpp"

namespace ark {
namespace {

using torch_ptr = std::uintptr_t;

void block_sparse_sdpa_bf16(
    torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask, torch_ptr lut,
    torch_ptr valid_block_num, int num_q_blocks, int num_k_blocks, int q_tile_override, int q_stride_s,
    int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b,
    int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h,
    int o_stride_b, int q_dtype, int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
    int head_dim, float softmax_scale, bool is_causal, torch_ptr execution_schedule,
    torch_ptr execution_schedule_rounds, int execution_schedule_max_rounds) {
  if (q_dtype != static_cast<int>(FlashAttnDtype::BF16)) {
    throw std::invalid_argument("BF16-only sparse build requires BF16 Q/K/V/O");
  }
  if (mask != 0 || is_causal) {
    throw std::invalid_argument("BF16-only sparse build supports non-causal, unmasked attention only");
  }
  if (lut == 0 || valid_block_num == 0) {
    throw std::invalid_argument("lut and valid_block_num must be provided");
  }
  if (head_dim != 128) {
    throw std::invalid_argument("BF16-only sparse build supports head_dim=128 only");
  }
  auto matches_block_size = [](int seq_len, int num_blocks, int block_size) {
    return block_size > 0 && num_blocks == ((seq_len + block_size - 1) / block_size);
  };
  if (!matches_block_size(seq_len_kv, num_k_blocks, 64) && !matches_block_size(seq_len_kv, num_k_blocks, 128)) {
    throw std::invalid_argument("BF16-only sparse build requires 64- or 128-token KV route blocks");
  }

  if (q_tile_override == 128 && matches_block_size(seq_len_q, num_q_blocks, 128)) {
    sdpa_impl_bf16_sparse_sdpa_qtile128_row64k(
        reinterpret_cast<sycl::queue*>(stream), reinterpret_cast<void*>(Q), reinterpret_cast<void*>(K),
        reinterpret_cast<void*>(V), reinterpret_cast<void*>(O), nullptr, reinterpret_cast<void*>(lut),
        reinterpret_cast<void*>(valid_block_num), num_q_blocks, num_k_blocks, q_tile_override, q_stride_s,
        q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
        v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
        seq_len_q, seq_len_kv, head_dim, softmax_scale, false, 128);
    return;
  }
  if (matches_block_size(seq_len_q, num_q_blocks, 256)) {
    if (q_tile_override != 256) {
      throw std::invalid_argument("256-token query blocks require q_tile_override=256");
    }
    sdpa_impl_bf16_sparse_sdpa_qtile256_row64k(
        reinterpret_cast<sycl::queue*>(stream), reinterpret_cast<void*>(Q), reinterpret_cast<void*>(K),
        reinterpret_cast<void*>(V), reinterpret_cast<void*>(O), nullptr, reinterpret_cast<void*>(lut),
        reinterpret_cast<void*>(valid_block_num), num_q_blocks, num_k_blocks, q_tile_override, q_stride_s,
        q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
        v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
        seq_len_q, seq_len_kv, head_dim, softmax_scale, false, 256, (int const*)execution_schedule,
        (int const*)execution_schedule_rounds, execution_schedule_max_rounds);
    return;
  }
  if (matches_block_size(seq_len_q, num_q_blocks, 64)) {
    if (q_tile_override == 256 || q_tile_override == 512) {
      sdpa_impl_bf16_sparse_sdpa_qtile256_row64k(
          reinterpret_cast<sycl::queue*>(stream), reinterpret_cast<void*>(Q), reinterpret_cast<void*>(K),
          reinterpret_cast<void*>(V), reinterpret_cast<void*>(O), nullptr, reinterpret_cast<void*>(lut),
          reinterpret_cast<void*>(valid_block_num), num_q_blocks, num_k_blocks, q_tile_override, q_stride_s,
          q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
          v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
          seq_len_q, seq_len_kv, head_dim, softmax_scale, false, 64, (int const*)execution_schedule,
          (int const*)execution_schedule_rounds, execution_schedule_max_rounds);
    } else if (q_tile_override == 128) {
      sdpa_impl_bf16_sparse_sdpa_qtile128_row64k(
          reinterpret_cast<sycl::queue*>(stream), reinterpret_cast<void*>(Q), reinterpret_cast<void*>(K),
          reinterpret_cast<void*>(V), reinterpret_cast<void*>(O), nullptr, reinterpret_cast<void*>(lut),
          reinterpret_cast<void*>(valid_block_num), num_q_blocks, num_k_blocks, q_tile_override, q_stride_s,
          q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
          v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
          seq_len_q, seq_len_kv, head_dim, softmax_scale, false, 64);
    } else {
      sdpa_impl_bf16_sparse_sdpa_row_linear(
          reinterpret_cast<sycl::queue*>(stream), reinterpret_cast<void*>(Q), reinterpret_cast<void*>(K),
          reinterpret_cast<void*>(V), reinterpret_cast<void*>(O), nullptr, reinterpret_cast<void*>(lut),
          reinterpret_cast<void*>(valid_block_num), num_q_blocks, num_k_blocks, q_tile_override, q_stride_s,
          q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
          v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
          seq_len_q, seq_len_kv, head_dim, softmax_scale, false);
    }
    return;
  }
  throw std::invalid_argument("BF16-only sparse build supports 64-, 128-, or 256-token query blocks");
}

}  // namespace
}  // namespace ark

PYBIND11_MODULE(PY_NAME, m) {
  m.doc() = "Temporary ARK BF16 sparse SDPA benchmark binding";
  m.def("block_sparse_sdpa", &ark::block_sparse_sdpa_bf16, pybind11::arg("stream"), pybind11::arg("Q"),
        pybind11::arg("K"), pybind11::arg("V"), pybind11::arg("O"), pybind11::arg("mask"),
        pybind11::arg("lut"), pybind11::arg("valid_block_num"), pybind11::arg("num_q_blocks"),
        pybind11::arg("num_k_blocks"), pybind11::arg("q_tile_override"), pybind11::arg("q_stride_s"),
        pybind11::arg("q_stride_d"), pybind11::arg("q_stride_h"), pybind11::arg("q_stride_b"),
        pybind11::arg("k_stride_s"), pybind11::arg("k_stride_d"), pybind11::arg("k_stride_h"),
        pybind11::arg("k_stride_b"), pybind11::arg("v_stride_d"), pybind11::arg("v_stride_s"),
        pybind11::arg("v_stride_h"), pybind11::arg("v_stride_b"), pybind11::arg("o_stride_s"),
        pybind11::arg("o_stride_d"), pybind11::arg("o_stride_h"), pybind11::arg("o_stride_b"),
        pybind11::arg("q_dtype"), pybind11::arg("batch"), pybind11::arg("num_heads_q"),
        pybind11::arg("num_heads_kv"), pybind11::arg("seq_len_q"), pybind11::arg("seq_len_kv"),
        pybind11::arg("head_dim"), pybind11::arg("softmax_scale"), pybind11::arg("is_causal"),
        pybind11::arg("execution_schedule") = 0, pybind11::arg("execution_schedule_rounds") = 0,
        pybind11::arg("execution_schedule_max_rounds") = 0);
  m.def("build_sparse_execution_schedule", &ark::build_sparse_execution_schedule,
        pybind11::arg("stream"), pybind11::arg("lut"), pybind11::arg("valid_block_num"),
        pybind11::arg("execution_schedule"), pybind11::arg("execution_schedule_rounds"),
        pybind11::arg("batch"), pybind11::arg("num_heads"), pybind11::arg("num_q_blocks"),
        pybind11::arg("num_k_blocks"), pybind11::arg("rows_per_group"),
        pybind11::arg("max_rounds"));
}

#endif  // ARK_SPARSE_BF16_ONLY
