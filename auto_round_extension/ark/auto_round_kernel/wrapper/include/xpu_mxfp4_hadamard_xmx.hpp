//
// Copyright (c) 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// XMX (Xe matrix / DPAS) fast path for the activation fused 32-point Hadamard
// transform + MXFP4 quantization. Opt-in via ``use_xmx``; deliberately a
// *relaxed* numerical contract (see xpu_mxfp4_hadamard_design_revised.md
// §11.4/§11.10): the Hadamard matrix is stored in FP16/BF16 (same dtype as the
// activation) and the transform runs on XMX DPAS with FP32 accumulation, which
// is neither bit-exact with the FWHT path nor with Path A. Acceptance is
// tolerance based: SQNR >= 15 dB, max relative error < 0.25.
//
// The fused kernel computes, for each 32-element group ``g`` (one row of the
// flattened activation ``[total_groups, 32]``):
//
//   y[g][i] = sum_j H_T[j][i] * x[g][j]          (DPAS, FP32 accumulate)
//
// i.e. ``y = H_T @ x^T`` with H_T the Hadamard matrix stored as T (fp16/bf16).
// Each XMX lane owns one full group (its 32 outputs live entirely in the lane's
// fragment), so absmax / E8M0 / E2M1 / packing / writeback are all lane-local:
// no SLM, no barriers, no cross-lane shuffles. The MMA's real N-tile is 64 (not
// the 32 passed to choose_tiled_mma_tile), so the global group index uses
// ``get<1>(mma.tile_mnk())`` as the per-workgroup stride.
//
// Measured on Arc Pro B60 (m = 262144 groups, BF16, memory-bound): ~335-418 GB/s
// vs ~186 GB/s Path A and ~395 GB/s streaming-copy baseline.

#pragma once

#include <cstdint>

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

// Only the cute layer is needed (XE_DPAS atoms, block-2D copies, tiled-MMA
// helper). The cutlass-facing sycl_tla_dense_gemm.hpp is deliberately NOT
// included: pulling in cutlass headers makes cute::half_t resolve to
// cutlass::half_t and breaks the bf16 reorder (ambiguous conversion).
// cute/tensor.hpp brings in the XE_DPAS traits; do not include
// cute/atom/mma_traits_xe.hpp explicitly (it pulls in cutlass::half_t).
#include "cute/tensor.hpp"

namespace ark {
namespace xmx_hadamard_detail {

using namespace cute;

// Branchless E2M1 magnitude index, equivalent to the piecewise thresholds
// <=.25 / <.75 / <=1.25 / <1.75 / <=2.5 / <3.5 / <=5 used by
// XpuMxfp4Hadamard::e2m1_magnitude_index. On the SPIR-V/OpenCL target vector
// comparisons return -1 (true) / 0 (false), hence the negation.
inline int e2m1_index(float a) {
  return int(a > 0.25f) + int(a >= 0.75f) + int(a > 1.25f) + int(a >= 1.75f) +
         int(a > 2.5f) + int(a >= 3.5f) + int(a > 5.0f);
}

// The DPAS atom is pinned to half input: bf16 values (8-bit mantissa) convert
// losslessly to half (10-bit mantissa) in range, so one atom serves both
// activation dtypes. Verified bit-exact against the fp32 reference for both
// fp16 and bf16 on Arc Pro B60 (see bench_xmx_final_fragquant.cpp).
template <typename TA, typename TB, typename TC>
auto choose_mma_op() {
  return XE_DPAS_TT<8, float, cute::half_t>{};
}

template <int TileM, int TileN, class SGLayout, class ATensor, class BTensor, class CTensor>
auto choose_tiled_mma_tile(ATensor const& A, BTensor const& B, CTensor const&) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  using TC = typename CTensor::element_type;

  auto op = choose_mma_op<TA, TB, TC>();

  constexpr bool byte = (cute::max(sizeof_bits_v<TA>, sizeof_bits_v<TB>) <= 8);
  constexpr bool a_t = is_constant_v<1, decltype(stride<0>(A))>;
  constexpr bool b_n = is_constant_v<1, decltype(stride<0>(B))>;
  constexpr bool use_1x_dpas_per_k = a_t || (byte && b_n);

  using _K = conditional_t<use_1x_dpas_per_k, C<op.K>, C<op.K * 2>>;
  using WGTile = Shape<Int<TileM>, Int<TileN>, _K>;
  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;

  return MMA{};
}

// A = H_T [32, 32] (row-major, T = fp16/bf16), B = activation [m, 32] (row-major).
// Each workgroup handles one n-tile of tile_n = 64 groups; each lane owns one
// full group (a 32-row column of the C tile), so the whole quant is lane-local.
template <int TileM, int TileN, class ATensor, class BTensor, class TiledMMA>
void fused_core_xmx(ATensor const& A, BTensor const& B, uint8_t* out_codes, uint8_t* out_scale, int64_t m,
                    TiledMMA const& mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto wg_n = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));

  auto C = make_tensor(make_gmem_ptr(static_cast<float*>(nullptr)), make_shape(m, Int<TileN>{}),
                       make_stride(Int<TileN>{}, Int<1>{}));

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));
  Tensor tCgC = thr_mma.partition_C(gC);

  clear(tCrC);
  copy(copy_a, tAgA(_, _, _, 0), tArA);
  copy(copy_b, tBgB(_, _, _, 0), tBrB);
  reorder(tArA, tCrA);
  reorder(tBrB, tCrB);
  gemm(mma, tCrA, tCrB, tCrC);

  // ---- lane-local quant (1 lane = 1 full group) ----
  const int col = get<1>(tCgC(0));  // group within the workgroup tile
  const int tile_n = get<1>(mma.tile_mnk());  // 64, NOT the template TileN
  const int64_t g = (int64_t)wg_n * tile_n + col;

  // absmax over the lane's 32 outputs (4 x float8 SIMD).
  float amax = 0.0f;
  for (int c = 0; c < 4; ++c) {
    sycl::vec<float, 8> v;
    for (int t = 0; t < 8; ++t) v[t] = tCrC(c * 8 + t);
    const sycl::vec<float, 8> av = sycl::fabs(v);
    amax = sycl::fmax(amax, sycl::fmax(sycl::fmax(av[0], av[1]), sycl::fmax(av[2], av[3])));
    amax = sycl::fmax(amax, sycl::fmax(sycl::fmax(av[4], av[5]), sycl::fmax(av[6], av[7])));
  }

  uint8_t e8m0 = 0;
  float inv = 0.0f;
  if (amax > 0.0f) {
    // e8m0 = biased_exponent - 2, read straight from the fp32 bits (no ilogb).
    const int b = int(sycl::bit_cast<uint32_t>(amax) >> 23) - 2;
    const int bcl = b < 0 ? 0 : (b > 254 ? 254 : b);
    e8m0 = static_cast<uint8_t>(bcl);
    inv = sycl::bit_cast<float>((254u - static_cast<uint32_t>(bcl)) << 23);  // 2^(127-b)
  }

  uint32_t packed[4] = {0u, 0u, 0u, 0u};
  if (amax > 0.0f) {
    for (int c = 0; c < 4; ++c) {
      sycl::vec<float, 8> v;
      for (int t = 0; t < 8; ++t) v[t] = tCrC(c * 8 + t);
      const sycl::vec<float, 8> aq = sycl::fabs(v) * inv;
      // NOTE: on this target vec comparisons return -1 (true) / 0 (false).
      const sycl::vec<int, 8> idx =
          -(aq > 0.25f) - (aq >= 0.75f) - (aq > 1.25f) - (aq >= 1.75f) -
          (aq > 2.5f) - (aq >= 3.5f) - (aq > 5.0f);            // 0..7
      const sycl::vec<uint32_t, 8> u = sycl::bit_cast<sycl::vec<uint32_t, 8>>(v);
      const sycl::vec<uint32_t, 8> iu = idx.template convert<uint32_t>();
      const sycl::vec<uint32_t, 8> iszero =
          sycl::vec<int, 8>(-(idx == sycl::vec<int, 8>(0))).template convert<uint32_t>();
      // Canonical zero: magnitude index 0 always encodes 0x0, never 0x8.
      const sycl::vec<uint32_t, 8> code = (iu | (((u >> 31) & 1u) << 3)) & ~(0u - iszero);
      for (int t = 0; t < 8; ++t) {
        const int row = c * 8 + t;
        packed[row >> 3] |= (code[t] & 0xF) << ((row & 7) * 4);
      }
    }
  }
  if (g < m) {
    auto* dst4 = reinterpret_cast<sycl::vec<uint32_t, 4>*>(out_codes + g * (32 / 2));
    *dst4 = sycl::vec<uint32_t, 4>{packed[0], packed[1], packed[2], packed[3]};
    out_scale[g] = e8m0;
  }
}

template <int TileM, int TileN, class SGLayout, class Element>
void fused_launch_xmx(sycl::queue* q, int64_t m, const Element* h_ptr, const Element* a_ptr, uint8_t* out_codes,
                      uint8_t* out_scale) {
  // A = H_T [32,32] (row-major), B = activation [m,32] (row-major).
  auto A = make_tensor(make_gmem_ptr(const_cast<Element*>(h_ptr)), make_shape(32, 32), make_stride(32, _1{}));
  auto B = make_tensor(make_gmem_ptr(const_cast<Element*>(a_ptr)), make_shape(m, 32), make_stride(32, _1{}));
  auto C = make_tensor(make_gmem_ptr(static_cast<float*>(nullptr)), make_shape(TileM, TileN), make_stride(TileN, _1{}));
  auto mma = choose_tiled_mma_tile<TileM, TileN, SGLayout>(A, B, C);

  sycl::range<2> local = {size(mma), 1};
  sycl::range<2> global = {local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
                           local[1] * ceil_div(shape<0>(A), get<0>(mma.tile_mnk()))};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<16>, intelex::grf_size<256>};

  q->parallel_for(sycl::nd_range<2>(global, local), kernel_props,
                  [=](sycl::nd_item<2>) { fused_core_xmx<TileM, TileN>(A, B, out_codes, out_scale, m, mma); });
}

// Convert the FP32 Hadamard matrix to the activation dtype AND transpose it.
// The fused GEMM is ``C = A @ B^T`` with A = the Hadamard matrix, so ``C[r][c]``
// = ``sum_k A[r][k] * x[c][k]``. We need ``y[group c][element r] = sum_k x[c][k]
// * H[k][r]``, which forces ``A[r][k] = H[k][r]`` -- i.e. A must be H^T. The
// transpose is applied here (once per call, 1024 elements, negligible) so the
// kernel entry point simply takes the logical (row-major) Hadamard matrix and
// always produces ``y = x @ H`` regardless of symmetry. The normalized Sylvester
// matrix is symmetric so a missing transpose is silently masked; non-symmetric
// custom matrices exposed the bug (see xpu_mxfp4_hadamard_design_revised.md
// §11.12).
template <typename T>
void convert_hadamard_to_dtype(sycl::queue* q, const float* h_fp32, T* h_t) {
  q->parallel_for(32 * 32, [=](sycl::id<1> i) {
    const int r = int(i) / 32;  // row of H
    const int c = int(i) % 32;  // column of H
    h_t[c * 32 + r] = static_cast<T>(h_fp32[i]);  // write H^T
  });
}

// Public entry point. x is the activation [num_rows, k] flattened to
// [total_groups, 32]; h is the Hadamard matrix [32,32] in the *same* dtype as
// the activation (T), already normalized by 1/sqrt(32), row-major. The kernel
// computes y = x @ h (the transpose for the C = A @ B^T GEMM convention is the
// caller's responsibility -- see convert_hadamard_to_dtype).
template <typename T>
void mxfp4_hadamard_quant_xmx(sycl::queue* q, const T* x, const T* h, uint8_t* out_codes, uint8_t* out_scale,
                              int64_t total_groups) {
  if (total_groups <= 0) {
    return;
  }
  using SmallTileSG = Layout<Shape<_1, _4, _1>, Stride<_0, _1, _0>>;
  fused_launch_xmx<32, 32, SmallTileSG, T>(q, total_groups, h, x, out_codes, out_scale);
}

}  // namespace xmx_hadamard_detail
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
