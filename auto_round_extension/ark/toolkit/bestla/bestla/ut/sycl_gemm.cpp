#include "bestla_ut.h"
#include "sycl_ut.h"
#include "../sycl/sycl_wrapper.h"
#include "../sycl/fp8_lut.h"
#include "bestla_prologue_b.h"
#include "kernel_wrapper.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#undef BTLA_UT_SYCL
namespace bestla {
using namespace ut;
using namespace utils;
using namespace sycl_utils;
using namespace sycl_gemm;
namespace syclex = sycl::ext::oneapi::experimental;
namespace syclintelex = sycl::ext::intel::experimental;
namespace sycl_ut {
namespace {
template <bool IsE4M3>
inline uint8_t sanitize_fp8_byte(uint8_t raw) {
  if constexpr (IsE4M3) {
    uint8_t exp = (raw >> 3) & 0xF;
    uint8_t mant = raw & 0x7;
    if (exp == 0xF && mant == 0x7) {
      raw = (raw & 0xF8) | 0x6;  // avoid NaN payload
    }
  } else {
    uint8_t exp = (raw >> 2) & 0x1F;
    if (exp == 0x1F) {
      raw = (raw & 0x83) | (0x1E << 2);  // drop Inf/NaN exponent
    }
  }
  return raw;
}

template <bool IsE4M3>
inline void sanitize_fp8_buffer(uint8_t* data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = sanitize_fp8_byte<IsE4M3>(data[i]);
  }
}

template <bool IsE4M3>
inline float decode_fp8_byte(uint8_t raw) {
  const uint8_t mag = raw & 0x7F;
  const float v = IsE4M3 ? sycl_prologue_b::fp8_lut::lut_e4m3_128[mag] : sycl_prologue_b::fp8_lut::lut_e5m2_128[mag];
  return (raw & 0x80) ? -v : v;
}
}  // namespace

class UT_SyclS4IGemm {
 public:
  UT_SyclS4IGemm() {
    UT_START();
    // ut_testT2();
    // ut_testT1(552, 552, 96);
    ut_testT3(4096, 4096, 4096, 64);
    ut_testT3(4096, 4096, 4096, 128);
    ut_testT3_half(4096, 4096, 4096, 64);
    ut_testT3_half(4096, 4096, 4096, 128);
    // ut_testT3(4096, 4096, 4096, 256);
    // ut_testT3(4096, 4096, 4096, 512);
    // ut_testT3(4096, 4096, 4096, 1024);
    // ut_testT3(4096, 4096, 4096, 4096);
    // ut_testT4(4096, 4096, 4096, 128);
    // ut_testT1(4096, 4096, 4096);
    // ut_testT1(4096, 11008, 4096);
    // ut_testT1(4096, 16384, 4096);
  }

  void ut_testT4(int m, int n, int k, int blocksize) {
    int constexpr runs = 100;
    int constexpr sg_size = 16;
    int constexpr sg_row = 8;                           // half threads of a XVE
    int constexpr sg_col = 4;                           // half threads of a XVE
    int constexpr wg_size = sg_size * sg_row * sg_col;  // half threads of a XVE
    int constexpr TM = 8;
    int constexpr TN = 16;
    int constexpr TK = 32;
    int constexpr SGM = 4;
    int constexpr SGN = 4;
    int constexpr UnrollK = 2;
    // A large GRF kernel
    // SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
    int constexpr PrefetchDis = 3;
    int constexpr G_NROW = 3;
    int constexpr BM = TM * sg_row * SGM;
    int constexpr BM_STRIDE = BM + 8;
    int constexpr BN = TN * sg_col * SGN;
    int constexpr BN_STRIDE = BN;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace syclintelex = sycl::ext::intel::experimental;

    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<float> matAfp(m * k), scaleA(m), matB(k * n), matC(m * n), ref(m * n);
    avector<int8_t> matA(m * k);
    fill_buffer_randn(scaleA.data(), scaleA.size(), 0.001f, 0.005f);
    fill_buffer_randn(matA.data(), matA.size(), int8_t(-128), int8_t(127));
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j += 1) {
        matAfp[i * k + j] = static_cast<float>(matA[i * k + j]) * scaleA[i];
      }
    }

    int blks = k / blocksize;
    avector<float> B_scale(size_t(blks) * n);
    avector<int8_t> B_s8(k * n), B_s8NT(k * n);
    fill_buffer_randn(B_s8.data(), B_s8.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(B_scale.data(), B_scale.size(), 0.001f, 0.005f);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 1) {
        auto noffset = i * blks + j / blocksize;
        matB[i * k + j] = static_cast<float>(B_s8[i * k + j]) * B_scale[noffset];
      }
    }
    avector<float> matBNT(k * n);
    kernel::wrapper::Transpose2D<float>::forward<BTLA_ISA::NoSIMD>(matB.data(), matBNT.data(), n, k, k, n);
    kernel::wrapper::Transpose2D<int8_t>::forward<BTLA_ISA::NoSIMD>(B_s8.data(), B_s8NT.data(), n, k, k, n);
    gemmref_fp32fp32fp32(m, n, k, matAfp.data(), matBNT.data(), ref.data(), k, n, n);
    avector<int> testRef(m * n), O32(m * n);
    gemmref_s8s8s32(m, n, k, matA.data(), B_s8NT.data(), testRef.data(), k, n, n);
    sycl_vector<float> dB(matB.size(), q), dC(matC.size(), q), dB_scale(B_scale.size(), q), dA_scale(m, q);
    sycl_vector<int8_t> dA(B_s8.size(), q), dBs8(B_s8.size(), q);
    sycl_vector<int> dO(m * n, q);
    sycl_vector<int> dws0(m * n, q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 1).wait();
    q->memcpy(dBs8.data(), B_s8.data(), B_s8.size() * 1).wait();
    q->memcpy(dB_scale.data(), B_scale.data(), B_scale.size() * 4).wait();
    q->memcpy(dA_scale.data(), scaleA.data(), scaleA.size() * 4).wait();
    auto A_d = dA.data();
    auto B_d = dBs8.data();
    auto A_scale_d = dA_scale.data();
    auto B_scale_d = dB_scale.data();
    auto C_d = dC.data();
    auto C_d_s32 = dO.data();
    auto ws0 = dws0.data();

    size_t g_ncnt = (n + BN - 1) / BN;
    size_t g_mcnt = (m + BM - 1) / BM;
    size_t wg_repeat = g_mcnt * g_ncnt;
    size_t m_tail = g_mcnt % G_NROW;
    size_t g_m_aligned = g_mcnt - m_tail;
    size_t aligned_wg = g_m_aligned * g_ncnt;

    auto ker = [&](sycl::handler& cgh) {
      // sycl::stream out(65536, 256, cgh);
      // sycl::local_accessor<int, 1> slm_b(sycl::range(BN * BM), cgh);
      // sycl::local_accessor<float, 1> slm_bf(sycl::range(BN * BM), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}),
          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(sg_size)]]
          {
            auto sg = it.get_sub_group();
            int g_id = it.get_group(0);
            size_t ws_off = (size_t)g_id * BN * BM;
            int g_n = g_id % g_ncnt;
            int g_m = g_id / g_ncnt;
            if (g_id <= aligned_wg) {
              int g_m_ = g_id % G_NROW;
              g_id /= G_NROW;
              g_n = g_id % g_ncnt;
              g_id /= g_ncnt;
              g_m = g_id * G_NROW + g_m_;
            }
            if (g_n >= g_ncnt || g_m >= g_mcnt) return;

            int sgSize = sg.get_local_range()[0];
            int sgGroupId = sg.get_group_id()[0];
            int sggid_col = sgGroupId % sg_col;
            int sggid_row = sgGroupId / sg_col;

            int sgId = sg.get_local_id()[0];
            int sg_idn = sggid_col * TN * SGN;
            int sg_idm = sggid_row * TM * SGM;
            int g_idn = sggid_col * TN * SGN + g_n * BN;
            int g_idm = sggid_row * TM * SGM + g_m * BM;
            // auto pA =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(A_d);
            auto pSA = syclex::annotated_ptr{
                A_scale_d, syclex::properties{syclintelex::read_hint<
                               syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L1>,
                               syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};
            auto pSB = syclex::annotated_ptr{
                B_scale_d, syclex::properties{syclintelex::read_hint<
                               syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L1>,
                               syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};
            auto pA = syclex::annotated_ptr{
                A_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};
            // auto pB =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(B_d);
            auto pB = syclex::annotated_ptr{
                B_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
            // auto pC = sycl::address_space_cast<sycl::access::address_space::global_space,
            // sycl::access::decorated::yes>(
            //     C_d_s32);
            auto pC = syclex::annotated_ptr{
                C_d, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                         syclintelex::cache_mode::write_back, syclex::cache_level::L1, syclex::cache_level::L3>>}};
            auto pCt = syclex::annotated_ptr{
                ws0 + ws_off,
                syclex::properties{syclintelex::write_hint<
                    syclintelex::cache_control<syclintelex::cache_mode::write_back, syclex::cache_level::L3>>}};
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::a, TM, TK,
                                                                  layout::row_major>
                sub_a[SGM];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::b, TK, TN,
                                                                  layout::col_major>
                sub_b[SGN];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int32_t, use::accumulator, TM, TN>
                sub_c[SGM * SGN];

            if (TK * UnrollK * (PrefetchDis + 1) <= k)
              for (size_t i = 0; i < PrefetchDis; i++) {
                for (int im = sggid_col; im < SGM; im += sg_col)
                  if ((g_idm + (im + 1) * TM) <= m)
                    joint_matrix_prefetch<TM, TK * UnrollK>(sg, A_d + (g_idm + im * TM) * k * 1 + TK * UnrollK * i, k,
                                                            layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});

                for (int in = sggid_row; in < SGN; in += sg_row)
                  if ((g_idn + (in + 1) * TN) <= n)
                    joint_matrix_prefetch<TN, TK * UnrollK>(sg, B_d + (g_idn + in * TN) * k + TK * UnrollK * i, k,
                                                            layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});
              }

            for (size_t ib = 0; ib < blks; ib++) {
#pragma unroll
              for (int im = 0; im < SGM; im++)
#pragma unroll
                for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], 0);
#pragma unroll(1)
              for (size_t ik = ib * blocksize; ik < (ib + 1) * blocksize; ik += TK * UnrollK) {
#pragma unroll
                for (size_t ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
                  for (int in = 0; in < SGN; in++) {
                    sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                        sg, sub_b[in], pB, k, n, k, (g_idn + in * TN) * 1, ik + ikk * TK);
                  }
#pragma unroll
                  for (int im = 0; im < SGM; im++) {
                    sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                        sg, sub_a[im], pA, k, m, k, (g_idm + im * TM) * 1, ik + TK * ikk);
#pragma unroll
                    for (int in = 0; in < SGN; in++) {
                      joint_matrix_mad(sg, sub_c[im * SGN + in], sub_a[im], sub_b[in], sub_c[im * SGN + in]);
                    }
                  }
                }
#pragma unroll
                for (int im = 0; im < SGM; im++)
#pragma unroll
                  for (int in = 0; in < SGN; in++)
                    joint_matrix_prefetch<TM, TN>(sg, C_d + (g_idm + im * TM) * n + g_idn + in * TN, n,
                                                  layout::row_major, syclex::properties{syclex::prefetch_hint_L3});
                if (ik + TK * UnrollK * (PrefetchDis + 1) <= k) {
                  for (int im = sggid_col; im < SGM; im += sg_col)
                    if ((g_idm + (im + 1) * TM) <= m)
                      joint_matrix_prefetch<TM, TK * UnrollK>(
                          sg, A_d + (g_idm + im * TM) * k * 1 + ik + TK * UnrollK * PrefetchDis, k, layout::row_major,
                          syclex::properties{syclex::prefetch_hint_L1});
                  for (int in = sggid_row; in < SGN; in += sg_row)
                    if ((g_idn + (in + 1) * TN) <= n)
                      joint_matrix_prefetch<TN, TK * UnrollK>(
                          sg, B_d + (g_idn + in * TN) * k + (ik + TK * UnrollK * PrefetchDis), k, layout::row_major,
                          syclex::properties{syclex::prefetch_hint_L1});
                }
              }
#pragma unroll
              for (int im = 0; im < SGM; im++) {
#pragma unroll
                for (int in = 0; in < SGN; in++) {
                  joint_matrix_store(sg, sub_c[im * SGN + in], pCt + (im * TM + sg_idm) * BN + sg_idn + in * TN, BN,
                                     layout::row_major);
                }
              }

#pragma unroll
              for (int im = 0; im < SGM; im++) {
#pragma unroll
                for (size_t imm = 0; imm < TM; imm++) {
                  auto tmp = *(sycl::vec<int, SGN>*)&ws0[(im * TM + imm + sg_idm) * BN + sg_idn + sgId * SGN + ws_off];
                  auto tmp2 = *(sycl::vec<float, SGN>*)&C_d[(g_idm + im * TM + imm) * n + g_idn + sgId * SGN];
              // sycl::vec<float, SGN> tmp2;
              // float scaleA = pSA[g_idm + im * TM + imm];
#pragma unroll
                  for (int in = 0; in < SGN; in++) {
                    float scaleB = pSB[(g_idn + sgId * SGN + in) * blks + ib];  // * scaleA;
                    tmp2[in] = tmp2[in] + static_cast<float>(tmp[in]) * scaleB;
                  }
                  *(sycl::vec<float, SGN>*)&C_d[(g_idm + im * TM + imm) * n + g_idn + sgId * SGN] = tmp2;
                }
              }
            }
#pragma unroll
            for (int im = 0; im < SGM; im++) {
#pragma unroll
              for (size_t imm = 0; imm < TM; imm++) {
                auto tmp2 = *(sycl::vec<float, SGN>*)&C_d[(g_idm + im * TM + imm) * n + g_idn + sgId * SGN];
                float scaleA = pSA[g_idm + im * TM + imm];
#pragma unroll
                for (int in = 0; in < SGN; in++) {
                  tmp2[in] = tmp2[in] * scaleA;
                }
                *(sycl::vec<float, SGN>*)&C_d[(g_idm + im * TM + imm) * n + g_idn + sgId * SGN] = tmp2;
              }
            }
          });  // parallel for
    };
    q->wait();
    for (size_t i = 0; i < runs; i++) {
      q->submit(ker);
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      q->submit(ker);
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2;
    printf("Time %f us, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    // q->memcpy(O32.data(), C_d_s32, matC.size() * 4).wait();
    // buffer_error(testRef.data(), O32.data(), testRef.size(), 0);
    buffer_error(ref.data(), matC.data(), ref.size(), 0.01f);
  }

  void ut_testT3_half(int m, int n, int k, int blocksize) {
    int constexpr runs = 100;
    int constexpr sg_size = 16;
    int constexpr sg_row = 8;                           // half threads of a XVE
    int constexpr sg_col = 4;                           // half threads of a XVE
    int constexpr wg_size = sg_size * sg_row * sg_col;  // half threads of a XVE
    int constexpr TM = 8;
    int constexpr TN = 16;
    int constexpr TK = 32;
    int constexpr SGM = 4;
    int constexpr SGN = 4;
    int constexpr UnrollK = 2;
    // A large GRF kernel
    // SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
    int constexpr PrefetchDis = 3;
    int constexpr G_NROW = 3;
    int constexpr BM = TM * sg_row * SGM;
    int constexpr BM_STRIDE = BM + 8;
    int constexpr BN = TN * sg_col * SGN;
    int constexpr BN_STRIDE = BN;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace syclintelex = sycl::ext::intel::experimental;

    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    int blks = k / blocksize;
    printf("Test Case: %d %d %d %d Device:%s\n", m, n, k, blocksize, dev->getName().c_str());
    avector<float> matAfp(m * k), matB(k * n), ref(m * n);
    avector<bestla::utils::fp16> B_scale(size_t(blks) * n), scaleA(m), matC(m * n);
    avector<int8_t> matA(m * k);
    fill_buffer_randn(scaleA.data(), scaleA.size(), bestla::utils::fp16(0.001f), bestla::utils::fp16(0.005f));
    fill_buffer_randn(matA.data(), matA.size(), int8_t(-128), int8_t(127));
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j += 1) {
        matAfp[i * k + j] = static_cast<float>(matA[i * k + j]) * float(scaleA[i]);
      }
    }

    avector<int8_t> B_s8(k * n), B_s8NT(k * n);
    fill_buffer_randn(B_s8.data(), B_s8.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(B_scale.data(), B_scale.size(), bestla::utils::fp16(0.001f), bestla::utils::fp16(0.005f));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 1) {
        auto noffset = i * blks + j / blocksize;
        matB[i * k + j] = static_cast<float>(B_s8[i * k + j]) * float(B_scale[noffset]);
      }
    }
    avector<float> matBNT(k * n);
    kernel::wrapper::Transpose2D<float>::forward<BTLA_ISA::NoSIMD>(matB.data(), matBNT.data(), n, k, k, n);
    kernel::wrapper::Transpose2D<int8_t>::forward<BTLA_ISA::NoSIMD>(B_s8.data(), B_s8NT.data(), n, k, k, n);
    gemmref_fp32fp32fp32(m, n, k, matAfp.data(), matBNT.data(), ref.data(), k, n, n);
    avector<bestla::utils::fp16> ref_half(ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
      ref_half[i] = ref[i];
    }

    avector<int> testRef(m * n), O32(m * n);
    gemmref_s8s8s32(m, n, k, matA.data(), B_s8NT.data(), testRef.data(), k, n, n);
    sycl_vector<sycl::half> dC(matC.size(), q), dB_scale(B_scale.size(), q), dA_scale(m, q);
    sycl_vector<int8_t> dA(B_s8.size(), q), dBs8(B_s8.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 1).wait();
    q->memcpy(dBs8.data(), B_s8.data(), B_s8.size() * 1).wait();
    q->memcpy(dB_scale.data(), B_scale.data(), B_scale.size() * 2).wait();
    q->memcpy(dA_scale.data(), scaleA.data(), scaleA.size() * 2).wait();
    auto A_d = dA.data();
    auto B_d = dBs8.data();
    auto A_scale_d = dA_scale.data();
    auto B_scale_d = dB_scale.data();
    auto C_d = dC.data();

    size_t g_ncnt = (n + BN - 1) / BN;
    size_t g_mcnt = (m + BM - 1) / BM;
    size_t wg_repeat = g_mcnt * g_ncnt;
    size_t m_tail = g_mcnt % G_NROW;
    size_t g_m_aligned = g_mcnt - m_tail;
    size_t aligned_wg = g_m_aligned * g_ncnt;

    auto ker = [&](sycl::handler& cgh) {
      // sycl::stream out(65536, 256, cgh);
      // sycl::local_accessor<int, 1> slm_b(sycl::range(BN * BM), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}),
          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(sg_size)]]
          {
            auto sg = it.get_sub_group();
            int g_id = it.get_group(0);
            int g_n = g_id % g_ncnt;
            int g_m = g_id / g_ncnt;
            if (g_id <= aligned_wg) {
              int g_m_ = g_id % G_NROW;
              g_id /= G_NROW;
              g_n = g_id % g_ncnt;
              g_id /= g_ncnt;
              g_m = g_id * G_NROW + g_m_;
            }
            if (g_n >= g_ncnt || g_m >= g_mcnt) return;

            int sgSize = sg.get_local_range()[0];
            int sgGroupId = sg.get_group_id()[0];
            int sggid_col = sgGroupId % sg_col;
            int sggid_row = sgGroupId / sg_col;

            int sgId = sg.get_local_id()[0];
            int g_idn = sggid_col * TN * SGN + g_n * BN;
            int g_idm = sggid_row * TM * SGM + g_m * BM;
            // auto pA =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(A_d);
            // auto pB =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(B_d);
            // auto pC = sycl::address_space_cast<sycl::access::address_space::global_space,
            // sycl::access::decorated::yes>(
            //     C_d_s32);
            auto pSA =
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
                    A_scale_d);
            // auto pSB =
            //     sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
            //         B_scale_d);
            // auto pSA = syclex::annotated_ptr{
            //     A_scale_d, syclex::properties{syclintelex::read_hint<
            //                    syclintelex::cache_control<syclintelex::cache_mode::streaming,
            //                    syclex::cache_level::L1>, syclintelex::cache_control<syclintelex::cache_mode::cached,
            //                    syclex::cache_level::L3>>}};
            auto pSB = syclex::annotated_ptr{
                B_scale_d, syclex::properties{syclintelex::read_hint<
                               syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};
            auto pA = syclex::annotated_ptr{
                A_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};

            auto pB = syclex::annotated_ptr{
                B_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};

            auto pC = syclex::annotated_ptr{
                C_d, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                         syclintelex::cache_mode::write_through, syclex::cache_level::L1, syclex::cache_level::L3>>}};
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::a, TM, TK,
                                                                  layout::row_major>
                sub_a[SGM];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::b, TK, TN,
                                                                  layout::col_major>
                sub_b[SGN];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int32_t, use::accumulator, TM, TN>
                sub_c[SGM * SGN];
            // sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, sycl::half, use::accumulator, TM,
            // TN>
            //     sub_c_fp[SGM * SGN];
            sycl::half acc_c[SGM * SGN * TM];
            for (size_t i = 0; i < SGM * SGN * TM; i++) {
              acc_c[i] = sycl::half(0);
            }

            if (TK * UnrollK * (PrefetchDis + 1) <= k)
              for (size_t i = 0; i < PrefetchDis; i++) {
                for (int im = sggid_col; im < SGM; im += sg_col)
                  if ((g_idm + (im + 1) * TM) <= m)
                    joint_matrix_prefetch<TM, TK * UnrollK>(sg, A_d + (g_idm + im * TM) * k * 1 + TK * UnrollK * i, k,
                                                            layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});

                for (int in = sggid_row; in < SGN; in += sg_row)
                  if ((g_idn + (in + 1) * TN) <= n)
                    joint_matrix_prefetch<TN, TK * UnrollK>(sg, B_d + (g_idn + in * TN) * k + TK * UnrollK * i, k,
                                                            layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});
              }

            for (size_t ib = 0; ib < blks; ib++) {
#pragma unroll
              for (int im = 0; im < SGM; im++)
#pragma unroll
                for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], 0);
#pragma unroll(1)
              for (size_t ik = ib * blocksize; ik < (ib + 1) * blocksize; ik += TK * UnrollK) {
#pragma unroll
                for (size_t ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
                  for (int in = 0; in < SGN; in++) {
                    sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                        sg, sub_b[in], pB, k, n, k, (g_idn + in * TN) * 1, ik + ikk * TK);
                  }
#pragma unroll
                  for (int im = 0; im < SGM; im++) {
                    sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                        sg, sub_a[im], pA, k, m, k, (g_idm + im * TM) * 1, ik + TK * ikk);
#pragma unroll
                    for (int in = 0; in < SGN; in++) {
                      joint_matrix_mad(sg, sub_c[im * SGN + in], sub_a[im], sub_b[in], sub_c[im * SGN + in]);
                    }
                  }
                }
                if (ik + TK * UnrollK * (PrefetchDis + 1) <= k) {
                  for (int im = sggid_col; im < SGM; im += sg_col)
                    if ((g_idm + (im + 1) * TM) <= m)
                      joint_matrix_prefetch<TM, TK * UnrollK>(
                          sg, A_d + (g_idm + im * TM) * k * 1 + ik + TK * UnrollK * PrefetchDis, k, layout::row_major,
                          syclex::properties{syclex::prefetch_hint_L1});
                  for (int in = sggid_row; in < SGN; in += sg_row)
                    if ((g_idn + (in + 1) * TN) <= n)
                      joint_matrix_prefetch<TN, TK * UnrollK>(
                          sg, B_d + (g_idn + in * TN) * k + (ik + TK * UnrollK * PrefetchDis), k, layout::row_major,
                          syclex::properties{syclex::prefetch_hint_L1});
                }
              }
#pragma unroll
              for (int im = 0; im < SGM; im++) {
#pragma unroll
                for (int in = 0; in < SGN; in++) {
                  auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(sg, sub_c[im * SGN + in]);
                  for (size_t imm = 0; imm < TM; imm++) {
                    auto element = wi_data_c[imm];
                    auto scaleB = pSB[(g_idn + in * TN + sgId) * blks + ib];
                    acc_c[(im * SGN + in) * TM + imm] += static_cast<sycl::half>(element) * scaleB;
                  }
                }
              }
            }

#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++) {
                for (size_t imm = 0; imm < TM; imm++) {
                  auto scaleA = pSA[g_idm + im * TM + imm];
                  acc_c[(im * SGN + in) * TM + imm] *= scaleA;
                  pC[(g_idm + im * TM + imm) * n + g_idn + in * TN + sgId] = acc_c[(im * SGN + in) * TM + imm];
                }
                // joint_matrix_store(sg, sub_c[im * SGN + in], pC + (g_idm + im * TM) * n + g_idn + in *TN,
                // n,layout::row_major);
                // sycl::ext::intel::experimental::matrix::joint_matrix_store_checked(
                //     sg, sub_c_fp[im * SGN + in], pC, n, layout::row_major, m, n, g_idm + im * TM, g_idn + in * TN);
              }
          });  // parallel for
    };
    q->wait();
    for (size_t i = 0; i < runs; i++) {
      // q->submit(ker);
      Launcher<xmx::IKblockGemmDQCfg<sycl::half>, xmx::IKblockGemmDQCore>::run(
          q, {A_d, B_d, C_d, m, n, k, k, k, n, nullptr, A_scale_d, B_scale_d, blocksize});
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      // q->submit(ker);
      Launcher<xmx::IKblockGemmDQCfg<sycl::half>, xmx::IKblockGemmDQCore>::run(
          q, {A_d, B_d, C_d, m, n, k, k, k, n, nullptr, A_scale_d, B_scale_d, blocksize});
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2;
    printf("Time %f us, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
    q->memcpy(matC.data(), C_d, matC.size() * 2).wait();
    // q->memcpy(O32.data(), C_d_s32, matC.size() * 4).wait();
    // buffer_error(testRef.data(), O32.data(), testRef.size(), 0);
    buffer_error(ref_half.data(), matC.data(), ref_half.size(), bestla::utils::fp16(0.01f));
  }

  template <typename T1, typename T2>
  struct KernelFunctor {
    T1* mPA;
    T2 mProp;
    KernelFunctor(T1* PA, T2 Prop) : mPA(PA), mProp(Prop) {}

    void operator()(sycl::id<1> i) const { mPA[i] += 2; }
    auto get(syclex::properties_tag) const { return mProp; }
  };

  void ut_testT3(int m, int n, int k, int blocksize) {
    int constexpr runs = 100;
    int constexpr sg_size = 16;
    int constexpr sg_row = 4;                           // half threads of a XVE
    int constexpr sg_col = 4;                           // half threads of a XVE
    int constexpr wg_size = sg_size * sg_row * sg_col;  // half threads of a XVE
    int constexpr TM = 8;
    int constexpr TN = 16;
    int constexpr TK = 32;
    int constexpr SGM = 4;
    int constexpr SGN = 2;
    int constexpr UnrollK = 2;
    // A large GRF kernel
    // SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
    int constexpr PrefetchDis = 3;
    int constexpr G_NROW = 3;
    int constexpr BM = TM * sg_row * SGM;
    int constexpr BM_STRIDE = BM + 8;
    int constexpr BN = TN * sg_col * SGN;
    int constexpr BN_STRIDE = BN;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;

    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Block:%d Device:%s\n", m, n, k, blocksize, dev->getName().c_str());
    avector<float> matAfp(m * k), scaleA(m), matB(k * n), matC(m * n), ref(m * n);
    avector<int8_t> matA(m * k);
    fill_buffer_randn(scaleA.data(), scaleA.size(), 0.001f, 0.005f);
    fill_buffer_randn(matA.data(), matA.size(), int8_t(-128), int8_t(127));
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j += 1) {
        matAfp[i * k + j] = static_cast<float>(matA[i * k + j]) * scaleA[i];
      }
    }

    int blks = k / blocksize;
    avector<float> B_scale(size_t(blks) * n);
    avector<int8_t> B_s8(k * n), B_s8NT(k * n);
    fill_buffer_randn(B_s8.data(), B_s8.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(B_scale.data(), B_scale.size(), 0.001f, 0.005f);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 1) {
        auto noffset = i * blks + j / blocksize;
        matB[i * k + j] = static_cast<float>(B_s8[i * k + j]) * B_scale[noffset];
      }
    }
    avector<float> matBNT(k * n), bias(n);
    fill_buffer_randn(bias.data(), bias.size(), 0.1f, 0.5f);

    kernel::wrapper::Transpose2D<float>::forward<BTLA_ISA::NoSIMD>(matB.data(), matBNT.data(), n, k, k, n);
    kernel::wrapper::Transpose2D<int8_t>::forward<BTLA_ISA::NoSIMD>(B_s8.data(), B_s8NT.data(), n, k, k, n);
    gemmref_fp32fp32fp32(m, n, k, matAfp.data(), matBNT.data(), ref.data(), k, n, n);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j += 1) {
        ref[i * n + j] += bias[j];
      }
    }

    avector<int> testRef(m * n), O32(m * n);
    gemmref_s8s8s32(m, n, k, matA.data(), B_s8NT.data(), testRef.data(), k, n, n);
    sycl_vector<float> dB(matB.size(), q), dC(matC.size(), q), dB_scale(B_scale.size(), q), dA_scale(m, q), dBias(n, q);
    sycl_vector<int8_t> dA(B_s8.size(), q), dBs8(B_s8.size(), q);
    sycl_vector<int> dO(m * n, q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 1).wait();
    q->memcpy(dBs8.data(), B_s8.data(), B_s8.size() * 1).wait();
    q->memcpy(dB_scale.data(), B_scale.data(), B_scale.size() * 4).wait();
    q->memcpy(dA_scale.data(), scaleA.data(), scaleA.size() * 4).wait();
    q->memcpy(dBias.data(), bias.data(), bias.size() * 4).wait();
    auto A_d = dA.data();
    auto B_d = dBs8.data();
    auto A_scale_d = dA_scale.data();
    auto B_scale_d = dB_scale.data();
    auto C_d = dC.data();
    auto C_d_s32 = dO.data();

    size_t g_ncnt = (n + BN - 1) / BN;
    size_t g_mcnt = (m + BM - 1) / BM;
    size_t wg_repeat = g_mcnt * g_ncnt;
    size_t m_tail = g_mcnt % G_NROW;
    size_t g_m_aligned = g_mcnt - m_tail;
    size_t aligned_wg = g_m_aligned * g_ncnt;
    // syclex::properties prop{syclintelex::grf_size<256>};
    // auto e = q->submit(
    //     [&](sycl::handler& cgh) { cgh.parallel_for<class SYCLKernelSpecifiedGRF>(32, KernelFunctor(A_d, prop)); });
    // e.wait();

    const auto ker = [=](sycl::handler& cgh) {
      // sycl::stream out(65536, 256, cgh);
      // sycl::local_accessor<int, 1> slm_b(sycl::range(BN * BM), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}),
          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(sg_size)]] [[sycl_grf_size]]
          {
            syclex::properties volatile kernel_properties{syclintelex::grf_size<256>};
            (void)kernel_properties;
            auto sg = it.get_sub_group();
            int g_id = it.get_group(0);
            int g_n = g_id % g_ncnt;
            int g_m = g_id / g_ncnt;
            if (g_id < aligned_wg) {
              int g_m_ = g_id % G_NROW;
              g_id /= G_NROW;
              g_n = g_id % g_ncnt;
              g_id /= g_ncnt;
              g_m = g_id * G_NROW + g_m_;
            }
            if (g_n >= g_ncnt || g_m >= g_mcnt) return;

            int sgSize = sg.get_local_range()[0];
            int sgGroupId = sg.get_group_id()[0];
            int sggid_col = sgGroupId % sg_col;
            int sggid_row = sgGroupId / sg_col;

            int sgId = sg.get_local_id()[0];
            int g_idn = sggid_col * TN * SGN + g_n * BN;
            int g_idm = sggid_row * TM * SGM + g_m * BM;
            // auto pA =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(A_d);
            // auto pB =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(B_d);
            // auto pC = sycl::address_space_cast<sycl::access::address_space::global_space,
            // sycl::access::decorated::yes>(
            //     C_d_s32);
            auto pSA =
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
                    A_scale_d);
            // auto pSB =
            //     sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
            //         B_scale_d);
            // auto pSA = syclex::annotated_ptr{
            //     A_scale_d, syclex::properties{syclintelex::read_hint<
            //                    syclintelex::cache_control<syclintelex::cache_mode::streaming,
            //                    syclex::cache_level::L1>, syclintelex::cache_control<syclintelex::cache_mode::cached,
            //                    syclex::cache_level::L3>>}};
            auto pSB = syclex::annotated_ptr{
                B_scale_d, syclex::properties{syclintelex::read_hint<
                               syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};
            auto pA = syclex::annotated_ptr{
                A_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};

            auto pB = syclex::annotated_ptr{
                B_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};

            auto pC = syclex::annotated_ptr{
                C_d, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                         syclintelex::cache_mode::write_through, syclex::cache_level::L1, syclex::cache_level::L3>>}};
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::a, TM, TK,
                                                                  layout::row_major>
                sub_a[SGM];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::b, TK, TN,
                                                                  layout::col_major>
                sub_b[SGN];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int32_t, use::accumulator, TM, TN>
                sub_c[SGM * SGN];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN>
                sub_c_fp[SGM * SGN];
            if (TK * UnrollK * (PrefetchDis + 1) <= k)
              for (size_t i = 0; i < PrefetchDis; i++) {
                for (int im = sggid_col; im < SGM; im += sg_col)
                  if ((g_idm + (im + 1) * TM) <= m)
                    joint_matrix_prefetch<TM, TK * UnrollK>(sg, A_d + (g_idm + im * TM) * k * 1 + TK * UnrollK * i, k,
                                                            layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});

                for (int in = sggid_row; in < SGN; in += sg_row)
                  if ((g_idn + (in + 1) * TN) <= n)
                    joint_matrix_prefetch<TN, TK * UnrollK>(sg, B_d + (g_idn + in * TN) * k + TK * UnrollK * i, k,
                                                            layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});
              }
#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c_fp[im * SGN + in], 0.f);
            sycl::group_barrier(sg);
            for (size_t ib = 0; ib < blks; ib++) {
#pragma unroll
              for (int im = 0; im < SGM; im++)
#pragma unroll
                for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], 0);
#pragma unroll(1)
              for (int ik = ib * blocksize; ik < (ib + 1) * blocksize; ik += TK * UnrollK) {
#pragma unroll
                for (int ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
                  for (int in = 0; in < SGN; in++) {
                    sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                        sg, sub_b[in], pB, k, n, k, (g_idn + in * TN) * 1, ik + ikk * TK);
                  }
#pragma unroll
                  for (int im = 0; im < SGM; im++) {
                    sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                        sg, sub_a[im], pA, k, m, k, (g_idm + im * TM) * 1, ik + TK * ikk);
#pragma unroll
                    for (int in = 0; in < SGN; in++) {
                      joint_matrix_mad(sg, sub_c[im * SGN + in], sub_a[im], sub_b[in], sub_c[im * SGN + in]);
                    }
                  }
                }
                if (ik + TK * UnrollK * (PrefetchDis + 1) <= k) {
                  for (int im = sggid_col; im < SGM; im += sg_col)
                    if ((g_idm + (im + 1) * TM) <= m)
                      joint_matrix_prefetch<TM, TK * UnrollK>(
                          sg, A_d + (g_idm + im * TM) * k * 1 + ik + TK * UnrollK * PrefetchDis, k, layout::row_major,
                          syclex::properties{syclex::prefetch_hint_L1});
                  for (int in = sggid_row; in < SGN; in += sg_row)
                    if ((g_idn + (in + 1) * TN) <= n)
                      joint_matrix_prefetch<TN, TK * UnrollK>(
                          sg, B_d + (g_idn + in * TN) * k + (ik + TK * UnrollK * PrefetchDis), k, layout::row_major,
                          syclex::properties{syclex::prefetch_hint_L1});
                }
              }
              sycl::group_barrier(sg);

#pragma unroll
              for (int im = 0; im < SGM; im++) {
#pragma unroll
                for (int in = 0; in < SGN; in++) {
                  auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(sg, sub_c[im * SGN + in]);
                  auto wi_data_c1 = sycl::ext::oneapi::detail::get_wi_data(sg, sub_c_fp[im * SGN + in]);
                  for (int i = 0; i < wi_data_c.length(); i++) {
                    auto element = wi_data_c[i];
                    auto element1 = wi_data_c1[i];
                    auto [row, col] = wi_data_c[i].get_coord();
                    float scaleB = pSB[(g_idn + in * TN + col) * blks + ib];
                    element1 = element1 + static_cast<float>(element) * scaleB;
                    wi_data_c1[i] = element1;
                  }
                }
              }
              sycl::group_barrier(sg);
            }
            sycl::group_barrier(sg);

#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++) {
                auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(sg, sub_c_fp[im * SGN + in]);
                for (int i = 0; i < wi_data_c.length(); i++) {
                  auto element = wi_data_c[i];
                  auto [row, col] = wi_data_c[i].get_coord();
                  float scaleA = pSA[g_idm + im * TM + row];
                  wi_data_c[i] = element * scaleA;
                }
                // joint_matrix_store(sg, sub_c[im * SGN + in], pC + (g_idm + im * TM) * n + g_idn + in * TN, n,
                //                    layout::row_major);
                sycl::ext::intel::experimental::matrix::joint_matrix_store_checked(
                    sg, sub_c_fp[im * SGN + in], pC, n, layout::row_major, m, n, g_idm + im * TM, g_idn + in * TN);
              }
          });  // parallel for
    };

    q->wait();
    for (size_t i = 0; i < runs; i++) {
      // q->submit(ker);
      Launcher<xmx::IKblockGemmDQCfg<float>, xmx::IKblockGemmDQCore>::run(
          q, {A_d, B_d, C_d, m, n, k, k, k, n, dBias.data(), A_scale_d, B_scale_d, blocksize});
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      // q->submit(ker);
      Launcher<xmx::IKblockGemmDQCfg<float>, xmx::IKblockGemmDQCore>::run(
          q, {A_d, B_d, C_d, m, n, k, k, k, n, dBias.data(), A_scale_d, B_scale_d, blocksize});
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2;
    printf("Time %f us, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    // q->memcpy(O32.data(), C_d_s32, matC.size() * 4).wait();
    // buffer_error(testRef.data(), O32.data(), testRef.size(), 0);
    buffer_error(ref.data(), matC.data(), ref.size(), 0.01f);
  }

  void ut_testT2() {
    int constexpr sg_size = 16;
    int constexpr TM = 8;
    int constexpr TN = 16;
    int constexpr TK = 32;
    int m = TM, n = TN, k = TK * 2;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace syclintelex = sycl::ext::intel::experimental;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<int8_t> matA(m * k), matB(k * n), matBT(k * n), matBT4(k * n / 2);
    avector<int32_t> matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), int8_t(-128), int8_t(127));
    fill_buffer_randn(matB.data(), matB.size(), int8_t(-8), int8_t(7));
    gemmref_s8s8s32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    bestla::kernel::wrapper::Transpose2D<int8_t>::forward_auto(matB.data(), matBT.data(), k, n, n, k);
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < k; j += 64) {
        for (size_t ik = 0; ik < 32; ik += 1) {
          auto v1 = uint8_t(matBT[i * k + j + ik] + 8);
          auto v2 = uint8_t(matBT[i * k + j + ik + 32] + 8);
          *(uint8_t*)&matBT4[(i * k + j) / 2 + ik] = v1 | (v2 << 4);
        }
      }
    }

    bestla::sycl_vector<int8_t> dA(matA.size(), q), dB4(matBT4.size(), q);
    bestla::sycl_vector<int32_t> dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 1).wait();
    q->memcpy(dB4.data(), matBT4.data(), matBT4.size() * 1).wait();
    auto A_d = dA.data();
    auto B4_d = dB4.data();
    auto C_d = dC.data();

    auto ker = [&](sycl::handler& cgh) {
      sycl::stream out(65536, 256, cgh);
      cgh.parallel_for(
          sycl::nd_range<2>({1, sg_size}, {1, sg_size}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(sg_size)]]
          {
            auto sg = it.get_sub_group();
            int g_id = it.get_group(0);

            int sgSize = sg.get_local_range()[0];
            int sgGroupId = sg.get_group_id()[0];
            int sgId = sg.get_local_id()[0];
            int g_idn = 0;
            int g_idm = 0;
            auto pA =
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(A_d);
            // auto pA = syclex::annotated_ptr{
            //     A_d, syclex::properties{syclintelex::read_hint<
            //              syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L1>,
            //              syclintelex::cache_control<syclintelex::cache_mode::uncached, syclex::cache_level::L3>>}};
            auto pB4 =
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(B4_d);
            // auto pB = syclex::annotated_ptr{
            //     B_d, syclex::properties{syclintelex::read_hint<syclintelex::cache_control<
            //              syclintelex::cache_mode::cached, syclex::cache_level::L1, syclex::cache_level::L3>>}};
            // auto pC =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(C_d);
            auto pC = syclex::annotated_ptr{
                C_d, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                         syclintelex::cache_mode::write_through, syclex::cache_level::L1, syclex::cache_level::L3>>}};
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::a, TM, TK,
                                                                  layout::row_major>
                sub_a, sub_a1;
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::b, TK, TN,
                                                                  layout::col_major>
                sub_b, sub_b1;
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int32_t, use::accumulator, TM, TN>
                sub_c;

            joint_matrix_fill(sg, sub_c, 0);
            joint_matrix_load(sg, sub_b, pB4, k / 2);
            joint_matrix_copy(sg, sub_b, sub_b1);
            // int a = sycl::ext::oneapi::detail::get_wi_data(sg, sub_b)[0];
            // a = *(uint8_t*)&a;
            // out << sgId << " " << a << "\n";
            joint_matrix_apply(sg, sub_b, [=](int8_t& val) {
              auto t = *(uint8_t*)&val;
              val = static_cast<int8_t>(t & 0xf) - 8;
            });
            // a = sycl::ext::oneapi::detail::get_wi_data(sg, sub_b)[0];
            // out << sgId << " " << a << "\n";

            // a = sycl::ext::oneapi::detail::get_wi_data(sg, sub_b1)[0];
            // a = *(uint8_t*)&a;
            // out << sgId << " " << a << "\n";
            joint_matrix_apply(sg, sub_b1, [=](int8_t& val) {
              auto t = *(uint8_t*)&val;
              val = static_cast<int8_t>(t >> 4) - 8;
            });
            // a = sycl::ext::oneapi::detail::get_wi_data(sg, sub_b1)[0];
            // out << sgId << " " << a << "\n";
            joint_matrix_load(sg, sub_a, pA, k);
            joint_matrix_load(sg, sub_a1, pA + TK, k);

            joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
            joint_matrix_mad(sg, sub_c, sub_a1, sub_b1, sub_c);

            // joint_matrix_apply(sg, sub_c, [=](Tc& x) { x *= ALPHA; });
            joint_matrix_store(sg, sub_c, pC, n, layout::row_major);
          });  // parallel for
    };
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < 1; i++) {
      q->submit(ker);
    }
    q->wait();
    auto t_ms = tm.stop() / 1;
    double ops = (double)m * n * k * 2;
    printf("Time %f us, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0);
  }

  void ut_testT1(int m, int n, int k) {
    int constexpr runs = 100;
    int constexpr sg_size = 16;
    int constexpr sg_row = 2;                           // half threads of a XVE
    int constexpr sg_col = 16;                          // half threads of a XVE
    int constexpr wg_size = sg_size * sg_row * sg_col;  // half threads of a XVE
    int constexpr TM = 8;
    int constexpr TN = 16;
    int constexpr TK = 32;
    int constexpr SGM = 16;
    int constexpr SGN = 1;
    int constexpr UnrollK = 2;
    // A large GRF kernel
    // SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
    int constexpr PrefetchDis = 3;
    int constexpr G_NROW = 3;
    int constexpr BM = TM * sg_row * SGM;
    int constexpr BM_STRIDE = BM + 8;
    int constexpr BN = TN * sg_col * SGN;
    int constexpr BN_STRIDE = BN;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace syclintelex = sycl::ext::intel::experimental;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<int8_t> matA(m * k), matB(k * n), matBT(k * n), matBT4(k * n / 2);
    avector<int32_t> matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), int8_t(-128), int8_t(127));
    fill_buffer_randn(matB.data(), matB.size(), int8_t(-8), int8_t(7));
    gemmref_s8s8s32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    bestla::kernel::wrapper::Transpose2D<int8_t>::forward_auto(matB.data(), matBT.data(), k, n, n, k);
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < k; j += 64) {
        for (size_t ik = 0; ik < 32; ik += 1) {
          auto v1 = uint8_t(matBT[i * k + j + ik] + 8);
          auto v2 = uint8_t(matBT[i * k + j + ik + 32] + 8);
          *(uint8_t*)&matBT4[(i * k + j) / 2 + ik] = v1 | (v2 << 4);
        }
      }
    }
    bestla::sycl_vector<int8_t> dA(matA.size(), q), dB(matBT4.size(), q);
    bestla::sycl_vector<int32_t> dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 1).wait();
    q->memcpy(dB.data(), matBT4.data(), matBT4.size() * 1).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    size_t g_ncnt = (n + BN - 1) / BN;
    size_t g_mcnt = (m + BM - 1) / BM;
    size_t wg_repeat = g_mcnt * g_ncnt;
    size_t m_tail = g_mcnt % G_NROW;
    size_t g_m_aligned = g_mcnt - m_tail;
    size_t aligned_wg = g_m_aligned * g_ncnt;

    auto ker = [&](sycl::handler& cgh) {
      // sycl::stream out(65536, 256, cgh);
      // sycl::local_accessor<int8_t, 2> slm_b(sycl::range(BN_STRIDE, TK * UnrollK), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}),
          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(sg_size)]]
          {
            auto sg = it.get_sub_group();
            int g_id = it.get_group(0);
            int g_n = g_id % g_ncnt;
            int g_m = g_id / g_ncnt;
            if (g_id <= aligned_wg) {
              int g_m_ = g_id % G_NROW;
              g_id /= G_NROW;
              g_n = g_id % g_ncnt;
              g_id /= g_ncnt;
              g_m = g_id * G_NROW + g_m_;
            }
            if (g_n >= g_ncnt || g_m >= g_mcnt) return;

            int sgSize = sg.get_local_range()[0];
            int sgGroupId = sg.get_group_id()[0];
            int sggid_col = sgGroupId % sg_col;
            int sggid_row = sgGroupId / sg_col;

            int sgId = sg.get_local_id()[0];
            int g_idn = sggid_col * TN * SGN + g_n * BN;
            int g_idm = sggid_row * TM * SGM + g_m * BM;
            // auto pA =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(A_d);
            auto pA = syclex::annotated_ptr{
                A_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};
            // auto pB =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(B_d);
            auto pB = syclex::annotated_ptr{
                B_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};
            // auto pC =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(C_d);
            auto pC = syclex::annotated_ptr{
                C_d, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                         syclintelex::cache_mode::write_through, syclex::cache_level::L1, syclex::cache_level::L3>>}};
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::a, TM, TK,
                                                                  layout::row_major>
                sub_a[SGM];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::b, TK, TN,
                                                                  layout::col_major>
                sub_b[SGN * UnrollK];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int32_t, use::accumulator, TM, TN>
                sub_c[SGM * SGN];
            if (TK * UnrollK * (PrefetchDis + 1) <= k)
              for (size_t i = 0; i < PrefetchDis; i++) {
                for (int im = sggid_col; im < SGM; im += sg_col)
                  if ((g_idm + (im + 1) * TM) <= m)
                    joint_matrix_prefetch<TM, TK * UnrollK>(sg, A_d + (g_idm + im * TM) * k * 1 + TK * UnrollK * i, k,
                                                            layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});

                for (int in = sggid_row; in < SGN; in += sg_row)
                  if ((g_idn + (in + 1) * TN) <= n)
                    joint_matrix_prefetch<TN, TK * UnrollK>(sg, B_d + (g_idn + in * TN) * k / 2 + TK * UnrollK * i / 2,
                                                            k / 2, layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});
              }

#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], 0);

#pragma unroll(1)
            for (size_t ik = 0; ik < k; ik += TK * UnrollK) {
#pragma unroll
              for (int in = 0; in < SGN; in++) {
                sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                    sg, sub_b[in * UnrollK + 0], pB, k / 2, n, k / 2, (g_idn + in * TN) * 1, ik / 2);
                joint_matrix_copy(sg, sub_b[in * UnrollK + 0], sub_b[in * UnrollK + 1]);
                joint_matrix_apply(sg, sub_b[in * UnrollK + 0], [=](int8_t& val) {
                  auto t = *(uint8_t*)&val;
                  val = static_cast<int8_t>(t & (uint8_t)0xf) - (int8_t)8;
                });
                joint_matrix_apply(sg, sub_b[in * UnrollK + 1], [=](int8_t& val) {
                  auto t = *(uint8_t*)&val;
                  val = static_cast<int8_t>(t >> (uint8_t)4) - (int8_t)8;
                  // val = val;  // >> (uint8_t)4;
                });
                // joint_matrix_apply(sg, sub_b[in * UnrollK + 0], sub_b[in * UnrollK + 1],
                //                    [=](int8_t& val, int8_t& val1) {
                //                      auto t = *(uint8_t*)&val;
                //                      val = static_cast<int8_t>(t & (uint8_t)0xf) - (int8_t)8;
                //                      val1 = static_cast<int8_t>(t >> (uint8_t)4) - (int8_t)8;
                //                    });

                // auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(sg, sub_b[in * UnrollK + 0]);
                // auto wi_data_c1 = sycl::ext::oneapi::detail::get_wi_data(sg, sub_b[in * UnrollK + 1]);
                // for (int i = 0; i < wi_data_c.length(); i++) {
                //   auto element = wi_data_c[i];
                //   auto element1 = wi_data_c1[i];
                //   auto t = *(uint8_t*)&element;
                //   element = static_cast<int8_t>(t & (uint8_t)0xf) - (int8_t)8;
                //   element1 = static_cast<int8_t>(t >> (uint8_t)4) - (int8_t)8;
                //   wi_data_c[i] = element;
                //   wi_data_c1[i] = element1;
                // }
              }
#pragma unroll
              for (size_t ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
                for (int im = 0; im < SGM; im++) {
                  sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                      sg, sub_a[im], pA, k, m, k, (g_idm + im * TM) * 1, ik + TK * ikk);
#pragma unroll
                  for (int in = 0; in < SGN; in++) {
                    joint_matrix_mad(sg, sub_c[im * SGN + in], sub_a[im], sub_b[in * UnrollK + ikk],
                                     sub_c[im * SGN + in]);
                  }
                }
              }
              if (ik + TK * UnrollK * (PrefetchDis + 1) <= k) {
                for (int im = sggid_col; im < SGM; im += sg_col)
                  if ((g_idm + (im + 1) * TM) <= m)
                    joint_matrix_prefetch<TM, TK * UnrollK>(
                        sg, A_d + (g_idm + im * TM) * k * 1 + ik + TK * UnrollK * PrefetchDis, k, layout::row_major,
                        syclex::properties{syclex::prefetch_hint_L1});
                for (int in = sggid_row; in < SGN; in += sg_row)
                  if ((g_idn + (in + 1) * TN) <= n)
                    joint_matrix_prefetch<TN, TK * UnrollK>(
                        sg, B_d + (g_idn + in * TN) * k / 2 + (ik + TK * UnrollK * PrefetchDis) / 2, k / 2,
                        layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
              }
            }
        // joint_matrix_apply(sg, sub_c, [=](Tc& x) { x *= ALPHA; });
#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++)
                // joint_matrix_store(sg, sub_c[im * SGN + in], pC + (g_idm + im * TM) * n + g_idn + in * TN, n,
                //                    layout::row_major);
                sycl::ext::intel::experimental::matrix::joint_matrix_store_checked(
                    sg, sub_c[im * SGN + in], pC, n, layout::row_major, m, n, g_idm + im * TM, g_idn + in * TN);
          });  // parallel for
    };
    q->wait();
    for (size_t i = 0; i < runs; i++) {
      q->submit(ker);
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      q->submit(ker);
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2;
    printf("Time %f us, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0);
  }

  void ut_testT2(int m, int n, int k) {
    int constexpr runs = 100;
    int constexpr sg_size = 16;
    int constexpr sg_row = 4;                           // half threads of a XVE
    int constexpr sg_col = 4;                           // half threads of a XVE
    int constexpr wg_size = sg_size * sg_row * sg_col;  // half threads of a XVE
    int constexpr TM = 8;
    int constexpr TN = 16;
    int constexpr TK = 32;
    int constexpr SGM = 4;
    int constexpr SGN = 4;
    int constexpr UnrollK = 2;
    // A large GRF kernel
    // SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
    int constexpr PrefetchDis = 1;
    int constexpr G_NROW = 1;
    int constexpr BM = TM * sg_row * SGM;
    int constexpr BM_STRIDE = BM + 8;
    int constexpr BN = TN * sg_col * SGN;
    int constexpr BN_STRIDE = BN;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace syclintelex = sycl::ext::intel::experimental;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<int8_t> matA(m * k), matB(k * n), matBT(k * n), matBT4(k * n / 2);
    avector<int32_t> matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), int8_t(-128), int8_t(127));
    fill_buffer_randn(matB.data(), matB.size(), int8_t(-8), int8_t(7));
    gemmref_s8s8s32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    bestla::kernel::wrapper::Transpose2D<int8_t>::forward_auto(matB.data(), matBT.data(), k, n, n, k);
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < k; j += 64) {
        for (size_t ik = 0; ik < 32; ik += 1) {
          auto v1 = uint8_t(matBT[i * k + j + ik] + 8);
          auto v2 = uint8_t(matBT[i * k + j + ik + 32] + 8);
          *(uint8_t*)&matBT4[(i * k + j) / 2 + ik] = v1 | (v2 << 4);
        }
      }
    }
    bestla::sycl_vector<int8_t> dA(matA.size(), q), dB(matBT4.size(), q);
    bestla::sycl_vector<int32_t> dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 1).wait();
    q->memcpy(dB.data(), matBT4.data(), matBT4.size() * 1).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    size_t g_ncnt = (n + BN - 1) / BN;
    size_t g_mcnt = (m + BM - 1) / BM;
    size_t wg_repeat = g_mcnt * g_ncnt;
    size_t m_tail = g_mcnt % G_NROW;
    size_t g_m_aligned = g_mcnt - m_tail;
    size_t aligned_wg = g_m_aligned * g_ncnt;

    auto ker = [&](sycl::handler& cgh) {
      // sycl::stream out(65536, 256, cgh);
      sycl::local_accessor<int8_t, 2> slm_b(sycl::range(BN_STRIDE, TK * UnrollK), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}),
          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(sg_size)]]
          {
            auto sg = it.get_sub_group();
            int g_id = it.get_group(0);
            int g_n = g_id % g_ncnt;
            int g_m = g_id / g_ncnt;
            if (g_id <= aligned_wg) {
              int g_m_ = g_id % G_NROW;
              g_id /= G_NROW;
              g_n = g_id % g_ncnt;
              g_id /= g_ncnt;
              g_m = g_id * G_NROW + g_m_;
            }
            if (g_n >= g_ncnt || g_m >= g_mcnt) return;

            int sgSize = sg.get_local_range()[0];
            int sgGroupId = sg.get_group_id()[0];
            int sggid_col = sgGroupId % sg_col;
            int sggid_row = sgGroupId / sg_col;

            int sgId = sg.get_local_id()[0];
            int g_idn = sggid_col * TN * SGN + g_n * BN;
            int g_idm = sggid_row * TM * SGM + g_m * BM;
            // auto pA =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(A_d);
            auto pA = syclex::annotated_ptr{
                A_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::uncached, syclex::cache_level::L3>>}};
            // auto pBl =
            //     sycl::address_space_cast<sycl::access::address_space::local_space,
            //     sycl::access::decorated::yes>(slm_b);
            auto pB = syclex::annotated_ptr{
                B_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};
            // auto pC =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(C_d);
            auto pC = syclex::annotated_ptr{
                C_d, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                         syclintelex::cache_mode::write_through, syclex::cache_level::L1, syclex::cache_level::L3>>}};
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::a, TM, TK,
                                                                  layout::row_major>
                sub_a[SGM];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::b, TK, TN,
                                                                  layout::col_major>
                sub_b[SGN];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int32_t, use::accumulator, TM, TN>
                sub_c[SGM * SGN];
            if (TK * UnrollK * (PrefetchDis + 1) <= k)
              for (size_t i = 0; i < PrefetchDis; i++) {
                for (int im = sggid_col; im < SGM; im += sg_col)
                  if ((g_idm + (im + 1) * TM) <= m)
                    joint_matrix_prefetch<TM, TK * UnrollK>(sg, A_d + (g_idm + im * TM) * k * 1 + TK * UnrollK * i, k,
                                                            layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});

                for (int in = sggid_row; in < SGN; in += sg_row)
                  if ((g_idn + (in + 1) * TN) <= n)
                    joint_matrix_prefetch<TN, TK * UnrollK>(sg, B_d + (g_idn + in * TN) * k / 2 + TK * UnrollK * i / 2,
                                                            k / 2, layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});
              }

#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], 0);

#pragma unroll(1)
            for (size_t ik = 0; ik < k; ik += TK * UnrollK) {
              // #pragma unroll
              //               for (int in = 0; in < SGN; in++) {
              //                 sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
              //                     sg, sub_b[in * UnrollK + 0], pBl, TK * UnrollK, n, k / 2, (g_idn + in * TN) * 1, ik
              //                     / 2);
              //                 joint_matrix_copy(sg, sub_b[in * UnrollK + 0], sub_b[in * UnrollK + 1]);
              //                 joint_matrix_apply(sg, sub_b[in * UnrollK + 0], [=](int8_t& val) {
              //                   auto t = *(uint8_t*)&val;
              //                   val = static_cast<int8_t>(t & 0xf) - 8;
              //                 });
              //                 joint_matrix_apply(sg, sub_b[in * UnrollK + 1], [=](int8_t& val) {
              //                   auto t = *(uint8_t*)&val;
              //                   val = static_cast<int8_t>(t >> 4) - 8;
              //                 });
              //               }
              it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
              for (size_t ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
                for (int in = 0; in < SGN; in++) {
                  joint_matrix_load(sg, sub_b[in],
                                    slm_b.get_multi_ptr<sycl::access::decorated::no>() + in * TN * TK + BN * TK * ikk,
                                    TK);
                  // joint_matrix_load(sg, sub_b[in], pB + ik + TK * ikk + (g_idn + in * TN) * k * 1, k);
                  // sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                  //     sg, sub_b[in], pB, k, n, k, (g_idn + in * TN) * 1, ik + TK * ikk);
                }
#pragma unroll
                for (int im = 0; im < SGM; im++) {
                  sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                      sg, sub_a[im], pA, k, m, k, (g_idm + im * TM) * 1, ik + TK * ikk);
#pragma unroll
                  for (int in = 0; in < SGN; in++) {
                    joint_matrix_mad(sg, sub_c[im * SGN + in], sub_a[im], sub_b[in * UnrollK + ikk],
                                     sub_c[im * SGN + in]);
                  }
                }
                it.barrier(sycl::access::fence_space::local_space);
              }
              // if (ik + TK * UnrollK * (PrefetchDis + 1) <= k) {
              //   for (int im = sggid_col; im < SGM; im += sg_col)
              //     if ((g_idm + (im + 1) * TM) <= m)
              //       joint_matrix_prefetch<TM, TK * UnrollK>(
              //           sg, A_d + (g_idm + im * TM) * k * 1 + ik + TK * UnrollK * PrefetchDis, k, layout::row_major,
              //           syclex::properties{syclex::prefetch_hint_L1});
              //   for (int in = sggid_row; in < SGN; in += sg_row)
              //     if ((g_idn + (in + 1) * TN) <= n)
              //       joint_matrix_prefetch<TN, TK * UnrollK>(
              //           sg, B_d + (g_idn + in * TN) * k / 2 + (ik + TK * UnrollK * PrefetchDis) / 2, k / 2,
              //           layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
              // }
            }
        // joint_matrix_apply(sg, sub_c, [=](Tc& x) { x *= ALPHA; });
#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++)
                // joint_matrix_store(sg, sub_c[im * SGN + in], pC + (g_idm + im * TM) * n + g_idn + in * TN, n,
                //                    layout::row_major);
                sycl::ext::intel::experimental::matrix::joint_matrix_store_checked(
                    sg, sub_c[im * SGN + in], pC, n, layout::row_major, m, n, g_idm + im * TM, g_idn + in * TN);
          });  // parallel for
    };
    q->wait();
    for (size_t i = 0; i < runs; i++) {
      q->submit(ker);
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      q->submit(ker);
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2;
    printf("Time %f us, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0);
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclS4IGemm sUT_SyclS4IGemm;
#endif

class UT_SyclIGemm {
 public:
  UT_SyclIGemm() {
    UT_START();
    ut_testT1(123, 128, 128);
    ut_dq<float>(123, 128, 128);
    ut_dq<float>(4096, 4096, 4096);
    ut_dq<utils::fp16, sycl::half>(4096, 4096, 4096);
    ut_dq<utils::bf16, sycl::ext::oneapi::bfloat16>(4096, 4096, 4096);
    ut_testT1(233, 256, 768);
    ut_testT1(4096, 4096, 4096);

    // ut_testT1(552, 552, 96);
    // ut_testT1(4096, 4096, 4096);
    // ut_testT1(4096, 11008, 4096);
    // ut_testT1(4096, 16384, 4096);
  }

  void ut_testT1(int m, int n, int k) {
    int constexpr runs = 100;
    int constexpr sg_size = 16;
    int constexpr sg_row = 8;                           // half threads of a XVE
    int constexpr sg_col = 4;                           // half threads of a XVE
    int constexpr wg_size = sg_size * sg_row * sg_col;  // half threads of a XVE
    int constexpr TM = 8;
    int constexpr TN = 16;
    int constexpr TK = 32;
    int constexpr SGM = 4;
    int constexpr SGN = 4;
    int constexpr UnrollK = 2;
    // A large GRF kernel
    // SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
    int constexpr PrefetchDis = 3;
    int constexpr G_NROW = 3;
    int constexpr G_NCOL = -1;
    int constexpr BM = TM * sg_row * SGM;
    int constexpr BM_STRIDE = BM + 8;
    int constexpr BN = TN * sg_col * SGN;
    int constexpr BN_STRIDE = BN;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace syclintelex = sycl::ext::intel::experimental;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<int8_t> matA(m * k), matB(k * n), matBT(k * n);
    avector<int32_t> matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), int8_t(-128), int8_t(127));
    fill_buffer_randn(matB.data(), matB.size(), int8_t(-128), int8_t(127));
    gemmref_s8s8s32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    bestla::kernel::wrapper::Transpose2D<int8_t>::forward_auto(matB.data(), matBT.data(), k, n, n, k);
    bestla::sycl_vector<int8_t> dA(matA.size(), q), dB(matB.size(), q);
    bestla::sycl_vector<int32_t> dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 1).wait();
    q->memcpy(dB.data(), matBT.data(), matBT.size() * 1).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    size_t g_ncnt = (n + BN - 1) / BN;
    size_t g_mcnt = (m + BM - 1) / BM;
    size_t wg_repeat = g_mcnt * g_ncnt;
    size_t m_tail = g_mcnt % G_NROW;
    size_t g_m_aligned = g_mcnt - m_tail;
    size_t aligned_wg = g_m_aligned * g_ncnt;

    auto ker = [&](sycl::handler& cgh) {
      // sycl::stream out(65536, 256, cgh);
      // sycl::local_accessor<int8_t, 2> slm_b(sycl::range(BN_STRIDE, TK * UnrollK), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}),
          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(sg_size)]]
          {
            auto sg = it.get_sub_group();
            int g_id = it.get_group(0);
            int g_n = g_id % g_ncnt;
            int g_m = g_id / g_ncnt;
            if (g_id <= aligned_wg) {
              int g_m_ = g_id % G_NROW;
              g_id /= G_NROW;
              g_n = g_id % g_ncnt;
              g_id /= g_ncnt;
              g_m = g_id * G_NROW + g_m_;
            }
            if (g_n >= g_ncnt || g_m >= g_mcnt) return;

            int sgSize = sg.get_local_range()[0];
            int sgGroupId = sg.get_group_id()[0];
            int sggid_col = sgGroupId % sg_col;
            int sggid_row = sgGroupId / sg_col;

            int sgId = sg.get_local_id()[0];
            int g_idn = sggid_col * TN * SGN + g_n * BN;
            int g_idm = sggid_row * TM * SGM + g_m * BM;
            // auto pA =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(A_d);
            auto pA = syclex::annotated_ptr{
                A_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};
            // auto pB =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(B_d);
            auto pB = syclex::annotated_ptr{
                B_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
            // auto pC =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(C_d);
            auto pC = syclex::annotated_ptr{
                C_d, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                         syclintelex::cache_mode::write_through, syclex::cache_level::L1, syclex::cache_level::L3>>}};
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::a, TM, TK,
                                                                  layout::row_major>
                sub_a[SGM];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int8_t, use::b, TK, TN,
                                                                  layout::col_major>
                sub_b[SGN];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, int32_t, use::accumulator, TM, TN>
                sub_c[SGM * SGN];
            if (TK * UnrollK * (PrefetchDis + 1) <= k)
              for (size_t i = 0; i < PrefetchDis; i++) {
                for (int im = sggid_col; im < SGM; im += sg_col)
                  if ((g_idm + (im + 1) * TM) <= m)
                    joint_matrix_prefetch<TM, TK * UnrollK>(sg, A_d + (g_idm + im * TM) * k * 1 + TK * UnrollK * i, k,
                                                            layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});

                for (int in = sggid_row; in < SGN; in += sg_row)
                  if ((g_idn + (in + 1) * TN) <= n)
                    joint_matrix_prefetch<TN, TK * UnrollK>(sg, B_d + (g_idn + in * TN) * k * 1 + TK * UnrollK * i, k,
                                                            layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});
              }

#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], 0);

#pragma unroll(1)
            for (size_t ik = 0; ik < k; ik += TK * UnrollK) {
#pragma unroll
              for (size_t ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
                for (int in = 0; in < SGN; in++) {
                  // joint_matrix_load(sg, sub_b[in], slm_b.get_multi_ptr<sycl::access::decorated::no>(), TK);
                  // joint_matrix_load(sg, sub_b[in], pB + ik + TK * ikk + (g_idn + in * TN) * k * 1, k);
                  sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                      sg, sub_b[in], pB, k, n, k, (g_idn + in * TN) * 1, ik + TK * ikk);
                }
#pragma unroll
                for (int im = 0; im < SGM; im++) {
                  // joint_matrix_load(sg, sub_a[im], pA + (g_idm + im * TM) * k * 1 + ik + TK * ikk, k);
                  sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                      sg, sub_a[im], pA, k, m, k, (g_idm + im * TM) * 1, ik + TK * ikk);
#pragma unroll
                  for (int in = 0; in < SGN; in++) {
                    joint_matrix_mad(sg, sub_c[im * SGN + in], sub_a[im], sub_b[in], sub_c[im * SGN + in]);
                  }
                }
              }
              if (ik + TK * UnrollK * (PrefetchDis + 1) <= k) {
                for (int im = sggid_col; im < SGM; im += sg_col)
                  if ((g_idm + (im + 1) * TM) <= m)
                    joint_matrix_prefetch<TM, TK * UnrollK>(
                        sg, A_d + (g_idm + im * TM) * k * 1 + ik + TK * UnrollK * PrefetchDis, k, layout::row_major,
                        syclex::properties{syclex::prefetch_hint_L1});
                for (int in = sggid_row; in < SGN; in += sg_row)
                  if ((g_idn + (in + 1) * TN) <= n)
                    joint_matrix_prefetch<TN, TK * UnrollK>(
                        sg, B_d + (g_idn + in * TN) * k * 1 + ik + TK * UnrollK * PrefetchDis, k, layout::row_major,
                        syclex::properties{syclex::prefetch_hint_L1});
              }
            }
        // joint_matrix_apply(sg, sub_c, [=](Tc& x) { x *= ALPHA; });
#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++)
                // joint_matrix_store(sg, sub_c[im * SGN + in], pC + (g_idm + im * TM) * n + g_idn + in * TN, n,
                //                    layout::row_major);
                sycl::ext::intel::experimental::matrix::joint_matrix_store_checked(
                    sg, sub_c[im * SGN + in], pC, n, layout::row_major, m, n, g_idm + im * TM, g_idn + in * TN);
          });  // parallel for
    };
    q->wait();
    for (size_t i = 0; i < runs; i++) {
      // q->submit(ker);
      Launcher<xmx::IGemmCfg, xmx::GemmCore>::run(q, {A_d, B_d, C_d, m, n, k, k, k, n});
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      // q->submit(ker);
      Launcher<xmx::IGemmCfg, xmx::GemmCore>::run(q, {A_d, B_d, C_d, m, n, k, k, k, n});
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2;
    printf("Time %f us, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0);
  }

  template <typename T, typename ST = T>
  void ut_dq(int m, int n, int k) {
    int constexpr runs = 100;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<int8_t> matA(m * k), matB(k * n), matBT(k * n);
    avector<int32_t> ref(m * n);
    avector<T> matC(m * n), ref0(m * n), scaleA(m), scaleB(n), bias(n);
    fill_buffer_randn(matA.data(), matA.size(), int8_t(-128), int8_t(127));
    fill_buffer_randn(matB.data(), matB.size(), int8_t(-128), int8_t(127));
    fill_buffer_randn(scaleA.data(), scaleA.size(), T(0.001), T(0.004));
    fill_buffer_randn(scaleB.data(), scaleB.size(), T(0.001), T(0.004));
    fill_buffer_randn(bias.data(), bias.size(), T(0.1), T(0.3));
    gemmref_s8s8s32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        ref0[i * n + j] = (float)ref[i * n + j] * (float)scaleA[i] * (float)scaleB[j] + (float)bias[j];
      }
    }

    bestla::kernel::wrapper::Transpose2D<int8_t>::forward_auto(matB.data(), matBT.data(), k, n, n, k);
    bestla::sycl_vector<int8_t> dA(matA.size(), q), dB(matB.size(), q);
    bestla::sycl_vector<T> dscaleA(m, q), dscaleB(n, q), dbias(n, q), dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 1).wait();
    q->memcpy(dB.data(), matBT.data(), matBT.size() * 1).wait();
    q->memcpy(dscaleA.data(), scaleA.data(), scaleA.size() * sizeof(T)).wait();
    q->memcpy(dscaleB.data(), scaleB.data(), scaleB.size() * sizeof(T)).wait();
    q->memcpy(dbias.data(), bias.data(), bias.size() * sizeof(T)).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();

    q->wait();
    for (size_t i = 0; i < runs; i++) {
      Launcher<xmx::IGemmDQCfg<ST>, xmx::IGemmDQCore>::run(
          q, {A_d, B_d, C_d, m, n, k, k, k, n, dbias.data(), dscaleA.data(), dscaleB.data()});
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      Launcher<xmx::IGemmDQCfg<ST>, xmx::IGemmDQCore>::run(
          q, {A_d, B_d, C_d, m, n, k, k, k, n, dbias.data(), dscaleA.data(), dscaleB.data()});
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2;
    printf("Time %f us, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
    q->memcpy(matC.data(), C_d, matC.size() * sizeof(T)).wait();
    buffer_error(ref0.data(), matC.data(), ref0.size(), T(0.2));
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclIGemm sUT_SyclIGemm;
#endif

class UT_SyclHGemm {
 public:
  UT_SyclHGemm() {
    UT_START();
    ut_test(256, 128, 2048);
    // ut_testT(256, 256, 1024 * 4);
    // ut_testT1(552, 552, 96);
    // ut_testT1(1, 1024, 768);
    // ut_testT1(300, 1024, 768);

    // ut_testT1(4096, 4096, 4096);
    // ut_fp32(4096, 4096, 4096);
    // ut_testT1(4096, 11008, 4096);
    // ut_testT1(4096, 16384, 4096);
  }
  using DT = sycl::half;
  void ut_test(int m, int n, int k) {
    int constexpr runs = 100;
    int constexpr sg_size = 16;
    int constexpr sg_row = 8;                           // half threads of a XVE
    int constexpr sg_col = 4;                           // half threads of a XVE
    int constexpr wg_size = sg_size * sg_row * sg_col;  // half threads of a XVE
    int constexpr TM = 16;
    int constexpr TN = 16;
    int constexpr TK = 16;
    int constexpr SGM = 2;
    int constexpr SGN = 2;
    int constexpr UnrollK = 2;
    int constexpr wg_repeat = 20 * 16;
    int constexpr BM = TM * sg_row * SGM;
    int constexpr BM_STRIDE = BM + 8;
    int constexpr BN = TN * sg_col * SGN;
    int constexpr BN_STRIDE = BN + 8;
    m = BM;
    n = BN;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace syclintelex = sycl::ext::intel::experimental;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<utils::fp16> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(matB.data(), matB.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    gemmref_fp16fp16fp16(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    sycl_vector<DT> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 2).wait();
    q->memcpy(dB.data(), matB.data(), matB.size() * 2).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();

    auto ker = [&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> slm_a(sycl::range(BM_STRIDE * TK), cgh);
      sycl::local_accessor<float, 1> slm_b(sycl::range(BN_STRIDE * TK), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}),
          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(sg_size)]]
          {
            auto sg = it.get_sub_group();
            int sgSize = sg.get_local_range()[0];
            int sgGroupId = sg.get_group_id()[0];
            int sggid_col = sgGroupId % sg_col;
            int sggid_row = sgGroupId / sg_col;
            int sgId = sg.get_local_id()[0];
            int g_idn = sggid_col * TN * SGN;
            int g_idm = sggid_row * TM * SGM;
            // auto pA =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::no>(A_d);
            // auto pB =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::no>(B_d);
            auto pC =
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(C_d);
            auto pA = syclex::annotated_ptr{
                A_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
            auto pB = syclex::annotated_ptr{
                B_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::a, TM, TK,
                                                                  layout::row_major>
                sub_a[SGM];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::b, TK, TN,
                                                                  layout::row_major>
                sub_b[SGN];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::accumulator, TM, TN>
                sub_c[SGM * SGN];
#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], DT(0.f));
#pragma unroll(1)
            for (size_t ik = 0; ik < k; ik += TK * UnrollK) {
#pragma unroll
              for (size_t ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
                for (int in = 0; in < SGN; in++) {
                  joint_matrix_load(sg, sub_b[in], pB + (ik + TK * ikk) * n + g_idn + in * TN, n);
                }
#pragma unroll
                for (int im = 0; im < SGM; im++) {
                  joint_matrix_load(sg, sub_a[im], pA + (g_idm + im * TM) * k + ik + TK * ikk, k);
#pragma unroll
                  for (int in = 0; in < SGN; in++) {
                    joint_matrix_mad(sg, sub_c[im * SGN + in], sub_a[im], sub_b[in], sub_c[im * SGN + in]);
                  }
                }
              }
            }
        // joint_matrix_apply(sg, sub_c, [=](Tc& x) { x *= ALPHA; });
#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++)
                joint_matrix_store(sg, sub_c[im * SGN + in], pC + (g_idm + im * TM) * n + g_idn + in * TN, n,
                                   layout::row_major);
          });  // parallel for
    };
    q->wait();
    for (size_t i = 0; i < runs; i++) {
      q->submit(ker);
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      q->submit(ker);
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2 * wg_repeat;
    printf("Time %f ms, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
    q->memcpy(matC.data(), C_d, matC.size() * 2).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), utils::fp16(0.2f));
  }

  template <typename CFG>
  struct Prefetcher {
    using DT = CFG::DT;
    static int constexpr SGM = CFG::SGM;
    static int constexpr SGN = CFG::SGN;
    static int constexpr TM = CFG::TM;
    static int constexpr TN = CFG::TN;
    static int constexpr TK = CFG::TK;
    static int constexpr UnrollK = CFG::UnrollK;
    static int constexpr PrefetchDis = CFG::PrefetchDis;

    static inline void next(sycl::sub_group sg, int ik, int m, int n, int k, int g_idm, int g_idn, int sggid_col,
                            int sg_col, int sggid_row, int sg_row, int lda, int ldb, const DT* Awg_d, const DT* Bwg_d) {
      using sycl::ext::oneapi::experimental::matrix::layout;
      using sycl::ext::oneapi::experimental::matrix::use;
      if (ik + TK * UnrollK * (PrefetchDis + 1) <= k) {
        for (int im = sggid_col; im < SGM; im += sg_col)
          if ((g_idm + im * TM) <= m)
            joint_matrix_prefetch<TM, TK * UnrollK>(sg, Awg_d + (im * TM) * lda + ik + TK * UnrollK * PrefetchDis, lda,
                                                    layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
        for (int in = sggid_row; in < SGN; in += sg_row)
          if ((g_idn + in * TN) <= n)
            joint_matrix_prefetch<TN, TK * UnrollK>(sg, Bwg_d + (in * TN) * ldb + ik + TK * UnrollK * PrefetchDis, ldb,
                                                    layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
      }
    }
  };

  template <typename TP, typename CFG>
  struct KernelFunctor {
    using DT = CFG::DT;
    using DT1 = CFG::DT_ACC;

    static int constexpr sg_size = CFG::sg_size;
    static int constexpr sg_row = CFG::sg_row;
    static int constexpr sg_col = CFG::sg_col;
    static int constexpr TM = CFG::TM;
    static int constexpr TN = CFG::TN;
    static int constexpr TK = CFG::TK;
    static int constexpr SGM = CFG::SGM;
    static int constexpr SGN = CFG::SGN;
    static int constexpr UnrollK = CFG::UnrollK;
    static int constexpr PrefetchDis = CFG::PrefetchDis;
    static int constexpr G_NROW = CFG::G_NROW;

    static int constexpr wg_size = sg_size * sg_row * sg_col;
    static int constexpr BM = TM * sg_row * SGM;
    static int constexpr BM_STRIDE = BM;
    static int constexpr BN = TN * sg_col * SGN;
    static int constexpr BN_STRIDE = BN;

    DT *A_d, *B_d;
    DT1* C_d;
    int m, n, k, lda, ldb, ldc;
    TP mProp;

    int g_ncnt, g_mcnt;
    size_t wg_repeat, aligned_wg;
    KernelFunctor(TP Prop, CFG _cfg, DT* A, DT* B, DT* C, int _m, int _n, int _k, int _lda = 0, int _ldb = 0,
                  int _ldc = 0)
        : mProp(Prop) {
      A_d = A;
      B_d = B;
      C_d = C;
      m = _m;
      n = _n;
      k = _k;
      lda = _lda ? _lda : k;
      ldb = _ldb ? _ldb : k;
      ldc = _ldc ? _ldc : n;
      g_ncnt = (n + BN - 1) / BN;
      g_mcnt = (m + BM - 1) / BM;
      wg_repeat = g_mcnt * g_ncnt;
      size_t m_tail = g_mcnt % G_NROW;
      size_t g_m_aligned = g_mcnt - m_tail;
      aligned_wg = g_m_aligned * g_ncnt;
    }

    sycl::nd_range<2> get_range() { return sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}); }

    void operator() [[sycl::reqd_sub_group_size(sg_size)]] (sycl::nd_item<2> it) const {
      auto sg = it.get_sub_group();
      int g_id = it.get_group(0);
      int g_n = g_id % g_ncnt;
      int g_m = g_id / g_ncnt;
      if (g_id <= aligned_wg) {
        int g_m_ = g_id % G_NROW;
        g_id /= G_NROW;
        g_n = g_id % g_ncnt;
        g_id /= g_ncnt;
        g_m = g_id * G_NROW + g_m_;
      }
      if (g_n >= g_ncnt || g_m >= g_mcnt) return;
      int sgSize = sg.get_local_range()[0];
      int sgGroupId = sg.get_group_id()[0];
      int sggid_col = sgGroupId % sg_col;
      int sggid_row = sgGroupId / sg_col;
      int sgId = sg.get_local_id()[0];
      int g_idn = sggid_col * TN * SGN + g_n * BN;
      int g_idm = sggid_row * TM * SGM + g_m * BM;
      auto Awg_d = A_d + (size_t)g_idm * lda;
      auto Bwg_d = B_d + (size_t)g_idn * ldb;
      auto Cwg_d = C_d + (size_t)g_idm * ldc + g_idn;
      using sycl::ext::oneapi::experimental::matrix::layout;
      using sycl::ext::oneapi::experimental::matrix::use;
      auto pA = syclex::annotated_ptr{
          Awg_d, syclex::properties{syclintelex::read_hint<
                     syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L1>,
                     syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};

      auto pB = syclex::annotated_ptr{
          Bwg_d, syclex::properties{syclintelex::read_hint<
                     syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                     syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};

      auto pC = syclex::annotated_ptr{
          Cwg_d, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                     syclintelex::cache_mode::write_through, syclex::cache_level::L1, syclex::cache_level::L3>>}};
      sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::a, TM, TK, layout::row_major>
          sub_a[SGM];
      sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::b, TK, TN, layout::col_major>
          sub_b[SGN];
      sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT1, use::accumulator, TM, TN>
          sub_c[SGM * SGN];

      Prefetcher<CFG>::next(sg, 0, m, n, k, g_idm, g_idn, sggid_col, sg_col, sggid_row, sg_row, lda, ldb, Awg_d, Bwg_d);

#pragma unroll
      for (int im = 0; im < SGM; im++)
#pragma unroll
        for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], DT(0.f));

#pragma unroll(1)
      for (size_t ik = 0; ik < k; ik += TK * UnrollK) {
#pragma unroll
        for (size_t ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
          for (int in = 0; in < SGN; in++) {
            sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(sg, sub_b[in], pB, ldb, n, k, in * TN,
                                                                              ik + TK * ikk);
          }
#pragma unroll
          for (int im = 0; im < SGM; im++) {
            sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(sg, sub_a[im], pA, lda, m, k, im * TM,
                                                                              ik + TK * ikk);
#pragma unroll
            for (int in = 0; in < SGN; in++) {
              joint_matrix_mad(sg, sub_c[im * SGN + in], sub_a[im], sub_b[in], sub_c[im * SGN + in]);
            }
          }
        }
        Prefetcher<CFG>::next(sg, ik, m, n, k, g_idm, g_idn, sggid_col, sg_col, sggid_row, sg_row, lda, ldb, Awg_d,
                              Bwg_d);
      }
#pragma unroll
      for (int im = 0; im < SGM; im++)
#pragma unroll
        for (int in = 0; in < SGN; in++)
          sycl::ext::intel::experimental::matrix::joint_matrix_store_checked(sg, sub_c[im * SGN + in], pC, ldc,
                                                                             layout::row_major, m, n, im * TM, in * TN);
    }

    auto get(syclex::properties_tag) const { return mProp; }
  };

  struct HGemmCfg {
    static int constexpr sg_size = 16;
    static int constexpr sg_row = 8;
    static int constexpr sg_col = 4;
    static int constexpr TM = 16;
    static int constexpr TN = 16;
    static int constexpr TK = 16;
    static int constexpr SGM = 2;
    static int constexpr SGN = 4;
    static int constexpr UnrollK = 2;
    static int constexpr PrefetchDis = 3;
    static int constexpr G_NROW = 3;
    using DT = sycl::half;
    using DT_ACC = sycl::half;
  };
  void ut_testT1(int m, int n, int k) {
    int constexpr runs = 100;
    int constexpr sg_size = 16;
    int constexpr sg_row = 8;                           // half threads of a XVE
    int constexpr sg_col = 4;                           // half threads of a XVE
    int constexpr wg_size = sg_size * sg_row * sg_col;  // half threads of a XVE
    int constexpr TM = 16;
    int constexpr TN = 16;
    int constexpr TK = 16;
    int constexpr SGM = 2;
    int constexpr SGN = 4;
    int constexpr UnrollK = 2;
    // A large GRF kernel
    // SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
    int constexpr PrefetchDis = 3;
    int constexpr G_NROW = 3;
    int constexpr BM = TM * sg_row * SGM;
    int constexpr BM_STRIDE = BM + 8;
    int constexpr BN = TN * sg_col * SGN;
    int constexpr BN_STRIDE = BN;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<utils::fp16> matA(m * k), matB(k * n), matBT(k * n), matC(m * n), ref(m * n), bias(n);
    fill_buffer_randn(matA.data(), matA.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(matB.data(), matB.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(bias.data(), bias.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    gemmref_fp16fp16fp16(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        ref[i * n + j] = (float)ref[i * n + j] + (float)bias[j];
      }
    }
    bestla::kernel::wrapper::Transpose2D<utils::fp16>::forward_auto(matB.data(), matBT.data(), k, n, n, k);
    bestla::sycl_vector<DT> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q), dBias(bias.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 2).wait();
    q->memcpy(dB.data(), matBT.data(), matBT.size() * 2).wait();
    q->memcpy(dBias.data(), bias.data(), bias.size() * 2).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    size_t g_ncnt = (n + BN - 1) / BN;
    size_t g_mcnt = (m + BM - 1) / BM;
    size_t wg_repeat = g_mcnt * g_ncnt;
    size_t m_tail = g_mcnt % G_NROW;
    size_t g_m_aligned = g_mcnt - m_tail;
    size_t aligned_wg = g_m_aligned * g_ncnt;
    auto ker = [&](sycl::handler& cgh) {
      // sycl::local_accessor<DT, 2> slm_b(sycl::range(BN_STRIDE, TK * UnrollK), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}),
          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(sg_size)]]
          {
            auto sg = it.get_sub_group();
            int g_id = it.get_group(0);
            int g_n = g_id % g_ncnt;
            int g_m = g_id / g_ncnt;
            if (g_id <= aligned_wg) {
              int g_m_ = g_id % G_NROW;
              g_id /= G_NROW;
              g_n = g_id % g_ncnt;
              g_id /= g_ncnt;
              g_m = g_id * G_NROW + g_m_;
            }
            if (g_n >= g_ncnt || g_m >= g_mcnt) return;
            int sgSize = sg.get_local_range()[0];
            int sgGroupId = sg.get_group_id()[0];
            int sggid_col = sgGroupId % sg_col;
            int sggid_row = sgGroupId / sg_col;
            int sgId = sg.get_local_id()[0];
            int g_idn = sggid_col * TN * SGN + g_n * BN;
            int g_idm = sggid_row * TM * SGM + g_m * BM;
            // auto pA =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(A_d);
            auto pA = syclex::annotated_ptr{
                A_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L3>>}};
            // auto pB =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(B_d);
            auto pB = syclex::annotated_ptr{
                B_d, syclex::properties{syclintelex::read_hint<
                         syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                         syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
            // auto pC =
            //     sycl::address_space_cast<sycl::access::address_space::global_space,
            //     sycl::access::decorated::yes>(C_d);
            auto pC = syclex::annotated_ptr{
                C_d, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                         syclintelex::cache_mode::write_through, syclex::cache_level::L1, syclex::cache_level::L3>>}};
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::a, TM, TK,
                                                                  layout::row_major>
                sub_a[SGM];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::b, TK, TN,
                                                                  layout::col_major>
                sub_b[SGN];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::accumulator, TM, TN>
                sub_c[SGM * SGN];
            if (TK * UnrollK * (PrefetchDis + 1) <= k)
              for (size_t i = 0; i < PrefetchDis; i++) {
                for (int im = sggid_col; im < SGM; im += sg_col)
                  if ((g_idm + im * TM) <= m)
                    joint_matrix_prefetch<TM, TK * UnrollK>(sg, A_d + (g_idm + im * TM) * k * 1 + TK * UnrollK * i, k,
                                                            layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});
                for (int in = sggid_row; in < SGN; in += sg_row)
                  if ((g_idn + in * TN) <= n)
                    joint_matrix_prefetch<TN, TK * UnrollK>(sg, B_d + (g_idn + in * TN) * k * 1 + TK * UnrollK * i, k,
                                                            layout::row_major,
                                                            syclex::properties{syclex::prefetch_hint_L1});
              }

#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], DT(0.f));

#pragma unroll(1)
            for (size_t ik = 0; ik < k; ik += TK * UnrollK) {
#pragma unroll
              for (size_t ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
                for (int in = 0; in < SGN; in++) {
                  // joint_matrix_load(sg, sub_b[in], slm_b.get_multi_ptr<sycl::access::decorated::no>(), TK);
                  // joint_matrix_load(sg, sub_b[in], pB + ik + TK * ikk + (g_idn + in * TN) * k * 1, k);
                  sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(sg, sub_b[in], pB, k, n, k,
                                                                                    g_idn + in * TN, ik + TK * ikk);
                }
#pragma unroll
                for (int im = 0; im < SGM; im++) {
                  // joint_matrix_load(sg, sub_a[im], pA + (g_idm + im * TM) * k * 1 + ik + TK * ikk, k);
                  sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(sg, sub_a[im], pA, k, m, k,
                                                                                    g_idm + im * TM, ik + TK * ikk);
#pragma unroll
                  for (int in = 0; in < SGN; in++) {
                    joint_matrix_mad(sg, sub_c[im * SGN + in], sub_a[im], sub_b[in], sub_c[im * SGN + in]);
                  }
                }
              }
              if (ik + TK * UnrollK * (PrefetchDis + 1) <= k) {
                for (int im = sggid_col; im < SGM; im += sg_col)
                  if ((g_idm + im * TM) <= m)
                    joint_matrix_prefetch<TM, TK * UnrollK>(
                        sg, A_d + (g_idm + im * TM) * k * 1 + ik + TK * UnrollK * PrefetchDis, k, layout::row_major,
                        syclex::properties{syclex::prefetch_hint_L1});
                for (int in = sggid_row; in < SGN; in += sg_row)
                  if ((g_idn + in * TN) <= n)
                    joint_matrix_prefetch<TN, TK * UnrollK>(
                        sg, B_d + (g_idn + in * TN) * k * 1 + ik + TK * UnrollK * PrefetchDis, k, layout::row_major,
                        syclex::properties{syclex::prefetch_hint_L1});
              }
            }
        // joint_matrix_apply(sg, sub_c, [=](Tc& x) { x *= ALPHA; });
#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++)
                // joint_matrix_store(sg, sub_c[im * SGN + in], pC + (g_idm + im * TM) * n + g_idn + in * TN, n,
                //                    layout::row_major);
                sycl::ext::intel::experimental::matrix::joint_matrix_store_checked(
                    sg, sub_c[im * SGN + in], pC, n, layout::row_major, m, n, g_idm + im * TM, g_idn + in * TN);
          });  // parallel for
    };
    auto ker1 = [&](sycl::handler& cgh) {
      syclex::properties prop{syclintelex::grf_size<256>};
      KernelFunctor largeker(prop, HGemmCfg(), A_d, B_d, C_d, m, n, k);
      cgh.parallel_for(largeker.get_range(), largeker);
    };
    using CFG = xmx::HGemmCfg;
    q->wait();
    for (size_t i = 0; i < runs; i++) {
      // q->submit(ker1);
      Launcher<CFG, xmx::GemmCore>::run(q, {A_d, B_d, C_d, m, n, k, k, k, n, dBias.data()});
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      // q->submit(ker1);
      Launcher<CFG, xmx::GemmCore>::run(q, {A_d, B_d, C_d, m, n, k, k, k, n, dBias.data()});
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2;
    printf("Time %f us, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
    q->memcpy(matC.data(), C_d, matC.size() * 2).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), utils::fp16(0.3f));
  }

  void ut_fp32(int m, int n, int k) {
    int constexpr runs = 100;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<utils::fp16> matA(m * k), matB(k * n), matBT(k * n);
    avector<float> matC(m * n), ref(m * n), bias(n);
    fill_buffer_randn(matA.data(), matA.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(matB.data(), matB.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(bias.data(), bias.size(), -0.5f, 0.5f);
    gemmref_fp16fp16fp32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        ref[i * n + j] += bias[j];
      }
    }

    bestla::kernel::wrapper::Transpose2D<utils::fp16>::forward_auto(matB.data(), matBT.data(), k, n, n, k);
    bestla::sycl_vector<DT> dA(matA.size(), q), dB(matB.size(), q);
    sycl_vector<float> dC(matC.size(), q), dBias(bias.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 2).wait();
    q->memcpy(dB.data(), matBT.data(), matBT.size() * 2).wait();
    q->memcpy(dBias.data(), bias.data(), bias.size() * 4).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();

    using CFG = xmx::HGemmAfp32Cfg;
    q->wait();
    for (size_t i = 0; i < runs; i++) {
      // q->submit(ker1);
      Launcher<CFG, xmx::GemmCore>::run(q, {A_d, B_d, C_d, m, n, k, k, k, n, dBias.data()});
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      // q->submit(ker1);
      Launcher<CFG, xmx::GemmCore>::run(q, {A_d, B_d, C_d, m, n, k, k, k, n, dBias.data()});
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2;
    printf("Time %f us, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), float(0.3f));
  }

  void ut_testT(int m, int n, int k) {
    int constexpr runs = 100;
    int constexpr sg_size = 16;
    int constexpr sg_row = 8;                           // half threads of a XVE
    int constexpr sg_col = 4;                           // half threads of a XVE
    int constexpr wg_size = sg_size * sg_row * sg_col;  // half threads of a XVE
    int constexpr TM = 16;
    int constexpr TN = 16;
    int constexpr TK = 16;
    int constexpr SGM = 2;
    int constexpr SGN = 2;
    int constexpr UnrollK = 2;
    int constexpr wg_repeat = 20 * 32;
    int constexpr BM = TM * sg_row * SGM;
    int constexpr BM_STRIDE = BM + 8;
    int constexpr BN = TN * sg_col * SGN;
    int constexpr BN_STRIDE = BN + 8;
    m = BM;
    n = BN;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<utils::fp16> matA(m * k), matB(k * n), matBT(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(matB.data(), matB.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    gemmref_fp16fp16fp16(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    bestla::kernel::wrapper::Transpose2D<utils::fp16>::forward_auto(matB.data(), matBT.data(), k, n, n, k);
    bestla::sycl_vector<DT> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 2).wait();
    q->memcpy(dB.data(), matBT.data(), matBT.size() * 2).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();

    auto ker = [&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> slm_a(sycl::range(BM_STRIDE * TK), cgh);
      sycl::local_accessor<float, 1> slm_b(sycl::range(BN_STRIDE * TK), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}),
          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(sg_size)]]
          {
            auto sg = it.get_sub_group();
            int sgSize = sg.get_local_range()[0];
            int sgGroupId = sg.get_group_id()[0];
            int sggid_col = sgGroupId % sg_col;
            int sggid_row = sgGroupId / sg_col;
            int sgId = sg.get_local_id()[0];
            int g_idn = sggid_col * TN * SGN;
            int g_idm = sggid_row * TM * SGM;
            auto pA =
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(A_d);
            auto pB =
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(B_d);
            auto pC =
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(C_d);
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::a, TM, TK,
                                                                  layout::row_major>
                sub_a[SGM];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::b, TK, TN,
                                                                  layout::col_major>
                sub_b[SGN];
            sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, DT, use::accumulator, TM, TN>
                sub_c[SGM * SGN];
#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++) joint_matrix_fill(sg, sub_c[im * SGN + in], DT(0.f));
#pragma unroll(1)
            for (size_t ik = 0; ik < k; ik += TK * UnrollK) {
#pragma unroll
              for (size_t ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
                for (int in = 0; in < SGN; in++) {
                  joint_matrix_load(sg, sub_b[in], pB + ik + TK * ikk + (g_idn + in * TN) * k, k);
                }
#pragma unroll
                for (int im = 0; im < SGM; im++) {
                  joint_matrix_load(sg, sub_a[im], pA + (g_idm + im * TM) * k + ik + TK * ikk, k);
#pragma unroll
                  for (int in = 0; in < SGN; in++) {
                    joint_matrix_mad(sg, sub_c[im * SGN + in], sub_a[im], sub_b[in], sub_c[im * SGN + in]);
                  }
                }
              }
            }
        // joint_matrix_apply(sg, sub_c, [=](Tc& x) { x *= ALPHA; });
#pragma unroll
            for (int im = 0; im < SGM; im++)
#pragma unroll
              for (int in = 0; in < SGN; in++)
                joint_matrix_store(sg, sub_c[im * SGN + in], pC + (g_idm + im * TM) * n + g_idn + in * TN, n,
                                   layout::row_major);
          });  // parallel for
    };
    q->wait();
    for (size_t i = 0; i < runs; i++) {
      q->submit(ker);
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      q->submit(ker);
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2 * wg_repeat;
    printf("Time %f ms, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
    q->memcpy(matC.data(), C_d, matC.size() * 2).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), utils::fp16(0.2f));
  }
};
#ifdef BTLA_UT_SYCL
#endif
static UT_SyclHGemm sUT_SyclHGemm;

class UT_SyclHGemmBF16 {
 public:
  UT_SyclHGemmBF16() {
    UT_START();
    ut_xmx(1, 1024, 768);
    ut_xmx(300, 1024, 1024);
    ut_xmx(1033, 1024, 1024);
    ut_xmx(4096, 4096, 4096);
    ut_xmx_fp32(4096, 4096, 4096);
  }

  void ut_xmx(int m, int n, int k) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<utils::bf16> matA(m * k), matB(k * n), matBT(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    fill_buffer_randn(matB.data(), matB.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    gemmref_bf16bf16bf16(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    bestla::kernel::wrapper::Transpose2D<utils::bf16>::forward_auto(matB.data(), matBT.data(), k, n, n, k);

    sycl_vector<sycl::ext::oneapi::bfloat16> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 2).wait();
    q->memcpy(dB.data(), matBT.data(), matBT.size() * 2).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    Launcher<xmx::HGemmBf16Cfg, xmx::GemmCore>::run(q, {A_d, B_d, C_d, m, n, k, k, k, n});
    q->memcpy(matC.data(), C_d, matC.size() * 2).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), utils::bf16(2.6f));
    int constexpr runs = 100;
    q->wait();
    for (size_t i = 0; i < runs; i++) {
      Launcher<xmx::HGemmBf16Cfg, xmx::GemmCore>::run(q, {A_d, B_d, C_d, m, n, k, k, k, n});
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      Launcher<xmx::HGemmBf16Cfg, xmx::GemmCore>::run(q, {A_d, B_d, C_d, m, n, k, k, k, n});
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2;
    printf("Time %f us, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
  }

  void ut_xmx_fp32(int m, int n, int k) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<utils::bf16> matA(m * k), matB(k * n), matBT(k * n);
    avector<float> matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    fill_buffer_randn(matB.data(), matB.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    gemmref_bf16bf16fp32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    bestla::kernel::wrapper::Transpose2D<utils::bf16>::forward_auto(matB.data(), matBT.data(), k, n, n, k);

    sycl_vector<sycl::ext::oneapi::bfloat16> dA(matA.size(), q), dB(matB.size(), q);
    sycl_vector<float> dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 2).wait();
    q->memcpy(dB.data(), matBT.data(), matBT.size() * 2).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    Launcher<xmx::HGemmBf16Fp32Cfg, xmx::GemmCore>::run(q, {A_d, B_d, C_d, m, n, k, k, k, n});
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), float(1.6f));
    int constexpr runs = 100;
    q->wait();
    for (size_t i = 0; i < runs; i++) {
      Launcher<xmx::HGemmBf16Fp32Cfg, xmx::GemmCore>::run(q, {A_d, B_d, C_d, m, n, k, k, k, n});
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      Launcher<xmx::HGemmBf16Fp32Cfg, xmx::GemmCore>::run(q, {A_d, B_d, C_d, m, n, k, k, k, n});
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double ops = (double)m * n * k * 2;
    printf("Time %f us, %f GFLOPS\n", t_ms, ops / t_ms / 1e3);
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclHGemmBF16 sUT_SyclHGemmBF16;
#endif

class UT_SyclInt4Dequant {
 public:
  UT_SyclInt4Dequant() {
    UT_START();
    ut_fp32_T(1024, 384, 32);
    ut_fp32_T(1024, 768, 32);
    ut_fp32_T(1024, 1024, 32);
  }

  void ut_fp32_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        ref[i * k + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * scale[noffset];
        ref[i * k + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * scale[noffset];
      }
    }
    using ProB = sycl_prologue_b::WeightS4T<float>;
    sycl_vector<float> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    auto ev = ProB::template dequant<ProB::CfgDequantF32>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
    ev.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), 0.001f);
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclInt4Dequant sUT_SyclInt4Dequant;
#endif

class UT_SyclInt8Dequant {
 public:
  UT_SyclInt8Dequant() {
    UT_START();
    ut_fp32_T(1024, 384, 32);
    ut_fp32_T(1024, 768, 32);
    ut_fp32_T(1024, 1024, 32);
  }

  void ut_fp32_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", n, k, blocksize, dev->getName().c_str());
    avector<int8_t> rawB(k * n);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), int8_t(-128), int8_t(127));
    auto srcptr = rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = *(std::array<int8_t, 2>*)&srcptr[i * k + j];
        auto noffset = i * blks + j / blocksize;
        ref[i * k + j + 0] = tmp[0] * scale[noffset];
        ref[i * k + j + 1] = tmp[1] * scale[noffset];
      }
    }
    using ProB = sycl_prologue_b::WeightS8T<float>;
    sycl_vector<float> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<int8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    auto ev = ProB::template dequant<ProB::CfgDequantF32>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
    ev.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), 0.001f);
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclInt8Dequant sUT_SyclInt8Dequant;
#endif

class UT_SyclInt8Quant {
 public:
  UT_SyclInt8Quant() {
    UT_START();
    ut_T<float>(4096, 4096, 1);
    ut_T<utils::fp16, sycl::half>(4096, 4096, 1);
    ut_T<utils::bf16, sycl::ext::oneapi::bfloat16>(4096, 4096, 1);
    ut_T<float>(4096, 4096, 0);
    ut_T<utils::fp16, sycl::half>(4096, 4096, 0);
    ut_T<utils::bf16, sycl::ext::oneapi::bfloat16>(4096, 4096, 0);
    // ut_fp32_T(1024, 768, 0);
    // ut_fp32_T(1024, 1024, 0);
  }

  template <typename T, typename SYCLT = T>
  void ut_T(int m, int k, int mask) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, k, mask, dev->getName().c_str());
    avector<int8_t> raw(k * m), q_dev(k * m);
    avector<T> scale(m), scale_dev(m), ref(m * k);
    fill_buffer_randn(scale.data(), scale.size(), T(0.01f), T(0.03f));
    if (mask == 1) {
      for (size_t i = 1; i < scale.size(); i++) {
        scale[i] = scale[0];
      }
    }
    fill_buffer_randn(raw.data(), raw.size(), int8_t(-128), int8_t(127));
    auto srcptr = raw.data();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j += 1) {
        ref[i * k + j] = raw[i * k + j] * (float)scale[i];
      }
    }
    using Pro = sycl_prologue_a::ActivationBase<SYCLT>;
    sycl_vector<T> dS(scale.size(), q), dequantB(ref.size(), q);
    sycl_vector<int8_t> dRaw(raw.size(), q);
    q->memcpy(dequantB.data(), ref.data(), ref.size() * sizeof(T)).wait();
    auto ev = Pro::template quant_s8<typename Pro::CfgQuantF32>(m, k, mask, {(SYCLT*)dequantB.data(), k}, dRaw.data(),
                                                                (SYCLT*)dS.data(), q);
    ev.wait();
    q->memcpy(q_dev.data(), dRaw.data(), q_dev.size()).wait();
    q->memcpy(scale_dev.data(), dS.data(), scale_dev.size() * sizeof(T)).wait();
    buffer_error(raw.data(), q_dev.data(), q_dev.size(), (int8_t)2);
    buffer_error(scale.data(), scale_dev.data(), scale_dev.size(), T(0.001));

    int constexpr runs = 1000;
    q->wait();
    for (size_t i = 0; i < runs; i++) {
      Pro::template quant_s8<typename Pro::CfgQuantF32>(m, k, mask, {(SYCLT*)dequantB.data(), k}, dRaw.data(),
                                                        (SYCLT*)dS.data(), q);
    }
    q->wait();
    bestla::utils::timer<bestla::utils::microseconds> tm;
    tm.start();
    for (size_t i = 0; i < runs; i++) {
      Pro::template quant_s8<typename Pro::CfgQuantF32>(m, k, mask, {(SYCLT*)dequantB.data(), k}, dRaw.data(),
                                                        (SYCLT*)dS.data(), q);
    }
    q->wait();
    auto t_ms = tm.stop() / runs;
    double memsize = (double)m * k * (sizeof(T) + 1);
    if (mask) memsize += m * k * 2;
    printf("Time %f us, %f GB/s\n", t_ms, memsize / t_ms / 1e3);
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclInt8Quant sUT_SyclInt8Quant;
#endif

#if 0
class UT_SyclF8Dequant {
 public:
  UT_SyclF8Dequant() {
    UT_START();
    ut_fp32_T<true>(1024, 384, 64);
    ut_fp32_T<false>(1024, 384, 64);
  }

  template <bool IsE4M3>
  void ut_fp32_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case FP8 %s dequant: %d %d %d Device:%s\n", IsE4M3 ? "E4M3" : "E5M2", n, k, blocksize,
           dev->getName().c_str());
    avector<uint8_t> rawB(k * n);
    int blks = k / blocksize;
    avector<float> scale(size_t(blks) * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    sanitize_fp8_buffer<IsE4M3>(rawB.data(), rawB.size());
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j++) {
        auto noffset = i * blks + j / blocksize;
        ref[i * k + j] = decode_fp8_byte<IsE4M3>(rawB[i * k + j]) * scale[noffset];
      }
    }
    using ProB = sycl_prologue_b::WeightF8T<xve::DefaultSGemmCore, float, IsE4M3>;
    sycl_vector<float> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size()).wait();
    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    auto ev = ProB::template dequant<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
    ev.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), 0.001f);

    avector<float> refNT(k * n);
    kernel::wrapper::Transpose2D<float>::forward<BTLA_ISA::NoSIMD>(ref.data(), refNT.data(), n, k, k, n);
    ev = ProB::template dequant_T<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
    ev.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(refNT.data(), dequant.data(), dequant.size(), 0.001f);
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclF8Dequant sUT_SyclF8Dequant;
#endif

class UT_SyclS8Gemv {
 public:
  UT_SyclS8Gemv() {
    UT_START();
    ut_T(1024, 11008, 32);
    ut_T(1024, 1024, 32);
    ut_half(1024, 11008, 32);
    ut_half(1024, 1024, 32);
  }

  void ut_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), C(n), dqB(n * k), A(k), refC(n);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(A.data(), A.size(), -0.1f, 0.3f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = *(std::array<int8_t, 2>*)&srcptr[i * k + j];
        auto noffset = i * blks + j / blocksize;
        dqB[i + (j + 0) * n] = tmp[0] * scale[noffset];
        dqB[i + (j + 1) * n] = tmp[1] * scale[noffset];
      }
    }
    gemmref_fp32fp32fp32(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<float> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<int8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 4).wait();
    auto S_d = dS.data();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto ev =
        sycl_prologue_b::WeightS8T<xve::DefaultSGemmCore, float>::gemv(A_d, {B_d, S_d, blks}, C_d, n, k, blocksize, q);
    ev.wait();
    q->memcpy(C.data(), C_d, C.size() * 4).wait();
    buffer_error(refC.data(), C.data(), C.size(), 0.001f);
  }

  void ut_half(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n);
    int blks = updiv(k, blocksize);
    avector<utils::fp16> scale(blks * n), C(n), dqB(n * k), A(k), refC(n);
    fill_buffer_randn(scale.data(), scale.size(), utils::fp16(0.01f), utils::fp16(0.03f));
    fill_buffer_randn(A.data(), A.size(), utils::fp16(-0.1f), utils::fp16(0.3f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = *(std::array<int8_t, 2>*)&srcptr[i * k + j];
        auto noffset = i * blks + j / blocksize;
        dqB[i + (j + 0) * n] = tmp[0] * float(scale[noffset]);
        dqB[i + (j + 1) * n] = tmp[1] * float(scale[noffset]);
      }
    }
    gemmref_fp16fp16fp16(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<sycl::half> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<int8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 2).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 2).wait();
    auto S_d = dS.data();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto ev = sycl_prologue_b::WeightS8T<xve::DefaultHGemmCore, sycl::half>::gemv(A_d, {B_d, S_d, blks}, C_d, n, k,
                                                                                  blocksize, q);
    ev.wait();
    q->memcpy(C.data(), C_d, C.size() * 2).wait();
    buffer_error(refC.data(), C.data(), C.size(), utils::fp16(0.5f));
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclS8Gemv sUT_SyclS8Gemv;
#endif

class UT_SyclF8Gemv {
 public:
  UT_SyclF8Gemv() {
    UT_START();
    ut_fp32_T<true>(1024, 11008, 64);
    ut_fp32_T<false>(1024, 11008, 64);
    ut_fp16<true>(1024, 4096, 64);
    ut_fp16<false>(1024, 4096, 64);
  }

  template <bool IsE4M3>
  void ut_fp32_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case FP8 %s gemv fp32: %d %d %d Device:%s\n", IsE4M3 ? "E4M3" : "E5M2", n, k, blocksize,
           dev->getName().c_str());
    avector<uint8_t> rawB(k * n);
    int blks = k / blocksize;
    avector<float> scale(size_t(blks) * n), bias(n), C(n), dqB(n * k), A(k), refC(n);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(bias.data(), bias.size(), -0.02f, 0.02f);
    fill_buffer_randn(A.data(), A.size(), -0.1f, 0.3f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    sanitize_fp8_buffer<IsE4M3>(rawB.data(), rawB.size());
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j++) {
        auto noffset = i * blks + j / blocksize;
        dqB[i + j * n] = decode_fp8_byte<IsE4M3>(rawB[i * k + j]) * scale[noffset];
      }
    }
    gemmref_fp32fp32fp32(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    for (int i = 0; i < n; i++) {
      refC[i] += bias[i];
    }
    sycl_vector<float> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<float> dBias(bias.size(), q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size()).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 4).wait();
    q->memcpy(dBias.data(), bias.data(), bias.size() * 4).wait();
    using ProB = sycl_prologue_b::WeightF8T<xve::DefaultSGemmCore, float, IsE4M3>;
    auto S_d = dS.data();
    auto B_d = dB.data();
    auto A_d = dA.data();
    auto C_d = dC.data();
    auto Bias_d = dBias.data();
    auto ev = ProB::gemv(A_d, {B_d, S_d, blks, Bias_d}, C_d, n, k, blocksize, q);
    ev.wait();
    q->memcpy(C.data(), C_d, C.size() * 4).wait();
    buffer_error(refC.data(), C.data(), C.size(), IsE4M3 ? 0.001f : 0.05f);
  }

  template <bool IsE4M3>
  void ut_fp16(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case FP8 %s gemv fp16: %d %d %d Device:%s\n", IsE4M3 ? "E4M3" : "E5M2", n, k, blocksize,
           dev->getName().c_str());
    avector<uint8_t> rawB(k * n);
    int blks = k / blocksize;
    avector<utils::fp16> scale(size_t(blks) * n), bias(n), C(n), dqB(n * k), A(k), refC(n);
    fill_buffer_randn(scale.data(), scale.size(), utils::fp16(0.01f), utils::fp16(0.03f));
    fill_buffer_randn(bias.data(), bias.size(), utils::fp16(-0.02f), utils::fp16(0.02f));
    fill_buffer_randn(A.data(), A.size(), utils::fp16(-0.1f), utils::fp16(0.3f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    sanitize_fp8_buffer<IsE4M3>(rawB.data(), rawB.size());
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j++) {
        auto noffset = i * blks + j / blocksize;
        float fscale = float(scale[noffset]);
        dqB[i + j * n] = decode_fp8_byte<IsE4M3>(rawB[i * k + j]) * fscale;
      }
    }
    gemmref_fp16fp16fp16(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    for (int i = 0; i < n; i++) {
      refC[i] = utils::fp16(float(refC[i]) + float(bias[i]));
    }
    sycl_vector<sycl::half> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<sycl::half> dBias(bias.size(), q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size()).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 2).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 2).wait();
    q->memcpy(dBias.data(), bias.data(), bias.size() * 2).wait();
    using ProB = sycl_prologue_b::WeightF8T<xve::DefaultHGemmCore, sycl::half, IsE4M3>;
    auto S_d = dS.data();
    auto B_d = dB.data();
    auto A_d = dA.data();
    auto C_d = dC.data();
    auto Bias_d = dBias.data();
    auto ev = ProB::gemv(A_d, {B_d, S_d, blks, Bias_d}, C_d, n, k, blocksize, q);
    ev.wait();
    q->memcpy(C.data(), C_d, C.size() * 2).wait();
    buffer_error(refC.data(), C.data(), C.size(), IsE4M3 ? utils::fp16(0.5f) : utils::fp16(10.f));
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclF8Gemv sUT_SyclF8Gemv;
#endif

void mha_sref(float* Q, float* K, float* V, float* S, float* O, int batch, int seq, int seqA, int hnum, int hsize) {
  avector<float> tmps(seqA);
  int nf = hnum * hsize;
  const float attn_scale = 1.0f / sqrtf(static_cast<float>(hsize));
  int n_past = seqA - seq;
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < seq; j++) {
      for (int ii = 0; ii < hnum; ii++) {
        float maxs = 0.f;
        for (int jj = 0; jj < seqA; jj++) {
          float tmp = 0.f;
          if (jj <= j + n_past) {
            for (int kk = 0; kk < hsize; kk++) {
              tmp +=
                  Q[i * seq * nf + j * nf + ii * hsize + kk] * K[i * nf * seqA + ii * seqA * hsize + jj * hsize + kk];
            }
            tmp *= attn_scale;
          } else {
            tmp = -INFINITY;
          }

          tmps[jj] = tmp;
          maxs = std::max(maxs, tmp);
        }
        float sums = 0.f;
        for (int jj = 0; jj < seqA; jj++) {
          tmps[jj] = std::exp(tmps[jj] - maxs);
          sums += tmps[jj];
        }
        sums = 1.f / sums;
        for (int jj = 0; jj < seqA; jj++) {
          tmps[jj] *= sums;
          S[i * seq * hnum * seqA + j * hnum * seqA + ii * seqA + jj] = tmps[jj];
        }
        for (int kk = 0; kk < hsize; kk++) {
          float tmp = 0.f;
          for (int jj = 0; jj < seqA; jj++) {
            tmp += tmps[jj] * V[i * nf * seqA + ii * hsize * seqA + kk * seqA + jj];
          }
          O[i * seq * nf + j * nf + ii * hsize + kk] = tmp;
        }
      }
    }
  }
}

class UT_MHASgemm {
 public:
  UT_MHASgemm() {
    UT_START();
    ut_T(1, 1, 1, 32, 128);
    ut_T(1, 1, 64, 32, 128);
    ut_T(4, 1, 64, 32, 128);
    ut_T(4, 64, 64, 32, 128);
  }
  template <typename T, typename T_DST>
  class MHA {
   public:
    template <bool Mask>
    static sycl::event forward(int batch, int seq, int seq_acc, int hnum, int hsize, const T* Q, const T* K, const T* V,
                               T_DST* O, sycl::queue* q) {
      const float attn_scale = 1.0f / sqrtf(static_cast<float>(hsize));
      int constexpr SgSize = 16;
      assert(hsize % SgSize == 0);
      int n_past = seq_acc - seq;
      if constexpr (Mask) {
        assert(seq > 1);
      }
      int WgSize = SgSize;
      int seq_acc_pad = utils::padto_le(seq_acc, WgSize * 2);
      int nf = hnum * hsize;
      auto ev = q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<T, 1> slm(sycl::range(std::max(seq_acc, 1024)), cgh);
        cgh.parallel_for(sycl::nd_range<1>(WgSize * batch * seq * hnum, WgSize),
                         [=](auto it) [[intel::reqd_sub_group_size(SgSize)]] {
                           auto sg = it.get_sub_group();
                           auto sg_idx = sg.get_group_id()[0];
                           auto wg_idx = it.get_group(0);
                           auto wg_loc_id = it.get_local_id();
                           auto lane_id = sg.get_local_id()[0];

                           int i = wg_idx;
                           int ih = i % hnum;
                           i /= hnum;
                           int is = i % seq;
                           i /= seq;
                           int ib = i % batch;
                           size_t Q_off = ib * seq * nf + is * nf + ih * hsize;
                           size_t K_off = ib * seq_acc * nf + ih * hsize * seq_acc;
                           size_t V_off = ib * seq_acc * nf + ih * hsize * seq_acc;
                           size_t O_off = ib * seq * nf + is * nf + ih * hsize;
                           typedef sycl::vec<T, 2> TC;
                           T maxs = -INFINITY;
                           for (int jj = 0; jj < seq_acc; jj++) {
                             TC tmp = {0, 0};
                             if constexpr (Mask) {
                               if (jj <= is + n_past) {
                                 for (int ik = wg_loc_id * 2; ik < hsize; ik += WgSize * 2) {
                                   tmp += *(TC*)&Q[Q_off + ik] * *(TC*)&K[K_off + jj * hsize + ik];
                                 }
                                 tmp *= attn_scale;
                               } else {
                                 tmp = {-INFINITY, -INFINITY};
                               }
                             } else {
                               for (int ik = wg_loc_id * 2; ik < hsize; ik += WgSize * 2) {
                                 tmp += *(TC*)&Q[Q_off + ik] * *(TC*)&K[K_off + jj * hsize + ik];
                               }
                               tmp *= attn_scale;
                             }
                             T tmp_sum = tmp[0] + tmp[1];
                             T sum = sycl::reduce_over_group(sg, tmp_sum, std::plus<>());
                             slm[jj] = sum;
                             maxs = std::max(maxs, sum);
                           }
                           float fsums = 0.f;
                           float fmax = float(maxs);
                           int jj = wg_loc_id * 2;
                           for (; jj < seq_acc_pad; jj += WgSize * 2) {
                             auto s2 = *(TC*)&slm[jj];
                             s2[0] = std::exp(s2[0] - fmax);
                             s2[1] = std::exp(s2[1] - fmax);
                             fsums += s2[0];
                             fsums += s2[1];
                             *(TC*)&slm[jj] = s2;
                           }
                           if (jj < seq_acc) {
                             slm[jj] = std::exp(float(slm[jj]) - fmax);
                             fsums += slm[jj];
                             if (jj + 1 < seq_acc) {
                               slm[jj + 1] = std::exp(float(slm[jj + 1]) - fmax);
                               fsums += slm[jj + 1];
                             }
                           }
                           float gsum = sycl::reduce_over_group(sg, fsums, std::plus<>());
                           T scale = 1.f / gsum;
                           jj = wg_loc_id * 2;
                           for (; jj < seq_acc_pad; jj += WgSize * 2) {
                             auto s2 = *(TC*)&slm[jj];
                             s2 *= scale;
                             *(TC*)&slm[jj] = s2;
                           }
                           if (jj < seq_acc) {
                             slm[jj] *= scale;
                             if (jj + 1 < seq_acc) {
                               slm[jj + 1] *= scale;
                             }
                           }

                           for (int kk = 0; kk < hsize; kk++) {
                             TC tmp = {0, 0};
                             jj = wg_loc_id * 2;
                             for (; jj < seq_acc_pad; jj += WgSize * 2) {
                               auto s2 = *(TC*)&slm[jj];
                               auto v2 = *(TC*)&V[V_off + kk * seq_acc + jj];
                               tmp += s2 * v2;
                             }
                             if (jj < seq_acc) {
                               tmp[0] += slm[jj] * V[V_off + kk * seq_acc + jj];
                               if (jj + 1 < seq_acc) {
                                 tmp[1] += slm[jj + 1] * V[V_off + kk * seq_acc + jj + 1];
                               }
                             }
                             T tmp_sum = tmp[0] + tmp[1];
                             T sum = sycl::reduce_over_group(sg, tmp_sum, std::plus<>());
                             O[O_off + kk] = sum;
                           }
                         });
      });
      return ev;
    }
  };

  void ut_T(int batch, int seq, int seqA, int hnum, int hsize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    assert(seqA >= seq);
    printf("Test Case %s: %d %d %d %d %d Device:%s\n", __FUNCTION__, batch, seq, seqA, hnum, hsize,
           dev->getName().c_str());
    avector<float> Q(batch * seq * hnum * hsize), K(batch * seqA * hnum * hsize), V(batch * seqA * hnum * hsize);
    fill_buffer_randn(Q.data(), Q.size(), -0.5f, 0.5f);
    fill_buffer_randn(K.data(), K.size(), -0.5f, 0.5f);
    fill_buffer_randn(V.data(), V.size(), -0.5f, 0.5f);
    avector<float> S(batch * seq * hnum * seqA), O(batch * seq * hnum * hsize);
    mha_sref(Q.data(), K.data(), V.data(), S.data(), O.data(), batch, seq, seqA, hnum, hsize);
    sycl_vector<float> dQ(batch * seq * hnum * hsize, q), dK(batch * seqA * hnum * hsize, q),
        dV(batch * seqA * hnum * hsize, q);
    sycl_vector<float> dS(batch * seq * hnum * seqA, q), dO(batch * seq * hnum * hsize, q);
    q->memcpy(dQ.data(), Q.data(), Q.size() * sizeof(Q[0]));
    q->memcpy(dK.data(), K.data(), K.size() * sizeof(K[0]));
    q->memcpy(dV.data(), V.data(), V.size() * sizeof(V[0]));
    q->wait();
    auto Qptr = dQ.data();
    auto Kptr = dK.data();
    auto Vptr = dV.data();
    auto Sptr = dS.data();
    auto Optr = dO.data();
    int nf = hnum * hsize;
    sycl::range<1> num_items{(size_t)batch * seq * hnum};
    int n_past = seqA - seq;
    const float attn_scale = 1.0f / sqrtf(static_cast<float>(hsize));
    if (seq > 1) {
      MHA<float, float>::forward<true>(batch, seq, seqA, hnum, hsize, Qptr, Kptr, Vptr, Optr, q).wait();
    } else {
      MHA<float, float>::forward<false>(batch, seq, seqA, hnum, hsize, Qptr, Kptr, Vptr, Optr, q).wait();
    }
    // auto ev = q->submit([&](sycl::handler& cgh) {
    //   cgh.parallel_for(num_items, [=](auto it) {
    //     int i = it;
    //     int ih = i % hnum;
    //     i /= hnum;
    //     int is = i % seq;
    //     i /= seq;
    //     int ib = i % batch;
    //     float maxs = 0.f;
    //     float tmps[64];
    //     for (int jj = 0; jj < seqA; jj++) {
    //       float tmp = 0.f;
    //       if (jj <= is + n_past) {
    //         for (int kk = 0; kk < hsize; kk++) {
    //           tmp += Qptr[ib * seq * nf + is * nf + ih * hsize + kk] *
    //                  Kptr[ib * nf * seqA + kk + ih * seqA * hsize + jj * hsize];
    //         }
    //         tmp *= attn_scale;
    //       } else {
    //         tmp = -INFINITY;
    //       }

    //      tmps[jj] = tmp;
    //      maxs = std::max(maxs, tmp);
    //    }
    //    float sums = 0.f;
    //    for (int jj = 0; jj < seqA; jj++) {
    //      tmps[jj] = std::exp(tmps[jj] - maxs);
    //      sums += tmps[jj];
    //    }
    //    sums = 1.f / sums;
    //    for (int jj = 0; jj < seqA; jj++) {
    //      tmps[jj] *= sums;
    //      Sptr[ib * seq * hnum * seqA + is * hnum * seqA + ih * seqA + jj] = tmps[jj];
    //    }
    //    for (int kk = 0; kk < hsize; kk++) {
    //      float tmp = 0.f;
    //      for (int jj = 0; jj < seqA; jj++) {
    //        tmp += tmps[jj] * Vptr[ib * seqA * nf + jj + ih * hsize * seqA + kk * seqA];
    //      }
    //      Optr[ib * seq * nf + is * nf + ih * hsize + kk] = tmp;
    //    }
    //  });
    //});
    q->wait();
    avector<float> STar(batch * seq * hnum * seqA), OTar(batch * seq * hnum * hsize);
    q->memcpy(STar.data(), Sptr, STar.size() * sizeof(STar[0]));
    q->memcpy(OTar.data(), Optr, OTar.size() * sizeof(OTar[0]));
    q->wait();
    // buffer_error(S.data(), STar.data(), S.size(), 0.001f);
    buffer_error(O.data(), OTar.data(), O.size(), 0.001f);
  }
};
#ifdef BTLA_UT_SYCL
#endif
#endif
// static UT_MHASgemm sUT_MHASgemm;
}  // namespace sycl_ut
}  // namespace bestla
