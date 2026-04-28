#include "bestla_gemm.h"
#include "bestla_gemm_kblock.h"
#include "bestla_utils.h"
#include "bestla_ut.h"

namespace bestla {
using namespace utils;

template <int NTILE>
void ref_bf16(utils::bf16* matA, utils::bf16* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride,
              int _cstride, int kpos) {
  int lda = _astride / sizeof(utils::bf16);
  int ldb = _bstride / sizeof(utils::bf16);
  int ldc = _cstride / sizeof(float);
  int constexpr KPack = 4 / sizeof(utils::bf16);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          continue;
        }
        float tmp = 0;
        for (int k = 0; k < _k; k += KPack) {
          for (int ik = 0; ik < KPack; ik++) {
            if (k + ik >= _k) {
              continue;
            }
            auto tmpA = utils::cast<utils::bf16, float>(utils::bf16{matA[i * lda + k + ik]});
            auto tmpB = utils::cast<utils::bf16, float>(utils::bf16{matB[k * NTILE + ij * KPack + ik + j * ldb]});
            tmp += tmpA * tmpB;
          }
        }
        matC[i * ldc + j + ij] = tmp;
      }
    }
  }
}

template <int NTILE>
void ref_fp16(utils::fp16* matA, utils::fp16* matB, utils::fp16* matC, int _m, int _n, int _k, int _astride,
              int _bstride, int _cstride, int kpos) {
  int lda = _astride / sizeof(utils::fp16);
  int ldb = _bstride / sizeof(utils::fp16);
  int ldc = _cstride / sizeof(utils::fp16);
  int constexpr KPack = 1;
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          continue;
        }
        utils::fp16 tmp = utils::fp16(0.f);
        for (int k = 0; k < _k; k += KPack) {
          for (int ik = 0; ik < KPack; ik++) {
            if (k + ik >= _k) {
              continue;
            }
            auto tmpA = utils::cast<utils::fp16, float>(matA[i * lda + k + ik]);
            auto tmpB = utils::cast<utils::fp16, float>(matB[k * NTILE + ij * KPack + ik + j * ldb]);
            tmp = float(tmp) + tmpA * tmpB;
          }
        }
        matC[i * ldc + j + ij] = tmp;
      }
    }
  }
}

template <int NTILE>
void ref_fp16_fp32(utils::fp16* matA, utils::fp16* matB, float* matC, int _m, int _n, int _k, int _astride,
                   int _bstride, int _cstride, int kpos) {
  int lda = _astride / sizeof(utils::fp16);
  int ldb = _bstride / sizeof(utils::fp16);
  int ldc = _cstride / sizeof(float);
  int constexpr KPack = 4 / sizeof(utils::fp16);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          continue;
        }
        float tmp = 0;
        for (int k = 0; k < _k; k += KPack) {
          for (int ik = 0; ik < KPack; ik++) {
            if (k + ik >= _k) {
              continue;
            }
            auto tmpA = utils::cast<utils::fp16, float>(utils::fp16{matA[i * lda + k + ik]});
            auto tmpB = utils::cast<utils::fp16, float>(utils::fp16{matB[k * NTILE + ij * KPack + ik + j * ldb]});
            tmp += tmpA * tmpB;
          }
        }
        matC[i * ldc + j + ij] = tmp;
      }
    }
  }
}

template <int NTILE>
void ref_fp32(float* matA, float* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
              int kpos) {
  int lda = _astride / sizeof(float);
  int ldb = _bstride / sizeof(float);
  int ldc = _cstride / sizeof(float);
  int constexpr KPack = 4 / sizeof(float);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          continue;
        }
        float tmp = 0;
        for (int k = 0; k < _k; k += KPack) {
          for (int ik = 0; ik < KPack; ik++) {
            if (k + ik >= _k) {
              continue;
            }
            auto tmpA = matA[i * lda + k + ik];
            auto tmpB = matB[k * NTILE + ij * KPack + ik + j * ldb];
            tmp += tmpA * tmpB;
          }
        }
        matC[i * ldc + j + ij] = tmp;
      }
    }
  }
}

template <int NTILE, class T_A_, class T_B_>
void ref_int8(T_A_* matA, T_B_* matB, int32_t* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
              int kpos) {
  int lda = _astride / sizeof(T_A_);
  int ldb = _bstride / sizeof(T_B_);
  int ldc = _cstride / sizeof(int32_t);
  int constexpr KPack = 4 / sizeof(T_B_);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          continue;
        }
        int32_t tmp = 0;
        for (int k = 0; k < _k; k += KPack) {
          for (int ik = 0; ik < KPack; ik++) {
            if (k + ik >= _k) {
              continue;
            }
            auto tmpA = utils::cast<T_A_, int32_t>(matA[i * lda + k + ik]);
            auto tmpB = utils::cast<T_B_, int32_t>(matB[k * NTILE + ij * KPack + ik + j * ldb]);
            tmp += tmpA * tmpB;
          }
        }
        matC[i * ldc + j + ij] = tmp;
      }
    }
  }
}

template <int NTILE, typename T_SB_>
void ref_kblock_int8(uint8_t* matA, int8_t* matB, float* matC, uint8_t* zpA, float* scaleA, int _ldsa, T_SB_* scaleB,
                     int* reduceB, int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride,
                     int _cstride, int kpos) {
  int lda = _astride / sizeof(matA[0]);
  int ldb = _bstride / sizeof(matB[0]);
  int ldc = _cstride / sizeof(matC[0]);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          break;
        }
        float tmpf = 0.f;
        for (int k = 0; k < _k; k += _kblock) {
          int tmp = 0;
          int zpval = int(zpA[i * _ldsa + k / _kblock]);
          for (int ik = 0; ik < _kblock; ik += 4) {
            if (k + ik >= _k) {
              break;
            }
            for (int ikk = 0; ikk < 4; ikk++) {
              tmp += (int(matA[i * lda + k + ik + ikk])) * int(matB[(k + ik) * NTILE + ij * 4 + ikk + j * ldb]);
            }
          }
          tmp -= zpval * reduceB[j + ij + k / _kblock * _ldsb];
          tmpf += tmp * scaleA[i * _ldsa + k / _kblock] * float(scaleB[j + ij + k / _kblock * _ldsb]);
        }
        matC[i * ldc + j + ij] = tmpf;
      }
    }
  }
}

template <int NTILE, typename T_SB_>
void ref_kblock_int8_ss(int8_t* matA, int8_t* matB, float* matC, float* scaleA, int _ldsa, T_SB_* scaleB, int _ldsb,
                        int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride, int kpos) {
  int lda = _astride / sizeof(matA[0]);
  int ldb = _bstride / sizeof(matB[0]);
  int ldc = _cstride / sizeof(matC[0]);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          break;
        }
        float tmpf = 0.f;
        for (int k = 0; k < _k; k += _kblock) {
          int tmp = 0;
          for (int ik = 0; ik < _kblock; ik += 4) {
            if (k + ik >= _k) {
              break;
            }
            for (int ikk = 0; ikk < 4; ikk++) {
              tmp += (int(matA[i * lda + k + ik + ikk])) * int(matB[(k + ik) * NTILE + ij * 4 + ikk + j * ldb]);
            }
          }
          tmpf += tmp * scaleA[i * _ldsa + k / _kblock] * float(scaleB[j + ij + k / _kblock * _ldsb]);
        }
        matC[i * ldc + j + ij] = tmpf;
      }
    }
  }
}
namespace ut {
class UT_GEMM_AVX2 {
 public:
  UT_GEMM_AVX2() {
    UT_START();
    CheckISA(AVX2);
    ut_24(4, 24, 3);
    ut_24(4, 48, 3);

    ut_48(1, 48, 3);
    ut_48(1, 144, 3);
  }

  void ut_24(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::SCoreRowNAvx2<24>;
    static Core gemm;
    if (n % Core::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    if (m > Core::Code::MTILE) {
      return;
    }
    avector<float> A(m * k), B(k * n), C(m * n, 0.f), RefC(m * n, 0.f);
    fill_buffer_randn(A.data(), A.size(), -0.5f, 0.5f);
    fill_buffer_randn(B.data(), B.size(), -0.5f, 0.5f);
    ref_fp32<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * 4, k * 4, n * 4, 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * 4, k * 4, n * 4, 0, cache, CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }

  void ut_48(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::SCoreRowNAvx2<48>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    if (m > Core::Code::MTILE) {
      return;
    }
    avector<float> A(m * k), B(k * n), C(m * n, 0.f), RefC(m * n, 0.f);
    fill_buffer_randn(A.data(), A.size(), -0.5f, 0.5f);
    fill_buffer_randn(B.data(), B.size(), -0.5f, 0.5f);
    ref_fp32<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * 4, k * 4, n * 4, 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * 4, k * 4, n * 4, 0, cache, CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }
};
#ifdef BTLA_UT_GEMM
static UT_GEMM_AVX2 sUT_GEMM_AVX2;
#endif

class UT_GEMM_AVX512F {
 public:
  UT_GEMM_AVX512F() {
    UT_START();
    CheckISA(AVX512F);
    ut_32(4, 32, 3);
    ut_32(4, 64, 3);

    ut_48(1, 48, 3);
    ut_48(1, 144, 3);
  }

  void ut_32(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::SCoreRowNAvx512f<32>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<float> A(m * k), B(k * n), C(m * n, 0.f), RefC(m * n, 0.f);
    fill_buffer_randn(A.data(), A.size(), -0.5f, 0.5f);
    fill_buffer_randn(B.data(), B.size(), -0.5f, 0.5f);
    ref_fp32<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * 4, k * 4, n * 4, 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * 4, k * 4, n * 4, 0, cache, CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }

  void ut_48(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::SCoreRowNAvx512f<48>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<float> A(m * k), B(k * n), C(m * n, 0.f), RefC(m * n, 0.f);
    fill_buffer_randn(A.data(), A.size(), -0.5f, 0.5f);
    fill_buffer_randn(B.data(), B.size(), -0.5f, 0.5f);
    ref_fp32<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * 4, k * 4, n * 4, 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * 4, k * 4, n * 4, 0, cache, CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }
};
#ifdef BTLA_UT_GEMM
static UT_GEMM_AVX512F sUT_GEMM_AVX512F;
#endif

class UT_GEMM_AVX512VNNI {
 public:
  UT_GEMM_AVX512VNNI() {
    UT_START();
    CheckISA(AVX512_VNNI);
    ut<32, 0>(4, 64, 12);
    ut<32, 12>(4, 64, 12);

    ut<48, 0>(4, 96, 12);
    ut<48, 8>(4, 96, 12);
  }

  template <int NTile, int MTile>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAvx512vnni<NTile, MTile>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<uint8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<int> C(m * n, 0), RefC(m * n, 0);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    ref_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]),
                                n * sizeof(C[0]), 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, cache,
                 CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 1);
  }
};
#ifdef BTLA_UT_GEMM
static UT_GEMM_AVX512VNNI sUT_GEMM_AVX512VNNI;
#endif

class UT_GEMM_AVX512BW {
 public:
  UT_GEMM_AVX512BW() {
    UT_START();
    CheckISA(AVX512BW);
    ut<32, 0>(4, 64, 12);
    ut<32, 12>(4, 64, 12);

    ut<48, 0>(4, 96, 12);
    ut<48, 8>(4, 96, 12);
  }

  template <int NTile, int MTile>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAvx512bw<NTile, MTile>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<uint8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<int> C(m * n, 0), RefC(m * n, 0);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)128);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    ref_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]),
                                n * sizeof(C[0]), 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, cache,
                 CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 1);
  }
};
#ifdef BTLA_UT_GEMM
static UT_GEMM_AVX512BW sUT_GEMM_AVX512BW;
#endif

class UT_GEMM_KBLOCK {
 public:
  UT_GEMM_KBLOCK() {
    UT_START();
    std::vector<std::array<int, 3>> cases = {{96, 36, 36}, {144, 128, 32}, {144, 128, 128}, {144, 256, 128}};
    GetCPUDevice();
    if (_cd->AVX2()) {
      using Core0 = gemm::KblockGemmAvx2vnni<24, 2>;
      using Core1 = gemm::KblockGemmAvx2vnni<24, 1>;
      using Core2 = gemm::KblockGemmAvx2vnniSS<24, 1>;
      for (size_t i = 0; i < cases.size(); i++) {
        ut<Core0>(cases[i][0], cases[i][1], cases[i][2]);
        ut<Core1>(cases[i][0], cases[i][1], cases[i][2]);
        ut<Core2>(cases[i][0], cases[i][1], cases[i][2]);
      }
    }
    if (_cd->AVX_VNNI()) {
      using Core0 = gemm::KblockGemmAvxvnni<24, 2>;
      using Core1 = gemm::KblockGemmAvxvnni<24, 1>;
      using Core2 = gemm::KblockGemmAvxvnniSS<24, 1>;
      for (size_t i = 0; i < cases.size(); i++) {
        ut<Core0>(cases[i][0], cases[i][1], cases[i][2]);
        ut<Core1>(cases[i][0], cases[i][1], cases[i][2]);
        ut<Core2>(cases[i][0], cases[i][1], cases[i][2]);
      }
    }
    if (_cd->AVX512BW()) {
      using Core0 = gemm::KblockGemmAvx512bw<48, 4>;
      for (size_t i = 0; i < cases.size(); i++) {
        ut<Core0>(cases[i][0], cases[i][1], cases[i][2]);
      }
    }
    if (_cd->AVX512_VNNI()) {
      using Core0 = gemm::KblockGemmAvx512vnni<48, 4>;
      using Core1 = gemm::KblockGemmAvx512vnni<48, 5>;
      for (size_t i = 0; i < cases.size(); i++) {
        ut<Core0>(cases[i][0], cases[i][1], cases[i][2]);
        ut<Core1>(cases[i][0], cases[i][1], cases[i][2]);
      }
    }
    if (_cd->AMX_INT8()) {
      using Core0 = gemm::KblockGemmAmxint8<48, 16>;
      using Core1 = gemm::KblockGemmAmxint8SS<48, 16>;
      for (size_t i = 0; i < cases.size(); i++) {
        ut<Core0>(cases[i][0], cases[i][1], cases[i][2]);
        ut<Core1>(cases[i][0], cases[i][1], cases[i][2]);
      }
    }
  }

  template <class Core>
  void ut(int n, int k, int kblock) {
    auto m = Core::MTILE;
    printf("Test Case %s: %d %d %d-%d\n", gemm::CoreAttr::to_str(Core::ID), m, n, k, kblock);
    if (n % Core::NTILE != 0) {
      return;
    }
    if (k % Core::KTILE != 0) {
      return;
    }
    if (kblock % Core::KTILE != 0) {
      return;
    }
    using AT = typename Core::AType;
    avector<AT> A(m * k);
    avector<int8_t> B(k * n);
    avector<float> C(m * n, 0), RefC(m * n, 0);
    int blk_num = utils::updiv(k, kblock);
    avector<float> scaleA(blk_num * m), scaleB(blk_num * n);
    avector<int> reduceB(blk_num * n, 0);
    avector<AT> zpA(m * blk_num);
    fill_buffer_randn(A.data(), A.size(), (AT)0, (AT)127);
    fill_buffer_randn(zpA.data(), zpA.size(), (AT)0, (AT)55);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    fill_buffer_randn(scaleA.data(), scaleA.size(), 0.003f, 0.005f);
    fill_buffer_randn(scaleB.data(), scaleB.size(), 0.003f, 0.005f);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        reduceB[i / kblock * n + j] += B[i * n + j];
      }
    }
    if constexpr (std::is_same_v<AT, uint8_t>) {
      ref_kblock_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), zpA.data(), scaleA.data(), blk_num,
                                         scaleB.data(), reduceB.data(), n, m, n, k, kblock, k * sizeof(A[0]),
                                         k * sizeof(B[0]), n * sizeof(C[0]), 0);
    } else {
      ref_kblock_int8_ss<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), scaleA.data(), blk_num, scaleB.data(), n,
                                            m, n, k, kblock, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0);
    }
    Core::configure(m, n, k);
    Core::forward(A.data(), B.data(), C.data(), std::is_same_v<AT, uint8_t> ? zpA.data() : nullptr, scaleA.data(),
                  blk_num, scaleB.data(), reduceB.data(), n, m, n, k, kblock, k * sizeof(A[0]), k * sizeof(B[0]),
                  n * sizeof(C[0]), 0, cache, CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }
};
#ifdef BTLA_UT_GEMM
static UT_GEMM_KBLOCK sUT_GEMM_KBLOCK;
#endif

class UT_GEMM_AVXVNNI {
 public:
  UT_GEMM_AVXVNNI() {
    UT_START();
    CheckISA(AVX_VNNI);
    ut<24>(4, 48, 12);
    ut<48>(2, 96, 12);

    ut_ss<24>(4, 48, 12);
    ut_ss<48>(2, 96, 12);
  }

  template <int NTile>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAvxvnni<NTile>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<uint8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<int> C(m * n, 0), RefC(m * n, 0);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    ref_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]),
                                n * sizeof(C[0]), 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, cache,
                 CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 1);
  }

  template <int NTile>
  void ut_ss(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAvxvnniSS<NTile>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<int8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<int> C(m * n, 0), RefC(m * n, 0);
    fill_buffer_randn(A.data(), A.size(), (int8_t)-127, (int8_t)127);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    ref_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]),
                                n * sizeof(C[0]), 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, cache,
                 CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 1);
  }
};
#ifdef BTLA_UT_GEMM
static UT_GEMM_AVXVNNI sUT_GEMM_AVXVNNI;
#endif

class UT_GEMM_AVX2VNNI {
 public:
  UT_GEMM_AVX2VNNI() {
    UT_START();
    CheckISA(AVX2);
    ut<24>(1, 48, 12);
    ut<24>(2, 48, 12);

    ut_ss<24>(1, 48, 12);
    ut_ss<24>(2, 96, 12);
  }

  template <int NTile>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAvx2vnni<NTile, 3>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<uint8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<int> C(m * n, 0), RefC(m * n, 0);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)127);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    ref_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]),
                                n * sizeof(C[0]), 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, cache,
                 CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 1);
  }

  template <int NTile>
  void ut_ss(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAvx2vnniSS<NTile, 3>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<int8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<int> C(m * n, 0), RefC(m * n, 0);
    fill_buffer_randn(A.data(), A.size(), (int8_t)-127, (int8_t)127);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    ref_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]),
                                n * sizeof(C[0]), 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, cache,
                 CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 1);
  }
};
#ifdef BTLA_UT_GEMM
static UT_GEMM_AVX2VNNI sUT_GEMM_AVX2VNNI;
#endif

class UT_GEMM_AVX512FP16 {
 public:
  UT_GEMM_AVX512FP16() {
    UT_START();
    CheckISA(AVX512_FP16);
    ut<32, 0>(4, 64, 3);
    ut<64, 0>(4, 128, 3);
  }

  template <int NTILE, int MTILE>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::HCoreRowNAvx512fp16<NTILE, MTILE>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }

    avector<utils::fp16> matAbf16(m * k), matBbf16(k * n), matC(m * n), refC(m * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    int reordered_bstride = k * 2;
    ref_fp16<Core::NTILE>(matAbf16.data(), matBbf16.data(), refC.data(), m, n, k, k * 2, reordered_bstride, n * 2, 0);
    gemm.forward(matAbf16.data(), matBbf16.data(), matC.data(), m, n, k, k * sizeof(fp16), k * sizeof(fp16),
                 n * sizeof(fp16), 0, cache, CacheSize);
    ut::buffer_error(refC.data(), matC.data(), refC.size(), fp16(FP16_ERR));
  }
};
#ifdef BTLA_UT_GEMM
static UT_GEMM_AVX512FP16 sUT_GEMM_AVX512FP16;
#endif

class UT_GEMM_AVX512BF16 {
 public:
  UT_GEMM_AVX512BF16() {
    UT_START();
    CheckISA(AVX512_BF16);
    ut<48, 0>(4, 96, 6);
    ut<48, 8>(4, 96, 6);
    ut<64, 0>(4, 128, 6);
  }

  template <int NTILE, int MTILE>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::HCoreRowNAvx512bf16<NTILE, MTILE>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }

    avector<utils::bf16> matAbf16(m * k), matBbf16(k * n);
    avector<float> matC(m * n), refC(m * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    ref_bf16<Core::NTILE>(matAbf16.data(), matBbf16.data(), refC.data(), m, n, k, k * 2, k * 2, n * 4, 0);
    gemm.forward(matAbf16.data(), matBbf16.data(), matC.data(), m, n, k, k * sizeof(bf16), k * sizeof(bf16),
                 n * sizeof(float), 0, cache, CacheSize);
    ut::buffer_error(refC.data(), matC.data(), refC.size(), 0.001f);
  }
};
#ifdef BTLA_UT_GEMM
static UT_GEMM_AVX512BF16 sUT_GEMM_AVX512BF16;
#endif

class UT_GEMM_AMXBF16 {
 public:
  UT_GEMM_AMXBF16() {
    UT_START();
    CheckISA(AMX_BF16);
    ut<32, 32>(32, 32, 64);
    ut<32, 32>(4, 96, 96);
    ut<48, 0>(4, 96, 96);
    ut<64, 16>(4, 128, 96);
  }

  template <int NTILE, int MTILE>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::HCoreRowNAmxbf16<NTILE, MTILE>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }

    avector<utils::bf16> matAbf16(m * k), matBbf16(k * n);
    avector<float> matC(Core::Code::MTILE * n), refC(Core::Code::MTILE * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    ref_bf16<Core::NTILE>(matAbf16.data(), matBbf16.data(), refC.data(), m, n, k, k * 2, k * 2, n * 4, 0);
    gemm.configure(m, n, k);

    gemm.forward(matAbf16.data(), matBbf16.data(), matC.data(), m, n, k, k * sizeof(bf16), k * sizeof(bf16),
                 n * sizeof(float), 0, cache, CacheSize);
    ut::buffer_error(refC.data(), matC.data(), m * n, 0.001f);
  }
};
#ifdef BTLA_UT_GEMM
static UT_GEMM_AMXBF16 sUT_GEMM_AMXBF16;
#endif

class UT_GEMM_AMXFP16 {
 public:
  UT_GEMM_AMXFP16() {
    UT_START();
    CheckISA(AMX_FP16);
    ut<32, 32>(32, 32, 64);
    ut<32, 32>(4, 96, 96);
    ut<48, 0>(4, 96, 96);
    ut<64, 16>(4, 128, 96);
  }

  template <int NTILE, int MTILE>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::HCoreRowNAmxfp16<NTILE, MTILE>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }

    avector<utils::fp16> matAfp16(m * k), matBfp16(k * n);
    avector<float> matC(Core::Code::MTILE * n), refC(Core::Code::MTILE * n);
    fill_buffer_randn(matAfp16.data(), matAfp16.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(matBfp16.data(), matBfp16.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    ref_fp16_fp32<Core::NTILE>(matAfp16.data(), matBfp16.data(), refC.data(), m, n, k, k * 2, k * 2, n * 4, 0);
    gemm.configure(m, n, k);

    gemm.forward(matAfp16.data(), matBfp16.data(), matC.data(), m, n, k, k * sizeof(fp16), k * sizeof(fp16),
                 n * sizeof(float), 0, cache, CacheSize);
    ut::buffer_error(refC.data(), matC.data(), m * n, 0.001f);
  }
};
#ifdef BTLA_UT_GEMM
static UT_GEMM_AMXFP16 sUT_GEMM_AMXFP16;
#endif

class UT_GEMM_AMXINT8 {
 public:
  UT_GEMM_AMXINT8() {
    UT_START();
    CheckISA(AMX_INT8);
    ut<32, 32>(32, 64, 64 * 3);
    ut<48, 16>(16, 96, 64 * 3);
    ut<32, 32>(4, 64, 64 * 3);
    ut<32, 32>(20, 64, 64 * 3);
    ut<32, 32>(32, 64, 64 * 3);

    ut<64, 16>(16, 128, 64 * 3);
  }

  template <int NTile, int MTile>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAmxint8<NTile, MTile>;
    static Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<uint8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<int> C(Core::Code::MTILE * n, 0), RefC(Core::Code::MTILE * n, 0);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    ref_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]),
                                n * sizeof(C[0]), 0);
    gemm.configure(m, n, k);
    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, cache,
                 CacheSize);
    ut::buffer_error(RefC.data(), C.data(), m * n, 1);
  }
};
#ifdef BTLA_UT_GEMM
static UT_GEMM_AMXINT8 sUT_GEMM_AMXINT8;
#endif
}  // namespace ut
}  // namespace bestla
