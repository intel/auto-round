#include <random>
#include <vector>

#include "bestla/bestla_device.h"
#pragma once

#include <random>
#include <stdexcept>
#include "bestla/bestla_utils.h"
#include "bestla/bestla_gemm.h"
#include "bestla/bestla_device.h"
#include "bestla/bestla_parallel.h"

#define CheckISA(ISA)                         \
  {                                           \
    GetCPUDevice();                           \
    if (!_cd->ISA()) {                        \
      printf("Wrong Device ISA: " #ISA "\n"); \
      return;                                 \
    }                                         \
  }

namespace bestla {
namespace ut {
using sAVX512F = gemm::SCoreRowNAvx512f<48, 8>;
using sAMX_BF16 = gemm::HCoreRowNAmxbf16<64, 16>;
using sAMX_FP16 = gemm::HCoreRowNAmxfp16<64, 16>;
using sAVX512_FP16 = gemm::HCoreRowNAvx512fp16<96, 8>;
using sAVX_VNNI = gemm::ICoreRowNAvxvnni<24, 4>;
using sAVX_VNNI_SS = gemm::ICoreRowNAvxvnniSS<24, 4>;
using sAVX512BW = gemm::ICoreRowNAvx512bw<48, 8>;
using sAVX512_VNNI = gemm::ICoreRowNAvx512vnni<48, 8>;
using sAMX_INT8_US = gemm::ICoreRowNAmxint8<64, 16>;
using sAMX_INT8_SS = gemm::ICoreRowNAmxint8SS<64, 16>;
using sAVX2 = gemm::SCoreRowNAvx2<24, 4>;

class UT_Threading {
 public:
  static bestla::parallel::IThreading* get() {
#if BTLA_OPENMP
    static bestla::parallel::OMPThreading DefaultThreading;
#else
    static bestla::parallel::StdThreading DefaultThreading;
#endif  // _OPNEMP
    return &DefaultThreading;
  }

  static void set_threads(int n_thread) { get()->set_threads(n_thread); }

  static std::vector<int> get_threads_config() {
    GetCPUDevice();
    if (_cd->isClient()) {
      if (_cd->isHybrid()) {
        return std::vector<int>{_cd->getThreads(), _cd->getCores(), int(_cd->getPcoreNum())};
      }
      return std::vector<int>{_cd->getCores() * 2, _cd->getCores()};
    }

    if (_cd->getThreads() == 56) {
      return std::vector<int>{48, 56};
    }
    if (_cd->getThreads() == 128) {
      return std::vector<int>{32};
    }
    return std::vector<int>{_cd->getThreads()};
  }
};
static inline size_t gemm_memsize(int m, int n, int k, BTLA_DTYPE dtA, BTLA_DTYPE dtB, BTLA_DTYPE dtC) {
  size_t total = 0;
  total += size_t(m) * k * utils::bestla_dtype_bits(dtA);
  total += size_t(n) * k * utils::bestla_dtype_bits(dtB);
  total += size_t(m) * n * utils::bestla_dtype_bits(dtC);
  return total / 8;
}

static inline int auto_batch(size_t memsize) {
  GetCPUDevice();
  auto L3 = _cd->getL3CacheSize();
  size_t constexpr Enlarge = 4;
  size_t constexpr TargetMem = 1LL << 30;
  auto batch = static_cast<int>(std::max(L3 * Enlarge, TargetMem) / memsize);
  return batch > 1 ? batch : 2;
}

constexpr size_t CacheSize = size_t(100) << 10;
static int8_t cache[CacheSize];

// UT Error definitions
// Activation uniform distribution range [-0.5f,0.5f]
// Weight uniform distribution range [-0.5f,0.5f]
// reduce dim: 4096
#define FP32_ERR 0.0001f
#define FP16_ERR 0.001f
#define BF16_ERR 0.02f
#define INT8_ERR 0.2f
#define F8_ERR 1.5f
#define INT4_ERR 3.5f
#define INT3_ERR 7.f
#define INT2_ERR 18.f
#define INT1_ERR 1e3f
#define FP4_ERR 3.5f

static inline float get_ut_err(BTLA_DTYPE qtype) {
  auto dbits = utils::bestla_dtype_bits(qtype);
  auto type = utils::bestla_dtype_type(qtype);
  auto err = FP32_ERR;
  auto constexpr dtype_int = utils::bestla_dtype_type(BTLA_DTYPE::TypeInt);
  if (type == dtype_int) {
    if (dbits == 8) {
      err = INT8_ERR;
    } else if (dbits == 3) {
      err = INT3_ERR;
    } else if (dbits == 2) {
      err = INT2_ERR;
    } else if (dbits == 1) {
      err = INT1_ERR;
    } else {
      err = INT4_ERR;
    }
  } else {
    if (dbits == 4) {
      err = FP4_ERR;
    } else if (dbits == 8) {
      err = F8_ERR;
    } else if (dbits == 16) {
      if (qtype == BTLA_DTYPE::F16) {
        err = FP16_ERR;
      } else {
        err = BF16_ERR;
      }
    }
  }
  return err;
}

template <typename _T>
inline _T randn(_T minval, _T maxval) {
  auto normval = (rand() + 0.5f) / (RAND_MAX + 1.f);
  auto _gap = maxval - minval;
  return static_cast<_T>(_gap * normval + minval);
}

template <>
inline utils::bf16 randn(utils::bf16 minval, utils::bf16 maxval) {
  auto normval = (rand() + 0.5f) / (RAND_MAX + 1.f);
  auto _gap = maxval.tofloat() - minval.tofloat();
  utils::bf16 tmp;
  tmp.fromfloat(_gap * normval + minval.tofloat());
  return tmp;
}

template <>
inline utils::fp16 randn(utils::fp16 minval, utils::fp16 maxval) {
  auto normval = (rand() + 0.5f) / (RAND_MAX + 1.f);
  auto _gap = float(maxval) - float(minval);
  return utils::fp16(_gap * normval + float(minval));
}

inline float remove_bf16_err(float raw) {
  return utils::cast<utils::bf16, float>(utils::cast<float, utils::bf16>(raw));
}

inline void buffer_remove_bf16_err(float* buf, size_t size) {
  for (size_t i = 0; i < size; i++) {
    buf[i] = remove_bf16_err(buf[i]);
  }
}

template <typename _T>
static void fill_buffer_randn(_T* buf, size_t size, _T minval, _T maxval) {
  if (size <= 1024) {
    for (size_t i = 0; i < size; i++) {
      buf[i] = randn(minval, maxval);
    }
    return;
  }
  int constexpr chunks = 128;
  auto sizeP = bestla::utils::padto_le(size, chunks);
  auto size_per_run = sizeP / chunks;
#pragma omp parallel for
  for (size_t i = 0; i < chunks; i++) {
    for (size_t j = 0; j < size_per_run; j++) {
      buf[i * size_per_run + j] = randn(minval, maxval);
    }
  }
  for (size_t i = sizeP; i < size; i++) {
    buf[i] = randn(minval, maxval);
  }
}

template <typename _T>
static void fill_buffer_faken(_T* buf, size_t size, _T minval, _T maxval) {
  int constexpr seed = 997;
  auto _gap = float(maxval) - float(minval);
#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    auto val = i % seed;
    auto normval = (val + 0.5f) / (seed + 1.f);
    buf[i] = _gap * normval + (float)minval;
  }
}

template <typename _T>
utils::aligned_vector<_T> readFile2Buffer(const char* filepath) {
  auto w1fp = fopen(filepath, "rb");
  if (w1fp == nullptr) {
    return utils::aligned_vector<_T>();
  }
  fseek(w1fp, 0, SEEK_END);
  auto size = ftell(w1fp);
  fseek(w1fp, 0, SEEK_SET);
  size_t memsize = utils::downdiv(size_t(size), sizeof(_T));
  utils::aligned_vector<_T> buf(memsize);
  fread(buf.data(), size, 1, w1fp);
  fclose(w1fp);
  return buf;
}

#define UT_START()                            \
  {                                           \
    UT_Threading::get()->set_threads(0);      \
    GetCPUDevice();                           \
    printf("Test Class: %s\n", __FUNCTION__); \
  }
template <typename _T>
static double buffer_error(_T* ref, _T* tar, size_t size, _T athres = _T(0), _T rthres = _T(0)) {
  double err = 0;
  int cnt = 0;
  float max_err = 0, max_a = 0.f, max_b = 0.f;
  int constexpr MAX_PRINT = 10;
  int max_i = 0;
  float _rthres = (float)rthres;
  float _athres = (float)athres;
  for (size_t i = 0; i < size; i++) {
    auto vtar = float(tar[i]);
    auto vref = float(ref[i]);
    auto diff = abs(vtar - vref);
    err += diff;
    bool _err = (diff > _rthres * abs(vref)) && (diff > _athres);
    if (_err && cnt < MAX_PRINT) {
      printf("%6i Ref %12.5f\tTar %12.5f\n", int(i), vref, vtar);
    }
    if (_err) cnt++;

    if (diff > max_err) {
      max_err = diff;
      max_a = vref;
      max_b = vtar;
      max_i = int(i);
    }
  }
  if (cnt == 0) {
    printf("Case Passed!\n");
  } else {
    printf("Case Failed!\n");
  }
  printf("Max Error @%d:%.3f=%.3f-%.3f\n", max_i, max_err, max_a, max_b);
  err /= size;
  printf("Average Error: %.3f\n", err);
  return err;
}

template <typename T>
static inline void print_buffer(const T* buf, size_t size) {
  if (size <= 16) {
    std::string str = "";
    for (size_t i = 0; i < size; i++) {
      str += std::to_string(float(buf[i])) + " ";
    }
    printf("%s\n", str.c_str());
    return;
  }
  float total_value = 0.;
  for (size_t i = 0; i < size; i++) {
    total_value += float(buf[i]);
  }
  printf("Total value is:  %f\n", float(total_value));
  constexpr size_t print_len = 6;
  std::string str = "";
  for (size_t i = 0; i < print_len; i++) {
    str += std::to_string(float(buf[i])) + " ";
  }
  str += "..., ";
  for (size_t i = 0; i < print_len; i++) {
    str += std::to_string(float(buf[size - print_len + i])) + " ";
  }
  printf("%s\n", str.c_str());
}

static void compare_topk(float* ref, float* tar, size_t size, int k = 8) {
  std::vector<int> indice(size);
  for (size_t i = 0; i < indice.size(); i++) {
    indice[i] = int(i);
  }
  auto comp = [&](const int& a, const int& b) { return abs(ref[a]) > abs(ref[b]); };
  if (k <= 128) {
    std::partial_sort(indice.begin(), indice.begin() + k, indice.end(), comp);
    printf("Top %d elements\n", k);
    for (size_t i = 0; i < k; i++) {
      printf("%6i Ref %12.5f\tTar %12.5f\n", int(indice[i]), float(ref[indice[i]]), float(tar[indice[i]]));
    }
    printf("\n");
  }
}

template <typename _T>
static double buffer_error_2d(_T* ref, _T* tar, size_t row, size_t col, size_t refstep, size_t tarstep,
                              _T thres = _T(0)) {
  double err = 0;
  int cnt = 0;
  int constexpr MAX_PRINT = 10;
  for (size_t i = 0; i < row; i++) {
    for (size_t j = 0; j < col; j++) {
      auto refv = ref[i * refstep + j];
      auto tarv = tar[i * tarstep + j];
      auto diff = std::abs(refv - tarv);
      err += diff;
      if (diff > thres && cnt < MAX_PRINT) {
        cnt++;
        printf("%i %d Ref %12.5f\tTar %12.5f\n", int(i), int(j), utils::cast<_T, float>(refv),
               utils::cast<_T, float>(tarv));
      }
    }
  }
  if (cnt == 0) {
    printf("Case Passed!\n");
  } else {
    printf("Case Failed!\n");
  }
  auto size = (size_t)row * col;
  err /= size;
  printf("Average Error: %.3f\n", err);
  return err;
}

struct UT_vector_s8 {
  utils::aligned_vector<int8_t> data_;
  void resize(size_t _size) { data_.resize(_size); }
  size_t size() { return data_.size(); }
  int8_t* data() { return data_.data(); }
  void fill_rand(int8_t minval, int8_t maxval) {
    for (size_t i = 0; i < data_.size(); i++) {
      data_[i] = randn(minval, maxval);
    }
  }
  void rand_scale(size_t n, float minval, float maxval) {
    scales.resize(n);
    for (size_t i = 0; i < n; i++) {
      scales[i] = randn(minval, maxval);
    }
  }
  utils::aligned_vector<float> scales;
};

struct UT_vector_u8 {
  utils::aligned_vector<uint8_t> data_;
  void resize(size_t _size) { data_.resize(_size); }
  size_t size() { return data_.size(); }
  uint8_t* data() { return data_.data(); }
  void fill_rand(uint8_t minval, uint8_t maxval) {
    for (size_t i = 0; i < data_.size(); i++) {
      data_[i] = ut::randn(minval, maxval);
    }
  }
  void rand_scale(size_t n, float minval, float maxval) {
    scales.resize(n);
    zeropoints.resize(n);
    for (size_t i = 0; i < n; i++) {
      scales[i] = ut::randn(minval, maxval);
      zeropoints[i] = 0;
    }
  }
  utils::aligned_vector<float> scales;
  utils::aligned_vector<int> zeropoints;
};

struct UT_GEMMData_Row_u8s8 {
  UT_GEMMData_Row_u8s8(int m, int n, int k, int lda, int ldb, int ldc, int ldd, int nscale = 1,
                       bool interleaved_ = false)
      : M(m), N(n), K(k), LDA(lda), LDB(ldb), LDC(ldc), LDD(ldd), interleaved(interleaved_) {
    matA.resize(m * lda);
    matB.resize((interleaved ? n : k) * ldb);
    matC.resize(m * ldc);
    if (ldd == 0) {
      matD.resize(n);
    } else {
      matD.resize(m * ldd);
    }
    matA.fill_rand(0, 255);
    matB.fill_rand(-127, 127);
    matD.fill_rand(0, 255);
    matA.rand_scale(nscale, 0.f, 0.01f);
    matB.rand_scale(nscale, 0.f, 0.01f);
    matD.rand_scale(nscale, 0.f, 1.f);
  }

  void calc_ref(float alpha, float beta) {
    if (interleaved) throw std::runtime_error("Only support plain data format!");
    float _cmin = std::numeric_limits<float>::max();
    float _cmax = std::numeric_limits<float>::min();
    matCRef.resize(M * LDC);
    auto tmpsrcscale = alpha * matA.scales[0] * matB.scales[0];
#pragma omp parallel for
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < N; i += 1) {
        int tmp = 0;
        for (int ik = 0; ik < K; ik++) {
          tmp += matA.data()[ik + j * LDA] * matB.data()[ik * LDB + i];
        }
        auto ftmp = tmp * tmpsrcscale;
        ftmp = ftmp + (matD.data()[i + j * LDD] - matD.zeropoints[0]) * matD.scales[0] * beta;
        matCRef[i + j * LDC] = ftmp;
        _cmin = ftmp < _cmin ? ftmp : _cmin;
        _cmax = ftmp > _cmax ? ftmp : _cmax;
      }
    }
    matC.scales.resize(1);
    matC.zeropoints.resize(1);
    matC.scales[0] = (_cmax - _cmin) / (255.f);
    matC.zeropoints[0] = int((0 - _cmin) / matC.scales[0]);
    auto tmpscale = 1.f / matC.scales[0];
#pragma omp parallel for
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < N; i += 1) {
        matC.data()[j * LDC + i] = utils::cast<float, uint8_t>(matCRef[j * LDC + i] * tmpscale + matC.zeropoints[0]);
      }
    }
    matCDequan.resize(matCRef.size());
#pragma omp parallel for
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < N; i += 1) {
        matCDequan.data()[j * LDC + i] = ((int)matC.data()[j * LDC + i] - matC.zeropoints[0]) * matC.scales[0];
      }
    }

    /*utils::ut::buffer_error(matCRef.data(), matCDequan.data(),
                                                    matCDequan.size(),
       matC.scales[0]);*/
  }

  UT_vector_u8 matA, matC, matD;
  utils::aligned_vector<float> matCRef, matCDequan;
  UT_vector_s8 matB;
  int M, N, K, LDA, LDB, LDC, LDD;
  bool interleaved;
};

static inline void gemmref_u8s8s32(int m, int n, int k, uint8_t* A, int8_t* B, int32_t* C, int lda, int ldb, int ldc) {
#pragma omp parallel for
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i += 1) {
      int tmp = 0;
      for (int ik = 0; ik < k; ik++) {
        tmp += int(A[ik + j * lda]) * int(B[ik * ldb + i]);
      }
      C[i + j * ldc] = tmp;
    }
  }
}

static inline void gemmref_s8s8s32(int m, int n, int k, int8_t* A, int8_t* B, int32_t* C, int lda, int ldb, int ldc) {
#pragma omp parallel for
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i += 1) {
      int tmp = 0;
      for (int ik = 0; ik < k; ik++) {
        tmp += int(A[ik + j * lda]) * int(B[ik * ldb + i]);
      }
      C[i + j * ldc] = tmp;
    }
  }
}

static inline void kblockgemmref_u8zp_s8_f32(int m, int n, int k, int kblock, uint8_t* A, uint8_t* zpA, float* scaleA,
                                             int8_t* B, float* scaleB, float* C, int lda, int ldsa, int ldb, int ldsb,
                                             int ldc) {
  int kblk = utils::padto_le(k, kblock);
#pragma omp parallel for
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i += 1) {
      float tmp = 0.f;
      int ik = 0;
      for (; ik < kblk; ik += kblock) {
        int stmp = 0;
        for (int ikk = 0; ikk < kblock; ikk++) {
          stmp += (int(A[(ik + ikk) + j * lda]) - int(zpA[j * ldsa + ik / kblock])) * int(B[(ik + ikk) * ldb + i]);
        }
        tmp += stmp * scaleA[j * ldsa + ik / kblock] * scaleB[ik / kblock * ldsb + i];
      }
      if (ik < k) {
        int stmp = 0;
        for (; ik < k; ik++) {
          stmp += (int(A[ik + j * lda]) - int(zpA[j * ldsa + ik / kblock])) * int(B[ik * ldb + i]);
        }
        tmp += stmp * scaleA[j * ldsa + ik / kblock] * scaleB[ik / kblock * ldsb + i];
      }
      C[i + j * ldc] = tmp;
    }
  }
}

static inline void kblockgemmref_u8zp_s8_f32(int m, int n, int k, int kblock, uint8_t* A, uint8_t* zpA, float* scaleA,
                                             int8_t* B, utils::bf16* scaleB, float* C, int lda, int ldsa, int ldb,
                                             int ldsb, int ldc) {
#pragma omp parallel for
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i += 1) {
      float tmp = 0.f;
      for (int ik = 0; ik < k; ik += kblock) {
        int stmp = 0;
        for (int ikk = 0; ikk < kblock; ikk++) {
          stmp += (int(A[(ik + ikk) + j * lda]) - int(zpA[j * ldsa + ik / kblock])) * int(B[(ik + ikk) * ldb + i]);
        }
        tmp += stmp * scaleA[j * ldsa + ik / kblock] * scaleB[ik / kblock * ldsb + i].tofloat();
      }
      C[i + j * ldc] = tmp;
    }
  }
}

struct UT_GEMMData_Row_bf16 {
  UT_GEMMData_Row_bf16(int m, int n, int k, int lda, int ldb, int ldc, int ldd)
      : M(m), N(n), K(k), LDA(lda), LDB(ldb), LDC(ldc), LDD(ldd) {
    matA.resize(m * lda);
    matB.resize(k * ldb);
    matC.resize(m * ldc);
    if (ldd == 0) {
      matD.resize(n);
    } else {
      matD.resize(m * ldd);
    }
    for (size_t i = 0; i < matA.size(); i++) {
      matA[i] = utils::cast<float, utils::bf16>(ut::randn(-0.5f, 0.5f));
    }
    for (size_t i = 0; i < matB.size(); i++) {
      matB[i] = utils::cast<float, utils::bf16>(ut::randn(-0.5f, 0.5f));
    }
    for (size_t i = 0; i < matD.size(); i++) {
      matD[i] = utils::cast<float, utils::bf16>(ut::randn(1.f, 10.f));
    }
  }

  void calc_ref(float alpha, float beta) {
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        auto tmp = 0.f;
        for (int ik = 0; ik < K; ik++) {
          auto tmpA = utils::cast<utils::bf16, float>(utils::bf16{matA.data()[i * LDA + ik]});
          auto tmpB = utils::cast<utils::bf16, float>(utils::bf16{matB.data()[ik * LDB + j]});
          tmp += tmpA * tmpB;
        }
        tmp = tmp * alpha + utils::cast<utils::bf16, float>({matD.data()[i * LDD + j]}) * beta;
        matC.data()[i * LDC + j] = utils::cast<float, utils::bf16>(tmp);
      }
    }
  }
  utils::aligned_vector<utils::bf16> matA, matC, matD;
  utils::aligned_vector<utils::bf16> matB;
  int M, N, K, LDA, LDB, LDC, LDD;
};

static inline void gemmref_fp32fp32fp32(int m, int n, int k, float* A, float* B, float* C, int lda, int ldb, int ldc,
                                        float* bias = nullptr, bool BT = false) {
#pragma omp parallel for
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i += 1) {
      float tmp = bias ? bias[i] : 0.f;
      if (BT) {
        for (int ik = 0; ik < k; ik++) {
          tmp += A[ik + j * lda] * B[ik + i * ldb];
        }
      } else {
        for (int ik = 0; ik < k; ik++) {
          tmp += A[ik + j * lda] * B[ik * ldb + i];
        }
      }

      C[i + j * ldc] = tmp;
    }
  }
}

static inline void gemmref_bf16bf16bf16(int m, int n, int k, utils::bf16* A, utils::bf16* B, utils::bf16* C, int lda,
                                        int ldb, int ldc, utils::bf16* bias = nullptr, bool BT = false) {
#pragma omp parallel for
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i += 1) {
      float tmp = bias ? float(bias[i]) : 0.f;
      if (BT) {
        for (int ik = 0; ik < k; ik++) {
          tmp += float(A[ik + j * lda]) * float(B[ik + i * ldb]);
        }
      } else {
        for (int ik = 0; ik < k; ik++) {
          tmp += float(A[ik + j * lda]) * float(B[ik * ldb + i]);
        }
      }
      C[i + j * ldc] = tmp;
    }
  }
}
static inline void gemmref_bf16bf16fp32(int m, int n, int k, utils::bf16* A, utils::bf16* B, float* C, int lda, int ldb,
                                        int ldc) {
#pragma omp parallel for
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i += 1) {
      float tmp = 0;
      for (int ik = 0; ik < k; ik++) {
        tmp += float(A[ik + j * lda]) * float(B[ik * ldb + i]);
      }
      C[i + j * ldc] = tmp;
    }
  }
}

static inline void gemmref_fp16fp16fp16(int m, int n, int k, utils::fp16* A, utils::fp16* B, utils::fp16* C, int lda,
                                        int ldb, int ldc, utils::fp16* bias = nullptr, bool BT = false) {
#pragma omp parallel for
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i += 1) {
      float tmp = bias ? float(bias[i]) : 0.f;
      if (BT) {
        for (int ik = 0; ik < k; ik++) {
          tmp += float(A[ik + j * lda]) * float(B[ik + i * ldb]);
        }
      } else {
        for (int ik = 0; ik < k; ik++) {
          tmp += float(A[ik + j * lda]) * float(B[ik * ldb + i]);
        }
      }

      C[i + j * ldc] = tmp;
    }
  }
}

static inline void gemmref_fp16fp16fp32(int m, int n, int k, utils::fp16* A, utils::fp16* B, float* C, int lda, int ldb,
                                        int ldc) {
#pragma omp parallel for
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i += 1) {
      float tmp = 0;
      for (int ik = 0; ik < k; ik++) {
        tmp += float(A[ik + j * lda]) * float(B[ik * ldb + i]);
      }
      C[i + j * ldc] = tmp;
    }
  }
}

struct UT_GEMMData_Row_fp16 {
  utils::aligned_vector<utils::fp16> matA, matB, matC, matD;
  int M, N, K, LDA, LDB, LDC, LDD;
  UT_GEMMData_Row_fp16(int m, int n, int k, int lda, int ldb, int ldc, int ldd)
      : M(m), N(n), K(k), LDA(lda), LDB(ldb), LDC(ldc), LDD(ldd) {
    matA.resize(m * lda);
    matB.resize(k * ldb);
    matC.resize(m * ldc);
    if (ldd == 0) {
      matD.resize(n);
    } else {
      matD.resize(m * ldd);
    }
    ut::fill_buffer_randn(matA.data(), m * lda, utils::fp16(-0.5f), utils::fp16(0.5f));
    ut::fill_buffer_randn(matB.data(), k * ldb, utils::fp16(-0.5f), utils::fp16(0.5f));
    ut::fill_buffer_randn(matD.data(), matD.size(), utils::fp16(0.f), utils::fp16(1.f));
  }

  void calc_ref(float alpha, float beta) {
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        utils::fp16 tmp = utils::fp16(0.f);
        for (int ik = 0; ik < K; ik++) {
          auto tmpA = utils::cast<utils::fp16, float>(matA.data()[i * LDA + ik]);
          auto tmpB = utils::cast<utils::fp16, float>(matB.data()[ik * LDB + j]);
          auto ret = utils::fp16(tmpA * tmpB);
          tmp = utils::fp16(float(tmp) + float(ret));
        }
        float ftmp = float(tmp);
        ftmp = ftmp * alpha + utils::cast<utils::fp16, float>(matD.data()[i * LDD + j]) * beta;
        matC.data()[i * LDC + j] = utils::cast<float, utils::fp16>(ftmp);
      }
    }
  }
};

struct UT_GEMMData_Row_f32 {
  utils::aligned_vector<float> matA, matB, matC, matD, matRef;
  int M, N, K, LDA, LDB, LDC, LDD;
  UT_GEMMData_Row_f32(int m, int n, int k, int lda, int ldb, int ldc, int ldd)
      : M(m), N(n), K(k), LDA(lda), LDB(ldb), LDC(ldc), LDD(ldd) {
    matA.resize(m * lda);
    matB.resize(k * ldb);
    matC.resize(m * ldc);
    if (ldd == 0) {
      matD.resize(n);
    } else {
      matD.resize(m * ldd);
    }
    ut::fill_buffer_randn(matA.data(), m * lda, -0.5f, 0.5f);
    ut::fill_buffer_randn(matB.data(), k * ldb, -0.5f, 0.5f);
    ut::fill_buffer_randn(matD.data(), matD.size(), 0.f, 1.f);
  }

  void calc_ref(float alpha, float beta) {
    matRef.resize(matC.size());
    ref_NN_f32(matA.data(), matB.data(), matRef.data(), matD.data(), M, N, K, LDA, LDB, LDC, LDD, alpha, beta);
  }
  static void ref_NN_f32(float* matA, float* matB, float* matC, float* matD, int m, int n, int k, int lda, int ldb,
                         int ldc, int ldd, float alpha, float beta) {
    int NBlock = 128;
#if 1
#pragma omp parallel for
    for (int i = 0; i < n; i += NBlock) {
      for (int j = 0; j < m; j++) {
        int remainn = i + NBlock <= n ? NBlock : n - i;
        for (int ii = 0; ii < remainn; ii++) {
          auto tmp = 0.f;
          for (int ik = 0; ik < k; ik++) {
            tmp += matA[ik + j * lda] * matB[ik * ldb + i + ii];
          }
          tmp = tmp * alpha + matD[(i + ii) + j * ldd] * beta;
          matC[(i + ii) + j * ldc] = tmp;
        }
      }
    }
#else
#pragma omp parallel for
    for (int i = 0; i < n; i += 1) {
      for (int j = 0; j < m; j++) {
        auto tmp = 0.f;
        for (int ik = 0; ik < k; ik++) {
          tmp += matA[ik + j * lda] * matB[ik * ldb + i];
        }
        tmp = tmp * alpha + matD[i + j * ldd] * beta;
        matC[i + j * ldc] = tmp;
      }
    }
#endif
  }
};

static inline void sdpa_ref(int head_num, int head_kv_num, int seq_cur, int seq_kv, int head_idim, int head_odim,
                            float scale, float* q_p, float* k_p, float* v_p, float* out, float* mask,
                            bool debug = false) {
  auto n_repeat = head_num / head_kv_num;
  // batch matmul host_Q and trans_K
  std::vector<float> qk_tmp(head_num * seq_cur * seq_kv);
  for (size_t i = 0; i < head_num; i++) {
    int i_kv = i / n_repeat;
    for (size_t j = 0; j < seq_cur; j++) {
      for (size_t k = 0; k < seq_kv; k++) {
        float sum = 0;
        for (size_t l = 0; l < head_idim; l++) {
          sum += q_p[i * head_idim + j * head_num * head_idim + l] *
                 k_p[i_kv * head_idim + k * head_kv_num * head_idim + l];
        }
        qk_tmp[i * seq_cur * seq_kv + j * seq_kv + k] =
            debug ? sum : sum * scale + mask[j * seq_kv + k];  // mul sqrt_dk
      }
    }
  }
  if (!debug) {
    // batch softmax host_tmp
    for (size_t i = 0; i < head_num; i++) {
      for (size_t j = 0; j < seq_cur; j++) {
        float sum = 0;
        float max = std::numeric_limits<float>::min();
        for (size_t k = 0; k < seq_kv; k++)
          max =
              qk_tmp[i * seq_cur * seq_kv + j * seq_kv + k] > max ? qk_tmp[i * seq_cur * seq_kv + j * seq_kv + k] : max;
        for (size_t k = 0; k < seq_kv; k++) {
          qk_tmp[i * seq_cur * seq_kv + j * seq_kv + k] = std::exp(qk_tmp[i * seq_cur * seq_kv + j * seq_kv + k] - max);
          sum += qk_tmp[i * seq_cur * seq_kv + j * seq_kv + k];
        }
        sum = 1.f / sum;
        for (size_t k = 0; k < seq_kv; k++) {
          qk_tmp[i * seq_cur * seq_kv + j * seq_kv + k] = qk_tmp[i * seq_cur * seq_kv + j * seq_kv + k] * sum;
        }
      }
    }
  }

  // batch matmul host_tmp and host_v
  for (size_t i = 0; i < head_num; i++) {
    auto i_kv = i / n_repeat;
    for (size_t j = 0; j < seq_cur; j++) {
      for (size_t k = 0; k < head_odim; k++) {
        float sum = 0;
        for (size_t l = 0; l < seq_kv; l++) {
          sum +=
              qk_tmp[i * seq_cur * seq_kv + j * seq_kv + l] * v_p[i_kv * head_odim + l * head_odim * head_kv_num + k];
        }
        out[i * head_odim + j * head_num * head_odim + k] = sum;
      }
    }
  }
}

};  // namespace ut
}  // namespace bestla
