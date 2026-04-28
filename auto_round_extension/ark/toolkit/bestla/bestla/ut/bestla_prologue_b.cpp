#include "bestla_gemm.h"
#include "bestla_gemm_kblock.h"
#include "bestla_prologue_b.h"
#include "bestla_parallel.h"
#include "bestla_device.h"
#include "bestla_wrapper.h"
#include "bestla_ut.h"

namespace bestla {
using namespace utils;
namespace ut {
class UT_BlockQunatize_INT8 {
 public:
  UT_BlockQunatize_INT8() {
    UT_START();
    CheckISA(AVX512F);
    ut(1024, 1024, 32);
    ut(1024, 1024, 32, true);
    ut(4128, 4096, 32);
    ut(4128, 4096, 32, true);
    ut(1024, 4096, 32);
    ut(4096, 1024, 32);

    ut_transpose(4096, 4096, 32);
    ut_transpose(4096, 4096, 32, true);
    ut_transpose(4128, 4096, 32);
    ut_transpose(4128, 4096, 32, true);
    ut_transpose(1024, 4096, 32);
    ut_transpose(4096, 1024, 32);
  }

  void ut(int n, int k, int blocksize, bool asym = false) {
    printf("%s: %d %d %d %s\n", __FUNCTION__, n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    utils::aligned_vector<float> dequanRef(n * k);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.003f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          quanW.data()[i * n + j] = 127;  // make sure each block has maximum value to quantize
        }
        if (i % blocksize == 1 && asym) {
          quanW.data()[i * n + j] = -128;  // make sure each block has minimum value to quantize if asym
        }
      }
    }

    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        if (asym) {
          dequanRef[j * ldb + i] = (float(quanW.data()[j * ldb + i]) - float(zero_points[j / blocksize * n + i])) *
                                   scales[j / blocksize * n + i];
        } else {
          dequanRef[j * ldb + i] = float(quanW.data()[j * ldb + i]) * scales[j / blocksize * n + i];
        }
      }
    }

    auto constexpr RuntimeISA = BTLA_ISA::AVX512F;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<gemm::SCoreRowNAvx512f<48, 8>>;
    auto ptr = PrologueB::createStorage(n, k, blocksize, BTLA_DTYPE::S8, bestla_dtype<float>, asym);
    avector<int8_t> buffer(ptr.mSize);
    ptr.assign(buffer.data());
    PrologueB::packWeight(n, k, dequanRef.data(), ldb, &ptr, UT_Threading::get());
    avector<float> dequant(n * k);
    PrologueB::unpackWeight(n, k, &ptr, dequant.data(), n, UT_Threading::get());
    ut::buffer_error(dequanRef.data(), dequant.data(), dequanRef.size(), 0.01f);
  }

  void ut_transpose(int n, int k, int blocksize, bool asym = false) {
    printf("%s: %d %d %d %s\n", __FUNCTION__, n, k, blocksize, asym ? "asym" : "sym");
    utils::aligned_vector<float> dequanRef(n * k);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.003f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          quanW.data()[i * n + j] = 127;  // make sure each block has maximum value to quantize
        }
        if (i % blocksize == 1 && asym) {
          quanW.data()[i * n + j] = -128;  // make sure each block has minimum value to quantize if asym
        }
      }
    }

    avector<float> dequanT(k * n);
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        if (asym) {
          dequanRef[j * n + i] = (float(quanW.data()[j * n + i]) - float(zero_points[j / blocksize * n + i])) *
                                 scales[j / blocksize * n + i];
        } else {
          dequanRef[j * n + i] = float(quanW.data()[j * n + i]) * scales[j / blocksize * n + i];
        }
        dequanT[j + i * k] = dequanRef[j * n + i];
      }
    }

    auto constexpr RuntimeISA = BTLA_ISA::AVX512F;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<gemm::SCoreRowNAvx512f<48, 8>>;
    auto ptr = PrologueB::createStorage(n, k, blocksize, BTLA_DTYPE::S8, bestla_dtype<float>, asym);
    avector<int8_t> buffer(ptr.mSize);
    ptr.assign(buffer.data());
    PrologueB::packTransposeWeight(n, k, dequanT.data(), k, &ptr, UT_Threading::get());
    avector<float> dequant(n * k), tardequanT(k * n);
    PrologueB::unpackWeight(n, k, &ptr, dequant.data(), n, UT_Threading::get());
    PrologueB::unpackTransposeWeight(n, k, &ptr, tardequanT.data(), k, UT_Threading::get());
    ut::buffer_error(dequanT.data(), tardequanT.data(), tardequanT.size(), 0.01f);
    ut::buffer_error(dequanRef.data(), dequant.data(), dequanRef.size(), 0.01f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_BlockQunatize_INT8 sUT_BlockQunatize_INT8;
#endif

class UT_BlockQunatize_SN {
 public:
  UT_BlockQunatize_SN() {
    UT_START();
    CheckISA(AVX2);
    ut<sAVX2>(4096, 4096, 32, BTLA_DTYPE::S1_CLIP, true);
    ut<sAVX2>(4096, 4096, 32, BTLA_DTYPE::S1_CLIP);
    ut<sAVX2>(4096, 4096, 128, BTLA_DTYPE::S1_CLIP);
    ut<sAVX2>(4096, 4096, 32, BTLA_DTYPE::S7_CLIP, true);
    ut<sAVX2>(4096, 4096, 32, BTLA_DTYPE::S7_CLIP);
    ut<sAVX2>(4096, 4096, 128, BTLA_DTYPE::S7_CLIP);
    ut<sAVX2>(4096, 4096, 32, BTLA_DTYPE::S6_CLIP, true);
    ut<sAVX2>(4096, 4096, 32, BTLA_DTYPE::S6_CLIP);
    ut<sAVX2>(4096, 4096, 128, BTLA_DTYPE::S6_CLIP);
    ut<sAVX2>(4096, 4096, 32, BTLA_DTYPE::S5_CLIP, true);
    ut<sAVX2>(4096, 4096, 32, BTLA_DTYPE::S5_CLIP);
    ut<sAVX2>(4096, 4096, 128, BTLA_DTYPE::S5_CLIP);
    ut<sAVX2>(4096, 4096, 32, BTLA_DTYPE::S4_CLIP, true);
    ut<sAVX2>(4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAVX2>(4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    ut<sAVX2>(127, 4096, 32, BTLA_DTYPE::S2_CLIP);
    ut<sAVX2>(127, 4096, 32, BTLA_DTYPE::S2_CLIP, true);
    ut<sAVX2>(4096, 4096, 32, BTLA_DTYPE::S2_CLIP);
    ut<sAVX2>(4096, 4096, 128, BTLA_DTYPE::S2_CLIP);
    ut<sAVX2>(127, 4096, 32, BTLA_DTYPE::S3_CLIP);
    ut<sAVX2>(127, 4096, 32, BTLA_DTYPE::S3_CLIP, true);
    ut<sAVX2>(4096, 4096, 32, BTLA_DTYPE::S3_CLIP);
    ut<sAVX2>(127, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAVX2>(127, 4096, 32, BTLA_DTYPE::S4_CLIP, true);

    CheckISA(AVX512F);
    ut<sAVX512F>(127, 4096, 32, BTLA_DTYPE::S3_CLIP);
    ut<sAVX512F>(4096, 4096, 32, BTLA_DTYPE::S3_CLIP);
    ut<sAVX512F>(4096, 4096, 128, BTLA_DTYPE::S3_CLIP);
    ut<sAVX512F>(127, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAVX512F>(4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAVX512F>(4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
  }
  template <class GemmCore>
  void ut(int n, int k, int blocksize, BTLA_DTYPE QUANT_T, bool isAsym = false) {
    auto constexpr RuntimeISA = GemmCore::ISA;
    printf("%s DType %s %d: %d %d %d Asym:%d\n", __FUNCTION__, utils::bestla_dtype_str(QUANT_T), int(RuntimeISA), n, k,
           blocksize, isAsym);
    int ldb = n;
    utils::aligned_vector<float> raw(n * k);
    ut::fill_buffer_randn(raw.data(), raw.size(), -0.5f, 0.5f);
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<GemmCore>;
    auto ptr = PrologueB::createStorage(n, k, blocksize, QUANT_T, BTLA_DTYPE::F32, isAsym);
    avector<int8_t> buffer(ptr.mSize);
    ptr.assign(buffer.data());
    PrologueB::packWeight(n, k, raw.data(), ldb, &ptr, UT_Threading::get());
    avector<float> dequant(n * k, 0);
    PrologueB::unpackWeight(n, k, &ptr, dequant.data(), n, UT_Threading::get());
    ut::buffer_error(raw.data(), dequant.data(), dequant.size(), 0.01f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
// no proper threshold for this UT
// static UT_BlockQunatize_SN sUT_BlockQunatize_SN;
#endif

class UT_TransposeBlockQuantize_F4 {
 public:
  UT_TransposeBlockQuantize_F4() {
    UT_START();
    CheckISA(AVX512F);
    ut(4096, 4096, 32, BTLA_DTYPE::F4_BNB, BTLA_DTYPE::F32);
    ut(1024, 4096, 32, BTLA_DTYPE::F4_BNB, BTLA_DTYPE::F32);
    ut(4096, 1024, 32, BTLA_DTYPE::F4_BNB, BTLA_DTYPE::F32);
    ut(48, 32, 32, BTLA_DTYPE::F4_BNB, BTLA_DTYPE::F32);
    ut(32, 32, 32, BTLA_DTYPE::F4_BNB, BTLA_DTYPE::F32);
    ut(48, 32, 32, BTLA_DTYPE::F4_BNB, BTLA_DTYPE::F32);
    ut(48, 32, 32, BTLA_DTYPE::F4_NF4, BTLA_DTYPE::F32);
    ut(48, 32, 32, BTLA_DTYPE::F4_E2M1, BTLA_DTYPE::F32);
    ut(16, 15, 8, BTLA_DTYPE::F4_NF4, BTLA_DTYPE::DQ8_BNB);
    ut(48, 32, 16, BTLA_DTYPE::F4_NF4, BTLA_DTYPE::DQ8_BNB);
    ut(1024, 4096, 32, BTLA_DTYPE::F4_NF4, BTLA_DTYPE::DQ8_BNB);
  }

  void ut(int n, int k, int blocksize, BTLA_DTYPE F4_T, BTLA_DTYPE SCA_T) {
    printf("Test Case: %d %d %d\n", n, k, blocksize);
    int ldb = n;
    utils::aligned_vector<float> dequanRef(n * k);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 1.f, 5.f);
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(0, 16);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          switch (F4_T) {
            case BTLA_DTYPE::F4_E2M1:
              quanW.data()[i * n + j] = 7;  // make sure each block has maximum fp4e2m1 value(0b111) to quantize
              break;
            case BTLA_DTYPE::F4_BNB:
              quanW.data()[i * n + j] = 3;  // make sure each block has maximum fp4bnb value(0b011) to quantize
              break;
            case BTLA_DTYPE::F4_NF4:
              quanW.data()[i * n + j] = 15;  // make sure each block has maximum nf4 value(0b1111) to quantize
              break;
            default:
              break;
          }
        }
      }
    }
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        switch (F4_T) {
          case BTLA_DTYPE::F4_E2M1:
            dequanRef[j + i * k] = kernel::ref::f4_dequantize<BTLA_DTYPE::F4_E2M1>(quanW.data()[j * ldb + i],
                                                                                   scales[j / blocksize * n + i]);
            quanW.data()[j * ldb + i] =
                kernel::ref::f4_quantize<BTLA_DTYPE::F4_E2M1>(dequanRef[j + i * k] / scales[j / blocksize * n + i]);
            break;
          case BTLA_DTYPE::F4_BNB:
            dequanRef[j + i * k] = kernel::ref::f4_dequantize<BTLA_DTYPE::F4_BNB>(quanW.data()[j * ldb + i],
                                                                                  scales[j / blocksize * n + i]);
            quanW.data()[j * ldb + i] =
                kernel::ref::f4_quantize<BTLA_DTYPE::F4_BNB>(dequanRef[j + i * k] / scales[j / blocksize * n + i]);
            break;
          case BTLA_DTYPE::F4_NF4:
            dequanRef[j + i * k] = kernel::ref::f4_dequantize<BTLA_DTYPE::F4_NF4>(quanW.data()[j * ldb + i],
                                                                                  scales[j / blocksize * n + i]);
            quanW.data()[j * ldb + i] =
                kernel::ref::f4_quantize<BTLA_DTYPE::F4_NF4>(dequanRef[j + i * k] / scales[j / blocksize * n + i]);
            break;
          default:
            break;
        }
      }
    }

    auto constexpr RuntimeISA = BTLA_ISA::AVX512F;
    using PrologueB = prologue_b::gemm::WeightKBlockNFloat<gemm::SCoreRowNAvx512f<48, 8>>;
    auto packedW = PrologueB::createStorage(n, k, blocksize, F4_T, SCA_T);
    auto packedW1 = PrologueB::createStorage(n, k, blocksize, F4_T, SCA_T);
    avector<int8_t> buf(packedW.mSize), buf1(packedW1.mSize);
    packedW.assign(buf.data());
    packedW1.assign(buf1.data());
    PrologueB::packTransposeWeight(n, k, dequanRef.data(), k, &packedW, UT_Threading::get());
    PrologueB::packQWeight(n, k, quanW.data(), ldb, scales.data(), nullptr, &packedW1, UT_Threading::get());
    avector<float> dequant(n * k);
    PrologueB::unpackTransposeWeight(n, k, &packedW1, dequant.data(), k, UT_Threading::get());
    if (SCA_T != BTLA_DTYPE::DQ8_BNB) {
      ut::buffer_error(packedW.SPtr<float>(), packedW1.SPtr<float>(), packedW1.CSize());
      ut::buffer_error(dequanRef.data(), dequant.data(), dequant.size());
    } else {
      ut::buffer_error(packedW.SPtr<int8_t>(), packedW1.SPtr<int8_t>(), packedW1.CSize());
      ut::buffer_error(dequanRef.data(), dequant.data(), dequant.size(), 0.1f);
    }
    ut::buffer_error(packedW.WPtr<int8_t>(), packedW1.WPtr<int8_t>(), packedW1.mQBuf.size<int8_t>());
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_TransposeBlockQuantize_F4 sUT_TransposeBlockQuantize_F4;
#endif

class UT_StorageMemCheck {
 public:
  UT_StorageMemCheck() {
    UT_START();
    ut_npack(1024, 768);
    ut_s4(4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    ut_f4(4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut_f4(4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
  }
  void ut_npack(int n, int k) {
    printf("Test C type Case: %d %d\n", n, k);
    int ldb = n;
    using GemmCore = gemm::SCoreRowNAvx512f<48, 8>;
    using PrologueB = prologue_b::gemm::WeightNPack<GemmCore>;

    auto stor = PrologueB::create_storage(n, k, storage::gemm::Layout::T);
    std::vector<float> src(n * k);
    fill_buffer_randn(src.data(), src.size(), -0.5f, 0.5f);
    stor.set_buffers((int8_t*)src.data());

    std::vector<int8_t> misc_mem(stor.misc_size_ + stor.buf_size_);

    stor.serialize(misc_mem.data());
    stor.copy_buffers(misc_mem.data() + stor.misc_size_);

    auto ptr = storage::CollectionParser::parse_buffer(misc_mem.data());
    if (ptr) {
      if (ptr->prologue_id_ == BTLA_PROLOGUEB_IDS::WeightPack) {
        auto wptr = (storage::gemm::StorageWeight*)ptr;
        wptr->set_buffers(misc_mem.data() + wptr->misc_size_);
        buffer_error(wptr->ptr(), (int8_t*)src.data(), wptr->buf_size_);
      }
      delete ptr;
    }
  }

  void ut_s4(int n, int k, int blocksize, BTLA_DTYPE qtype, bool asym = false) {
    printf("Test C type Case: %d %d %d %s\n", n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    using GemmCore = gemm::SCoreRowNAvx512f<48, 8>;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<GemmCore>;

    auto packedW = PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<float>, asym);
    avector<int8_t> buf0(packedW.mSize), buf1(packedW.mSize);
    packedW.assign(buf0.data());
    storage::gemm::StorageWeightKBlockNInteger tmp(GemmCore::ID);
    tmp.deserialize(buf0.data());
    tmp.serialize(buf1.data());
    buffer_error(buf0.data(), buf1.data(), buf0.size());
  }

  void ut_f4(int n, int k, int blocksize, BTLA_DTYPE qtype) {
    printf("Test C type Case: %d %d %d\n", n, k, blocksize);
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    using GemmCore = gemm::HCoreRowNAmxbf16<64, 16>;
    using PrologueB = prologue_b::gemm::WeightKBlockNFloat<GemmCore>;

    auto packedW = PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<float>);
    avector<int8_t> buf0(packedW.mSize), buf1(packedW.mSize);
    packedW.assign(buf0.data());
    storage::gemm::StorageWeightKBlockNFloat tmp(GemmCore::ID);
    tmp.deserialize(buf0.data());
    tmp.serialize(buf1.data());
    buffer_error(buf0.data(), buf1.data(), buf0.size());
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_StorageMemCheck sUT_StorageMemCheck;
#endif

class UT_WeightNPack {
 public:
  UT_WeightNPack() {
    UT_START();
    ut_npack<gemm::SCoreRowNAvx2<24, 4>>(1024, 768);
  }
  template <class _Gc>
  void ut_npack(int n, int k) {
    printf("Test C type Case: %d %d\n", n, k);
    int ldb = n;
    using PrologueB = prologue_b::gemm::WeightNPack<_Gc>;
    using T = typename _Gc::BType;

    std::vector<T> src(n * k), vdst(k * n);
    fill_buffer_randn(src.data(), src.size(), T(-0.5f), T(0.5f));

    storage::gemm::StorageWeight raw;
    raw.set_plain_data(n, k, k, utils::bestla_dtype<T>, src.data());
    auto stor = PrologueB::create_storage(n, k, storage::gemm::Layout::NPack);
    std::vector<int8_t> tmp(stor.buf_size_);
    stor.set_buffers(tmp.data());
    PrologueB::convert(raw, stor, UT_Threading::get());

    storage::gemm::StorageWeight tar = raw;
    tar.set_buffers((int8_t*)vdst.data());
    PrologueB::convert(stor, tar, UT_Threading::get());
    buffer_error(src.data(), vdst.data(), src.size());
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_WeightNPack sUT_WeightNPack;
#endif

class UT_WeightNInt {
 public:
  UT_WeightNInt() {
    UT_START();
    ut_qdq<gemm::SCoreRowNAvx2<24, 4>>(1024, 768, 32, BTLA_DTYPE::S4_CLIP, false);
    ut_qdq<gemm::SCoreRowNAvx2<24, 4>>(1024, 768, 32, BTLA_DTYPE::S4_CLIP, true);
    ut_qdq<gemm::SCoreRowNAvx2<24, 4>>(1024, 768, -1, BTLA_DTYPE::S8, false);
    ut_convert<gemm::SCoreRowNAvx2<24, 4>>(1024, 768, 32, BTLA_DTYPE::S4_CLIP, false);
  }
  template <class _Gc>
  void ut_qdq(int n, int k, int block, BTLA_DTYPE qdtype, bool is_asym) {
    printf("Test C type Case: %d %d\n", n, k);
    int ldb = n;
    using PrologueB = prologue_b::gemm::WeightKBlockNInt<_Gc>;
    using T = typename _Gc::BType;

    std::vector<T> src(n * k), vdst(k * n);
    fill_buffer_randn(src.data(), src.size(), T(-0.5f), T(0.5f));

    storage::gemm::StorageWeightNInt raw, deq;
    raw.set_plain_data(n, k, k, utils::bestla_dtype<T>, src.data());
    deq.set_plain_data(n, k, k, utils::bestla_dtype<T>, vdst.data());
    auto stor = PrologueB::create_storage(n, k, block, qdtype, BTLA_DTYPE::F32, is_asym);
    std::vector<int8_t> tmp(stor.buf_size_);
    stor.set_buffers(tmp.data());
    PrologueB::convert(raw, stor, UT_Threading::get());
    PrologueB::convert(stor, deq, UT_Threading::get());
    buffer_error(src.data(), vdst.data(), src.size(), 0.1f);
  }

  template <class _Gc>
  void ut_convert(int n, int k, int block, BTLA_DTYPE qdtype, bool is_asym) {
    printf("Test C type Case: %d %d\n", n, k);
    int ldb = n;
    using PrologueB = prologue_b::gemm::WeightKBlockNInt<_Gc>;
    using T = typename _Gc::BType;

    std::vector<T> src(n * k), vdst(k * n);
    fill_buffer_randn(src.data(), src.size(), T(-0.5f), T(0.5f));
    std::vector<T> srcNT(k * n);
    prologue_b::gemm::transpose(n, k, (int8_t*)src.data(), utils::bestla_dtype<T>, k, (int8_t*)srcNT.data(), n,
                                UT_Threading::get());
    storage::gemm::StorageWeightNInt raw, t;
    raw.set_plain_data(n, k, k, utils::bestla_dtype<T>, src.data());
    auto stor = PrologueB::create_storage(n, k, block, qdtype, BTLA_DTYPE::F32, is_asym);
    std::vector<int8_t> tmp(stor.buf_size_);
    stor.set_buffers(tmp.data());
    t.t_from(stor);
    std::vector<int8_t> tmp2(t.buf_size_);
    t.set_buffers(tmp2.data());
    PrologueB::convert(raw, stor, UT_Threading::get());
    PrologueB::convert(stor, t, UT_Threading::get());
    storage::gemm::StorageWeightNInt stor2 = stor;
    std::vector<int8_t> tmp3(stor2.buf_size_);
    stor2.set_buffers(tmp3.data());
    PrologueB::convert(t, stor2, UT_Threading::get());
    std::vector<int8_t> upk0(n * k), upk1(k * n);
    PrologueB::template unpack_weight<int8_t>(n, k, stor, upk0.data(), n, UT_Threading::get());
    PrologueB::decompress_weight(n, k, t, upk1.data(), n, UT_Threading::get());
    buffer_error(tmp.data(), tmp3.data(), tmp.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_WeightNInt sUT_WeightNInt;
#endif

class UT_ShuffleIndices {
 public:
  UT_ShuffleIndices() {
    UT_START();
    CheckISA(AVX2);
    // ut_file();
    ut_s4(4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
  }

  void ut_s4(int n, int k, int blocksize, BTLA_DTYPE qtype, bool asym = false) {
    printf("Test C type Case: %d %d %d %s\n", n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    using GemmCore = gemm::SCoreRowNAvx2<24, 4>;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<GemmCore>;
    auto packedW = PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<float>, asym);
    PrologueB::enableShuffle(&packedW);
    avector<int> groupindices(k, 0);
    auto groupsize = utils::updiv(k, blocksize);
    avector<int> reflut(k, 0);
    for (size_t i = 0; i < k; i++) {
      groupindices[i] = i % groupsize;
      auto offset = i / groupsize;
      reflut[groupindices[i] * blocksize + offset] = static_cast<int>(i);
    }
    avector<int8_t> buf0(packedW.mSize), buf1(packedW.mSize);
    packedW.assign(buf0.data());
    PrologueB::setShuffleIndices(groupindices.data(), &packedW, UT_Threading::get());
    buffer_error(reflut.data(), packedW.ShfIndice(), reflut.size());

    storage::gemm::StorageWeightKBlockNInteger tmp(GemmCore::ID);
    tmp.deserialize(buf0.data());
    tmp.serialize(buf1.data());
    buffer_error(buf0.data(), buf1.data(), buf0.size());
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_ShuffleIndices sUT_ShuffleIndices;
#endif

class UT_CompFp32 {
 public:
  UT_CompFp32() {
    UT_START();
    ut_int_new<sAVX2>(1, 4096, 4096, 16, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F16, true);
    ut_int_new<sAVX512F>(1, 4096, 4096, 16, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F16, true);
    ut_s4_full();
    ut_new_type(BTLA_DTYPE::S1_CLIP);
    ut_new_type(BTLA_DTYPE::S2_CLIP);
    ut_new_type(BTLA_DTYPE::S3_CLIP);
    ut_new_type(BTLA_DTYPE::S5_CLIP);
    ut_new_type(BTLA_DTYPE::S6_CLIP);
    ut_new_type(BTLA_DTYPE::S7_CLIP);
    ut_new_type(BTLA_DTYPE::S8);

    ut_f4();
    ut_f8();
  }

  void ut_new_type(BTLA_DTYPE qtype) {
    GetCPUDevice();
    if (_cd->AVX2()) {
      ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(1, 4096, 4096, 16, qtype, BTLA_DTYPE::F16, true);
      ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(1, 4096, 4096, 16, qtype, BTLA_DTYPE::BF16, true);
      ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(4, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
      ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F16, true);
      ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
      ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(8, 4096, 4096, -1, qtype, BTLA_DTYPE::BF16, false);
    }
    if (_cd->AVX512F()) {
      ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(1, 4096, 4096, 16, qtype, BTLA_DTYPE::BF16, true);
      ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(4, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
      ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
      ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(8, 4096, 4096, -1, qtype, BTLA_DTYPE::BF16, false);
    }
  }

  void ut_f8() {
    CheckISA(AVX2);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E4M3, BTLA_DTYPE::F8_E8M0);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E4M3, BTLA_DTYPE::F32);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E5M2, BTLA_DTYPE::F8_E8M0);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E5M2, BTLA_DTYPE::F32);
    CheckISA(AVX512F);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E4M3, BTLA_DTYPE::F8_E8M0);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E4M3, BTLA_DTYPE::F32);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E5M2, BTLA_DTYPE::F8_E8M0);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E5M2, BTLA_DTYPE::F32);
  }

  void ut_s4_full() {
    BTLA_DTYPE qtype = BTLA_DTYPE::S4_CLIP;
    CheckISA(AVX2);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(1, 4096, 4096, 16, qtype, BTLA_DTYPE::F16, true);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(1, 4096, 4096, 16, qtype, BTLA_DTYPE::BF16, true);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(4, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F16, true);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(8, 4096, 4096, -1, qtype, BTLA_DTYPE::BF16, false);

    CheckISA(AVX512F);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F16, false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, -1, qtype, BTLA_DTYPE::F32, false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(8, 4096, 4096, 32, qtype, BTLA_DTYPE::BF16, false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(8, 4096, 4096, 32, qtype, BTLA_DTYPE::F16, false);
  }

  void ut_f4() {
    CheckISA(AVX2);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB, BTLA_DTYPE::F32);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, -1, BTLA_DTYPE::F4_BNB, BTLA_DTYPE::F32);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1, BTLA_DTYPE::F32);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, -1, BTLA_DTYPE::F4_E2M1, BTLA_DTYPE::F32);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4, BTLA_DTYPE::F32);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, -1, BTLA_DTYPE::F4_NF4, BTLA_DTYPE::F32);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB, BTLA_DTYPE::BF16);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1, BTLA_DTYPE::BF16);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4, BTLA_DTYPE::BF16);

    CheckISA(AVX512F);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB, BTLA_DTYPE::F32);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, -1, BTLA_DTYPE::F4_BNB, BTLA_DTYPE::F32);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1, BTLA_DTYPE::F32);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, -1, BTLA_DTYPE::F4_E2M1, BTLA_DTYPE::F32);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4, BTLA_DTYPE::F32);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, -1, BTLA_DTYPE::F4_NF4, BTLA_DTYPE::F32);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB, BTLA_DTYPE::BF16);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1, BTLA_DTYPE::BF16);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4, BTLA_DTYPE::BF16);
  }

  template <class GemmCore_T>
  void ut_int_new(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, BTLA_DTYPE stype, bool isAsym) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s Asym:%d\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), bestla_dtype_str(stype), isAsym);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher =
        wrapper::gemm::LauncherBaseV2<GemmCore_T, prologue_a::gemm::ActivationKBlockBaseF32,
                                      prologue_b::gemm::WeightKBlockNInt, epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
    blocksize = blocksize == -1 ? k : blocksize;
    auto packedw = Launcher::PrologueB::create_storage(n, k, blocksize, qtype, stype, isAsym);
    utils::avector<int8_t> buffer(packedw.buf_size_);
    packedw.set_buffers(buffer.data());
    auto reduceA = Launcher::PrologueA::createStorage(m, k, blocksize);
    utils::avector<int8_t> bufferA(reduceA.mSize);
    reduceA.assign(bufferA.data());
    avector<float> matBf32(k * n), matAf32(m * k), match(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    fill_buffer_randn(matAf32.data(), matAf32.size(), -0.5f, 0.5f);
    storage::gemm::StorageWeightNInt rawB, dstB;
    rawB.set_nt_data(n, k, k, utils::bestla_dtype<float>, matBf32.data());
    Launcher::PrologueB::convert(rawB, packedw, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    Launcher::PrologueB::convert(packedw, rawB, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refCupk.data(), k, n, n);

    Launcher::PrologueA::reduce({matAf32.data(), k, &reduceA}, m, k, blocksize, UT_Threading::get());
    utils::GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp, {matAf32.data(), k, &reduceA}, {&packedw}, {match.data(), n}};
    parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
    auto err = get_ut_err(qtype);
    auto dbits = bestla_dtype_bits(qtype);
    auto type = bestla_dtype_type(qtype);
    auto constexpr dtype_int = bestla_dtype_type(BTLA_DTYPE::TypeInt);
    buffer_error(refC.data(), match.data(), refC.size(), err);
    buffer_error(refCupk.data(), match.data(), refCupk.size(), 0.001f);
  }

  template <class GemmCore_T, template <class _T> class Wei>
  void ut_int(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, BTLA_DTYPE stype, bool isAsym) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s Asym:%d\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), bestla_dtype_str(stype), isAsym);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher =
        wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ActivationKBlockBaseF32,
                                    prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;

    blocksize = blocksize == -1 ? k : blocksize;
    using WType = typename Launcher::PrologueB::StorageWeight;
    WType packedw(0);
    if constexpr (std::is_same_v<Wei<GemmCore_T>, prologue_b::gemm::WeightKBlockNInteger<GemmCore_T>>) {
      packedw = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, stype, isAsym);
    } else if constexpr (std::is_same_v<Wei<GemmCore_T>, prologue_b::gemm::WeightKBlockNFloat<GemmCore_T>>) {
      packedw = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, stype);
    }

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    auto reduceA = Launcher::PrologueA::createStorage(m, k, blocksize);
    utils::avector<int8_t> bufferA(packedw.mSize);
    reduceA.assign(bufferA.data());
    avector<float> matBf32(k * n), matAf32(m * k), match(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    fill_buffer_randn(matAf32.data(), matAf32.size(), -0.5f, 0.5f);
    Launcher::PrologueB::packWeight(n, k, matBf32.data(), n, &packedw, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    Launcher::PrologueB::unpackWeight(n, k, &packedw, matBf32.data(), n, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refCupk.data(), k, n, n);

    Launcher::PrologueA::reduce({matAf32.data(), k, &reduceA}, m, k, blocksize, UT_Threading::get());
    utils::GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp, {matAf32.data(), k, &reduceA}, {&packedw}, {match.data(), n}};
    parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
    auto err = get_ut_err(qtype);
    auto dbits = bestla_dtype_bits(qtype);
    auto type = bestla_dtype_type(qtype);
    auto constexpr dtype_int = bestla_dtype_type(BTLA_DTYPE::TypeInt);
    buffer_error(refC.data(), match.data(), refC.size(), err);
    buffer_error(refCupk.data(), match.data(), refCupk.size(), 0.001f);
  }

  template <class GemmCore_T, template <class _T> class Wei>
  void ut(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, BTLA_DTYPE stype) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), bestla_dtype_str(stype));
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher = wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ActivationBase, Wei,
                                                 epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;

    blocksize = blocksize == -1 ? k : blocksize;
    using WType = typename Wei<GemmCore_T>::StorageWeight;
    WType packedw(0);
    packedw = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, stype);
    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), match(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    fill_buffer_randn(matAf32.data(), matAf32.size(), -0.5f, 0.5f);
    Launcher::PrologueB::packWeight(n, k, matBf32.data(), n, &packedw, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    Launcher::PrologueB::unpackWeight(n, k, &packedw, matBf32.data(), n, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refCupk.data(), k, n, n);
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp, {matAf32.data(), k}, {&packedw}, {match.data(), n}};
    parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
    auto err = get_ut_err(qtype);
    buffer_error(refC.data(), match.data(), refC.size(), err);
    buffer_error(refCupk.data(), match.data(), refCupk.size(), 0.001f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_CompFp32 sUT_CompFp32;
#endif

class UT_CompInt8 {
 public:
  UT_CompInt8() {
    UT_START();
    ut_newkblock_v2<gemm::KblockGemmAvx512vnni<48, 4>>(1, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F16, true);
    ut_s4_full();
    ut_new_dtype(BTLA_DTYPE::S1_CLIP);
    ut_new_dtype(BTLA_DTYPE::S2_CLIP);
    ut_new_dtype(BTLA_DTYPE::S3_CLIP);
    ut_new_dtype(BTLA_DTYPE::S5_CLIP);
    ut_new_dtype(BTLA_DTYPE::S6_CLIP);
    ut_new_dtype(BTLA_DTYPE::S7_CLIP);
  }

  void ut_new_dtype(BTLA_DTYPE qtype) {
    GetCPUDevice();
    if (_cd->AVX2()) {
      ut_newkblock<gemm::KblockGemmAvx2vnni<24, 2>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvx2vnni<24, 2>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F16, true);
      ut_newkblock<gemm::KblockGemmAvx2vnni<24, 2>>(4, 4096, 4096, 64, qtype, BTLA_DTYPE::BF16);
      ut_newkblock<gemm::KblockGemmAvx2vnni<24, 2>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
    }
    if (_cd->AVX_VNNI()) {
      ut_newkblock<gemm::KblockGemmAvxvnni<24, 2>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvxvnni<24, 2>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F16, true);
      ut_newkblock<gemm::KblockGemmAvxvnni<24, 2>>(4, 4096, 4096, 64, qtype, BTLA_DTYPE::BF16);
      ut_newkblock<gemm::KblockGemmAvxvnni<24, 2>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
    }
    if (_cd->AVX512_VNNI()) {
      ut_newkblock<gemm::KblockGemmAvx512vnni<48, 4>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvx512vnni<48, 4>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F16, true);
      ut_newkblock<gemm::KblockGemmAvx512vnni<48, 4>>(4, 4096, 4096, 64, qtype, BTLA_DTYPE::BF16);
      ut_newkblock<gemm::KblockGemmAvx512vnni<48, 4>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
    }
    if (_cd->AVX512BW()) {
      ut_newkblock<gemm::KblockGemmAvx512bw<48, 4>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvx512bw<48, 4>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F16, true);
      ut_newkblock<gemm::KblockGemmAvx512bw<48, 4>>(4, 4096, 4096, 64, qtype, BTLA_DTYPE::BF16);
      ut_newkblock<gemm::KblockGemmAvx512bw<48, 4>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
    }
    if (_cd->AMX_INT8()) {
      ut_newkblock<gemm::KblockGemmAmxint8<64, 16>>(1, 4096, 4096, 64, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAmxint8<64, 16>>(1, 4096, 4096, 64, qtype, BTLA_DTYPE::F16, true);
      ut_newkblock<gemm::KblockGemmAmxint8<64, 16>>(8, 4096, 4096, 64, qtype, BTLA_DTYPE::BF16);
      ut_newkblock<gemm::KblockGemmAmxint8<64, 16>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
    }
  }

  void ut_s4_full() {
    GetCPUDevice();
    auto qtype = BTLA_DTYPE::S4_CLIP;
    if (_cd->AVX2()) {
      ut_newkblock<gemm::KblockGemmAvx2vnni<24, 2>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvx2vnni<24, 2>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F16);
      ut_newkblock<gemm::KblockGemmAvx2vnni<24, 2>>(4, 4096, 4096, 64, qtype, BTLA_DTYPE::BF16);
      ut_newkblock<gemm::KblockGemmAvx2vnni<24, 2>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F16, true);
      ut_newkblock<gemm::KblockGemmAvx2vnni<24, 2>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvx2vnni<24, 2>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::DQ8_BNB);
      ut_newkblock_pc<gemm::ICoreRowNAvx2vnni<24, 4>>(1, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock_pc<gemm::ICoreRowNAvx2vnni<24, 4>>(8, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);

      ut_newkblock<gemm::KblockGemmAvx2vnniSS<24, 1>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvx2vnniSS<24, 1>>(4, 4096, 4096, 64, qtype, BTLA_DTYPE::BF16);
      ut_newkblock<gemm::KblockGemmAvx2vnniSS<24, 1>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvx2vnniSS<24, 1>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::DQ8_BNB);
      ut_newkblock_pc<gemm::ICoreRowNAvx2vnniSS<24, 1>>(1, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock_pc<gemm::ICoreRowNAvx2vnniSS<24, 1>>(8, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
    }
    if (_cd->AVX_VNNI()) {
      ut_newkblock<gemm::KblockGemmAvxvnni<24, 2>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvxvnni<24, 2>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F16, true);
      ut_newkblock<gemm::KblockGemmAvxvnni<24, 2>>(4, 4096, 4096, 64, qtype, BTLA_DTYPE::BF16);
      ut_newkblock<gemm::KblockGemmAvxvnni<24, 2>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvxvnni<24, 2>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::DQ8_BNB);
      ut_newkblock<gemm::KblockGemmAvxvnni<24, 2>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, false);
      ut_newkblock<gemm::KblockGemmAvxvnni<24, 2>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock_pc<gemm::ICoreRowNAvxvnni<24, 4>>(1, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock_pc<gemm::ICoreRowNAvxvnni<24, 4>>(8, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);

      ut_newkblock<gemm::KblockGemmAvxvnniSS<24, 2>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvxvnniSS<24, 2>>(4, 4096, 4096, 64, qtype, BTLA_DTYPE::BF16);
      ut_newkblock<gemm::KblockGemmAvxvnniSS<24, 2>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvxvnniSS<24, 2>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::DQ8_BNB);
      ut_newkblock_pc<gemm::ICoreRowNAvxvnniSS<24, 4>>(1, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock_pc<gemm::ICoreRowNAvxvnniSS<24, 4>>(8, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
    }

    if (_cd->AVX512_VNNI()) {
      ut_newkblock<gemm::KblockGemmAvx512vnni<48, 4>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvx512vnni<48, 4>>(1, 4096, 4096, 64, qtype, BTLA_DTYPE::F16);
      ut_newkblock<gemm::KblockGemmAvx512vnni<48, 4>>(4, 4096, 4096, 64, qtype, BTLA_DTYPE::BF16);
      ut_newkblock<gemm::KblockGemmAvx512vnni<48, 4>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, false);
      ut_newkblock<gemm::KblockGemmAvx512vnni<48, 4>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvx512vnni<48, 4>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, false);
      ut_newkblock<gemm::KblockGemmAvx512vnni<48, 4>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::DQ8_BNB);
      ut_newkblock_pc<gemm::ICoreRowNAvx512vnni<48, 8>>(1, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock_pc<gemm::ICoreRowNAvx512vnni<48, 8>>(8, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
    }

    if (_cd->AVX512BW()) {
      ut_newkblock<gemm::KblockGemmAvx512bw<48, 4>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvx512bw<48, 4>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::F16, true);
      ut_newkblock<gemm::KblockGemmAvx512bw<48, 4>>(4, 4096, 4096, 64, qtype, BTLA_DTYPE::BF16);
      ut_newkblock<gemm::KblockGemmAvx512bw<48, 4>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAvx512bw<48, 4>>(1, 4096, 4096, 32, qtype, BTLA_DTYPE::DQ8_BNB);
      ut_newkblock_pc<gemm::ICoreRowNAvx512bw<48, 8>>(1, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock_pc<gemm::ICoreRowNAvx512bw<48, 8>>(8, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
    }

    if (_cd->AMX_INT8()) {
      ut_newkblock<gemm::KblockGemmAmxint8SS<64, 16>>(1, 4096, 4096, 64, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAmxint8SS<64, 16>>(1, 4096, 4096, 64, qtype, BTLA_DTYPE::F16);
      ut_newkblock<gemm::KblockGemmAmxint8SS<64, 16>>(8, 4096, 4096, 64, qtype, BTLA_DTYPE::BF16);
      ut_newkblock<gemm::KblockGemmAmxint8SS<64, 16>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAmxint8SS<64, 16>>(1, 4096, 4096, 128, qtype, BTLA_DTYPE::DQ8_BNB);
      ut_newkblock_pc<gemm::ICoreRowNAmxint8SS<64, 16>>(1, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock_pc<gemm::ICoreRowNAmxint8SS<64, 16>>(8, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);

      ut_newkblock<gemm::KblockGemmAmxint8<64, 16>>(1, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAmxint8<64, 16>>(1, 4096, 4096, 64, qtype, BTLA_DTYPE::F32);
      ut_newkblock<gemm::KblockGemmAmxint8<64, 16>>(8, 4096, 4096, 128, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock<gemm::KblockGemmAmxint8<64, 16>>(128, 4096, 4096, 128, qtype, BTLA_DTYPE::DQ8_BNB);
      ut_newkblock_pc<gemm::ICoreRowNAmxint8<64, 16>>(1, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
      ut_newkblock_pc<gemm::ICoreRowNAmxint8<64, 16>>(8, 4096, 4096, 4096, qtype, BTLA_DTYPE::F32, true);
    }
  }

  using PcWriteBack = epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue::gemm::AccumulatorWriteBackFp32>;

  template <class GemmCore_T>
  void ut_newkblock(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, BTLA_DTYPE stype, bool isAsym = false) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s Asym:%d\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), bestla_dtype_str(stype), isAsym);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher = wrapper::gemm::LauncherIntKBlock<GemmCore_T, prologue_a::gemm::ActivationF32KBlockQuantize,
                                                      prologue_b::gemm::WeightKBlockNInteger,
                                                      epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerIntKBlock<GemmCore_T>;

    blocksize = blocksize == -1 ? k : blocksize;
    int kblks = updiv(k, blocksize);
    using WType = typename Launcher::PrologueB::StorageWeight;
    WType packedw = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, stype, isAsym);
    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), match(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    avector<uint8_t> matAu8(m * k), zpAu8(m * kblks);
    avector<float> scaleAf32(m * kblks);
    fill_buffer_randn(matAu8.data(), matAu8.size(), uint8_t(0), uint8_t(127));
    fill_buffer_randn(zpAu8.data(), zpAu8.size(), uint8_t(60), uint8_t(64));
    fill_buffer_randn(scaleAf32.data(), scaleAf32.size(), 0.001f, 0.005f);
    ut::fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        matAf32[i * k + j] =
            (float(matAu8[i * k + j]) - zpAu8[i * kblks + j / blocksize]) * scaleAf32[i * kblks + j / blocksize];
      }
    }

    Launcher::PrologueB::packWeight(n, k, matBf32.data(), n, &packedw, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    Launcher::PrologueB::unpackWeight(n, k, &packedw, matBf32.data(), n, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refCupk.data(), k, n, n);
    auto quanA = Launcher::PrologueA::createStorage(m, k, blocksize, isAsym);
    utils::avector<int8_t> bufferA(quanA.mSize);
    quanA.assign(bufferA.data());
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp, {matAf32.data(), k, &quanA}, {&packedw}, {match.data(), n}};
    parallel::GemmRunWithA<Parallel, Launcher>(args, UT_Threading::get());
    auto err = get_ut_err(qtype);
    buffer_error(refC.data(), match.data(), refC.size(), err);
    if (stype != BTLA_DTYPE::DQ8_BNB) {
      buffer_error(refCupk.data(), match.data(), refCupk.size(), INT8_ERR);  // dynamic quant error
    } else {
      auto DQ_INT8_ERR = 0.8f;
      buffer_error(refCupk.data(), match.data(), refCupk.size(), DQ_INT8_ERR);  // dynamic quant error
    }
  }

  template <class GemmCore_T>
  void ut_newkblock_v2(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, BTLA_DTYPE stype, bool isAsym = false) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s Asym:%d\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), bestla_dtype_str(stype), isAsym);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher = wrapper::gemm::LauncherIntKBlockV2<GemmCore_T, prologue_a::gemm::ActivationF32KBlockQuantize,
                                                        prologue_b::gemm::WeightKBlockNInt,
                                                        epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerIntKBlock<GemmCore_T>;

    blocksize = blocksize == -1 ? k : blocksize;
    int kblks = updiv(k, blocksize);
    auto packedw = Launcher::PrologueB::create_storage(n, k, blocksize, qtype, stype, isAsym);
    utils::avector<int8_t> buffer(packedw.buf_size_);
    packedw.set_buffers(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), match(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    avector<uint8_t> matAu8(m * k), zpAu8(m * kblks);
    avector<float> scaleAf32(m * kblks);
    fill_buffer_randn(matAu8.data(), matAu8.size(), uint8_t(0), uint8_t(127));
    fill_buffer_randn(zpAu8.data(), zpAu8.size(), uint8_t(60), uint8_t(64));
    fill_buffer_randn(scaleAf32.data(), scaleAf32.size(), 0.001f, 0.005f);
    ut::fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        matAf32[i * k + j] =
            (float(matAu8[i * k + j]) - zpAu8[i * kblks + j / blocksize]) * scaleAf32[i * kblks + j / blocksize];
      }
    }
    storage::gemm::StorageWeightNInt rawB, dstB;
    rawB.set_nt_data(n, k, k, utils::bestla_dtype<float>, matBf32.data());
    Launcher::PrologueB::convert(rawB, packedw, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    Launcher::PrologueB::convert(packedw, rawB, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refCupk.data(), k, n, n);
    auto quanA = Launcher::PrologueA::createStorage(m, k, blocksize, isAsym);
    utils::avector<int8_t> bufferA(quanA.mSize);
    quanA.assign(bufferA.data());
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp, {matAf32.data(), k, &quanA}, {&packedw}, {match.data(), n}};
    parallel::GemmRunWithA<Parallel, Launcher>(args, UT_Threading::get());
    auto err = get_ut_err(qtype);
    buffer_error(refC.data(), match.data(), refC.size(), err);
    if (stype != BTLA_DTYPE::DQ8_BNB) {
      buffer_error(refCupk.data(), match.data(), refCupk.size(), INT8_ERR);  // dynamic quant error
    } else {
      auto DQ_INT8_ERR = 0.8f;
      buffer_error(refCupk.data(), match.data(), refCupk.size(), DQ_INT8_ERR);  // dynamic quant error
    }
  }

  template <class GemmCore_T>
  void ut_newkblock_pc(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, BTLA_DTYPE stype, bool isAsym = false) {
    assert(blocksize >= k);
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s Asym:%d\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), bestla_dtype_str(stype), isAsym);
    auto constexpr ISA = GemmCore_T::ISA;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
    using Launcher = wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ActivationF32KBlockQuantize,
                                                 prologue_b::gemm::WeightKBlockNInteger, PcWriteBack>;

    blocksize = blocksize == -1 ? k : blocksize;
    int kblks = updiv(k, blocksize);
    using WType = typename Launcher::PrologueB::StorageWeight;
    WType packedw = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, stype, isAsym);
    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), match(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    avector<uint8_t> matAu8(m * k), zpAu8(m * kblks);
    avector<float> scaleAf32(m * kblks);
    fill_buffer_randn(matAu8.data(), matAu8.size(), uint8_t(0), uint8_t(127));
    fill_buffer_randn(zpAu8.data(), zpAu8.size(), uint8_t(60), uint8_t(64));
    fill_buffer_randn(scaleAf32.data(), scaleAf32.size(), 0.001f, 0.005f);
    ut::fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    avector<float> reduceAf32(m * kblks, 0.f);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        matAf32[i * k + j] =
            (float(matAu8[i * k + j]) - zpAu8[i * kblks + j / blocksize]) * scaleAf32[i * kblks + j / blocksize];
        reduceAf32[i * kblks + j / blocksize] += matAf32[i * k + j];
      }
    }
    Launcher::PrologueB::packWeight(n, k, matBf32.data(), n, &packedw, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    Launcher::PrologueB::unpackWeight(n, k, &packedw, matBf32.data(), n, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refCupk.data(), k, n, n);
    auto quanA = Launcher::PrologueA::createStorage(m, k, blocksize, isAsym);
    utils::avector<int8_t> bufferA(quanA.mSize);
    quanA.assign(bufferA.data());
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{
        gp,
        {matAf32.data(), k, &quanA},
        {&packedw},
        {{packedw.template SPtr<char>(), packedw.SDtype(), quanA.template SPtr<float>(), quanA.template ZPtr<uint8_t>(),
          packedw.template RPtr<int32_t>(), nullptr, nullptr, k},
         {match.data(), n}}};
    parallel::GemmRunWithA<Parallel, Launcher>(args, UT_Threading::get());
    auto err = get_ut_err(qtype);
    buffer_error(refC.data(), match.data(), refC.size(), err);

    if (stype != BTLA_DTYPE::DQ8_BNB) {
      buffer_error(refCupk.data(), match.data(), refCupk.size(), INT8_ERR);  // dynamic quant error
    } else {
      auto DQ_INT8_ERR = 0.8f;
      buffer_error(refCupk.data(), match.data(), refCupk.size(), DQ_INT8_ERR);  // dynamic quant error
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_CompInt8 sUT_CompInt8;
#endif

class UT_CompBf16 {
 public:
  UT_CompBf16() {
    UT_START();
    CheckISA(AMX_BF16);
    request_perm_xtile_data();
    ut_s4();
    ut_s8();
    ut_f4();
    ut_f8();
  }

  void ut_f8() {
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, f8>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E4M3);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E4M3);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, f8>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E5M2);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E5M2);
  }

  void ut_s4() {
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, utils::fp16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
  }

  void ut_s8() {
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, 128, BTLA_DTYPE::S8);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, -1, BTLA_DTYPE::S8);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
  }

  void ut_f4() {
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_BNB);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_E2M1);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_NF4);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
  }

  template <class GemmCore_T, template <class _T> class Wei, typename Scale_T>
  void ut(int m, int n, int k, int blocksize, BTLA_DTYPE qtype) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), type_str<Scale_T>);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher = wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ActivationBase, Wei,
                                                 epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;

    blocksize = blocksize == -1 ? k : blocksize;
    using WType = typename Wei<GemmCore_T>::StorageWeight;
    WType packedw(0);
    if constexpr (std::is_same_v<Wei<GemmCore_T>, prologue_b::gemm::WeightKBlockNInteger<GemmCore_T>>) {
      packedw = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, false);
    } else if constexpr (std::is_same_v<Wei<GemmCore_T>, prologue_b::gemm::WeightKBlockNFloat<GemmCore_T>>) {
      packedw = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);
    }

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<utils::bf16> matAbf16(m * k), matBbf16(k * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    avector<float> matBf32(k * n), matAf32(m * k), match(m * n), refC(m * n), refCupk(m * n);
    for (size_t i = 0; i < matBf32.size(); i++) {
      matBf32[i] = matBbf16[i];
    }
    Launcher::PrologueB::packWeight(n, k, matBf32.data(), n, &packedw, UT_Threading::get());
    gemmref_bf16bf16fp32(m, n, k, matAbf16.data(), matBbf16.data(), refC.data(), k, n, n);
    Launcher::PrologueB::unpackWeight(n, k, &packedw, matBf32.data(), n, UT_Threading::get());
    for (size_t i = 0; i < matBf32.size(); i++) {
      matBbf16[i] = static_cast<utils::bf16>(matBf32[i]);
    }
    gemmref_bf16bf16fp32(m, n, k, matAbf16.data(), matBbf16.data(), refCupk.data(), k, n, n);
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp, {matAbf16.data(), k}, {&packedw}, {match.data(), n}};
    parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
    auto err = get_ut_err(qtype);
    buffer_error(refC.data(), match.data(), refC.size(), err);
    buffer_error(refCupk.data(), match.data(), refCupk.size(), 0.05f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_CompBf16 sUT_CompBf16;
#endif

class UT_ORT_NBits {
 public:
  UT_ORT_NBits() {
    UT_START();
    ut_s4();
  }

  void ut_s4() {
    CheckISA(AVX2);
    ut<sAVX2>(1, 14336, 4096, 32, BTLA_DTYPE::S4_CLIP, true);
    ut<sAVX2>(1, 1, 32, 32, BTLA_DTYPE::S4_CLIP, false);
    ut<sAVX2>(1, 2, 32, 32, BTLA_DTYPE::S4_CLIP, true);
    ut<sAVX2>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, true);
    ut<sAVX2>(1, 11008, 4096, 32, BTLA_DTYPE::S4_CLIP, false);
    ut<sAVX2>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, false);
    ut<sAVX2>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP, false);
    ut<sAVX2>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP, false);
    CheckISA(AVX512F);
    ut<sAVX512F>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, true);
    ut<sAVX512F>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, false);
    ut<sAVX512F>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP, false);
    ut<sAVX512F>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP, false);
  }

  template <class GemmCore_T>
  void ut(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, bool isasym) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s asym:%d \n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), isasym);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher =
        wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ActivationKBlockBaseF32,
                                    prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;

    blocksize = blocksize == -1 ? k : blocksize;
    using WType = storage::gemm::StorageWeightKBlockNInteger;
    WType packedw(0);
    packedw = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<float>, isasym);

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), match(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matAf32.data(), matAf32.size(), -0.5f, 0.5f);
    avector<uint8_t> matBs8(n * k);
    avector<int4x2> matBs4(n * updiv(k, 2));
    int blks = updiv(k, blocksize);
    avector<float> scalesB(n * blks);
    avector<uint8_t> zpBs8(n * blks, 8);
    auto blk_padding = updiv(blks, 2);
    avector<int4x2> zpBs4(n * blk_padding, uint8_t(0x88));
    fill_buffer_randn(matBs8.data(), matBs8.size(), uint8_t(0), uint8_t(15));
    if (isasym) {
      fill_buffer_randn(zpBs8.data(), zpBs8.size(), uint8_t(0), uint8_t(15));
    }
    fill_buffer_randn(scalesB.data(), scalesB.size(), 0.001f, 0.005f);
    avector<float> reduceA(m * blks, 0.f);

    auto rA = Launcher::PrologueA::createStorage(m, k, blocksize);
    avector<int8_t> tmpA(rA.mSize);
    if (isasym) {
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
          reduceA[i * blks + j / blocksize] += matAf32[i * k + j];
        }
      }
      rA.assign(tmpA.data());
      Launcher::PrologueA::reduce({matAf32.data(), k, &rA}, m, k, blocksize,
                                  UT_Threading::get());  // for reduce UT
      buffer_error(reduceA.data(), rA.template RPtr<float>(), reduceA.size(), FP32_ERR);
      memset(tmpA.data(), 0, tmpA.size());  // clear
    }
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < k; j += 2) {
        *(uint8_t*)&matBs4[i * k / 2 + j / 2] = matBs8[i * k + j] | matBs8[i * k + j + 1] << 4;
        auto koff = j / blocksize + i * blks;
        auto koff1 = (j + 1) / blocksize + i * blks;
        matBf32[j * n + i] = (float(matBs8[i * k + j]) - zpBs8[koff]) * scalesB[koff];
        matBf32[(j + 1) * n + i] = (float(matBs8[i * k + j + 1]) - zpBs8[koff1]) * scalesB[koff1];
      }
      for (size_t j = 0; j < k; j += blocksize * 2) {
        *(uint8_t*)&zpBs4[i * blk_padding + j / blocksize / 2] =
            zpBs8[i * blks + j / blocksize] | zpBs8[i * blks + j / blocksize + 1] << 4;
      }
    }
    Launcher::PrologueB::packNbitsWeightQ4(n, k, isasym, (uint8_t*)matBs4.data(), k, scalesB.data(),
                                           isasym ? (uint8_t*)zpBs4.data() : nullptr, &packedw, UT_Threading::get());
    Launcher::PrologueB::reduceWeight(&packedw, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    avector<float> revB(matBf32.size());
    Launcher::PrologueB::unpackWeight(n, k, &packedw, revB.data(), n, UT_Threading::get());
    buffer_error(matBf32.data(), revB.data(), revB.size(), FP32_ERR);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), revB.data(), refCupk.data(), k, n, n);
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp, {matAf32.data(), k, &rA}, {&packedw}, {match.data(), n}};
    if (isasym) {
      parallel::GemmRunWithA<Parallel, Launcher>(args, UT_Threading::get());
    } else {
      parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
    }
    auto err = INT4_ERR;
    buffer_error(refC.data(), match.data(), refC.size(), err);
    buffer_error(refCupk.data(), match.data(), refCupk.size(), 0.001f);
  }

  template <class GemmCore_T>
  void ut_file(int m) {
    int n = 14336, k = 4096, blocksize = 32;
    BTLA_DTYPE qtype = BTLA_DTYPE::S4_CLIP;
    bool isasym = true;
    printf("Test Case %s: %d %d %d-%d type:%s core:%s asym:%d \n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), isasym);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher =
        wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ActivationKBlockBaseF32,
                                    prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerIntKBlock<GemmCore_T>;

    const char *qfile = "int_weight.bin", *sfile = "scales.bin", *zfile = "zeros.bin";
    auto qdata = ut::readFile2Buffer<int8_t>(qfile);
    auto sdata = readFile2Buffer<float>(sfile);
    auto zdata = readFile2Buffer<int8_t>(zfile);
    using WType = storage::gemm::StorageWeightKBlockNInteger;
    WType packedw = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<float>,
                                                       bestla_dtype<utils::bf16>, isasym);

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), match(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matAf32.data(), matAf32.size(), -0.5f, 0.5f);
    int blks = updiv(k, blocksize);
    avector<float> reduceA(m * blks, 0.f);

    auto rA = Launcher::PrologueA::createStorage(m, k, blocksize);
    avector<int8_t> tmpA(rA.mSize);
    if (isasym) {
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
          reduceA[i * blks + j / blocksize] += matAf32[i * k + j];
        }
      }
      rA.assign(tmpA.data());
      Launcher::PrologueA::template reduce<ISA>({matAf32.data(), k, &rA}, m, k, blocksize,
                                                UT_Threading::get());  // for reduce UT
      buffer_error(reduceA.data(), rA.template RPtr<float>(), reduceA.size(), FP32_ERR);
      memset(tmpA.data(), 0, tmpA.size());  // clear
    }
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        matBf32[i * n + j] = (float(qdata[i * n + j]) - zdata[i / blocksize * n + j]) * sdata[i / blocksize * n + j];
      }
    }

    Launcher::PrologueB::packQWeight(n, k, qdata.data(), n, sdata.data(), zdata.data(), &packedw, UT_Threading::get());

    auto bfile = readFile2Buffer<int8_t>("bestla_w3.weight.bin");
    WType packedfile(0);
    packedfile.deserialize(bfile.data());
    buffer_error(packedw.WPtr<int8_t>(), packedfile.WPtr<int8_t>(), packedw.mQBuf.size<int8_t>());
    buffer_error(packedw.SPtr<float>(), packedfile.SPtr<float>(), packedw.CSize());
    buffer_error(packedw.ZPtr<int8_t>(), packedfile.ZPtr<int8_t>(), packedw.CSize());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    avector<float> revB(matBf32.size());
    Launcher::PrologueB::unpackWeight(n, k, &packedw, revB.data(), n, UT_Threading::get());
    buffer_error(matBf32.data(), revB.data(), revB.size(), FP32_ERR);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), revB.data(), refCupk.data(), k, n, n);
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp, {matAf32.data(), k, &rA}, {&packedw}, {match.data(), n}};
    if (isasym) {
      parallel::GemmRunWithA<Parallel, Launcher>(args, UT_Threading::get());
    } else {
      parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
    }
    auto err = INT4_ERR;
    buffer_error(refC.data(), match.data(), refC.size(), err);
    buffer_error(refCupk.data(), match.data(), refCupk.size(), 0.001f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_ORT_NBits sUT_ORT_NBits;
#endif

class UT_CompFp16 {
 public:
  UT_CompFp16() {
    UT_START();
    CheckISA(AMX_FP16);
    ut_s4();
  }

  void ut_s4() {
    ut<sAMX_FP16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_FP16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_FP16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_FP16, prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_FP16, prologue_b::gemm::WeightKBlockNInteger, utils::fp16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_FP16, prologue_b::gemm::WeightKBlockNInteger, float>(16, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
  }

  template <class GemmCore_T, template <class _T> class Wei, typename Scale_T>
  void ut(int m, int n, int k, int blocksize, BTLA_DTYPE qtype) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), type_str<Scale_T>);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher = wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ActivationBase, Wei,
                                                 epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;

    blocksize = blocksize == -1 ? k : blocksize;
    using WType = typename Wei<GemmCore_T>::StorageWeight;
    WType packedw(0);
    if constexpr (std::is_same_v<Wei<GemmCore_T>, prologue_b::gemm::WeightKBlockNInteger<GemmCore_T>>) {
      packedw = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, false);
    } else if constexpr (std::is_same_v<Wei<GemmCore_T>, prologue_b::gemm::WeightKBlockNFloat<GemmCore_T>>) {
      packedw = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);
    }

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<utils::fp16> matAfp16(m * k), matBfp16(k * n);
    fill_buffer_randn(matAfp16.data(), matAfp16.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(matBfp16.data(), matBfp16.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    avector<float> matBf32(k * n), matAf32(m * k), match(m * n), refC(m * n), refCupk(m * n);
    kernel::wrapper::Memcpy2DFp16CvtFp32::forward<ISA>(matBfp16.data(), matBf32.data(), k, n, n * sizeof(matBfp16[0]),
                                                       n * sizeof(matBf32[0]), false);
    Launcher::PrologueB::packWeight(n, k, matBf32.data(), n, &packedw, UT_Threading::get());
    gemmref_fp16fp16fp32(m, n, k, matAfp16.data(), matBfp16.data(), refC.data(), k, n, n);
    Launcher::PrologueB::unpackWeight(n, k, &packedw, matBf32.data(), n, UT_Threading::get());
    kernel::wrapper::Memcpy2DFp32CvtFp16::forward<ISA>(matBf32.data(), matBfp16.data(), k, n, n * sizeof(matBf32[0]),
                                                       n * sizeof(matBfp16[0]), false);
    gemmref_fp16fp16fp32(m, n, k, matAfp16.data(), matBfp16.data(), refCupk.data(), k, n, n);
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp, {matAfp16.data(), k}, {&packedw}, {match.data(), n}};
    parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
    auto err = get_ut_err(qtype);
    buffer_error(refC.data(), match.data(), refC.size(), err);
    buffer_error(refCupk.data(), match.data(), refCupk.size(), 0.05f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_CompFp16 sUT_CompFp16;
#endif
}  // namespace ut
}  // namespace bestla
