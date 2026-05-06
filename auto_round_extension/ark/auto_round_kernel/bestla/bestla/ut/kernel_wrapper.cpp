#include "kernel_ut.h"
#include "kernel_wrapper.h"
namespace bestla {
using namespace utils;
namespace ut {
#if 0
class UT_DecompressKBlockS4FP {
 public:
  UT_DecompressKBlockS4FP() {
    UT_START();
    CheckISA(AVX2);
    ut_avx2<BTLA_DTYPE::S4_CLIP, 1, float, float>(410, 24, 24, 24, 0, 128, 24);
    ut_avx2<BTLA_DTYPE::S4_CLIP, 1, float, float>(410, 48, 48, 48, 0, 128, 48);
  }


  template <BTLA_DTYPE S4_T, int PACK_ROW, typename ST_T, typename DST_T>
  void ut_avx2(int row, int col, int ld_src, int ld_dst, int k_offset, int kblock, int NPad, bool asym = false) {
    printf("Test Case %s_%d_%d: %d %d %d %d %d %d %d %d\n", __FUNCTION__, int(S4_T), PACK_ROW, row, col, ld_src, ld_dst,
           k_offset, kblock, NPad, asym);
    int nk_blk = updiv(row, kblock);
    std::vector<utils::int4x2> s4_wei(row * col / 2);
    std::vector<int8_t> s8_wei(col * row);
    std::vector<DST_T> bf16_wei(ld_dst * row);
    std::vector<DST_T> ref_wei(ld_dst * row);
    std::vector<ST_T> scales(NPad * nk_blk);
    std::vector<int8_t> zero_points(NPad * nk_blk);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-128), int8_t(127));
    fill_buffer_randn(scales.data(), scales.size(), ST_T(0.01f), ST_T(0.02f));
    fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));

    for (int i = 0; i < col * row; i += 2) {
      s4_wei[i / 2].x = utils::int4x2::convert(s8_wei[i]);
      s4_wei[i / 2].y = utils::int4x2::convert(s8_wei[i + 1]);
    }
    kernel::wrapper::DecompressKBlockS4Fp<DST_T, PACK_ROW>::template forward<BTLA_ISA::NoSIMD, ST_T, S4_T>(
        s4_wei.data(), bf16_wei.data(), row, col, ld_src, ld_dst, scales.data(), asym ? zero_points.data() : nullptr,
        k_offset, kblock, NPad, cache, CacheSize);
    kernel::wrapper::DecompressKBlockS4Fp<DST_T, PACK_ROW>::template forward<BTLA_ISA::AVX2, ST_T, S4_T>(
        s4_wei.data(), ref_wei.data(), row, col, ld_src, ld_dst, scales.data(), asym ? zero_points.data() : nullptr,
        k_offset, kblock, NPad, cache, CacheSize);
    DST_T thres = DST_T(0.01f);
    if constexpr (std::is_same_v<DST_T, utils::bf16>) {
      thres = DST_T(BF16_ERR);
    }
    ut::buffer_error(ref_wei.data(), bf16_wei.data(), bf16_wei.size(), thres);
  }
};
#ifdef BTLA_UT_KERNEL_WRAPPER
static UT_DecompressKBlockS4FP sUT_DecompressKBlockS4FP;
#endif
#endif

class UT_DecompressKBlockF4FP {
 public:
  UT_DecompressKBlockF4FP() {
    UT_START();
    CheckISA(AVX2);
    ut<float, 1, BTLA_DTYPE::F4_BNB, BTLA_ISA::AVX2>(35, 48, 48, 48, 0, 12, 48);
    ut<float, 1, BTLA_DTYPE::F4_BNB, BTLA_ISA::AVX2>(11, 48, 48, 48, 0, 20, 48);
    CheckISA(AVX512F);
    ut<float, 1, BTLA_DTYPE::F4_BNB, BTLA_ISA::AVX512F>(35, 48, 48, 48, 0, 12, 48);
    ut<float, 1, BTLA_DTYPE::F4_BNB, BTLA_ISA::AVX512F>(11, 48, 48, 48, 0, 20, 48);
  }

  template <typename T, int PACK_ROW, BTLA_DTYPE F4_T, BTLA_ISA ISA_T>
  void ut(int row, int col, int ld_src, int ld_dst, int k_offset, int kblock, int NPad) {
    std::vector<utils::f4x2> f4_wei(row * col / 2);
    std::vector<int8_t> s8_wei(col * row);
    std::vector<T> wei(col * row);
    std::vector<T> ref_wei(col * row);
    std::vector<T> scales(col * updiv(row, kblock));
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(scales.data(), scales.size(), T(1.f), T(10.f));
    for (int i = 0; i < col * row; i += 2) {
      f4_wei[i / 2].x = utils::int4x2::convert(s8_wei[i]);
      f4_wei[i / 2].y = utils::int4x2::convert(s8_wei[i + 1]);
    }
    kernel::wrapper::DecompressKBlockF4Fp<T, PACK_ROW>::template forward<ISA_T, T, F4_T>(
        f4_wei.data(), wei.data(), row, col, ld_src, ld_dst, scales.data(), k_offset, kblock, NPad, cache, CacheSize);
    kernel::wrapper::DecompressKBlockF4Fp<T, PACK_ROW>::template forward<BTLA_ISA::NoSIMD, T, F4_T>(
        f4_wei.data(), ref_wei.data(), row, col, ld_src, ld_dst, scales.data(), k_offset, kblock, NPad, cache,
        CacheSize);
    ut::buffer_error(ref_wei.data(), wei.data(), wei.size(), T(0.01f));
  }
};
#ifdef BTLA_UT_KERNEL_WRAPPER
static UT_DecompressKBlockF4FP sUT_DecompressKBlockF4FP;
#endif

class UT_memcpy2d {
 public:
  UT_memcpy2d() {
    UT_START();
    CheckISA(AVX2);
    ut<utils::bf16, BTLA_ISA::AVX2>(32, 64);
    ut<utils::bf16, BTLA_ISA::AVX2>(32, 64 + 3);
    ut<utils::fp16, BTLA_ISA::AVX2>(32, 64);
    ut<utils::fp16, BTLA_ISA::AVX2>(32, 64 + 3);
    CheckISA(AVX512F);
    ut<utils::bf16, BTLA_ISA::AVX512F>(32, 64);
    ut<utils::bf16, BTLA_ISA::AVX512F>(32, 64 + 3);
    ut<utils::fp16, BTLA_ISA::AVX512F>(32, 64);
    ut<utils::fp16, BTLA_ISA::AVX512F>(32, 64 + 3);
  }

  template <typename T_SRC, BTLA_ISA ISA_T>
  void ut(int row, int col) {
    printf("%s %d %d\n", __FUNCTION__, row, col);
    std::vector<T_SRC> src(row * col);
    std::vector<float> dst(row * col);
    std::vector<T_SRC> src_revert(row * col);

    fill_buffer_randn(src.data(), src.size(), T_SRC(-127), T_SRC(127));
    if (std::is_same_v<T_SRC, utils::bf16>) {
      kernel::wrapper::Memcpy2DBf16CvtFp32::forward<ISA_T>(src.data(), dst.data(), row, col, col * sizeof(T_SRC),
                                                           col * sizeof(float), true);
      kernel::wrapper::Memcpy2DFp32CvtBf16::forward<ISA_T>(dst.data(), src_revert.data(), row, col, col * sizeof(float),
                                                           col * sizeof(T_SRC), true);
    }
    if (std::is_same_v<T_SRC, utils::fp16>) {
      kernel::wrapper::Memcpy2DFp16CvtFp32::forward<ISA_T>(src.data(), dst.data(), row, col, col * sizeof(T_SRC),
                                                           col * sizeof(float), true);
      kernel::wrapper::Memcpy2DFp32CvtFp16::forward<ISA_T>(dst.data(), src_revert.data(), row, col, col * sizeof(float),
                                                           col * sizeof(T_SRC), true);
    }
    ut::buffer_error(src.data(), src_revert.data(), src.size(), T_SRC(1.f));
  }
};
#ifdef BTLA_UT_KERNEL_WRAPPER
static UT_memcpy2d sUT_memcpy2d;
#endif

class UT_PaddingInterleaveMN {
 public:
  UT_PaddingInterleaveMN() {
    UT_START();
    // ut<48, 2, bf16, bf16>(128, 128, 2);  // TO IMPLEMENT
    CheckISA(AVX512_FP16);
    ut<32, 2, fp16, bf16>(128, 128, 2);
  }
  template <int NTile, int RowPack, typename T_SRC, typename T_DST>
  void ut(int row, int col, int row_tile) {
    printf("%s %d %d %d\n", __FUNCTION__, row, col, row_tile);
    int row_pad = padto(row, row_tile);
    int col_pad = padto(col, NTile);

    aligned_vector<T_SRC> src(row * col);
    aligned_vector<T_DST> dst(row_pad * col_pad), ref(row_pad * col_pad);
    for (size_t i = 0; i < src.size(); i++) src[i] = static_cast<T_SRC>(float(i));

    kernel::wrapper::PaddingInterleaveMN<NTile, RowPack, T_SRC, T_DST>::template forward<BTLA_ISA::NoSIMD>(
        src.data(), ref.data(), row, col, row_pad, col_pad, row_pad, col);
    kernel::wrapper::PaddingInterleaveMN<NTile, RowPack, T_SRC, T_DST>::template forward<BTLA_ISA::AVX512_FP16>(
        src.data(), dst.data(), row, col, row_pad, col_pad, col, row_pad);
    ut::buffer_error(dst.data(), ref.data(), dst.size(), T_DST(100));
  }
};
#ifdef BTLA_UT_KERNEL_WRAPPER
static UT_PaddingInterleaveMN sUT_PaddingInterleaveMN;
#endif

class UT_PaddingTransInterleaveMN {
 public:
  UT_PaddingTransInterleaveMN() {
    UT_START();
    CheckISA(AVX512_FP16);
    // ut<48, 2, bf16, bf16>(128, 128, 2);  // TO IMPLEMENT
    ut<32, 2, fp16, bf16>(128, 128, 2);
  }
  template <int MTile, int ColPack, typename T_SRC, typename T_DST>
  void ut(int row, int col, int col_tile) {
    printf("%s %d %d %d\n", __FUNCTION__, row, col, col_tile);
    int row_pad = padto(row, MTile);
    int col_pad = padto(col, col_tile);

    aligned_vector<T_SRC> src(row * col);
    aligned_vector<T_DST> dst(col_pad * row_pad), ref(col_pad * row_pad);
    for (size_t i = 0; i < src.size(); i++) src[i] = static_cast<T_SRC>(float(i));

    kernel::wrapper::PaddingTransInterleaveMN<MTile, ColPack, T_SRC, T_DST>::template forward<BTLA_ISA::NoSIMD>(
        src.data(), ref.data(), row, col, row_pad, col_pad, row_pad, col);
    kernel::wrapper::PaddingTransInterleaveMN<MTile, ColPack, T_SRC, T_DST>::template forward<BTLA_ISA::AVX512_FP16>(
        src.data(), dst.data(), row, col, row_pad, col_pad, col, row_pad);
    ut::buffer_error(dst.data(), ref.data(), dst.size(), T_DST(100));
  }
};
#ifdef BTLA_UT_KERNEL_WRAPPER
static UT_PaddingTransInterleaveMN sUT_PaddingTransInterleaveMN;
#endif

class UT_RevertPaddingInterleaveMN {
 public:
  UT_RevertPaddingInterleaveMN() {
    UT_START();
    ut<48, 4, char>(128, 128, 4);   // vnni
    ut<48, 1, float>(128, 128, 1);  // 512f
    ut<48, 4, char>(128, 32, 64);   // amxint8
  }
  template <int NTile, int PackRow, typename T>
  void ut(int row, int col, int rowtile) {
    printf("%s %d %d %d\n", __FUNCTION__, row, col, rowtile);
    int rowpad = padto(row, rowtile);
    int colpad = padto(col, NTile);
    aligned_vector<T> src(row * col), packed(rowpad * colpad);
    for (size_t i = 0; i < src.size(); i++) {
      src[i] = static_cast<T>(i);
    }
    aligned_vector<T> reverted(row * col);
    kernel::wrapper::PaddingInterleaveMN<NTile, PackRow, T>::template forward<BTLA_ISA::NoSIMD>(
        src.data(), packed.data(), row, col, rowpad, colpad, col, rowpad);
    kernel::wrapper::RevertPaddingInterleaveMN<NTile, PackRow, T>::template forward<BTLA_ISA::NoSIMD>(
        packed.data(), reverted.data(), row, col, rowpad, colpad, rowpad, col);
    ut::buffer_error(src.data(), reverted.data(), reverted.size());
  }
};
#ifdef BTLA_UT_KERNEL_WRAPPER
static UT_RevertPaddingInterleaveMN sUT_RevertPaddingInterleaveMN;
#endif

class UT_LayerNormalization {
 public:
  UT_LayerNormalization() {
    UT_START();
    CheckISA(AVX2);
    ut<float, BTLA_ISA::AVX2>(4096, false, true, true);
    ut<float, BTLA_ISA::AVX2>(4096, false, false, false);
    ut<float, BTLA_ISA::AVX2>(111, false, true, true);
    ut<float, BTLA_ISA::AVX2>(111, true, true, true);
    CheckISA(AVX512F);
    ut<float, BTLA_ISA::AVX512F>(4096, false, true, true);
    ut<float, BTLA_ISA::AVX512F>(4096, false, false, false);
    ut<float, BTLA_ISA::AVX512F>(111, false, true, true);
    ut<float, BTLA_ISA::AVX512F>(111, true, true, true);
  }
  template <typename T, BTLA_ISA ISA>
  void ut(int norm_size, bool simplified, bool hasscale, bool hasbias) {
    printf("%s %d\n", __FUNCTION__, norm_size);
    aligned_vector<T> src(norm_size), dst(norm_size), bias(norm_size), scale(norm_size), ref(norm_size);
    fill_buffer_randn(src.data(), src.size(), -0.5f, 0.5f);
    fill_buffer_randn(bias.data(), bias.size(), -0.5f, 0.5f);
    fill_buffer_randn(scale.data(), scale.size(), 0.1f, 1.f);
    T mean = 0.f, mean_square = 0.f;
    kernel::wrapper::LayerNormalization::forward<BTLA_ISA::NoSIMD>(src.data(), hasscale ? scale.data() : nullptr,
                                                                   hasbias ? bias.data() : nullptr, 0.00001f, norm_size,
                                                                   ref.data(), &mean, &mean_square, simplified);
    kernel::wrapper::LayerNormalization::forward<ISA>(src.data(), hasscale ? scale.data() : nullptr,
                                                      hasbias ? bias.data() : nullptr, 0.00001f, norm_size, dst.data(),
                                                      &mean, &mean_square, simplified);
    buffer_error(ref.data(), dst.data(), ref.size(), 0.01f);
  }
};
#ifdef BTLA_UT_KERNEL_WRAPPER
UT_LayerNormalization sUT_LayerNormalization;
#endif
}  // namespace ut
}  // namespace bestla
