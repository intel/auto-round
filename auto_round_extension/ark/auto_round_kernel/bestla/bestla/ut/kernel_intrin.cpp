#include "kernel_ut.h"
#include "kernel_avx2.h"
#include "kernel_avx_vnni.h"
#include "kernel_avx512f.h"
#include "kernel_avx512_vnni.h"

namespace bestla {
using namespace utils;
namespace ut {

#if CompileAVX512F()
class UT_avx512_decompress_s1_s8 {
 public:
  UT_avx512_decompress_s1_s8() {
    UT_START();
    CheckISA(AVX512F);
    ut<1, 48>(32);
    ut<4, 48>(32);
    ut<1, 48>(32, true);
    ut<2, 48>(32, true);
    ut<4, 48>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, row, col, blocksize);
    avector<utils::bit1x8> s1_wei(row * col / 8);

    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = 8;
    assert(blocksize % 8 == 0);
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-1), int8_t(0));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-1), int8_t(0));

    for (int i = 0; i < col * row; i += 8) {
      memcpy(&s8_ref[i], &s8_wei[i], 8 * sizeof(int8_t));
      s1_wei[i / 8].a = s8_wei[i + 0] + 1;
      s1_wei[i / 8].b = s8_wei[i + 1] + 1;
      s1_wei[i / 8].c = s8_wei[i + 2] + 1;
      s1_wei[i / 8].d = s8_wei[i + 3] + 1;
      s1_wei[i / 8].e = s8_wei[i + 4] + 1;
      s1_wei[i / 8].f = s8_wei[i + 5] + 1;
      s1_wei[i / 8].g = s8_wei[i + 6] + 1;
      s1_wei[i / 8].h = s8_wei[i + 7] + 1;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx512f::decompress_kblock_s1_s8<PackRow, NTILE>(s1_wei.data(), isasym ? zp.data() : nullptr, rev.data(),
                                                             blocksize, NTILE, 0, 0, row_offset, NTILE, cache,
                                                             CacheSize);
    kernel::avx512f::decompress_kblock_s1_s8<PackRow, NTILE>(
        s1_wei.data() + row_offset * NTILE / 8, isasym ? zp.data() : nullptr, rev.data() + row_offset * NTILE,
        blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx512_decompress_s1_s8 sUT_avx512_decompress_s1_s8;
#endif

class UT_avx512_decompress_s4_s8 {
 public:
  UT_avx512_decompress_s4_s8() {
    UT_START();
    CheckISA(AVX512F);
    ut<1, 48>(32);
    ut<4, 48>(32);
    ut<1, 48>(32, true);
    ut<2, 48>(32, true);
    ut<4, 48>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, row, col, blocksize);
    std::vector<utils::int4x2> s4_wei(row * col / 2);
    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = PackRow;
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-8), int8_t(7));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-8), int8_t(7));

    for (int i = 0; i < col * row; i += 2) {
      s8_ref[i] = s8_wei[i];
      s8_ref[i + 1] = s8_wei[i + 1];
      s4_wei[i / 2].x = utils::int4x2::convert(s8_wei[i]) + 8;
      s4_wei[i / 2].y = utils::int4x2::convert(s8_wei[i + 1]) + 8;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx512f::decompress_kblock_s4_s8<PackRow, NTILE>(s4_wei.data(), isasym ? zp.data() : nullptr, rev.data(),
                                                             blocksize, NTILE, 0, 0, row_offset, NTILE, cache,
                                                             CacheSize);
    kernel::avx512f::decompress_kblock_s4_s8<PackRow, NTILE>(
        s4_wei.data() + row_offset * NTILE / 2, isasym ? zp.data() : nullptr, rev.data() + row_offset * NTILE,
        blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx512_decompress_s4_s8 sUT_avx512_decompress_s4_s8;
#endif

class UT_avx512_decompress_s3_s8 {
 public:
  UT_avx512_decompress_s3_s8() {
    UT_START();
    CheckISA(AVX512F);
    ut<1, 48>(32);
    ut<4, 48>(32);
    ut<1, 48>(32, true);
    ut<2, 48>(32, true);
    ut<4, 48>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, PackRow, row, col, blocksize, isasym);
    std::vector<utils::bit2x4> s2_wei(row * col / 4);
    avector<utils::bit1x8> s1_wei(row * col / 8);

    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = 8;
    assert(blocksize % 8 == 0);
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-4), int8_t(3));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-4), int8_t(3));

    for (int i = 0; i < col * row; i += 8) {
      memcpy(&s8_ref[i], &s8_wei[i], 8 * sizeof(int8_t));
      s2_wei[i / 4].a = (s8_wei[i + 0] + 4) & 0x3;
      s2_wei[i / 4].b = (s8_wei[i + 1] + 4) & 0x3;
      s2_wei[i / 4].c = (s8_wei[i + 2] + 4) & 0x3;
      s2_wei[i / 4].d = (s8_wei[i + 3] + 4) & 0x3;

      s2_wei[i / 4 + 1].a = (s8_wei[i + 4] + 4) & 0x3;
      s2_wei[i / 4 + 1].b = (s8_wei[i + 5] + 4) & 0x3;
      s2_wei[i / 4 + 1].c = (s8_wei[i + 6] + 4) & 0x3;
      s2_wei[i / 4 + 1].d = (s8_wei[i + 7] + 4) & 0x3;

      s1_wei[i / 8].a = ((s8_wei[i + 0] + 4) & 0x4) >> 2;
      s1_wei[i / 8].b = ((s8_wei[i + 1] + 4) & 0x4) >> 2;
      s1_wei[i / 8].c = ((s8_wei[i + 2] + 4) & 0x4) >> 2;
      s1_wei[i / 8].d = ((s8_wei[i + 3] + 4) & 0x4) >> 2;
      s1_wei[i / 8].e = ((s8_wei[i + 4] + 4) & 0x4) >> 2;
      s1_wei[i / 8].f = ((s8_wei[i + 5] + 4) & 0x4) >> 2;
      s1_wei[i / 8].g = ((s8_wei[i + 6] + 4) & 0x4) >> 2;
      s1_wei[i / 8].h = ((s8_wei[i + 7] + 4) & 0x4) >> 2;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx512f::decompress_kblock_s3_s8<PackRow, NTILE>(s2_wei.data(), s1_wei.data(), isasym ? zp.data() : nullptr,
                                                             rev.data(), blocksize, NTILE, 0, 0, row_offset, NTILE,
                                                             cache, CacheSize);
    kernel::avx512f::decompress_kblock_s3_s8<PackRow, NTILE>(
        s2_wei.data() + row_offset * NTILE / 4, s1_wei.data() + row_offset * NTILE / 8, isasym ? zp.data() : nullptr,
        rev.data() + row_offset * NTILE, blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx512_decompress_s3_s8 sUT_avx512_decompress_s3_s8;
#endif

class UT_avx512_decompress_s5_s8 {
 public:
  UT_avx512_decompress_s5_s8() {
    UT_START();
    CheckISA(AVX512F);
    ut<1, 48>(32);
    ut<4, 48>(32);
    ut<1, 48>(32, true);
    ut<2, 48>(32, true);
    ut<4, 48>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, row, col, blocksize);
    std::vector<utils::bit4x2> s4_wei(row * col / 2);
    avector<utils::bit1x8> s1_wei(row * col / 8);

    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = 8;
    assert(blocksize % 8 == 0);
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-16), int8_t(15));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-16), int8_t(15));

    for (int i = 0; i < col * row; i += 8) {
      memcpy(&s8_ref[i], &s8_wei[i], 8 * sizeof(int8_t));
      s4_wei[i / 2].x = (s8_wei[i + 0] + 16) & 0xf;
      s4_wei[i / 2].y = (s8_wei[i + 1] + 16) & 0xf;
      s4_wei[i / 2 + 1].x = (s8_wei[i + 2] + 16) & 0xf;
      s4_wei[i / 2 + 1].y = (s8_wei[i + 3] + 16) & 0xf;
      s4_wei[i / 2 + 2].x = (s8_wei[i + 4] + 16) & 0xf;
      s4_wei[i / 2 + 2].y = (s8_wei[i + 5] + 16) & 0xf;
      s4_wei[i / 2 + 3].x = (s8_wei[i + 6] + 16) & 0xf;
      s4_wei[i / 2 + 3].y = (s8_wei[i + 7] + 16) & 0xf;

      s1_wei[i / 8].a = ((s8_wei[i + 0] + 16) & 0x10) >> 4;
      s1_wei[i / 8].b = ((s8_wei[i + 1] + 16) & 0x10) >> 4;
      s1_wei[i / 8].c = ((s8_wei[i + 2] + 16) & 0x10) >> 4;
      s1_wei[i / 8].d = ((s8_wei[i + 3] + 16) & 0x10) >> 4;
      s1_wei[i / 8].e = ((s8_wei[i + 4] + 16) & 0x10) >> 4;
      s1_wei[i / 8].f = ((s8_wei[i + 5] + 16) & 0x10) >> 4;
      s1_wei[i / 8].g = ((s8_wei[i + 6] + 16) & 0x10) >> 4;
      s1_wei[i / 8].h = ((s8_wei[i + 7] + 16) & 0x10) >> 4;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx512f::decompress_kblock_s5_s8<PackRow, NTILE>(s4_wei.data(), s1_wei.data(), isasym ? zp.data() : nullptr,
                                                             rev.data(), blocksize, NTILE, 0, 0, row_offset, NTILE,
                                                             cache, CacheSize);
    kernel::avx512f::decompress_kblock_s5_s8<PackRow, NTILE>(
        s4_wei.data() + row_offset * NTILE / 2, s1_wei.data() + row_offset * NTILE / 8, isasym ? zp.data() : nullptr,
        rev.data() + row_offset * NTILE, blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx512_decompress_s5_s8 sUT_avx512_decompress_s5_s8;
#endif

class UT_avx512_decompress_s6_s8 {
 public:
  UT_avx512_decompress_s6_s8() {
    UT_START();
    CheckISA(AVX512F);
    ut<1, 48>(32);
    ut<4, 48>(32);
    ut<1, 48>(32, true);
    ut<2, 48>(32, true);
    ut<4, 48>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr FullRange = 1 << (6 - 1);
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, row, col, blocksize);
    std::vector<utils::bit4x2> s4_wei(row * col / 2);
    avector<utils::bit2x4> s2_wei(row * col / 4);

    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = 8;
    assert(blocksize % 8 == 0);
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-FullRange), int8_t(FullRange - 1));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-FullRange), int8_t(FullRange - 1));

    for (int i = 0; i < col * row; i += 4) {
      memcpy(&s8_ref[i], &s8_wei[i], 4 * sizeof(int8_t));
      s4_wei[i / 2].x = (s8_wei[i + 0] + FullRange) & 0xf;
      s4_wei[i / 2].y = (s8_wei[i + 1] + FullRange) & 0xf;
      s4_wei[i / 2 + 1].x = (s8_wei[i + 2] + FullRange) & 0xf;
      s4_wei[i / 2 + 1].y = (s8_wei[i + 3] + FullRange) & 0xf;

      s2_wei[i / 4].a = ((s8_wei[i + 0] + FullRange) & 0x30) >> 4;
      s2_wei[i / 4].b = ((s8_wei[i + 1] + FullRange) & 0x30) >> 4;
      s2_wei[i / 4].c = ((s8_wei[i + 2] + FullRange) & 0x30) >> 4;
      s2_wei[i / 4].d = ((s8_wei[i + 3] + FullRange) & 0x30) >> 4;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx512f::decompress_kblock_s6_s8<PackRow, NTILE>(s4_wei.data(), s2_wei.data(), isasym ? zp.data() : nullptr,
                                                             rev.data(), blocksize, NTILE, 0, 0, row_offset, NTILE,
                                                             cache, CacheSize);
    kernel::avx512f::decompress_kblock_s6_s8<PackRow, NTILE>(
        s4_wei.data() + row_offset * NTILE / 2, s2_wei.data() + row_offset * NTILE / 4, isasym ? zp.data() : nullptr,
        rev.data() + row_offset * NTILE, blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx512_decompress_s6_s8 sUT_avx512_decompress_s6_s8;
#endif

class UT_avx512_decompress_s7_s8 {
 public:
  UT_avx512_decompress_s7_s8() {
    UT_START();
    CheckISA(AVX512F);
    ut<1, 48>(32);
    ut<4, 48>(32);
    ut<1, 48>(32, true);
    ut<2, 48>(32, true);
    ut<4, 48>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, row, col, blocksize);
    std::vector<utils::bit4x2> s4_wei(row * col / 2);
    avector<utils::bit1x8> s1_wei(row * col / 8);
    avector<utils::bit2x4> s2_wei(row * col / 4);

    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = 8;
    assert(blocksize % 8 == 0);
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-64), int8_t(63));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-64), int8_t(63));

    for (int i = 0; i < col * row; i += 8) {
      memcpy(&s8_ref[i], &s8_wei[i], 8 * sizeof(int8_t));
      s4_wei[i / 2].x = (s8_wei[i + 0] + 64) & 0xf;
      s4_wei[i / 2].y = (s8_wei[i + 1] + 64) & 0xf;
      s4_wei[i / 2 + 1].x = (s8_wei[i + 2] + 64) & 0xf;
      s4_wei[i / 2 + 1].y = (s8_wei[i + 3] + 64) & 0xf;
      s4_wei[i / 2 + 2].x = (s8_wei[i + 4] + 64) & 0xf;
      s4_wei[i / 2 + 2].y = (s8_wei[i + 5] + 64) & 0xf;
      s4_wei[i / 2 + 3].x = (s8_wei[i + 6] + 64) & 0xf;
      s4_wei[i / 2 + 3].y = (s8_wei[i + 7] + 64) & 0xf;

      s2_wei[i / 4].a = ((s8_wei[i + 0] + 64) & 0x30) >> 4;
      s2_wei[i / 4].b = ((s8_wei[i + 1] + 64) & 0x30) >> 4;
      s2_wei[i / 4].c = ((s8_wei[i + 2] + 64) & 0x30) >> 4;
      s2_wei[i / 4].d = ((s8_wei[i + 3] + 64) & 0x30) >> 4;
      s2_wei[i / 4 + 1].a = ((s8_wei[i + 4] + 64) & 0x30) >> 4;
      s2_wei[i / 4 + 1].b = ((s8_wei[i + 5] + 64) & 0x30) >> 4;
      s2_wei[i / 4 + 1].c = ((s8_wei[i + 6] + 64) & 0x30) >> 4;
      s2_wei[i / 4 + 1].d = ((s8_wei[i + 7] + 64) & 0x30) >> 4;

      s1_wei[i / 8].a = ((s8_wei[i + 0] + 64) & 0x40) >> 6;
      s1_wei[i / 8].b = ((s8_wei[i + 1] + 64) & 0x40) >> 6;
      s1_wei[i / 8].c = ((s8_wei[i + 2] + 64) & 0x40) >> 6;
      s1_wei[i / 8].d = ((s8_wei[i + 3] + 64) & 0x40) >> 6;
      s1_wei[i / 8].e = ((s8_wei[i + 4] + 64) & 0x40) >> 6;
      s1_wei[i / 8].f = ((s8_wei[i + 5] + 64) & 0x40) >> 6;
      s1_wei[i / 8].g = ((s8_wei[i + 6] + 64) & 0x40) >> 6;
      s1_wei[i / 8].h = ((s8_wei[i + 7] + 64) & 0x40) >> 6;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx512f::decompress_kblock_s7_s8<PackRow, NTILE>(s4_wei.data(), s2_wei.data(), s1_wei.data(),
                                                             isasym ? zp.data() : nullptr, rev.data(), blocksize, NTILE,
                                                             0, 0, row_offset, NTILE, cache, CacheSize);
    kernel::avx512f::decompress_kblock_s7_s8<PackRow, NTILE>(
        s4_wei.data() + row_offset * NTILE / 2, s2_wei.data() + row_offset * NTILE / 4,
        s1_wei.data() + row_offset * NTILE / 8, isasym ? zp.data() : nullptr, rev.data() + row_offset * NTILE,
        blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx512_decompress_s7_s8 sUT_avx512_decompress_s7_s8;
#endif

class UT_avx512_decompress_s2_s8 {
 public:
  UT_avx512_decompress_s2_s8() {
    UT_START();
    CheckISA(AVX512F);
    ut<1, 48>(32);
    ut<4, 48>(32);
    ut<1, 48>(32, true);
    ut<2, 48>(32, true);
    ut<4, 48>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, PackRow, row, col, blocksize, isasym);
    std::vector<utils::bit2x4> s2_wei(row * col / 4);

    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = blocksize;
    assert(blocksize % 8 == 0);
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-2), int8_t(1));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-2), int8_t(1));

    for (int i = 0; i < col * row; i += 4) {
      memcpy(&s8_ref[i], &s8_wei[i], 4 * sizeof(int8_t));
      s2_wei[i / 4].a = (s8_wei[i + 0] + 2) & 0x3;
      s2_wei[i / 4].b = (s8_wei[i + 1] + 2) & 0x3;
      s2_wei[i / 4].c = (s8_wei[i + 2] + 2) & 0x3;
      s2_wei[i / 4].d = (s8_wei[i + 3] + 2) & 0x3;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx512f::decompress_kblock_s2_s8<PackRow, NTILE>(s2_wei.data(), isasym ? zp.data() : nullptr, rev.data(),
                                                             blocksize, NTILE, 0, 0, row_offset, NTILE, cache,
                                                             CacheSize);
    kernel::avx512f::decompress_kblock_s2_s8<PackRow, NTILE>(
        s2_wei.data() + row_offset * NTILE / 4, isasym ? zp.data() : nullptr, rev.data() + row_offset * NTILE,
        blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx512_decompress_s2_s8 sUT_avx512_decompress_s2_s8;
#endif

class UT_avx512_decompress_s4_fp {
 public:
  UT_avx512_decompress_s4_fp() {
    UT_START();
    CheckISA(AVX512F);
    ut<1, 48, float>(32);
    ut<2, 48, float>(32);
    ut<4, 48, float>(32);
    ut<4, 48, utils::bf16>(32);
    ut<4, 48, utils::bf16, utils::bf16>(32);
    ut<1, 48, float>(32, true);
    ut<2, 48, float>(32, true);
    ut<4, 48, float>(32, true);
    ut<4, 48, utils::bf16>(32, true);
    ut<4, 48, utils::bf16, utils::bf16>(32, true);
  }

  template <int PackRow, int NTILE, typename T, typename ScaleT = float>
  void ut(int blocksize, bool isasym = false) {
    auto dst_dtype = bestla_dtype<T>;
    auto scale_dtype = bestla_dtype<ScaleT>;
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d Asym:%d Pack:%d %s %s\n", __FUNCTION__, row, col, blocksize, isasym, PackRow,
           utils::bestla_dtype_str(dst_dtype), bestla_dtype_str(scale_dtype));
    std::vector<utils::int4x2> s4_wei(row * col / 2);
    std::vector<int8_t> s8_wei(col * row);
    std::vector<T> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = PackRow;
    std::vector<int8_t> zp(col * blks);
    avector<ScaleT> scale(col * blks);
    fill_buffer_randn(scale.data(), scale.size(), ScaleT(0.01f), ScaleT(0.03f));
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-8), int8_t(7));
    std::vector<T> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-8), int8_t(7));

    for (int i = 0; i < col * row; i += 2) {
      s8_ref[i] = float(s8_wei[i]);
      s8_ref[i + 1] = float(s8_wei[i + 1]);
      s4_wei[i / 2].x = utils::int4x2::convert(s8_wei[i]) + 8;
      s4_wei[i / 2].y = utils::int4x2::convert(s8_wei[i + 1]) + 8;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          int corr_offset = i / blocksize * NTILE + j;
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] = float(s8_ref[i * NTILE + j * PackRow + ip]) - float(zp[corr_offset]);
          }
        }
      }
    }

    for (int i = 0; i < row; i += PackRow) {
      for (int j = 0; j < NTILE; j++) {
        int corr_offset = i / blocksize * NTILE + j;
        for (int ip = 0; ip < PackRow; ip++) {
          s8_ref[i * NTILE + j * PackRow + ip] =
              float(s8_ref[i * NTILE + j * PackRow + ip]) * float(scale[corr_offset]);
        }
      }
    }
    kernel::avx512f::decompress_kblock_s4_fp<PackRow, NTILE>(s4_wei.data(), rev.data(), row_offset, NTILE, scale.data(),
                                                             scale_dtype, isasym ? zp.data() : nullptr, 0, 0, blocksize,
                                                             NTILE, cache, CacheSize);
    kernel::avx512f::decompress_kblock_s4_fp<PackRow, NTILE>(
        s4_wei.data() + row_offset * NTILE / 2, rev.data() + row_offset * NTILE, row - row_offset, NTILE, scale.data(),
        scale_dtype, isasym ? zp.data() : nullptr, row_offset, 0, blocksize, NTILE, cache, CacheSize);
    float err = get_ut_err(dst_dtype);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), T(err));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx512_decompress_s4_fp sUT_avx512_decompress_s4_fp;
#endif
#endif

#if CompileAVX512VNNI()
class UT_avx512_gemv {
 public:
  UT_avx512_gemv() {
    UT_START();
    CheckISA(AVX512F);
    ut_1bit_fp32<1>(48, 128, 32, true);
    ut_1bit_fp32<1>(48, 128, 32, false);
    ut_1bit_fp32<4>(48, 128, 32, true);
    ut_1bit_fp32<4>(48, 128, 32, false);

    ut_7bit_fp32<1>(48, 128, 32, true);
    ut_7bit_fp32<1>(48, 128, 32, false);
    ut_7bit_fp32<4>(48, 128, 32, true);
    ut_7bit_fp32<4>(48, 128, 32, false);

    ut_6bit_fp32<1>(48, 128, 32, true);
    ut_6bit_fp32<1>(48, 128, 32, false);
    ut_6bit_fp32<4>(48, 128, 32, true);
    ut_6bit_fp32<4>(48, 128, 32, false);

    ut_5bit_fp32<1>(48, 128, 32, true);
    ut_5bit_fp32<1>(48, 128, 32, false);
    ut_5bit_fp32<4>(48, 128, 32, true);
    ut_5bit_fp32<4>(48, 128, 32, false);

    ut_4bit_fp32<1>(48, 128, 32, true);
    ut_4bit_fp32<1>(48, 128, 32, false);
    ut_4bit_fp32<4>(48, 128, 32, true);
    ut_4bit_fp32<4>(48, 128, 32, false);

    ut_3bit_fp32<1>(48, 128, 32, true);
    ut_3bit_fp32<1>(48, 128, 32, false);
    ut_3bit_fp32<4>(48, 128, 32, true);
    ut_3bit_fp32<4>(48, 128, 32, false);

    ut_2bit_fp32<1>(48, 128, 32, true);
    ut_2bit_fp32<1>(48, 128, 32, false);
    ut_2bit_fp32<4>(48, 128, 32, true);
    ut_2bit_fp32<4>(48, 128, 32, false);

    CheckISA(AVX512_VNNI);
    ut_1bit_u8s8<1>(48, 128, 32, true);
    ut_1bit_u8s8<1>(48, 128, 32, false);
    ut_1bit_u8s8<4>(48, 128, 32, true);
    ut_1bit_u8s8<4>(48, 128, 32, false);

    ut_1bit_s8s8<1>(48, 128, 32, true);
    ut_1bit_s8s8<1>(48, 128, 32, false);
    ut_1bit_s8s8<4>(48, 128, 32, true);
    ut_1bit_s8s8<4>(48, 128, 32, false);

    ut_4bit<1>(48, 128, 32, true);
    ut_4bit<1>(48, 128, 32, false);
    ut_4bit<4>(48, 128, 32, false);
    ut_4bit<4>(48, 128, 32, true);

    ut_4bit_s8s8<1>(48, 128, 32, true);
    ut_4bit_s8s8<1>(48, 128, 32, false);
    ut_4bit_s8s8<4>(48, 128, 32, true);
    ut_4bit_s8s8<4>(48, 128, 32, false);

    ut_2bit<1>(48, 128, 32, true);
    ut_2bit<1>(48, 128, 32, false);
    ut_2bit<4>(48, 128, 32, true);
    ut_2bit<4>(48, 128, 32, false);

    ut_2bit_s8s8<1>(48, 128, 32, true);
    ut_2bit_s8s8<1>(48, 128, 32, false);
    ut_2bit_s8s8<4>(48, 128, 32, true);
    ut_2bit_s8s8<4>(48, 128, 32, false);

    ut_3bit_u8s8<1>(48, 128, 32, true);
    ut_3bit_u8s8<1>(48, 128, 32, false);
    ut_3bit_u8s8<4>(48, 128, 32, true);
    ut_3bit_u8s8<4>(48, 128, 32, false);

    ut_3bit_s8s8<1>(48, 128, 32, true);
    ut_3bit_s8s8<1>(48, 128, 32, false);
    ut_3bit_s8s8<4>(48, 128, 32, true);
    ut_3bit_s8s8<4>(48, 128, 32, false);

    ut_6bit_u8s8<1>(48, 128, 32, true);
    ut_6bit_u8s8<1>(48, 128, 32, false);
    ut_6bit_u8s8<4>(48, 128, 32, true);
    ut_6bit_u8s8<4>(48, 128, 32, false);

    ut_6bit_s8s8<1>(48, 128, 32, true);
    ut_6bit_s8s8<1>(48, 128, 32, false);
    ut_6bit_s8s8<4>(48, 128, 32, true);
    ut_6bit_s8s8<4>(48, 128, 32, false);

    ut_5bit_u8s8<1>(48, 128, 32, true);
    ut_5bit_u8s8<1>(48, 128, 32, false);
    ut_5bit_u8s8<4>(48, 128, 32, true);
    ut_5bit_u8s8<4>(48, 128, 32, false);

    ut_5bit_s8s8<1>(48, 128, 32, true);
    ut_5bit_s8s8<1>(48, 128, 32, false);
    ut_5bit_s8s8<4>(48, 128, 32, true);
    ut_5bit_s8s8<4>(48, 128, 32, false);

    ut_7bit_s8s8<1>(48, 128, 32, true);
    ut_7bit_s8s8<1>(48, 128, 32, false);
    ut_7bit_s8s8<4>(48, 128, 32, true);
    ut_7bit_s8s8<4>(48, 128, 32, false);

    ut_7bit_u8s8<1>(48, 128, 32, true);
    ut_7bit_u8s8<1>(48, 128, 32, false);
    ut_7bit_u8s8<4>(48, 128, 32, true);
    ut_7bit_u8s8<4>(48, 128, 32, false);
  }

  template <int MTILE>
  void ut_1bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-1), int8_t(0));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s1_s8(b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 4) {
        if (iasym) {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0] - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1] - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2] - bzp[bid * n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3] - bzp[bid * n + j + 3]) * scaleb[bid * n + j + 3];
        } else {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3]) * scaleb[bid * n + j + 3];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{nullptr, nullptr, (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr,
                               1,       n};
    kernel::avx512f::gemv_1bit_fp32_fp32<float, 48, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache,
                                                           CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_1bit_u8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-1), int8_t(0));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j]) - azp[bid]) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s1_s8(b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{nullptr, nullptr, (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr,
                               2,       n};
    kernel::avx512f::vnni::gemv_1bit_u8s8_fp32<float, 48, MTILE>({A.data(), scalea.data(), azp.data(), k, blks}, B,
                                                                 Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_1bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-1), int8_t(0));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> A(MTILE * k);
    fill_buffer_randn(A.data(), A.size(), int8_t(0), int8_t(127));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j])) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s1_s8(b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{nullptr, nullptr, (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr,
                               2,       n};
    kernel::avx512f::vnni::gemv_1bit_s8s8_fp32<float, 48, MTILE>({(uint8_t*)A.data(), scalea.data(), nullptr, k, blks},
                                                                 B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_7bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit1x8> b1(n * k / 8);
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-64), int8_t(63));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s7_s8(b4.data(), b2.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 4) {
        if (iasym) {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0] - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1] - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2] - bzp[bid * n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3] - bzp[bid * n + j + 3]) * scaleb[bid * n + j + 3];
        } else {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3]) * scaleb[bid * n + j + 3];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{(uint8_t*)b4.data(),
                               (uint8_t*)b2.data(),
                               (uint8_t*)b1.data(),
                               scaleb.data(),
                               iasym ? bzp.data() : nullptr,
                               7,
                               n};
    kernel::avx512f::gemv_7bit_fp32_fp32<float, 48, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache,
                                                           CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_7bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit2x4> b2(n * k / 4);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-64), int8_t(63));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> A(MTILE * k);
    fill_buffer_randn(A.data(), A.size(), int8_t(0), int8_t(127));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j])) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s7_s8(b4.data(), b2.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{(uint8_t*)b4.data(),
                               (uint8_t*)b2.data(),
                               (uint8_t*)b1.data(),
                               scaleb.data(),
                               iasym ? bzp.data() : nullptr,
                               7,
                               n};
    kernel::avx512f::vnni::gemv_7bit_s8s8_fp32<float, 48, MTILE>({(uint8_t*)A.data(), scalea.data(), nullptr, k, blks},
                                                                 B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_7bit_u8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit2x4> b2(n * k / 4);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-64), int8_t(63));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(128));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(50), uint8_t(64));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j]) - azp[bid]) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s7_s8(b4.data(), b2.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{(uint8_t*)b4.data(),
                               (uint8_t*)b2.data(),
                               (uint8_t*)b1.data(),
                               scaleb.data(),
                               iasym ? bzp.data() : nullptr,
                               7,
                               n};
    kernel::avx512f::vnni::gemv_7bit_u8s8_fp32<float, 48, MTILE>({A.data(), scalea.data(), azp.data(), k, blks}, B,
                                                                 Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_4bit(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b2(n * k / 2);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j]) - azp[bid]) * scalea[bid];
        }
      }
    }

    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        auto b24 = b2[(i * n + j * 4) / 2];
        auto b42 = b2[(i * n + j * 4 + 2) / 2];
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b24.x - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.y - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b42.x - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b42.y - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b24.x - 8)) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.y - 8)) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b42.x - 8)) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b42.y - 8)) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{(uint8_t*)b2.data(),          nullptr, nullptr, scaleb.data(),
                               iasym ? bzp.data() : nullptr, 2,       n};
    kernel::avx512f::vnni::gemv_4bit_u8s8_fp32<float, 48, MTILE>({A.data(), scalea.data(), azp.data(), k, blks}, B,
                                                                 Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_4bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b2(n * k / 2);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);

    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 2) {
        auto b24 = b2[(i * n + j) / 2];
        if (iasym) {
          Bf32[(i)*n + j + 0] = (int(b24.x - 8) - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (int(b24.y - 8) - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
        } else {
          Bf32[(i)*n + j + 0] = (int(b24.x - 8)) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (int(b24.y - 8)) * scaleb[bid * n + j + 1];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{(uint8_t*)b2.data(),          nullptr, nullptr, scaleb.data(),
                               iasym ? bzp.data() : nullptr, 4,       n};
    kernel::avx512f::gemv_4bit_fp32_fp32<float, 48, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache,
                                                           CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_4bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b2(n * k / 2);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<int8_t> A(MTILE * k);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(A.data(), A.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j])) * scalea[bid];
        }
      }
    }

    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        auto b24 = b2[(i * n + j * 4) / 2];
        auto b42 = b2[(i * n + j * 4 + 2) / 2];
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b24.x - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.y - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b42.x - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b42.y - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b24.x - 8)) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.y - 8)) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b42.x - 8)) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b42.y - 8)) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{(uint8_t*)b2.data(),          nullptr, nullptr, scaleb.data(),
                               iasym ? bzp.data() : nullptr, 2,       n};
    kernel::avx512f::vnni::gemv_4bit_s8s8_fp32<float, 48, MTILE>({(uint8_t*)A.data(), scalea.data(), nullptr, k, blks},
                                                                 B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_2bit(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-2), int8_t(1));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int i = 0; i < MTILE; i++) {
      for (int j = 0; j < k; j++) {
        Af32[i * k + j] = (int(A[i * k + j]) - azp[i * blks + j / kblock]) * scalea[i * blks + j / kblock];
      }
    }
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        auto b24 = b2[(i * n + j * 4) / 4];
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b24.a - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.b - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b24.c - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b24.d - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b24.a - 2)) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.b - 2)) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b24.c - 2)) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b24.d - 2)) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    kernel::avx512f::vnni::gemv_2bit_u8s8_fp32<float, 48, MTILE>(
        {A.data(), scalea.data(), azp.data(), k, blks},
        {nullptr, (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 2, n}, Cf32.data(), n, k,
        kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_2bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<int8_t> A(MTILE * k);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(A.data(), A.size(), int8_t(0), int8_t(127));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-2), int8_t(1));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int i = 0; i < MTILE; i++) {
      for (int j = 0; j < k; j++) {
        Af32[i * k + j] = (int(A[i * k + j])) * scalea[i * blks + j / kblock];
      }
    }
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        auto b24 = b2[(i * n + j * 4) / 4];
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b24.a - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.b - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b24.c - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b24.d - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b24.a - 2)) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.b - 2)) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b24.c - 2)) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b24.d - 2)) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    kernel::avx512f::vnni::gemv_2bit_s8s8_fp32<float, 48, MTILE>(
        {(uint8_t*)A.data(), scalea.data(), nullptr, k, blks},
        {nullptr, (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 2, n}, Cf32.data(), n, k,
        kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_2bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);

    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 4) {
        auto b24 = b2[(i * n + j) / 4];
        if (iasym) {
          Bf32[(i)*n + j + 0] = (int(b24.a - 2) - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (int(b24.b - 2) - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (int(b24.c - 2) - bzp[bid * n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (int(b24.d - 2) - bzp[bid * n + j + 3]) * scaleb[bid * n + j + 3];
        } else {
          Bf32[(i)*n + j + 0] = (int(b24.a - 2)) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (int(b24.b - 2)) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (int(b24.c - 2)) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (int(b24.d - 2)) * scaleb[bid * n + j + 3];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{nullptr, (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 2,
                               n};
    kernel::avx512f::gemv_2bit_fp32_fp32<float, 48, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache,
                                                           CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_3bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s3_s8(b2.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 4) {
        if (iasym) {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0] - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1] - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2] - bzp[bid * n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3] - bzp[bid * n + j + 3]) * scaleb[bid * n + j + 3];
        } else {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3]) * scaleb[bid * n + j + 3];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        nullptr, (uint8_t*)b2.data(), (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr, 2, n};
    kernel::avx512f::gemv_3bit_fp32_fp32<float, 48, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache,
                                                           CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_3bit_u8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j]) - azp[bid]) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s3_s8(b2.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        nullptr, (uint8_t*)b2.data(), (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr, 2, n};
    kernel::avx512f::vnni::gemv_3bit_u8s8_fp32<float, 48, MTILE>({A.data(), scalea.data(), azp.data(), k, blks}, B,
                                                                 Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_3bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> A(MTILE * k);
    fill_buffer_randn(A.data(), A.size(), int8_t(0), int8_t(127));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j])) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s3_s8(b2.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        nullptr, (uint8_t*)b2.data(), (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr, 2, n};
    kernel::avx512f::vnni::gemv_3bit_s8s8_fp32<float, 48, MTILE>({(uint8_t*)A.data(), scalea.data(), nullptr, k, blks},
                                                                 B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_5bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-16), int8_t(15));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s5_s8(b4.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 4) {
        if (iasym) {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0] - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1] - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2] - bzp[bid * n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3] - bzp[bid * n + j + 3]) * scaleb[bid * n + j + 3];
        } else {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3]) * scaleb[bid * n + j + 3];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        (uint8_t*)b4.data(), nullptr, (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr, 5, n};
    kernel::avx512f::gemv_5bit_fp32_fp32<float, 48, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache,
                                                           CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_5bit_u8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-16), int8_t(15));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j]) - azp[bid]) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s5_s8(b4.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        (uint8_t*)b4.data(), nullptr, (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr, 2, n};
    kernel::avx512f::vnni::gemv_5bit_u8s8_fp32<float, 48, MTILE>({A.data(), scalea.data(), azp.data(), k, blks}, B,
                                                                 Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_5bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-16), int8_t(15));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> A(MTILE * k);
    fill_buffer_randn(A.data(), A.size(), int8_t(0), int8_t(127));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j])) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s5_s8(b4.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        (uint8_t*)b4.data(), nullptr, (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr, 5, n};
    kernel::avx512f::vnni::gemv_5bit_s8s8_fp32<float, 48, MTILE>({(uint8_t*)A.data(), scalea.data(), nullptr, k, blks},
                                                                 B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_6bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-32), int8_t(31));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s6_s8(b4.data(), b2.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 4) {
        if (iasym) {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0] - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1] - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2] - bzp[bid * n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3] - bzp[bid * n + j + 3]) * scaleb[bid * n + j + 3];
        } else {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3]) * scaleb[bid * n + j + 3];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        (uint8_t*)b4.data(), (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 6, n};
    kernel::avx512f::gemv_6bit_fp32_fp32<float, 48, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache,
                                                           CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_6bit_u8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-32), int8_t(31));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j]) - azp[bid]) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s6_s8(b4.data(), b2.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        (uint8_t*)b4.data(), (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 6, n};
    kernel::avx512f::vnni::gemv_6bit_u8s8_fp32<float, 48, MTILE>({A.data(), scalea.data(), azp.data(), k, blks}, B,
                                                                 Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_6bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-32), int8_t(31));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> A(MTILE * k);
    fill_buffer_randn(A.data(), A.size(), int8_t(0), int8_t(127));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j])) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s6_s8(b4.data(), b2.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        (uint8_t*)b4.data(), (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 5, n};
    kernel::avx512f::vnni::gemv_6bit_s8s8_fp32<float, 48, MTILE>({(uint8_t*)A.data(), scalea.data(), nullptr, k, blks},
                                                                 B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
UT_avx512_gemv sUT_avx512_gemv;
#endif
#endif

#if CompileAVX2()
class UT_avx2_decompress_s6_s8 {
 public:
  UT_avx2_decompress_s6_s8() {
    UT_START();
    CheckISA(AVX2);
    ut<1, 24>(32);
    ut<4, 24>(32);
    ut<1, 24>(32, true);
    ut<2, 24>(32, true);
    ut<4, 24>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr FullRange = 1 << (6 - 1);
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, row, col, blocksize);
    std::vector<utils::bit4x2> s4_wei(row * col / 2);
    avector<utils::bit2x4> s2_wei(row * col / 4);

    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = 8;
    assert(blocksize % 8 == 0);
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-FullRange), int8_t(FullRange - 1));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-FullRange), int8_t(FullRange - 1));

    for (int i = 0; i < col * row; i += 4) {
      memcpy(&s8_ref[i], &s8_wei[i], 4 * sizeof(int8_t));
      s4_wei[i / 2].x = (s8_wei[i + 0] + FullRange) & 0xf;
      s4_wei[i / 2].y = (s8_wei[i + 1] + FullRange) & 0xf;
      s4_wei[i / 2 + 1].x = (s8_wei[i + 2] + FullRange) & 0xf;
      s4_wei[i / 2 + 1].y = (s8_wei[i + 3] + FullRange) & 0xf;

      s2_wei[i / 4].a = ((s8_wei[i + 0] + FullRange) & 0x30) >> 4;
      s2_wei[i / 4].b = ((s8_wei[i + 1] + FullRange) & 0x30) >> 4;
      s2_wei[i / 4].c = ((s8_wei[i + 2] + FullRange) & 0x30) >> 4;
      s2_wei[i / 4].d = ((s8_wei[i + 3] + FullRange) & 0x30) >> 4;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx2::decompress_kblock_s6_s8<PackRow, NTILE>(s4_wei.data(), s2_wei.data(), isasym ? zp.data() : nullptr,
                                                          rev.data(), blocksize, NTILE, 0, 0, row_offset, NTILE, cache,
                                                          CacheSize);
    kernel::avx2::decompress_kblock_s6_s8<PackRow, NTILE>(
        s4_wei.data() + row_offset * NTILE / 2, s2_wei.data() + row_offset * NTILE / 4, isasym ? zp.data() : nullptr,
        rev.data() + row_offset * NTILE, blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx2_decompress_s6_s8 sUT_avx2_decompress_s6_s8;
#endif

class UT_avx2_decompress_s7_s8 {
 public:
  UT_avx2_decompress_s7_s8() {
    UT_START();
    CheckISA(AVX2);
    ut<1, 24>(32);
    ut<4, 24>(32);
    ut<1, 24>(32, true);
    ut<2, 24>(32, true);
    ut<4, 24>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, row, col, blocksize);
    std::vector<utils::bit4x2> s4_wei(row * col / 2);
    avector<utils::bit1x8> s1_wei(row * col / 8);
    avector<utils::bit2x4> s2_wei(row * col / 4);

    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = 8;
    assert(blocksize % 8 == 0);
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-64), int8_t(63));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-64), int8_t(63));

    for (int i = 0; i < col * row; i += 8) {
      memcpy(&s8_ref[i], &s8_wei[i], 8 * sizeof(int8_t));
      s4_wei[i / 2].x = (s8_wei[i + 0] + 64) & 0xf;
      s4_wei[i / 2].y = (s8_wei[i + 1] + 64) & 0xf;
      s4_wei[i / 2 + 1].x = (s8_wei[i + 2] + 64) & 0xf;
      s4_wei[i / 2 + 1].y = (s8_wei[i + 3] + 64) & 0xf;
      s4_wei[i / 2 + 2].x = (s8_wei[i + 4] + 64) & 0xf;
      s4_wei[i / 2 + 2].y = (s8_wei[i + 5] + 64) & 0xf;
      s4_wei[i / 2 + 3].x = (s8_wei[i + 6] + 64) & 0xf;
      s4_wei[i / 2 + 3].y = (s8_wei[i + 7] + 64) & 0xf;

      s2_wei[i / 4].a = ((s8_wei[i + 0] + 64) & 0x30) >> 4;
      s2_wei[i / 4].b = ((s8_wei[i + 1] + 64) & 0x30) >> 4;
      s2_wei[i / 4].c = ((s8_wei[i + 2] + 64) & 0x30) >> 4;
      s2_wei[i / 4].d = ((s8_wei[i + 3] + 64) & 0x30) >> 4;
      s2_wei[i / 4 + 1].a = ((s8_wei[i + 4] + 64) & 0x30) >> 4;
      s2_wei[i / 4 + 1].b = ((s8_wei[i + 5] + 64) & 0x30) >> 4;
      s2_wei[i / 4 + 1].c = ((s8_wei[i + 6] + 64) & 0x30) >> 4;
      s2_wei[i / 4 + 1].d = ((s8_wei[i + 7] + 64) & 0x30) >> 4;

      s1_wei[i / 8].a = ((s8_wei[i + 0] + 64) & 0x40) >> 6;
      s1_wei[i / 8].b = ((s8_wei[i + 1] + 64) & 0x40) >> 6;
      s1_wei[i / 8].c = ((s8_wei[i + 2] + 64) & 0x40) >> 6;
      s1_wei[i / 8].d = ((s8_wei[i + 3] + 64) & 0x40) >> 6;
      s1_wei[i / 8].e = ((s8_wei[i + 4] + 64) & 0x40) >> 6;
      s1_wei[i / 8].f = ((s8_wei[i + 5] + 64) & 0x40) >> 6;
      s1_wei[i / 8].g = ((s8_wei[i + 6] + 64) & 0x40) >> 6;
      s1_wei[i / 8].h = ((s8_wei[i + 7] + 64) & 0x40) >> 6;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx2::decompress_kblock_s7_s8<PackRow, NTILE>(s4_wei.data(), s2_wei.data(), s1_wei.data(),
                                                          isasym ? zp.data() : nullptr, rev.data(), blocksize, NTILE, 0,
                                                          0, row_offset, NTILE, cache, CacheSize);
    kernel::avx2::decompress_kblock_s7_s8<PackRow, NTILE>(
        s4_wei.data() + row_offset * NTILE / 2, s2_wei.data() + row_offset * NTILE / 4,
        s1_wei.data() + row_offset * NTILE / 8, isasym ? zp.data() : nullptr, rev.data() + row_offset * NTILE,
        blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx2_decompress_s7_s8 sUT_avx2_decompress_s7_s8;
#endif

class UT_avx2_decompress_s5_s8 {
 public:
  UT_avx2_decompress_s5_s8() {
    UT_START();
    CheckISA(AVX2);
    ut<1, 24>(32);
    ut<4, 24>(32);
    ut<1, 24>(32, true);
    ut<2, 24>(32, true);
    ut<4, 24>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, row, col, blocksize);
    std::vector<utils::bit4x2> s4_wei(row * col / 2);
    avector<utils::bit1x8> s1_wei(row * col / 8);

    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = 8;
    assert(blocksize % 8 == 0);
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-16), int8_t(15));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-16), int8_t(15));

    for (int i = 0; i < col * row; i += 8) {
      memcpy(&s8_ref[i], &s8_wei[i], 8 * sizeof(int8_t));
      s4_wei[i / 2].x = (s8_wei[i + 0] + 16) & 0xf;
      s4_wei[i / 2].y = (s8_wei[i + 1] + 16) & 0xf;
      s4_wei[i / 2 + 1].x = (s8_wei[i + 2] + 16) & 0xf;
      s4_wei[i / 2 + 1].y = (s8_wei[i + 3] + 16) & 0xf;
      s4_wei[i / 2 + 2].x = (s8_wei[i + 4] + 16) & 0xf;
      s4_wei[i / 2 + 2].y = (s8_wei[i + 5] + 16) & 0xf;
      s4_wei[i / 2 + 3].x = (s8_wei[i + 6] + 16) & 0xf;
      s4_wei[i / 2 + 3].y = (s8_wei[i + 7] + 16) & 0xf;

      s1_wei[i / 8].a = ((s8_wei[i + 0] + 16) & 0x10) >> 4;
      s1_wei[i / 8].b = ((s8_wei[i + 1] + 16) & 0x10) >> 4;
      s1_wei[i / 8].c = ((s8_wei[i + 2] + 16) & 0x10) >> 4;
      s1_wei[i / 8].d = ((s8_wei[i + 3] + 16) & 0x10) >> 4;
      s1_wei[i / 8].e = ((s8_wei[i + 4] + 16) & 0x10) >> 4;
      s1_wei[i / 8].f = ((s8_wei[i + 5] + 16) & 0x10) >> 4;
      s1_wei[i / 8].g = ((s8_wei[i + 6] + 16) & 0x10) >> 4;
      s1_wei[i / 8].h = ((s8_wei[i + 7] + 16) & 0x10) >> 4;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx2::decompress_kblock_s5_s8<PackRow, NTILE>(s4_wei.data(), s1_wei.data(), isasym ? zp.data() : nullptr,
                                                          rev.data(), blocksize, NTILE, 0, 0, row_offset, NTILE, cache,
                                                          CacheSize);
    kernel::avx2::decompress_kblock_s5_s8<PackRow, NTILE>(
        s4_wei.data() + row_offset * NTILE / 2, s1_wei.data() + row_offset * NTILE / 8, isasym ? zp.data() : nullptr,
        rev.data() + row_offset * NTILE, blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx2_decompress_s5_s8 sUT_avx2_decompress_s5_s8;
#endif

class UT_avx2_decompress_s4_s8 {
 public:
  UT_avx2_decompress_s4_s8() {
    UT_START();
    CheckISA(AVX2);
    ut<1, 24>(32);
    ut<4, 24>(32);
    ut<1, 24>(32, true);
    ut<2, 24>(32, true);
    ut<4, 24>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, row, col, blocksize);
    std::vector<utils::int4x2> s4_wei(row * col / 2);
    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = PackRow;
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-8), int8_t(7));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-8), int8_t(7));

    for (int i = 0; i < col * row; i += 2) {
      s8_ref[i] = s8_wei[i];
      s8_ref[i + 1] = s8_wei[i + 1];
      s4_wei[i / 2].x = utils::int4x2::convert(s8_wei[i]) + 8;
      s4_wei[i / 2].y = utils::int4x2::convert(s8_wei[i + 1]) + 8;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx2::decompress_kblock_s4_s8<PackRow, NTILE>(s4_wei.data(), isasym ? zp.data() : nullptr, rev.data(),
                                                          blocksize, NTILE, 0, 0, row_offset, NTILE, cache, CacheSize);
    kernel::avx2::decompress_kblock_s4_s8<PackRow, NTILE>(
        s4_wei.data() + row_offset * NTILE / 2, isasym ? zp.data() : nullptr, rev.data() + row_offset * NTILE,
        blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx2_decompress_s4_s8 sUT_avx2_decompress_s4_s8;
#endif

class UT_avx2_decompress_s3_s8 {
 public:
  UT_avx2_decompress_s3_s8() {
    UT_START();
    CheckISA(AVX2);
    ut<1, 24>(32);
    ut<4, 24>(32);
    ut<1, 24>(32, true);
    ut<2, 24>(32, true);
    ut<4, 24>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, row, col, blocksize);
    std::vector<utils::bit2x4> s2_wei(row * col / 4);
    avector<utils::bit1x8> s1_wei(row * col / 8);

    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = 8;
    assert(blocksize % 8 == 0);
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-4), int8_t(3));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-4), int8_t(3));

    for (int i = 0; i < col * row; i += 8) {
      memcpy(&s8_ref[i], &s8_wei[i], 8 * sizeof(int8_t));
      s2_wei[i / 4].a = (s8_wei[i + 0] + 4) & 0x3;
      s2_wei[i / 4].b = (s8_wei[i + 1] + 4) & 0x3;
      s2_wei[i / 4].c = (s8_wei[i + 2] + 4) & 0x3;
      s2_wei[i / 4].d = (s8_wei[i + 3] + 4) & 0x3;

      s2_wei[i / 4 + 1].a = (s8_wei[i + 4] + 4) & 0x3;
      s2_wei[i / 4 + 1].b = (s8_wei[i + 5] + 4) & 0x3;
      s2_wei[i / 4 + 1].c = (s8_wei[i + 6] + 4) & 0x3;
      s2_wei[i / 4 + 1].d = (s8_wei[i + 7] + 4) & 0x3;

      s1_wei[i / 8].a = ((s8_wei[i + 0] + 4) & 0x4) >> 2;
      s1_wei[i / 8].b = ((s8_wei[i + 1] + 4) & 0x4) >> 2;
      s1_wei[i / 8].c = ((s8_wei[i + 2] + 4) & 0x4) >> 2;
      s1_wei[i / 8].d = ((s8_wei[i + 3] + 4) & 0x4) >> 2;
      s1_wei[i / 8].e = ((s8_wei[i + 4] + 4) & 0x4) >> 2;
      s1_wei[i / 8].f = ((s8_wei[i + 5] + 4) & 0x4) >> 2;
      s1_wei[i / 8].g = ((s8_wei[i + 6] + 4) & 0x4) >> 2;
      s1_wei[i / 8].h = ((s8_wei[i + 7] + 4) & 0x4) >> 2;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx2::decompress_kblock_s3_s8<PackRow, NTILE>(s2_wei.data(), s1_wei.data(), isasym ? zp.data() : nullptr,
                                                          rev.data(), blocksize, NTILE, 0, 0, row_offset, NTILE, cache,
                                                          CacheSize);
    kernel::avx2::decompress_kblock_s3_s8<PackRow, NTILE>(
        s2_wei.data() + row_offset * NTILE / 4, s1_wei.data() + row_offset * NTILE / 8, isasym ? zp.data() : nullptr,
        rev.data() + row_offset * NTILE, blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx2_decompress_s3_s8 sUT_avx2_decompress_s3_s8;
#endif

class UT_avx2_decompress_s1_s8 {
 public:
  UT_avx2_decompress_s1_s8() {
    UT_START();
    CheckISA(AVX2);
    ut<1, 24>(32);
    ut<4, 24>(32);
    ut<1, 24>(32, true);
    ut<2, 24>(32, true);
    ut<4, 24>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, row, col, blocksize);
    avector<utils::bit1x8> s1_wei(row * col / 8);

    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = 8;
    assert(blocksize % 8 == 0);
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-1), int8_t(0));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-1), int8_t(0));

    for (int i = 0; i < col * row; i += 8) {
      memcpy(&s8_ref[i], &s8_wei[i], 8 * sizeof(int8_t));
      s1_wei[i / 8].a = s8_wei[i + 0] + 1;
      s1_wei[i / 8].b = s8_wei[i + 1] + 1;
      s1_wei[i / 8].c = s8_wei[i + 2] + 1;
      s1_wei[i / 8].d = s8_wei[i + 3] + 1;
      s1_wei[i / 8].e = s8_wei[i + 4] + 1;
      s1_wei[i / 8].f = s8_wei[i + 5] + 1;
      s1_wei[i / 8].g = s8_wei[i + 6] + 1;
      s1_wei[i / 8].h = s8_wei[i + 7] + 1;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx2::decompress_kblock_s1_s8<PackRow, NTILE>(s1_wei.data(), isasym ? zp.data() : nullptr, rev.data(),
                                                          blocksize, NTILE, 0, 0, row_offset, NTILE, cache, CacheSize);
    kernel::avx2::decompress_kblock_s1_s8<PackRow, NTILE>(
        s1_wei.data() + row_offset * NTILE / 8, isasym ? zp.data() : nullptr, rev.data() + row_offset * NTILE,
        blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx2_decompress_s1_s8 sUT_avx2_decompress_s1_s8;
#endif

class UT_avx2_decompress_s2_s8 {
 public:
  UT_avx2_decompress_s2_s8() {
    UT_START();
    CheckISA(AVX2);
    ut<1, 24>(32);
    ut<4, 24>(32);
    ut<1, 24>(32, true);
    ut<2, 24>(32, true);
    ut<4, 24>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, PackRow, row, col, blocksize, isasym);
    std::vector<utils::bit2x4> s2_wei(row * col / 4);

    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = blocksize;
    assert(blocksize % 8 == 0);
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-2), int8_t(1));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-2), int8_t(1));

    for (int i = 0; i < col * row; i += 4) {
      memcpy(&s8_ref[i], &s8_wei[i], 4 * sizeof(int8_t));
      s2_wei[i / 4].a = (s8_wei[i + 0] + 2) & 0x3;
      s2_wei[i / 4].b = (s8_wei[i + 1] + 2) & 0x3;
      s2_wei[i / 4].c = (s8_wei[i + 2] + 2) & 0x3;
      s2_wei[i / 4].d = (s8_wei[i + 3] + 2) & 0x3;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx2::decompress_kblock_s2_s8<PackRow, NTILE>(s2_wei.data(), isasym ? zp.data() : nullptr, rev.data(),
                                                          blocksize, NTILE, 0, 0, row_offset, NTILE, cache, CacheSize);
    kernel::avx2::decompress_kblock_s2_s8<PackRow, NTILE>(
        s2_wei.data() + row_offset * NTILE / 4, isasym ? zp.data() : nullptr, rev.data() + row_offset * NTILE,
        blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx2_decompress_s2_s8 sUT_avx2_decompress_s2_s8;
#endif

class UT_avx2_decompress_s4_fp {
 public:
  UT_avx2_decompress_s4_fp() {
    UT_START();
    CheckISA(AVX2);
    ut<4, 24, utils::bf16>(32);
    ut<1, 24, float>(32);
    ut<2, 24, float>(32);
    ut<4, 24, float>(32);
    ut<4, 24, utils::bf16, utils::bf16>(32);
    ut<1, 24, float>(32, true);
    ut<2, 24, float>(32, true);
    ut<4, 24, float>(32, true);
    ut<4, 24, utils::bf16>(32, true);
    ut<4, 24, utils::bf16, utils::bf16>(32, true);
  }

  template <int PackRow, int NTILE, typename T, typename ScaleT = float>
  void ut(int blocksize, bool isasym = false) {
    auto dst_dtype = bestla_dtype<T>;
    auto scale_dtype = bestla_dtype<ScaleT>;
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d Asym:%d Pack:%d %s %s\n", __FUNCTION__, row, col, blocksize, isasym, PackRow,
           utils::bestla_dtype_str(dst_dtype), bestla_dtype_str(scale_dtype));
    std::vector<utils::int4x2> s4_wei(row * col / 2);
    std::vector<int8_t> s8_wei(col * row);
    std::vector<T> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = PackRow;
    std::vector<int8_t> zp(col * blks);
    avector<ScaleT> scale(col * blks);
    fill_buffer_randn(scale.data(), scale.size(), ScaleT(0.01f), ScaleT(0.03f));
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-8), int8_t(7));
    std::vector<T> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-8), int8_t(7));

    for (int i = 0; i < col * row; i += 2) {
      s8_ref[i] = float(s8_wei[i]);
      s8_ref[i + 1] = float(s8_wei[i + 1]);
      s4_wei[i / 2].x = utils::int4x2::convert(s8_wei[i]) + 8;
      s4_wei[i / 2].y = utils::int4x2::convert(s8_wei[i + 1]) + 8;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          int corr_offset = i / blocksize * NTILE + j;
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] = float(s8_ref[i * NTILE + j * PackRow + ip]) - float(zp[corr_offset]);
          }
        }
      }
    }

    for (int i = 0; i < row; i += PackRow) {
      for (int j = 0; j < NTILE; j++) {
        int corr_offset = i / blocksize * NTILE + j;
        for (int ip = 0; ip < PackRow; ip++) {
          s8_ref[i * NTILE + j * PackRow + ip] =
              float(s8_ref[i * NTILE + j * PackRow + ip]) * float(scale[corr_offset]);
        }
      }
    }
    kernel::avx2::decompress_kblock_s4_fp<PackRow, NTILE>(s4_wei.data(), rev.data(), row_offset, NTILE, scale.data(),
                                                          scale_dtype, isasym ? zp.data() : nullptr, 0, 0, blocksize,
                                                          NTILE, cache, CacheSize);
    kernel::avx2::decompress_kblock_s4_fp<PackRow, NTILE>(
        s4_wei.data() + row_offset * NTILE / 2, rev.data() + row_offset * NTILE, row - row_offset, NTILE, scale.data(),
        scale_dtype, isasym ? zp.data() : nullptr, row_offset, 0, blocksize, NTILE, cache, CacheSize);
    float err = get_ut_err(dst_dtype);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), T(err));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx2_decompress_s4_fp sUT_avx2_decompress_s4_fp;
#endif

// s<N>_fp share the same process: s<N>->s8->fp, so it's not necessary to test all s<N>_fp cases.
// test s<N>_s8 is just fine.
#endif

#if CompileAVXVNNI()
class UT_avx2_gemv {
 public:
  UT_avx2_gemv() {
    UT_START();
    CheckISA(AVX2);
    ut_1bit_fp32<1>(24, 128, 32, true);
    ut_1bit_fp32<1>(24, 128, 32, false);
    ut_1bit_fp32<4>(24, 128, 32, true);
    ut_1bit_fp32<4>(24, 128, 32, false);

    ut_7bit_fp32<1>(24, 128, 32, true);
    ut_7bit_fp32<1>(24, 128, 32, false);
    ut_7bit_fp32<4>(24, 128, 32, true);
    ut_7bit_fp32<4>(24, 128, 32, false);

    ut_6bit_fp32<1>(24, 128, 32, true);
    ut_6bit_fp32<1>(24, 128, 32, false);
    ut_6bit_fp32<4>(24, 128, 32, true);
    ut_6bit_fp32<4>(24, 128, 32, false);

    ut_5bit_fp32<1>(24, 128, 32, true);
    ut_5bit_fp32<1>(24, 128, 32, false);
    ut_5bit_fp32<4>(24, 128, 32, true);
    ut_5bit_fp32<4>(24, 128, 32, false);

    ut_4bit_fp32<1>(24, 128, 32, true);
    ut_4bit_fp32<1>(24, 128, 32, false);
    ut_4bit_fp32<4>(24, 128, 32, true);
    ut_4bit_fp32<4>(24, 128, 32, false);

    ut_3bit_fp32<1>(24, 128, 32, true);
    ut_3bit_fp32<1>(24, 128, 32, false);
    ut_3bit_fp32<4>(24, 128, 32, true);
    ut_3bit_fp32<4>(24, 128, 32, false);

    ut_2bit_fp32<1>(24, 128, 32, true);
    ut_2bit_fp32<1>(24, 128, 32, false);
    ut_2bit_fp32<4>(24, 128, 32, true);
    ut_2bit_fp32<4>(24, 128, 32, false);

    CheckISA(AVX_VNNI);
    ut_1bit_u8s8<1>(24, 128, 32, true);
    ut_1bit_u8s8<1>(24, 128, 32, false);
    ut_1bit_u8s8<4>(24, 128, 32, true);
    ut_1bit_u8s8<4>(24, 128, 32, false);

    ut_1bit_s8s8<1>(24, 128, 32, true);
    ut_1bit_s8s8<1>(24, 128, 32, false);
    ut_1bit_s8s8<4>(24, 128, 32, true);
    ut_1bit_s8s8<4>(24, 128, 32, false);

    ut_7bit_u8s8<1>(24, 128, 32, true);
    ut_7bit_u8s8<1>(24, 128, 32, false);
    ut_7bit_u8s8<4>(24, 128, 32, true);
    ut_7bit_u8s8<4>(24, 128, 32, false);

    ut_7bit_s8s8<1>(24, 128, 32, true);
    ut_7bit_s8s8<1>(24, 128, 32, false);
    ut_7bit_s8s8<4>(24, 128, 32, true);
    ut_7bit_s8s8<4>(24, 128, 32, false);

    ut_4bit<1>(24, 128, 32, true);
    ut_4bit<1>(24, 128, 32, false);
    ut_4bit<4>(24, 128, 32, false);
    ut_4bit<4>(24, 128, 32, true);

    ut_4bit_s8s8<1>(24, 128, 32, true);
    ut_4bit_s8s8<1>(24, 128, 32, false);
    ut_4bit_s8s8<4>(24, 128, 32, true);
    ut_4bit_s8s8<4>(24, 128, 32, false);

    ut_2bit<1>(24, 128, 32, true);
    ut_2bit<1>(24, 128, 32, false);
    ut_2bit<4>(24, 128, 32, true);
    ut_2bit<4>(24, 128, 32, false);

    ut_2bit_s8s8<1>(24, 128, 32, true);
    ut_2bit_s8s8<1>(24, 128, 32, false);
    ut_2bit_s8s8<4>(24, 128, 32, true);
    ut_2bit_s8s8<4>(24, 128, 32, false);

    ut_3bit_u8s8<1>(24, 128, 32, true);
    ut_3bit_u8s8<1>(24, 128, 32, false);
    ut_3bit_u8s8<4>(24, 128, 32, true);
    ut_3bit_u8s8<4>(24, 128, 32, false);

    ut_3bit_s8s8<1>(24, 128, 32, true);
    ut_3bit_s8s8<1>(24, 128, 32, false);
    ut_3bit_s8s8<4>(24, 128, 32, true);
    ut_3bit_s8s8<4>(24, 128, 32, false);

    ut_6bit_u8s8<1>(24, 128, 32, true);
    ut_6bit_u8s8<1>(24, 128, 32, false);
    ut_6bit_u8s8<4>(24, 128, 32, true);
    ut_6bit_u8s8<4>(24, 128, 32, false);

    ut_6bit_s8s8<1>(24, 128, 32, true);
    ut_6bit_s8s8<1>(24, 128, 32, false);
    ut_6bit_s8s8<4>(24, 128, 32, true);
    ut_6bit_s8s8<4>(24, 128, 32, false);

    ut_5bit_u8s8<1>(24, 128, 32, true);
    ut_5bit_u8s8<1>(24, 128, 32, false);
    ut_5bit_u8s8<4>(24, 128, 32, true);
    ut_5bit_u8s8<4>(24, 128, 32, false);

    ut_5bit_s8s8<1>(24, 128, 32, true);
    ut_5bit_s8s8<1>(24, 128, 32, false);
    ut_5bit_s8s8<4>(24, 128, 32, true);
    ut_5bit_s8s8<4>(24, 128, 32, false);
  }

  template <int MTILE>
  void ut_4bit(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b2(n * k / 2);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j]) - azp[bid]) * scalea[bid];
        }
      }
    }

    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        auto b24 = b2[(i * n + j * 4) / 2];
        auto b42 = b2[(i * n + j * 4 + 2) / 2];
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b24.x - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.y - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b42.x - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b42.y - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b24.x - 8)) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.y - 8)) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b42.x - 8)) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b42.y - 8)) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{(uint8_t*)b2.data(),          nullptr, nullptr, scaleb.data(),
                               iasym ? bzp.data() : nullptr, 2,       n};
    kernel::avx2::vnni::gemv_4bit_u8s8_fp32<float, 24, MTILE>({A.data(), scalea.data(), azp.data(), k, blks}, B,
                                                              Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_4bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b2(n * k / 2);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);

    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 2) {
        auto b24 = b2[(i * n + j) / 2];
        if (iasym) {
          Bf32[(i)*n + j + 0] = (int(b24.x - 8) - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (int(b24.y - 8) - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
        } else {
          Bf32[(i)*n + j + 0] = (int(b24.x - 8)) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (int(b24.y - 8)) * scaleb[bid * n + j + 1];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{(uint8_t*)b2.data(),          nullptr, nullptr, scaleb.data(),
                               iasym ? bzp.data() : nullptr, 4,       n};
    kernel::avx2::gemv_4bit_fp32_fp32<float, 24, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_4bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b2(n * k / 2);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<int8_t> A(MTILE * k);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(A.data(), A.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j])) * scalea[bid];
        }
      }
    }

    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        auto b24 = b2[(i * n + j * 4) / 2];
        auto b42 = b2[(i * n + j * 4 + 2) / 2];
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b24.x - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.y - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b42.x - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b42.y - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b24.x - 8)) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.y - 8)) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b42.x - 8)) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b42.y - 8)) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{(uint8_t*)b2.data(),          nullptr, nullptr, scaleb.data(),
                               iasym ? bzp.data() : nullptr, 2,       n};
    kernel::avx2::vnni::gemv_4bit_s8s8_fp32<float, 24, MTILE>({(uint8_t*)A.data(), scalea.data(), nullptr, k, blks}, B,
                                                              Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_2bit(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-2), int8_t(1));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int i = 0; i < MTILE; i++) {
      for (int j = 0; j < k; j++) {
        Af32[i * k + j] = (int(A[i * k + j]) - azp[i * blks + j / kblock]) * scalea[i * blks + j / kblock];
      }
    }
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        auto b24 = b2[(i * n + j * 4) / 4];
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b24.a - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.b - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b24.c - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b24.d - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b24.a - 2)) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.b - 2)) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b24.c - 2)) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b24.d - 2)) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    kernel::avx2::vnni::gemv_2bit_u8s8_fp32<float, 24, MTILE>(
        {A.data(), scalea.data(), azp.data(), k, blks},
        {nullptr, (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 2, n}, Cf32.data(), n, k,
        kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_2bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<int8_t> A(MTILE * k);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(A.data(), A.size(), int8_t(0), int8_t(127));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-2), int8_t(1));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int i = 0; i < MTILE; i++) {
      for (int j = 0; j < k; j++) {
        Af32[i * k + j] = (int(A[i * k + j])) * scalea[i * blks + j / kblock];
      }
    }
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        auto b24 = b2[(i * n + j * 4) / 4];
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b24.a - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.b - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b24.c - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b24.d - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b24.a - 2)) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.b - 2)) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b24.c - 2)) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b24.d - 2)) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    kernel::avx2::vnni::gemv_2bit_s8s8_fp32<float, 24, MTILE>(
        {(uint8_t*)A.data(), scalea.data(), nullptr, k, blks},
        {nullptr, (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 2, n}, Cf32.data(), n, k,
        kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_2bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);

    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 4) {
        auto b24 = b2[(i * n + j) / 4];
        if (iasym) {
          Bf32[(i)*n + j + 0] = (int(b24.a - 2) - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (int(b24.b - 2) - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (int(b24.c - 2) - bzp[bid * n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (int(b24.d - 2) - bzp[bid * n + j + 3]) * scaleb[bid * n + j + 3];
        } else {
          Bf32[(i)*n + j + 0] = (int(b24.a - 2)) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (int(b24.b - 2)) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (int(b24.c - 2)) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (int(b24.d - 2)) * scaleb[bid * n + j + 3];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{nullptr, (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 2,
                               n};
    kernel::avx2::gemv_2bit_fp32_fp32<float, 24, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_3bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s3_s8(b2.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 4) {
        if (iasym) {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0] - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1] - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2] - bzp[bid * n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3] - bzp[bid * n + j + 3]) * scaleb[bid * n + j + 3];
        } else {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3]) * scaleb[bid * n + j + 3];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        nullptr, (uint8_t*)b2.data(), (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr, 2, n};
    kernel::avx2::gemv_3bit_fp32_fp32<float, 24, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_1bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-1), int8_t(0));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s1_s8(b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 4) {
        if (iasym) {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0] - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1] - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2] - bzp[bid * n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3] - bzp[bid * n + j + 3]) * scaleb[bid * n + j + 3];
        } else {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3]) * scaleb[bid * n + j + 3];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{nullptr, nullptr, (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr,
                               1,       n};
    kernel::avx2::gemv_1bit_fp32_fp32<float, 24, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_1bit_u8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-1), int8_t(0));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j]) - azp[bid]) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s1_s8(b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{nullptr, nullptr, (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr,
                               2,       n};
    kernel::avx2::vnni::gemv_1bit_u8s8_fp32<float, 24, MTILE>({A.data(), scalea.data(), azp.data(), k, blks}, B,
                                                              Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_1bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-1), int8_t(0));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> A(MTILE * k);
    fill_buffer_randn(A.data(), A.size(), int8_t(0), int8_t(127));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j])) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s1_s8(b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{nullptr, nullptr, (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr,
                               2,       n};
    kernel::avx2::vnni::gemv_1bit_s8s8_fp32<float, 24, MTILE>({(uint8_t*)A.data(), scalea.data(), nullptr, k, blks}, B,
                                                              Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_3bit_u8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j]) - azp[bid]) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s3_s8(b2.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        nullptr, (uint8_t*)b2.data(), (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr, 2, n};
    kernel::avx2::vnni::gemv_3bit_u8s8_fp32<float, 24, MTILE>({A.data(), scalea.data(), azp.data(), k, blks}, B,
                                                              Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_3bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> A(MTILE * k);
    fill_buffer_randn(A.data(), A.size(), int8_t(0), int8_t(127));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j])) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s3_s8(b2.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        nullptr, (uint8_t*)b2.data(), (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr, 2, n};
    kernel::avx2::vnni::gemv_3bit_s8s8_fp32<float, 24, MTILE>({(uint8_t*)A.data(), scalea.data(), nullptr, k, blks}, B,
                                                              Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_6bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-32), int8_t(31));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s6_s8(b4.data(), b2.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 4) {
        if (iasym) {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0] - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1] - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2] - bzp[bid * n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3] - bzp[bid * n + j + 3]) * scaleb[bid * n + j + 3];
        } else {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3]) * scaleb[bid * n + j + 3];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        (uint8_t*)b4.data(), (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 6, n};
    kernel::avx2::gemv_6bit_fp32_fp32<float, 24, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_6bit_u8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-32), int8_t(31));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j]) - azp[bid]) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s6_s8(b4.data(), b2.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        (uint8_t*)b4.data(), (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 6, n};
    kernel::avx2::vnni::gemv_6bit_u8s8_fp32<float, 24, MTILE>({A.data(), scalea.data(), azp.data(), k, blks}, B,
                                                              Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_6bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-32), int8_t(31));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> A(MTILE * k);
    fill_buffer_randn(A.data(), A.size(), int8_t(0), int8_t(127));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j])) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s6_s8(b4.data(), b2.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        (uint8_t*)b4.data(), (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 5, n};
    kernel::avx2::vnni::gemv_6bit_s8s8_fp32<float, 24, MTILE>({(uint8_t*)A.data(), scalea.data(), nullptr, k, blks}, B,
                                                              Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_5bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-16), int8_t(15));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s5_s8(b4.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 4) {
        if (iasym) {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0] - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1] - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2] - bzp[bid * n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3] - bzp[bid * n + j + 3]) * scaleb[bid * n + j + 3];
        } else {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3]) * scaleb[bid * n + j + 3];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        (uint8_t*)b4.data(), nullptr, (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr, 5, n};
    kernel::avx2::gemv_5bit_fp32_fp32<float, 24, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_5bit_u8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-16), int8_t(15));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j]) - azp[bid]) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s5_s8(b4.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        (uint8_t*)b4.data(), nullptr, (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr, 2, n};
    kernel::avx2::vnni::gemv_5bit_u8s8_fp32<float, 24, MTILE>({A.data(), scalea.data(), azp.data(), k, blks}, B,
                                                              Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_5bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-16), int8_t(15));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> A(MTILE * k);
    fill_buffer_randn(A.data(), A.size(), int8_t(0), int8_t(127));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j])) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s5_s8(b4.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{
        (uint8_t*)b4.data(), nullptr, (uint8_t*)b1.data(), scaleb.data(), iasym ? bzp.data() : nullptr, 5, n};
    kernel::avx2::vnni::gemv_5bit_s8s8_fp32<float, 24, MTILE>({(uint8_t*)A.data(), scalea.data(), nullptr, k, blks}, B,
                                                              Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_7bit_fp32(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit1x8> b1(n * k / 8);
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(Af32.data(), Af32.size(), -0.5f, 0.5f);
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-64), int8_t(63));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s7_s8(b4.data(), b2.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 1) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 4) {
        if (iasym) {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0] - bzp[bid * n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1] - bzp[bid * n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2] - bzp[bid * n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3] - bzp[bid * n + j + 3]) * scaleb[bid * n + j + 3];
        } else {
          Bf32[(i)*n + j + 0] = (b8[(i)*n + j + 0]) * scaleb[bid * n + j + 0];
          Bf32[(i)*n + j + 1] = (b8[(i)*n + j + 1]) * scaleb[bid * n + j + 1];
          Bf32[(i)*n + j + 2] = (b8[(i)*n + j + 2]) * scaleb[bid * n + j + 2];
          Bf32[(i)*n + j + 3] = (b8[(i)*n + j + 3]) * scaleb[bid * n + j + 3];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{(uint8_t*)b4.data(),
                               (uint8_t*)b2.data(),
                               (uint8_t*)b1.data(),
                               scaleb.data(),
                               iasym ? bzp.data() : nullptr,
                               7,
                               n};
    kernel::avx2::gemv_7bit_fp32_fp32<float, 24, MTILE>(Af32.data(), k, B, Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_7bit_u8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit2x4> b2(n * k / 4);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-64), int8_t(63));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<uint8_t> A(MTILE * k), azp(MTILE * blks);
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(128));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(50), uint8_t(64));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j]) - azp[bid]) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s7_s8(b4.data(), b2.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{(uint8_t*)b4.data(),
                               (uint8_t*)b2.data(),
                               (uint8_t*)b1.data(),
                               scaleb.data(),
                               iasym ? bzp.data() : nullptr,
                               7,
                               n};
    kernel::avx2::vnni::gemv_7bit_u8s8_fp32<float, 24, MTILE>({A.data(), scalea.data(), azp.data(), k, blks}, B,
                                                              Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  template <int MTILE>
  void ut_7bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s_%d: %d %d %d Asym:%d\n", __FUNCTION__, MTILE, n, k, kblock, iasym);
    int blks = k / kblock;
    avector<bit4x2> b4(n * k / 2);
    avector<bit2x4> b2(n * k / 4);
    avector<bit1x8> b1(n * k / 8);
    avector<float> scaleb(n * blks), scalea(MTILE * blks);
    avector<int8_t> bzp(n * blks);
    avector<float> Af32(MTILE * k), Bf32(n * k), Cf32(MTILE * n), Cref(MTILE * n);
    fill_buffer_randn((uint8_t*)b4.data(), b4.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn((uint8_t*)b1.data(), b1.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-64), int8_t(63));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    avector<int8_t> A(MTILE * k);
    fill_buffer_randn(A.data(), A.size(), int8_t(0), int8_t(127));
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);
    for (int im = 0; im < MTILE; im++) {
      for (int i = 0; i < k; i += 4) {
        int bid = i / kblock + im * blks;
        for (int j = 0; j < 4; j++) {
          Af32[im * k + i + j] = (int(A[im * k + i + j])) * scalea[bid];
        }
      }
    }

    avector<int8_t> b8(n * k);
    kernel::ref::decompress_s7_s8(b4.data(), b2.data(), b1.data(), b8.data(), b8.size(), cache, CacheSize);
    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < n; j += 1) {
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2]) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3]) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b8[i * n + j * 4 + 0])) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b8[i * n + j * 4 + 1])) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b8[i * n + j * 4 + 2])) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b8[i * n + j * 4 + 3])) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(MTILE, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    utils::GemvParamB<float> B{(uint8_t*)b4.data(),
                               (uint8_t*)b2.data(),
                               (uint8_t*)b1.data(),
                               scaleb.data(),
                               iasym ? bzp.data() : nullptr,
                               7,
                               n};
    kernel::avx2::vnni::gemv_7bit_s8s8_fp32<float, 24, MTILE>({(uint8_t*)A.data(), scalea.data(), nullptr, k, blks}, B,
                                                              Cf32.data(), n, k, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
UT_avx2_gemv sUT_avx2_gemv;
#endif
#endif

}  // namespace ut
}  // namespace bestla
