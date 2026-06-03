#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "common.hpp"
#include "../include/xpu_wrapper.hpp"

// Set ARK_S3_BENCH=1 to run only the int3 (S3) GEMV/GEMM throughput benchmarks and skip every other
// case (functional GEMM/WOQ tests here and the SDPA suite in main). Useful for quick perf sweeps.
inline bool ark_s3_bench_only() {
  const char* e = std::getenv("ARK_S3_BENCH");
  return e && e[0] && e[0] != '0';
}

#ifdef ARK_XPU
// ---------------------------------------------------------------------------------------------
// W10 int3 layout perf spike (Phase 0). Self-contained: a candidate "10 three-bit values per
// uint32 word" packing + GEMV that lives entirely here so we can measure throughput before
// productionizing. Unlike the dense S3 straddle blob (32 vals -> 3 words, fields cross word
// boundaries -> strided gather), W10 packs 10 vals into one word (30 bits used, 2 wasted) so:
//   * decode is branchless: (w >> 3*i) & 0x7,
//   * lane L reads word L -> 32 lanes read 32 contiguous words -> one coalesced 128B load.
// Scale-block alignment (Scheme A): each scale-block of `blocksize` values is padded to whole
// words, wpb = ceil(blocksize/10), so every word belongs to exactly one block (scale = sptr[w/wpb]).
// ---------------------------------------------------------------------------------------------
namespace w10_spike {

inline int words_per_block(int blocksize) { return (blocksize + 9) / 10; }

// Host packer: raw signed int3 weights row-major [k, n] -> per-column contiguous W10 word stream.
// Each block contributes wpb words; value stored as (q+4)&0x7 at bits [3*i, 3*i+3); padded slots = 4
// (which decodes to 0). Output layout: column-major over n, words[col * (blks*wpb) + block*wpb + wb].
inline std::vector<uint32_t> pack(const std::vector<int8_t>& raw, int n, int k, int blocksize) {
  int blks = k / blocksize;
  int wpb = words_per_block(blocksize);
  size_t wpc = size_t(blks) * wpb;  // words per column
  std::vector<uint32_t> out(wpc * n, 0u);
  for (int col = 0; col < n; ++col) {
    for (int b = 0; b < blks; ++b) {
      for (int wb = 0; wb < wpb; ++wb) {
        uint32_t word = 0u;
        for (int i = 0; i < 10; ++i) {
          int local = wb * 10 + i;  // index within the block
          uint32_t v;
          if (local < blocksize) {
            int kk = b * blocksize + local;
            v = uint32_t(raw[size_t(kk) * n + col] + 4) & 0x7u;
          } else {
            v = 4u;  // pad -> decodes to 0
          }
          word |= (v << (3 * i));
        }
        out[size_t(col) * wpc + b * wpb + wb] = word;
      }
    }
  }
  return out;
}

// Branchless decode of 10 three-bit values from one word.
static __attribute__((always_inline)) inline void unpack10(uint32_t w, int8_t* out) {
#pragma unroll
  for (int i = 0; i < 10; ++i) out[i] = int8_t((w >> (3 * i)) & 0x7u) - 4;
}

// Candidate GEMV: C[col] = sum_k A[k] * (q-4) * scale[col, k/blocksize]. One sub-group per column.
// Nested block->word loop (no per-word division), each lane reads a contiguous run of the column's
// words (coalesced within the sub-group), branchless 10-value decode.
template <typename T>
sycl::event gemv(const T* A, const uint32_t* Bwords, const T* scale, T* C, int n, int k, int blocksize,
                 sycl::queue* q) {
  int blks = k / blocksize;
  int wpb = words_per_block(blocksize);
  int wpc = blks * wpb;
  constexpr int SgSize = 32;
  sycl::range<1> group{SgSize};
  sycl::range<1> problem{size_t(n) * SgSize};
  return q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
      int g_n = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      const uint32_t* bptr = Bwords + size_t(g_n) * wpc;
      const T* sptr = scale + size_t(g_n) * blks;
      T tmpAcc = 0.f;
      // Each lane strides over words; block index advances without division.
      for (int w = sg_id; w < wpc; w += SgSize) {
        uint32_t word = bptr[w];  // coalesced: lane L reads word L of the current 32-word slice
        int8_t vals[10];
        unpack10(word, vals);
        int block = w / wpb;
        int wb = w - block * wpb;
        int base = block * blocksize + wb * 10;
        T scl = sptr[block];
        int valid = blocksize - wb * 10;
        if (valid > 10) valid = 10;
        T gacc = 0.f;
#pragma unroll
        for (int i = 0; i < 10; ++i) {
          if (i < valid) gacc += A[base + i] * static_cast<T>(vals[i]);
        }
        tmpAcc += gacc * scl;
      }
      sycl::group_barrier(sg);
      T sum = sycl::reduce_over_group(sg, tmpAcc, std::plus<>());
      if (sg_id == 0) C[g_n] = sum;
    });
  });
}

// Same as gemv() but with joint_prefetch over the CONTIGUOUS word stream. W10 is coalesced (lane L
// reads word L), so unlike the s3 straddle gather the prefetch covers a contiguous region — tests
// whether latency-hiding helps once the access pattern is already coalesced.
template <typename T>
sycl::event gemv_pf(const T* A, const uint32_t* Bwords, const T* scale, T* C, int n, int k, int blocksize,
                    sycl::queue* q) {
  int blks = k / blocksize;
  int wpb = words_per_block(blocksize);
  int wpc = blks * wpb;
  constexpr int SgSize = 32;
  constexpr int PrefetchDis = 3;
  sycl::range<1> group{SgSize};
  sycl::range<1> problem{size_t(n) * SgSize};
  return q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
      int g_n = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      const uint32_t* bptr = Bwords + size_t(g_n) * wpc;
      const T* sptr = scale + size_t(g_n) * blks;
      T tmpAcc = 0.f;
#pragma unroll
      for (int j = 1; j <= PrefetchDis; ++j)
        if (j * SgSize < wpc)
          sycl::ext::oneapi::experimental::joint_prefetch(sg, bptr + j * SgSize, SgSize * sizeof(uint32_t));
      for (int w = sg_id; w < wpc; w += SgSize) {
        int pw = w + PrefetchDis * SgSize;
        if (pw < wpc) sycl::ext::oneapi::experimental::joint_prefetch(sg, bptr + pw, SgSize * sizeof(uint32_t));
        uint32_t word = bptr[w];
        int8_t vals[10];
        unpack10(word, vals);
        int block = w / wpb;
        int wb = w - block * wpb;
        int base = block * blocksize + wb * 10;
        T scl = sptr[block];
        int valid = blocksize - wb * 10;
        if (valid > 10) valid = 10;
        T gacc = 0.f;
#pragma unroll
        for (int i = 0; i < 10; ++i)
          if (i < valid) gacc += A[base + i] * static_cast<T>(vals[i]);
        tmpAcc += gacc * scl;
      }
      sycl::group_barrier(sg);
      T sum = sycl::reduce_over_group(sg, tmpAcc, std::plus<>());
      if (sg_id == 0) C[g_n] = sum;
    });
  });
}

// Nested block->word GEMV. Each lane owns WHOLE blocks (lane L handles blocks L, L+32, ...), so the
// inner wb loop runs 0..WPB-1 with WPB a compile-time constant and fully unrolls: every bit offset
// (3*i), the per-word valid count (min(10, BLK - wb*10)), and the activation base are compile-time
// constants. No integer division, no modulo, no straddle branch; scale is loaded ONCE per block.
// Tradeoff: lane L reads block L's WPB words, so the global load is strided by WPB across lanes (not
// coalesced) — this variant isolates "does removing all dynamic-index/decode cost help, even at the
// price of a strided load?".
template <int BLK, typename T>
sycl::event gemv_nested(const T* A, const uint32_t* Bwords, const T* scale, T* C, int n, int k, sycl::queue* q) {
  constexpr int WPB = (BLK + 9) / 10;
  int blks = k / BLK;
  int wpc = blks * WPB;
  constexpr int SgSize = 32;
  sycl::range<1> group{SgSize};
  sycl::range<1> problem{size_t(n) * SgSize};
  return q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
      int g_n = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      const uint32_t* bptr = Bwords + size_t(g_n) * wpc;
      const T* sptr = scale + size_t(g_n) * blks;
      T tmpAcc = 0.f;
      for (int block = sg_id; block < blks; block += SgSize) {
        const uint32_t* wblock = bptr + size_t(block) * WPB;
        T scl = sptr[block];
        int kbase0 = block * BLK;
        T bacc = 0.f;
        // Unroll over wb with wb as a TEMPLATE param so bit offsets and valid count are constexpr.
        [&]<int... WB>(std::integer_sequence<int, WB...>) {
          (
              [&] {
                constexpr int valid = (BLK - WB * 10) < 10 ? (BLK - WB * 10) : 10;
                uint32_t word = wblock[WB];
                int base = kbase0 + WB * 10;
#pragma unroll
                for (int i = 0; i < valid; ++i)
                  bacc += A[base + i] * static_cast<T>(int8_t((word >> (3 * i)) & 0x7u) - 4);
              }(),
              ...);
        }(std::make_integer_sequence<int, WPB>{});
        tmpAcc += bacc * scl;  // scale applied once per block
      }
      sycl::group_barrier(sg);
      T sum = sycl::reduce_over_group(sg, tmpAcc, std::plus<>());
      if (sg_id == 0) C[g_n] = sum;
    });
  });
}

// ILP variant of gemv_nested: identical layout, decode, and per-lane-owns-whole-blocks scheme, but
// the per-block FMAs are split across NACC INDEPENDENT accumulators instead of one serial `bacc`.
// In gemv_nested every `bacc += a*v` reads the value the previous FMA wrote, so the sub-group issues
// one FMA per FMA-latency (~L cycles) regardless of how cheap decode is — a serial dependency chain
// that caps IPC at ~1/L of peak. Here position i feeds acc[i % NACC]; consecutive FMAs target
// different accumulators (no RAW hazard between them), so up to NACC can be in flight at once. The
// slot is compile-time (i is a constexpr loop index), so this stays division/modulo-free in codegen.
// Tradeoff: NACC live accumulators raise register pressure; too many can spill and erase the gain —
// which is exactly what sweeping NACC measures. Partials are summed once per block before scaling.
template <int BLK, int NACC, typename T>
sycl::event gemv_nested_ilp(const T* A, const uint32_t* Bwords, const T* scale, T* C, int n, int k,
                            sycl::queue* q) {
  constexpr int WPB = (BLK + 9) / 10;
  int blks = k / BLK;
  int wpc = blks * WPB;
  constexpr int SgSize = 32;
  sycl::range<1> group{SgSize};
  sycl::range<1> problem{size_t(n) * SgSize};
  return q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
      int g_n = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      const uint32_t* bptr = Bwords + size_t(g_n) * wpc;
      const T* sptr = scale + size_t(g_n) * blks;
      T tmpAcc = 0.f;
      for (int block = sg_id; block < blks; block += SgSize) {
        const uint32_t* wblock = bptr + size_t(block) * WPB;
        T scl = sptr[block];
        int kbase0 = block * BLK;
        T acc[NACC];
#pragma unroll
        for (int a = 0; a < NACC; ++a) acc[a] = 0.f;
        // Unroll over wb with wb as a TEMPLATE param so bit offsets and valid count are constexpr.
        [&]<int... WB>(std::integer_sequence<int, WB...>) {
          (
              [&] {
                constexpr int valid = (BLK - WB * 10) < 10 ? (BLK - WB * 10) : 10;
                uint32_t word = wblock[WB];
                int base = kbase0 + WB * 10;
#pragma unroll
                for (int i = 0; i < valid; ++i)
                  acc[(WB * 10 + i) % NACC] += A[base + i] * static_cast<T>(int8_t((word >> (3 * i)) & 0x7u) - 4);
              }(),
              ...);
        }(std::make_integer_sequence<int, WPB>{});
        T bacc = 0.f;  // tree-free reduce of the independent partials, then scale once per block
#pragma unroll
        for (int a = 0; a < NACC; ++a) bacc += acc[a];
        tmpAcc += bacc * scl;
      }
      sycl::group_barrier(sg);
      T sum = sycl::reduce_over_group(sg, tmpAcc, std::plus<>());
      if (sg_id == 0) C[g_n] = sum;
    });
  });
}

// Coalesced-load nested GEMV. The ISA dump proved gemv_nested's plateau is a memory-feed stall, not
// FMA latency: lane L owns whole blocks, so its WPB words are strided by WPB across the sub-group and
// IGC emits ~40 per-lane gathers (load.ugm.a64 (32|M0)) per step instead of block loads. This variant
// keeps the compile-time-position decode but fixes the load: the sub-group cooperatively stages
// SgSize whole blocks (SgSize*WPB contiguous words of the column's stream) into SLM with a COALESCED
// block load (lane L reads words L, L+SgSize, ...), barriers, then each lane decodes its own block
// from SLM. Same per-lane-owns-a-block decode as gemv_nested, but the global traffic is now coalesced
// like S4's. NACC independent accumulators are retained as a free orthogonal knob (ISA showed no
// spill through nacc=8); NACC=1 reproduces the serial chain.
template <int BLK, int NACC, typename T>
sycl::event gemv_nested_coal(const T* A, const uint32_t* Bwords, const T* scale, T* C, int n, int k,
                             sycl::queue* q) {
  constexpr int WPB = (BLK + 9) / 10;
  constexpr int SgSize = 32;
  constexpr int WPSTEP = SgSize * WPB;  // contiguous words staged per barrier round (SgSize blocks)
  int blks = k / BLK;
  int wpc = blks * WPB;
  int blk_main = (blks / SgSize) * SgSize;  // blocks covered by the coalesced/staged main loop
  sycl::range<1> group{SgSize};
  sycl::range<1> problem{size_t(n) * SgSize};
  return q->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(WPSTEP), cgh);
    cgh.parallel_for(sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
      int g_n = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      const uint32_t* bptr = Bwords + size_t(g_n) * wpc;
      const T* sptr = scale + size_t(g_n) * blks;
      T tmpAcc = 0.f;

      for (int bbase = 0; bbase < blk_main; bbase += SgSize) {
        const uint32_t* base = bptr + size_t(bbase) * WPB;
        // Coalesced staging: the WPSTEP contiguous words for these SgSize blocks, lane L reads L,L+32,...
#pragma unroll
        for (int p = 0; p < WPB; p++) slm[p * SgSize + sg_id] = base[p * SgSize + sg_id];
        sycl::group_barrier(sg);

        int block = bbase + sg_id;  // this lane's block
        const uint32_t* wblock = &slm[sg_id * WPB];
        T scl = sptr[block];
        int kbase0 = block * BLK;
        T acc[NACC];
#pragma unroll
        for (int a = 0; a < NACC; ++a) acc[a] = 0.f;
        [&]<int... WB>(std::integer_sequence<int, WB...>) {
          (
              [&] {
                constexpr int valid = (BLK - WB * 10) < 10 ? (BLK - WB * 10) : 10;
                uint32_t word = wblock[WB];
                int kb = kbase0 + WB * 10;
#pragma unroll
                for (int i = 0; i < valid; ++i)
                  acc[(WB * 10 + i) % NACC] += A[kb + i] * static_cast<T>(int8_t((word >> (3 * i)) & 0x7u) - 4);
              }(),
              ...);
        }(std::make_integer_sequence<int, WPB>{});
        T bacc = 0.f;
#pragma unroll
        for (int a = 0; a < NACC; ++a) bacc += acc[a];
        tmpAcc += bacc * scl;
        sycl::group_barrier(sg);  // SLM reuse next round
      }

      // Tail: blocks beyond blk_main, one per lane, decoded direct from global (small remainder).
      for (int block = blk_main + sg_id; block < blks; block += SgSize) {
        const uint32_t* wblock = bptr + size_t(block) * WPB;
        T scl = sptr[block];
        int kbase0 = block * BLK;
        T bacc = 0.f;
        [&]<int... WB>(std::integer_sequence<int, WB...>) {
          (
              [&] {
                constexpr int valid = (BLK - WB * 10) < 10 ? (BLK - WB * 10) : 10;
                uint32_t word = wblock[WB];
                int kb = kbase0 + WB * 10;
#pragma unroll
                for (int i = 0; i < valid; ++i)
                  bacc += A[kb + i] * static_cast<T>(int8_t((word >> (3 * i)) & 0x7u) - 4);
              }(),
              ...);
        }(std::make_integer_sequence<int, WPB>{});
        tmpAcc += bacc * scl;
      }

      sycl::group_barrier(sg);
      T sum = sycl::reduce_over_group(sg, tmpAcc, std::plus<>());
      if (sg_id == 0) C[g_n] = sum;
    });
  });
}

// Padded packer + vector-load nested GEMV. The coal attempt failed because hand-rolled scalar SLM
// staging (slm[p*32+sg_id]=...) still lowered to gathers. S4's speed instead comes from each lane
// doing ONE per-lane vector load of a contiguous power-of-two chunk: consecutive lanes read
// consecutive contiguous blocks, so the sub-group span is contiguous and IGC emits a transpose block
// load (load.ugm.d32xNt). To get that here, each block's WPB words are padded up to VPB (next pow2,
// 16 for BLK=128) so lane L reads vec<uint32_t, VPB> at base + L*VPB with stride == width == VPB.
// No SLM, no barrier on the load. NACC independent accumulators retained as the orthogonal ILP knob.

inline int vpb_padded(int blocksize) {
  int wpb = words_per_block(blocksize);
  int v = 1;
  while (v < wpb) v <<= 1;
  return v;  // next power of two >= wpb
}

// Per-column contiguous stream, each block occupies VPB words (real WPB + padding zeros that decode
// to 0). Layout: words[col * (blks*VPB) + block*VPB + wb].
inline std::vector<uint32_t> pack_padded(const std::vector<int8_t>& raw, int n, int k, int blocksize) {
  int blks = k / blocksize;
  int wpb = words_per_block(blocksize);
  int vpb = vpb_padded(blocksize);
  size_t wpc = size_t(blks) * vpb;
  std::vector<uint32_t> out(wpc * n, 0u);
  for (int col = 0; col < n; ++col) {
    for (int b = 0; b < blks; ++b) {
      for (int wb = 0; wb < wpb; ++wb) {
        uint32_t word = 0u;
        for (int i = 0; i < 10; ++i) {
          int local = wb * 10 + i;
          uint32_t v = (local < blocksize) ? (uint32_t(raw[size_t(b * blocksize + local) * n + col] + 4) & 0x7u) : 4u;
          word |= (v << (3 * i));
        }
        out[size_t(col) * wpc + b * vpb + wb] = word;
      }
      // words [wpb, vpb) stay 0 -> decode (0&7)-4 = -4; guarded out by the valid count below.
    }
  }
  return out;
}

// VPB compile-time so the per-lane load is vec<uint32_t, VPB> (contiguous, power-of-two -> block load).
template <int BLK, int NACC, typename T>
sycl::event gemv_nested_vec(const T* A, const uint32_t* Bwords, const T* scale, T* C, int n, int k,
                            sycl::queue* q) {
  constexpr int WPB = (BLK + 9) / 10;
  constexpr int VPB = WPB <= 16 ? 16 : 32;  // next pow2 >= WPB for BLK<=160; BLK=128 -> 16
  constexpr int SgSize = 32;
  int blks = k / BLK;
  int wpc = blks * VPB;
  sycl::range<1> group{SgSize};
  sycl::range<1> problem{size_t(n) * SgSize};
  return q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
      int g_n = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      const uint32_t* bptr = Bwords + size_t(g_n) * wpc;
      const T* sptr = scale + size_t(g_n) * blks;
      T tmpAcc = 0.f;
      for (int block = sg_id; block < blks; block += SgSize) {
        // One contiguous per-lane vector load of the whole padded block (lane L reads [block*VPB ..]).
        sycl::vec<uint32_t, VPB> wv = *(const sycl::vec<uint32_t, VPB>*)(bptr + size_t(block) * VPB);
        T scl = sptr[block];
        int kbase0 = block * BLK;
        T acc[NACC];
#pragma unroll
        for (int a = 0; a < NACC; ++a) acc[a] = 0.f;
        [&]<int... WB>(std::integer_sequence<int, WB...>) {
          (
              [&] {
                constexpr int valid = (BLK - WB * 10) < 10 ? (BLK - WB * 10) : 10;
                uint32_t word = wv[WB];
                int kb = kbase0 + WB * 10;
#pragma unroll
                for (int i = 0; i < valid; ++i)
                  acc[(WB * 10 + i) % NACC] += A[kb + i] * static_cast<T>(int8_t((word >> (3 * i)) & 0x7u) - 4);
              }(),
              ...);
        }(std::make_integer_sequence<int, WPB>{});
        T bacc = 0.f;
#pragma unroll
        for (int a = 0; a < NACC; ++a) bacc += acc[a];
        tmpAcc += bacc * scl;
      }
      sycl::group_barrier(sg);
      T sum = sycl::reduce_over_group(sg, tmpAcc, std::plus<>());
      if (sg_id == 0) C[g_n] = sum;
    });
  });
}

// ------------------------------------------------------------------------------------------------
// Transposed (lane-interleaved) layout — the structural fix. Both prior coalescing attempts failed
// for the same reason: lane L read block L, so across the 32-lane sub-group the word addresses were
// strided (by WPB / by VPB*4 bytes), non-contiguous, and IGC kept a per-lane gather. S4 gets its
// d32xNt transpose block loads because consecutive lanes read consecutive contiguous words.
//
// Fix the LAYOUT so that holds here too. Process blocks in tiles of SgSize (=32). Within a tile, the
// words are stored word-major then lane-minor: for word index wb in [0,WPB) and lane L in [0,32),
// tile word (wb, L) lives at tilebase + wb*32 + L. So when all 32 lanes read their own block's word
// wb via `tilebase + wb*32 + sg_id`, the 32 addresses are CONTIGUOUS -> coalesced sub-group load ->
// block load. Lane L still owns block (tile*32 + L); the decode is the same compile-time unroll.
// ------------------------------------------------------------------------------------------------

// Host packer: per column, blocks grouped into tiles of 32; within a tile, word-major/lane-minor.
// Tail blocks (blks % 32) are stored in a final partial tile the same way (lanes >= rem unused).
inline std::vector<uint32_t> pack_transposed(const std::vector<int8_t>& raw, int n, int k, int blocksize) {
  int blks = k / blocksize;
  int wpb = words_per_block(blocksize);
  int tiles = (blks + 31) / 32;
  size_t wpc = size_t(tiles) * 32 * wpb;  // words per column (padded up to whole tiles)
  std::vector<uint32_t> out(wpc * n, 0u);
  for (int col = 0; col < n; ++col) {
    for (int b = 0; b < blks; ++b) {
      int tile = b / 32, lane = b % 32;
      for (int wb = 0; wb < wpb; ++wb) {
        uint32_t word = 0u;
        for (int i = 0; i < 10; ++i) {
          int local = wb * 10 + i;
          uint32_t v = (local < blocksize) ? (uint32_t(raw[size_t(b * blocksize + local) * n + col] + 4) & 0x7u) : 4u;
          word |= (v << (3 * i));
        }
        // tile base for this column, then word-major/lane-minor within the tile.
        out[size_t(col) * wpc + (size_t(tile) * 32 + 0) * wpb + size_t(wb) * 32 + lane] = word;
      }
    }
  }
  return out;
}

template <int BLK, int NACC, typename T>
sycl::event gemv_nested_trans(const T* A, const uint32_t* Bwords, const T* scale, T* C, int n, int k,
                              sycl::queue* q) {
  constexpr int WPB = (BLK + 9) / 10;
  constexpr int SgSize = 32;
  int blks = k / BLK;
  int tiles = (blks + 31) / 32;
  int wpc = tiles * 32 * WPB;
  sycl::range<1> group{SgSize};
  sycl::range<1> problem{size_t(n) * SgSize};
  return q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
      int g_n = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      const uint32_t* bptr = Bwords + size_t(g_n) * wpc;
      const T* sptr = scale + size_t(g_n) * blks;
      T tmpAcc = 0.f;
      for (int tile = 0; tile < tiles; ++tile) {
        int block = tile * 32 + sg_id;
        const uint32_t* tilebase = bptr + size_t(tile) * 32 * WPB;
        // Read this lane's WPB words; lane stride is 1 within each word slab -> contiguous sub-group
        // span -> block load. wb slabs are 32 apart.
        uint32_t wv[WPB];
#pragma unroll
        for (int wb = 0; wb < WPB; ++wb) wv[wb] = tilebase[wb * 32 + sg_id];
        if (block >= blks) continue;  // partial tail tile: unused lanes contribute nothing
        T scl = sptr[block];
        int kbase0 = block * BLK;
        T acc[NACC];
#pragma unroll
        for (int a = 0; a < NACC; ++a) acc[a] = 0.f;
        [&]<int... WB>(std::integer_sequence<int, WB...>) {
          (
              [&] {
                constexpr int valid = (BLK - WB * 10) < 10 ? (BLK - WB * 10) : 10;
                uint32_t word = wv[WB];
                int kb = kbase0 + WB * 10;
#pragma unroll
                for (int i = 0; i < valid; ++i)
                  acc[(WB * 10 + i) % NACC] += A[kb + i] * static_cast<T>(int8_t((word >> (3 * i)) & 0x7u) - 4);
              }(),
              ...);
        }(std::make_integer_sequence<int, WPB>{});
        T bacc = 0.f;
#pragma unroll
        for (int a = 0; a < NACC; ++a) bacc += acc[a];
        tmpAcc += bacc * scl;
      }
      sycl::group_barrier(sg);
      T sum = sycl::reduce_over_group(sg, tmpAcc, std::plus<>());
      if (sg_id == 0) C[g_n] = sum;
    });
  });
}

// Transposed layout + SOFTWARE PREFETCH. ISA correction: S4's "block loads" (d32xNt, null:0 dest)
// are joint_prefetch software prefetches, NOT its weight loads — S4 demand-loads with the SAME
// d32x4.a64 (32|M0) gather family as our int3 kernels. The real structural difference is that S4
// issues 4 joint_prefetch calls to hide memory latency and every int3 variant here issues ZERO.
// This variant adds that: prefetch whole tiles (32*WPB contiguous words) PrefetchDis tiles ahead of
// the one being consumed, on top of the best (transposed) layout. Same decode, same NACC knob.
template <int BLK, int NACC, typename T>
sycl::event gemv_nested_trans_pf(const T* A, const uint32_t* Bwords, const T* scale, T* C, int n, int k,
                                 sycl::queue* q) {
  constexpr int WPB = (BLK + 9) / 10;
  constexpr int SgSize = 32;
  constexpr int TileWords = SgSize * WPB;  // contiguous words per tile
  constexpr int PrefetchDis = 3;           // tiles ahead, mirrors S4's prefetch distance
  int blks = k / BLK;
  int tiles = (blks + 31) / 32;
  int wpc = tiles * TileWords;
  sycl::range<1> group{SgSize};
  sycl::range<1> problem{size_t(n) * SgSize};
  return q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
      int g_n = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      const uint32_t* bptr = Bwords + size_t(g_n) * wpc;
      const T* sptr = scale + size_t(g_n) * blks;
      T tmpAcc = 0.f;
      // Prime the pipeline: prefetch the first PrefetchDis tiles.
#pragma unroll
      for (int j = 1; j <= PrefetchDis; ++j)
        if (j * TileWords < wpc)
          sycl::ext::oneapi::experimental::joint_prefetch(sg, bptr + size_t(j) * TileWords,
                                                          TileWords * sizeof(uint32_t));
      for (int tile = 0; tile < tiles; ++tile) {
        const uint32_t* tilebase = bptr + size_t(tile) * TileWords;
        // Prefetch the tile PrefetchDis ahead while we consume this one.
        size_t pf = size_t(tile + PrefetchDis) * TileWords;
        if (pf < size_t(wpc))
          sycl::ext::oneapi::experimental::joint_prefetch(sg, bptr + pf, TileWords * sizeof(uint32_t));
        int block = tile * 32 + sg_id;
        uint32_t wv[WPB];
#pragma unroll
        for (int wb = 0; wb < WPB; ++wb) wv[wb] = tilebase[wb * 32 + sg_id];
        if (block >= blks) continue;
        T scl = sptr[block];
        int kbase0 = block * BLK;
        T acc[NACC];
#pragma unroll
        for (int a = 0; a < NACC; ++a) acc[a] = 0.f;
        [&]<int... WB>(std::integer_sequence<int, WB...>) {
          (
              [&] {
                constexpr int valid = (BLK - WB * 10) < 10 ? (BLK - WB * 10) : 10;
                uint32_t word = wv[WB];
                int kb = kbase0 + WB * 10;
#pragma unroll
                for (int i = 0; i < valid; ++i)
                  acc[(WB * 10 + i) % NACC] += A[kb + i] * static_cast<T>(int8_t((word >> (3 * i)) & 0x7u) - 4);
              }(),
              ...);
        }(std::make_integer_sequence<int, WPB>{});
        T bacc = 0.f;
#pragma unroll
        for (int a = 0; a < NACC; ++a) bacc += acc[a];
        tmpAcc += bacc * scl;
      }
      sycl::group_barrier(sg);
      T sum = sycl::reduce_over_group(sg, tmpAcc, std::plus<>());
      if (sg_id == 0) C[g_n] = sum;
    });
  });
}

}  // namespace w10_spike

// ---------------------------------------------------------------------------------------------
// S3 dense straddle layout, but read DIRECT from global with prefetch — NO SLM staging, NO
// barriers. Tests whether dropping the SLM round-trip (and accepting the stride-12B gather, since
// lane L reads its 3 words 3L,3L+1,3L+2) beats the committed SLM kernel now that we know S3 is
// compute-bound, not memory-bound. Same dense blob as packq_int3: per column, kgroups*3 words.
// ---------------------------------------------------------------------------------------------
namespace s3_direct_spike {

// Dense straddle pack (host), mirrors packq_int3: 32 vals -> 3 contiguous words, (q+4)&0x7,
// fields cross word boundaries. Output per column: kgroups*3 words.
inline std::vector<uint32_t> pack(const std::vector<int8_t>& raw, int n, int k) {
  int kgroups = k / 32;
  std::vector<uint32_t> out(size_t(kgroups) * 3 * n, 0u);
  for (int col = 0; col < n; ++col) {
    for (int gk = 0; gk < kgroups; ++gk) {
      uint32_t words[3] = {0u, 0u, 0u};
      for (int i = 0; i < 32; ++i) {
        uint32_t v = uint32_t(raw[size_t(gk * 32 + i) * n + col] + 4) & 0x7u;
        int index = i * 3, word_idx = index >> 5, bit_offset = index & 31;
        words[word_idx] |= (v << bit_offset);
        if (bit_offset + 3 > 32) words[word_idx + 1] |= (v >> (32 - bit_offset));
      }
      uint32_t* dst = out.data() + (size_t(col) * kgroups + gk) * 3;
      dst[0] = words[0];
      dst[1] = words[1];
      dst[2] = words[2];
    }
  }
  return out;
}

static __attribute__((always_inline)) inline void unpack32(uint32_t w0, uint32_t w1, uint32_t w2, int8_t* out) {
  const uint32_t w[3] = {w0, w1, w2};
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    int index = i * 3, word_idx = index >> 5, bit_offset = index & 31;
    int part1 = (32 - bit_offset) < 3 ? (32 - bit_offset) : 3;
    int part2 = 3 - part1;
    uint32_t val = (w[word_idx] >> bit_offset) & ((1u << part1) - 1u);
    if (part2 > 0) val |= (w[word_idx + 1] & ((1u << part2) - 1u)) << part1;
    out[i] = int8_t(val) - 4;
  }
}

// Direct-from-global GEMV with prefetch, no SLM. Lane L handles strided groups gk=L, L+32, ...;
// reads its 3 words directly (the stride-12B gather), prefetches groups PrefetchDis ahead.
template <typename T>
sycl::event gemv(const T* A, const uint32_t* Bwords, const T* scale, T* C, int n, int k, int blocksize,
                 sycl::queue* q) {
  int kgroups = k / 32;
  constexpr int SgSize = 32;
  constexpr int PrefetchDis = 3;
  sycl::range<1> group{SgSize};
  sycl::range<1> problem{size_t(n) * SgSize};
  return q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
      int g_n = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      const uint32_t* bptr = Bwords + size_t(g_n) * kgroups * 3;
      const T* sptr = scale + size_t(g_n) * (k / blocksize);
      T tmpAcc = 0.f;
      // Prime: prefetch the sub-group's first few 96-byte group-slices.
#pragma unroll
      for (int j = 1; j <= PrefetchDis; ++j) {
        int pg = j * SgSize;
        if (pg < kgroups)
          sycl::ext::oneapi::experimental::joint_prefetch(sg, bptr + size_t(pg) * 3, SgSize * 3 * sizeof(uint32_t));
      }
      for (int gk = sg_id; gk < kgroups; gk += SgSize) {
        int pg = gk + PrefetchDis * SgSize;
        if (pg < kgroups)
          sycl::ext::oneapi::experimental::joint_prefetch(sg, bptr + size_t(pg) * 3, SgSize * 3 * sizeof(uint32_t));
        const uint32_t* wp = bptr + size_t(gk) * 3;
        int8_t vals[32];
        unpack32(wp[0], wp[1], wp[2], vals);
        int kbase = gk * 32;
        T tmpA[32];
#pragma unroll
        for (int c = 0; c < 32; c += 8) *(sycl::vec<T, 8>*)&tmpA[c] = *(sycl::vec<T, 8>*)&A[kbase + c];
        T scl = sptr[kbase / blocksize];
        T gacc = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) gacc += tmpA[i] * static_cast<T>(vals[i]);
        tmpAcc += gacc * scl;
      }
      sycl::group_barrier(sg);
      T sum = sycl::reduce_over_group(sg, tmpAcc, std::plus<>());
      if (sg_id == 0) C[g_n] = sum;
    });
  });
}

}  // namespace s3_direct_spike
#endif

struct TestGemm {
  TestGemm() {
#ifdef ARK_XPU
    if (ark_s3_bench_only()) {
      run_s3_benchmarks();
      return;
    }
#endif
    test<float>(128, 128, 128);
#ifdef ARK_XPU
    test<bf16>(128, 128, 128);
    test<fp16>(128, 128, 128);
    test_s8s8<float>(128, 128, 128);
    test_woqs8<float>(128, 128, 128);
    test_woq_s3<float>(128, 128, 128);
    test_woq_s3<float>(256, 96, 32);
    test_woq_s3<float>(64, 256, 64);
    test_woq_s3_gemm<float>(4, 128, 128, 128);
    test_woq_s3_gemm<float>(8, 256, 128, 128);
    test_woq_s3_gemm<float>(16, 64, 256, 64);
    run_s3_benchmarks();
#endif
  }

  template <typename T>
  void test(size_t m, size_t n, size_t k) {
    GETQ();
    LOG_LINE();
    auto dt = ark::to_dt<T>();
    auto A = ctx->allocate(m * k * sizeof(T));
    auto B = ctx->allocate(n * k * sizeof(T));
    auto C = ctx->allocate(m * n * sizeof(T));
    ark::DnnlWrapper::gemm(q, m, n, k, A, dt, B, dt, false, C, dt, nullptr);
#ifdef ARK_XPU
    q->wait();
#endif
    ctx->deallocate(A);
    ctx->deallocate(B);
    ctx->deallocate(C);
  }

  template <typename T>
  void test_s8s8(size_t m, size_t n, size_t k) {
    GETQ();
    LOG_LINE();
    auto dt = ark::to_dt<T>();
    auto A = ctx->allocate(m * k * sizeof(int8_t));
    auto B = ctx->allocate(n * k * sizeof(int8_t));
    auto C = ctx->allocate(m * n * sizeof(T));
    auto scaleA = ctx->allocate(1 * sizeof(T));
    auto scaleB = ctx->allocate(n * sizeof(T));
    ark::DnnlWrapper::igemm_s8s8(q, m, n, k, A, B, true, C, dt, scaleA, scaleB, nullptr);
    ctx->deallocate(A);
    ctx->deallocate(B);
    ctx->deallocate(C);
    ctx->deallocate(scaleA);
    ctx->deallocate(scaleB);
  }

  template <typename T>
  void test_woqs8(size_t m, size_t n, size_t k) {
    GETQ();
    LOG_LINE();
    auto dt = ark::to_dt<T>();
    auto A = ctx->allocate(m * k * sizeof(T));
    auto B = ctx->allocate(n * k * sizeof(int8_t));
    auto C = ctx->allocate(m * n * sizeof(T));
    auto scaleB = ctx->allocate(n * sizeof(T));
    ark::DnnlWrapper::woq_s8(q, m, n, k, A, B, true, C, dt, scaleB, nullptr, k);
    ctx->deallocate(A);
    ctx->deallocate(B);
    ctx->deallocate(C);
    ctx->deallocate(scaleB);
  }

#ifdef ARK_XPU
  // int3 (S3) throughput benchmarks. GEMV (m==1, dedicated S3 kernel) and GEMM (m>1, fp-dequant +
  // DNNL fp GEMM fallback) at LLM-projection-like shapes (Qwen3 hidden/intermediate sizes). Edit the
  // shapes here; gated standalone via ARK_S3_BENCH=1. S4 runs the SAME shapes as a reference: its
  // GEMV uses the dedicated S4 kernel and its m>1 GEMM takes the native int8-XMX path (woq_s8),
  // whereas S3 m>1 falls back to fp-dequant + fp GEMM — so the S4 numbers are the speed-of-light
  // reference S3 is measured against.
  void run_s3_benchmarks() {
    struct Shape {
      const char* tag;
      size_t n, k;
      int blk;
    };
    const Shape gemv_shapes[] = {{"n4096_k4096", 4096, 4096, 128}, {"n4096_k11008", 4096, 11008, 128}};
    struct GShape {
      const char* tag;
      size_t m, n, k;
      int blk;
    };
    const GShape gemm_shapes[] = {{"m32_n4096_k4096", 32, 4096, 4096, 128},
                                  {"m128_n4096_k4096", 128, 4096, 4096, 128},
                                  {"m512_n4096_k11008", 512, 4096, 11008, 128}};
    // (weight_type, output tag) pairs: S3 is under test, S4 is the reference at the same shapes.
    // S2 added as a sub-byte calibration point: is ANY narrow-bit kernel here bandwidth-fast?
    const std::pair<BTLA_DTYPE, const char*> wtypes[] = {
        {BTLA_DTYPE::S2, "s2"}, {BTLA_DTYPE::S3, "s3"}, {BTLA_DTYPE::S4, "s4"}};

    for (auto& w : wtypes) {
      for (auto& s : gemv_shapes) {
        bench_woq_gemv<float>(w.first, w.second, std::string("bench_") + w.second + "_gemv_" + s.tag, s.n, s.k, s.blk,
                              10, 50);
      }
    }
    // W10 spike (Phase 0): same blk=128 shapes as s3/s4, plus a blk=32 case to see the padding cost.
    bench_woq_w10_spike<float>("bench_w10_gemv_n4096_k4096", 4096, 4096, 128, 10, 50);
    bench_woq_w10_spike<float>("bench_w10_gemv_n4096_k11008", 4096, 11008, 128, 10, 50);
    bench_woq_w10_spike<float>("bench_w10_gemv_n4096_k4096_blk32", 4096, 4096, 32, 10, 50);
    // W10 + joint_prefetch over the contiguous word stream (coalesced, unlike s3 straddle gather).
    bench_woq_w10_spike<float>("bench_w10pf_gemv_n4096_k4096", 4096, 4096, 128, 10, 50, /*use_pf=*/true);
    bench_woq_w10_spike<float>("bench_w10pf_gemv_n4096_k11008", 4096, 11008, 128, 10, 50, /*use_pf=*/true);
    // W10 nested block->word (BLK=128): all positions compile-time, no div/mod, scale once per block.
    bench_woq_w10_nested<128, float>("bench_w10nest_gemv_n4096_k4096", 4096, 4096, 10, 50);
    bench_woq_w10_nested<128, float>("bench_w10nest_gemv_n4096_k11008", 4096, 11008, 10, 50);
    // W10 nested + ILP: sweep independent-accumulator count to break the serial bacc FMA chain.
    bench_woq_w10_nested_ilp<128, 2, float>("bench_w10nestilp2_gemv_n4096_k4096", 4096, 4096, 10, 50);
    bench_woq_w10_nested_ilp<128, 4, float>("bench_w10nestilp4_gemv_n4096_k4096", 4096, 4096, 10, 50);
    bench_woq_w10_nested_ilp<128, 8, float>("bench_w10nestilp8_gemv_n4096_k4096", 4096, 4096, 10, 50);
    bench_woq_w10_nested_ilp<128, 2, float>("bench_w10nestilp2_gemv_n4096_k11008", 4096, 11008, 10, 50);
    bench_woq_w10_nested_ilp<128, 4, float>("bench_w10nestilp4_gemv_n4096_k11008", 4096, 11008, 10, 50);
    bench_woq_w10_nested_ilp<128, 8, float>("bench_w10nestilp8_gemv_n4096_k11008", 4096, 11008, 10, 50);
    // W10 nested + COALESCED SLM staging: turns the nested variant's per-lane gathers into block loads.
    // nacc=1 isolates the pure coalescing win vs serial nested; nacc=2 stacks ILP on top.
    bench_woq_w10_nested_coal<128, 1, float>("bench_w10nestcoal1_gemv_n4096_k4096", 4096, 4096, 10, 50);
    bench_woq_w10_nested_coal<128, 2, float>("bench_w10nestcoal2_gemv_n4096_k4096", 4096, 4096, 10, 50);
    bench_woq_w10_nested_coal<128, 1, float>("bench_w10nestcoal1_gemv_n4096_k11008", 4096, 11008, 10, 50);
    bench_woq_w10_nested_coal<128, 2, float>("bench_w10nestcoal2_gemv_n4096_k11008", 4096, 11008, 10, 50);
    // W10 nested + VECTOR per-lane block load (padded WPB->VPB=16): tests whether a contiguous pow2
    // vector load lowers to a transpose block load (the actual source of S4's speed).
    bench_woq_w10_nested_vec<128, 1, float>("bench_w10nestvec1_gemv_n4096_k4096", 4096, 4096, 10, 50);
    bench_woq_w10_nested_vec<128, 2, float>("bench_w10nestvec2_gemv_n4096_k4096", 4096, 4096, 10, 50);
    bench_woq_w10_nested_vec<128, 1, float>("bench_w10nestvec1_gemv_n4096_k11008", 4096, 11008, 10, 50);
    bench_woq_w10_nested_vec<128, 2, float>("bench_w10nestvec2_gemv_n4096_k11008", 4096, 11008, 10, 50);
    // W10 nested + TRANSPOSED lane-interleaved layout (structural fix): contiguous sub-group span ->
    // block load. nacc=1 isolates the layout win; nacc=2 stacks ILP.
    bench_woq_w10_nested_trans<128, 1, float>("bench_w10nesttrans1_gemv_n4096_k4096", 4096, 4096, 10, 50);
    bench_woq_w10_nested_trans<128, 2, float>("bench_w10nesttrans2_gemv_n4096_k4096", 4096, 4096, 10, 50);
    bench_woq_w10_nested_trans<128, 1, float>("bench_w10nesttrans1_gemv_n4096_k11008", 4096, 11008, 10, 50);
    bench_woq_w10_nested_trans<128, 2, float>("bench_w10nesttrans2_gemv_n4096_k11008", 4096, 11008, 10, 50);
    // W10 transposed + software prefetch: the lever the ISA showed S4 has (joint_prefetch) and every
    // int3 variant lacked. Stacks on the best (transposed) layout.
    bench_woq_w10_nested_trans_pf<128, 1, float>("bench_w10transpf1_gemv_n4096_k4096", 4096, 4096, 10, 50);
    bench_woq_w10_nested_trans_pf<128, 2, float>("bench_w10transpf2_gemv_n4096_k4096", 4096, 4096, 10, 50);
    bench_woq_w10_nested_trans_pf<128, 1, float>("bench_w10transpf1_gemv_n4096_k11008", 4096, 11008, 10, 50);
    bench_woq_w10_nested_trans_pf<128, 2, float>("bench_w10transpf2_gemv_n4096_k11008", 4096, 11008, 10, 50);
    // S3 dense direct-from-global + prefetch, no SLM — does dropping SLM beat the committed kernel?
    bench_s3_direct_spike<float>("bench_s3_direct_gemv_n4096_k4096", 4096, 4096, 128, 10, 50);
    bench_s3_direct_spike<float>("bench_s3_direct_gemv_n4096_k11008", 4096, 11008, 128, 10, 50);
    // for (auto& w : wtypes) {
    //   for (auto& s : gemm_shapes) {
    //     bench_woq_gemm<float>(w.first, w.second, std::string("bench_") + w.second + "_gemm_" + s.tag, s.m, s.n, s.k,
    //                           s.blk, 5, 20);
    //   }
    // }
  }

  // m==1 int3 (S3) symmetric WOQ GEMV accuracy check: pack random 3-bit weights via the kernel
  // packer, run woq_gemv, and compare against a host fp reference dequant-then-matvec.
  template <typename T>
  void test_woq_s3(size_t n, size_t k, int blocksize) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % blocksize != 0 || blocksize % 32 != 0) {
      throw std::runtime_error("test_woq_s3 requires k % 32 == 0, k % blocksize == 0, blocksize % 32 == 0");
    }
    int blks = int(k) / blocksize;

    std::mt19937 rng(7u + uint32_t(n) + uint32_t(k));
    std::uniform_int_distribution<int> wdist(-4, 3);   // symmetric int3 range
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.05f);

    // raw signed weights, row-major [k, n]; activations [k]; per-(n,blk) scales.
    std::vector<int8_t> raw(k * n);
    for (auto& w : raw) w = int8_t(wdist(rng));
    std::vector<float> hostA(k);
    for (auto& a : hostA) a = adist(rng);
    // scale source layout for packscale: row-major [blks, n].
    std::vector<float> hostScale(size_t(blks) * n);
    for (auto& s : hostScale) s = sdist(rng);

    // Host reference: C[j] = sum_k A[k] * raw[k,j] * scale[k/blocksize, j].
    std::vector<float> refC(n, 0.0f);
    for (size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (size_t kk = 0; kk < k; ++kk) {
        float sc = hostScale[(kk / blocksize) * n + j];
        acc += hostA[kk] * float(raw[kk * n + j]) * sc;
      }
      refC[j] = acc;
    }

    ark::QuantParam param{int(n), int(k), blocksize, (int)BTLA_DTYPE::F32, (int)BTLA_DTYPE::S3, (int)BTLA_DTYPE::F32,
                          false};
    size_t blob_size = ark::XpuWrapper::get_packw_size(&param);

    auto* dRaw = reinterpret_cast<int8_t*>(ctx->allocate(raw.size() * sizeof(int8_t)));
    auto* dScale = reinterpret_cast<float*>(ctx->allocate(hostScale.size() * sizeof(float)));
    auto* dBlob = reinterpret_cast<int8_t*>(ctx->allocate(blob_size));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(n * sizeof(T)));

    std::vector<T> hostAt(k);
    for (size_t i = 0; i < k; ++i) hostAt[i] = T(hostA[i]);

    q->memcpy(dRaw, raw.data(), raw.size() * sizeof(int8_t)).wait();
    q->memcpy(dScale, hostScale.data(), hostScale.size() * sizeof(float)).wait();
    q->memcpy(dA, hostAt.data(), k * sizeof(T)).wait();

    // Pack (raw int8 + scale) into the device blob, then run the m==1 GEMV.
    ark::XpuWrapper::packq(dRaw, (void*)dScale, nullptr, dBlob, &param, q);
    q->wait();
    int ret = ark::XpuWrapper::woq_gemv(q, 1, &param, dA, dBlob, dC, nullptr, BTLA_DTYPE::F32);
    q->wait();
    if (ret != 0) {
      throw std::runtime_error("woq_gemv(S3) returned non-zero: " + std::to_string(ret));
    }

    std::vector<T> hostC(n);
    q->memcpy(hostC.data(), dC, n * sizeof(T)).wait();

    float max_diff = 0.0f;
    for (size_t j = 0; j < n; ++j) {
      max_diff = std::max(max_diff, std::fabs(float(hostC[j]) - refC[j]));
    }
    std::cout << "[woq_s3][accuracy] n=" << n << " k=" << k << " blk=" << blocksize << " max_diff=" << max_diff << "\n";
    if (max_diff > 1e-2f) {
      throw std::runtime_error("woq_s3 accuracy check failed");
    }

    ctx->deallocate(dRaw);
    ctx->deallocate(dScale);
    ctx->deallocate(dBlob);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }

  // m>1 int3 (S3) symmetric WOQ GEMM accuracy check. Drives the full woq_gemm path with m>1 and
  // compute_type=S8 (matching the Python qlinear cdt="int8") to exercise the fp-compute fallback
  // that decodes the dense 3-bit blob via WeightS3T::dequant and runs a DNNL fp GEMM. Compares
  // against a host fp reference dequant-then-matmul.
  template <typename T>
  void test_woq_s3_gemm(size_t m, size_t n, size_t k, int blocksize) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % blocksize != 0 || blocksize % 32 != 0) {
      throw std::runtime_error("test_woq_s3_gemm requires k % 32 == 0, k % blocksize == 0, blocksize % 32 == 0");
    }
    int blks = int(k) / blocksize;

    std::mt19937 rng(11u + uint32_t(m) + uint32_t(n) + uint32_t(k));
    std::uniform_int_distribution<int> wdist(-4, 3);  // symmetric int3 range
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.05f);

    // raw signed weights, row-major [k, n]; activations [m, k]; per-(blk, n) scales [blks, n].
    std::vector<int8_t> raw(k * n);
    for (auto& w : raw) w = int8_t(wdist(rng));
    std::vector<float> hostA(m * k);
    for (auto& a : hostA) a = adist(rng);
    std::vector<float> hostScale(size_t(blks) * n);
    for (auto& s : hostScale) s = sdist(rng);

    // Host reference: C[i,j] = sum_k A[i,k] * raw[k,j] * scale[k/blocksize, j].
    std::vector<float> refC(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (size_t kk = 0; kk < k; ++kk) {
          float sc = hostScale[(kk / blocksize) * n + j];
          acc += hostA[i * k + kk] * float(raw[kk * n + j]) * sc;
        }
        refC[i * n + j] = acc;
      }
    }

    // compute_type=S8 mirrors the Python XPU path (cdt="int8"); woq_gemm must force S3 to fp.
    ark::QuantParam param{int(n), int(k), blocksize, (int)BTLA_DTYPE::S8, (int)BTLA_DTYPE::S3, (int)BTLA_DTYPE::F32,
                          false};
    size_t blob_size = ark::XpuWrapper::get_packw_size(&param);

    auto* dRaw = reinterpret_cast<int8_t*>(ctx->allocate(raw.size() * sizeof(int8_t)));
    auto* dScale = reinterpret_cast<float*>(ctx->allocate(hostScale.size() * sizeof(float)));
    auto* dBlob = reinterpret_cast<int8_t*>(ctx->allocate(blob_size));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(m * k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(m * n * sizeof(T)));

    std::vector<T> hostAt(m * k);
    for (size_t i = 0; i < m * k; ++i) hostAt[i] = T(hostA[i]);

    q->memcpy(dRaw, raw.data(), raw.size() * sizeof(int8_t)).wait();
    q->memcpy(dScale, hostScale.data(), hostScale.size() * sizeof(float)).wait();
    q->memcpy(dA, hostAt.data(), m * k * sizeof(T)).wait();

    ark::XpuWrapper::packq(dRaw, (void*)dScale, nullptr, dBlob, &param, q);
    q->wait();
    ark::XpuWrapper::woq_gemm(int(m), dA, dBlob, dC, nullptr, BTLA_DTYPE::F32, &param, q);
    q->wait();

    std::vector<T> hostC(m * n);
    q->memcpy(hostC.data(), dC, m * n * sizeof(T)).wait();

    float max_diff = 0.0f;
    for (size_t idx = 0; idx < m * n; ++idx) {
      max_diff = std::max(max_diff, std::fabs(float(hostC[idx]) - refC[idx]));
    }
    std::cout << "[woq_s3][gemm] m=" << m << " n=" << n << " k=" << k << " blk=" << blocksize
              << " max_diff=" << max_diff << "\n";
    if (max_diff > 1e-2f) {
      throw std::runtime_error("woq_s3 gemm accuracy check failed");
    }

    ctx->deallocate(dRaw);
    ctx->deallocate(dScale);
    ctx->deallocate(dBlob);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }

  // Time `fn` over `iters` runs (after `warmup`), returning mean ms/iter. Mirrors TestSDPA::run_bench.
  static double run_bench(const std::function<void()>& fn, sycl::queue* q, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    q->wait();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    q->wait();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / double(iters);
  }

  // BTLA_DTYPE of the activation/output scalar T, for the woq_gemv/woq_gemm `acdt` argument.
  template <typename T>
  static constexpr BTLA_DTYPE acdt_of() {
    if constexpr (std::is_same_v<T, float>)
      return BTLA_DTYPE::F32;
    else if constexpr (std::is_same_v<T, fp16>)
      return BTLA_DTYPE::F16;
    else if constexpr (std::is_same_v<T, bf16>)
      return BTLA_DTYPE::BF16;
    else
      static_assert(sizeof(T) == 0, "unsupported acdt scalar");
  }

  // WOQ GEMV (m==1) throughput for a low-bit symmetric weight type (S3/S4/...). Weight-bound, so
  // report effective GB/s over the packed blob plus the 2*n*k matvec FLOPs. Packs once, times only
  // the woq_gemv call. `tag` (e.g. "s3"/"s4") prefixes the output label.
  template <typename T>
  void bench_woq_gemv(BTLA_DTYPE wtype, const char* tag, const std::string& name, size_t n, size_t k, int blocksize,
                      int warmup, int iters) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % blocksize != 0 || blocksize % 32 != 0) {
      throw std::runtime_error("bench_woq_gemv requires k % 32 == 0, k % blocksize == 0, blocksize % 32 == 0");
    }
    int blks = int(k) / blocksize;
    std::vector<int8_t> raw(k * n, int8_t(1));
    std::vector<float> hostScale(size_t(blks) * n, 0.02f);

    ark::QuantParam param{int(n), int(k), blocksize, (int)BTLA_DTYPE::F32, (int)wtype, (int)BTLA_DTYPE::F32, false};
    size_t blob_size = ark::XpuWrapper::get_packw_size(&param);

    auto* dRaw = reinterpret_cast<int8_t*>(ctx->allocate(raw.size() * sizeof(int8_t)));
    auto* dScale = reinterpret_cast<float*>(ctx->allocate(hostScale.size() * sizeof(float)));
    auto* dBlob = reinterpret_cast<int8_t*>(ctx->allocate(blob_size));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(n * sizeof(T)));
    q->memcpy(dRaw, raw.data(), raw.size() * sizeof(int8_t)).wait();
    q->memcpy(dScale, hostScale.data(), hostScale.size() * sizeof(float)).wait();
    q->memset(dA, 0, k * sizeof(T)).wait();
    ark::XpuWrapper::packq(dRaw, (void*)dScale, nullptr, dBlob, &param, q);
    q->wait();

    double ms = run_bench(
        [&]() { ark::XpuWrapper::woq_gemv(q, 1, &param, dA, dBlob, dC, nullptr, acdt_of<T>()); }, q, warmup, iters);

    double flops = 2.0 * double(n) * double(k);
    double tflops = flops / (ms * 1e-3) / 1e12;
    double gbps = double(blob_size) / (ms * 1e-3) / 1e9;  // weight-bound: packed blob bytes
    std::cout << std::fixed << std::setprecision(4) << "[woq_" << tag << "][gemv_bench] " << name << " n=" << n
              << " k=" << k << " blk=" << blocksize << " ms=" << ms << " TFLOPS=" << tflops << " GBps=" << gbps
              << "\n";

    ctx->deallocate(dRaw);
    ctx->deallocate(dScale);
    ctx->deallocate(dBlob);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }

  // WOQ GEMM (m>1) throughput for a low-bit symmetric weight type. compute_type=S8 mirrors the Python
  // cdt="int8": S4 then takes the native int8-XMX path (woq_s8), while S3 is forced down the
  // fp-dequant + DNNL fp GEMM fallback (no S8-compute unpack for the dense 3-bit blob). Reports
  // 2*m*n*k FLOPs; packs once, times only the woq_gemm call.
  template <typename T>
  void bench_woq_gemm(BTLA_DTYPE wtype, const char* tag, const std::string& name, size_t m, size_t n, size_t k,
                      int blocksize, int warmup, int iters) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % blocksize != 0 || blocksize % 32 != 0) {
      throw std::runtime_error("bench_woq_gemm requires k % 32 == 0, k % blocksize == 0, blocksize % 32 == 0");
    }
    int blks = int(k) / blocksize;
    std::vector<int8_t> raw(k * n, int8_t(1));
    std::vector<float> hostScale(size_t(blks) * n, 0.02f);

    ark::QuantParam param{int(n), int(k), blocksize, (int)BTLA_DTYPE::S8, (int)wtype, (int)BTLA_DTYPE::F32, false};
    size_t blob_size = ark::XpuWrapper::get_packw_size(&param);

    auto* dRaw = reinterpret_cast<int8_t*>(ctx->allocate(raw.size() * sizeof(int8_t)));
    auto* dScale = reinterpret_cast<float*>(ctx->allocate(hostScale.size() * sizeof(float)));
    auto* dBlob = reinterpret_cast<int8_t*>(ctx->allocate(blob_size));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(m * k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(m * n * sizeof(T)));
    q->memcpy(dRaw, raw.data(), raw.size() * sizeof(int8_t)).wait();
    q->memcpy(dScale, hostScale.data(), hostScale.size() * sizeof(float)).wait();
    q->memset(dA, 0, m * k * sizeof(T)).wait();
    ark::XpuWrapper::packq(dRaw, (void*)dScale, nullptr, dBlob, &param, q);
    q->wait();

    double ms = run_bench(
        [&]() { ark::XpuWrapper::woq_gemm(int(m), dA, dBlob, dC, nullptr, acdt_of<T>(), &param, q); }, q, warmup,
        iters);

    double flops = 2.0 * double(m) * double(n) * double(k);
    double tflops = flops / (ms * 1e-3) / 1e12;
    std::cout << std::fixed << std::setprecision(4) << "[woq_" << tag << "][gemm_bench] " << name << " m=" << m
              << " n=" << n << " k=" << k << " blk=" << blocksize << " ms=" << ms << " TFLOPS=" << tflops << "\n";

    ctx->deallocate(dRaw);
    ctx->deallocate(dScale);
    ctx->deallocate(dBlob);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }

  // W10 int3 GEMV perf spike. Host-packs the W10 blob, times only w10_spike::gemv, and prints
  // [woq_w10_spike][gemv_bench] with GB/s over the TRUE W10 blob bytes (blks*wpb*4*n). Also runs a
  // one-shot correctness cross-check against a host dequant-matvec and prints max_diff, so the spike
  // proves correct AND fast in one shot. Uses random weights/activations (unlike the s3/s4 bench
  // which uses constant inputs) so the accuracy number is meaningful.
  template <typename T>
  void bench_woq_w10_spike(const std::string& name, size_t n, size_t k, int blocksize, int warmup, int iters,
                           bool use_pf = false) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % blocksize != 0 || blocksize % 32 != 0) {
      throw std::runtime_error("bench_woq_w10_spike requires k % 32 == 0, k % blocksize == 0, blocksize % 32 == 0");
    }
    int blks = int(k) / blocksize;
    int wpb = w10_spike::words_per_block(blocksize);
    size_t wpc = size_t(blks) * wpb;
    size_t blob_bytes = wpc * n * sizeof(uint32_t);

    std::mt19937 rng(23u + uint32_t(n) + uint32_t(k) + uint32_t(blocksize));
    std::uniform_int_distribution<int> wdist(-4, 3);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.05f);

    std::vector<int8_t> raw(k * n);
    for (auto& w : raw) w = int8_t(wdist(rng));
    std::vector<float> hostA(k);
    for (auto& a : hostA) a = adist(rng);
    std::vector<float> hostScale(size_t(blks) * n);  // [blks, n]
    for (auto& s : hostScale) s = sdist(rng);

    // Host reference: C[j] = sum_k A[k] * raw[k,j] * scale[k/blocksize, j].
    std::vector<float> refC(n, 0.0f);
    for (size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (size_t kk = 0; kk < k; ++kk) acc += hostA[kk] * float(raw[kk * n + j]) * hostScale[(kk / blocksize) * n + j];
      refC[j] = acc;
    }

    // W10 pack on host; scale needs [n, blks] for the kernel's sptr = scale + col*blks layout.
    std::vector<uint32_t> blob = w10_spike::pack(raw, int(n), int(k), blocksize);
    std::vector<T> scaleNK(size_t(blks) * n);
    for (int b = 0; b < blks; ++b)
      for (size_t j = 0; j < n; ++j) scaleNK[j * blks + b] = T(hostScale[size_t(b) * n + j]);
    std::vector<T> hostAt(k);
    for (size_t i = 0; i < k; ++i) hostAt[i] = T(hostA[i]);

    auto* dBlob = reinterpret_cast<uint32_t*>(ctx->allocate(blob_bytes));
    auto* dScale = reinterpret_cast<T*>(ctx->allocate(scaleNK.size() * sizeof(T)));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(n * sizeof(T)));
    q->memcpy(dBlob, blob.data(), blob_bytes).wait();
    q->memcpy(dScale, scaleNK.data(), scaleNK.size() * sizeof(T)).wait();
    q->memcpy(dA, hostAt.data(), k * sizeof(T)).wait();

    // Correctness cross-check (one run).
    auto kern = [&]() {
      if (use_pf)
        w10_spike::gemv_pf<T>(dA, dBlob, dScale, dC, int(n), int(k), blocksize, q);
      else
        w10_spike::gemv<T>(dA, dBlob, dScale, dC, int(n), int(k), blocksize, q);
    };
    kern();
    q->wait();
    std::vector<T> hostC(n);
    q->memcpy(hostC.data(), dC, n * sizeof(T)).wait();
    float max_diff = 0.0f;
    for (size_t j = 0; j < n; ++j) max_diff = std::max(max_diff, std::fabs(float(hostC[j]) - refC[j]));

    double ms = run_bench(kern, q, warmup, iters);

    double flops = 2.0 * double(n) * double(k);
    double tflops = flops / (ms * 1e-3) / 1e12;
    double gbps = double(blob_bytes) / (ms * 1e-3) / 1e9;  // weight-bound: true W10 blob bytes
    std::cout << std::fixed << std::setprecision(4) << "[woq_w10" << (use_pf ? "_pf" : "") << "_spike][gemv_bench] "
              << name << " n=" << n << " k=" << k << " blk=" << blocksize << " ms=" << ms << " TFLOPS=" << tflops
              << " GBps=" << gbps << " max_diff=" << max_diff << "\n";

    ctx->deallocate(dBlob);
    ctx->deallocate(dScale);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }

  // W10 nested block->word variant (BLK compile-time): all bit positions / valid counts compile-time,
  // no div/mod, scale loaded once per block. Lane owns whole blocks. GBps over true W10 blob bytes.
  template <int BLK, typename T>
  void bench_woq_w10_nested(const std::string& name, size_t n, size_t k, int warmup, int iters) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % BLK != 0) {
      throw std::runtime_error("bench_woq_w10_nested requires k % 32 == 0 and k % BLK == 0");
    }
    int blks = int(k) / BLK;
    int wpb = w10_spike::words_per_block(BLK);
    size_t wpc = size_t(blks) * wpb;
    size_t blob_bytes = wpc * n * sizeof(uint32_t);

    std::mt19937 rng(29u + uint32_t(n) + uint32_t(k) + uint32_t(BLK));
    std::uniform_int_distribution<int> wdist(-4, 3);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.05f);

    std::vector<int8_t> raw(k * n);
    for (auto& w : raw) w = int8_t(wdist(rng));
    std::vector<float> hostA(k);
    for (auto& a : hostA) a = adist(rng);
    std::vector<float> hostScale(size_t(blks) * n);  // [blks, n]
    for (auto& s : hostScale) s = sdist(rng);

    std::vector<float> refC(n, 0.0f);
    for (size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (size_t kk = 0; kk < k; ++kk) acc += hostA[kk] * float(raw[kk * n + j]) * hostScale[(kk / BLK) * n + j];
      refC[j] = acc;
    }

    std::vector<uint32_t> blob = w10_spike::pack(raw, int(n), int(k), BLK);
    std::vector<T> scaleNK(size_t(blks) * n);  // kernel wants [n, blks]
    for (int b = 0; b < blks; ++b)
      for (size_t j = 0; j < n; ++j) scaleNK[j * blks + b] = T(hostScale[size_t(b) * n + j]);
    std::vector<T> hostAt(k);
    for (size_t i = 0; i < k; ++i) hostAt[i] = T(hostA[i]);

    auto* dBlob = reinterpret_cast<uint32_t*>(ctx->allocate(blob_bytes));
    auto* dScale = reinterpret_cast<T*>(ctx->allocate(scaleNK.size() * sizeof(T)));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(n * sizeof(T)));
    q->memcpy(dBlob, blob.data(), blob_bytes).wait();
    q->memcpy(dScale, scaleNK.data(), scaleNK.size() * sizeof(T)).wait();
    q->memcpy(dA, hostAt.data(), k * sizeof(T)).wait();

    w10_spike::gemv_nested<BLK, T>(dA, dBlob, dScale, dC, int(n), int(k), q);
    q->wait();
    std::vector<T> hostC(n);
    q->memcpy(hostC.data(), dC, n * sizeof(T)).wait();
    float max_diff = 0.0f;
    for (size_t j = 0; j < n; ++j) max_diff = std::max(max_diff, std::fabs(float(hostC[j]) - refC[j]));

    double ms = run_bench([&]() { w10_spike::gemv_nested<BLK, T>(dA, dBlob, dScale, dC, int(n), int(k), q); }, q, warmup,
                          iters);
    double flops = 2.0 * double(n) * double(k);
    double tflops = flops / (ms * 1e-3) / 1e12;
    double gbps = double(blob_bytes) / (ms * 1e-3) / 1e9;
    std::cout << std::fixed << std::setprecision(4) << "[woq_w10_nested][gemv_bench] " << name << " n=" << n
              << " k=" << k << " blk=" << BLK << " ms=" << ms << " TFLOPS=" << tflops << " GBps=" << gbps
              << " max_diff=" << max_diff << "\n";

    ctx->deallocate(dBlob);
    ctx->deallocate(dScale);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }

  // W10 nested + ILP: identical blob/layout to bench_woq_w10_nested, but the per-block FMAs run over
  // NACC independent accumulators to break the serial bacc dependency chain. Sweeping NACC isolates
  // "does more FMA ILP push int3 past the ~150 GBps nested plateau toward the S4 reference?".
  template <int BLK, int NACC, typename T>
  void bench_woq_w10_nested_ilp(const std::string& name, size_t n, size_t k, int warmup, int iters) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % BLK != 0) {
      throw std::runtime_error("bench_woq_w10_nested_ilp requires k % 32 == 0 and k % BLK == 0");
    }
    int blks = int(k) / BLK;
    int wpb = w10_spike::words_per_block(BLK);
    size_t wpc = size_t(blks) * wpb;
    size_t blob_bytes = wpc * n * sizeof(uint32_t);

    std::mt19937 rng(29u + uint32_t(n) + uint32_t(k) + uint32_t(BLK));
    std::uniform_int_distribution<int> wdist(-4, 3);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.05f);

    std::vector<int8_t> raw(k * n);
    for (auto& w : raw) w = int8_t(wdist(rng));
    std::vector<float> hostA(k);
    for (auto& a : hostA) a = adist(rng);
    std::vector<float> hostScale(size_t(blks) * n);  // [blks, n]
    for (auto& s : hostScale) s = sdist(rng);

    std::vector<float> refC(n, 0.0f);
    for (size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (size_t kk = 0; kk < k; ++kk) acc += hostA[kk] * float(raw[kk * n + j]) * hostScale[(kk / BLK) * n + j];
      refC[j] = acc;
    }

    std::vector<uint32_t> blob = w10_spike::pack(raw, int(n), int(k), BLK);
    std::vector<T> scaleNK(size_t(blks) * n);  // kernel wants [n, blks]
    for (int b = 0; b < blks; ++b)
      for (size_t j = 0; j < n; ++j) scaleNK[j * blks + b] = T(hostScale[size_t(b) * n + j]);
    std::vector<T> hostAt(k);
    for (size_t i = 0; i < k; ++i) hostAt[i] = T(hostA[i]);

    auto* dBlob = reinterpret_cast<uint32_t*>(ctx->allocate(blob_bytes));
    auto* dScale = reinterpret_cast<T*>(ctx->allocate(scaleNK.size() * sizeof(T)));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(n * sizeof(T)));
    q->memcpy(dBlob, blob.data(), blob_bytes).wait();
    q->memcpy(dScale, scaleNK.data(), scaleNK.size() * sizeof(T)).wait();
    q->memcpy(dA, hostAt.data(), k * sizeof(T)).wait();

    w10_spike::gemv_nested_ilp<BLK, NACC, T>(dA, dBlob, dScale, dC, int(n), int(k), q);
    q->wait();
    std::vector<T> hostC(n);
    q->memcpy(hostC.data(), dC, n * sizeof(T)).wait();
    float max_diff = 0.0f;
    for (size_t j = 0; j < n; ++j) max_diff = std::max(max_diff, std::fabs(float(hostC[j]) - refC[j]));

    double ms = run_bench(
        [&]() { w10_spike::gemv_nested_ilp<BLK, NACC, T>(dA, dBlob, dScale, dC, int(n), int(k), q); }, q, warmup, iters);
    double flops = 2.0 * double(n) * double(k);
    double tflops = flops / (ms * 1e-3) / 1e12;
    double gbps = double(blob_bytes) / (ms * 1e-3) / 1e9;
    std::cout << std::fixed << std::setprecision(4) << "[woq_w10_nested_ilp][gemv_bench] " << name << " n=" << n
              << " k=" << k << " blk=" << BLK << " nacc=" << NACC << " ms=" << ms << " TFLOPS=" << tflops
              << " GBps=" << gbps << " max_diff=" << max_diff << "\n";

    ctx->deallocate(dBlob);
    ctx->deallocate(dScale);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }

  // W10 nested + COALESCED SLM staging (+ optional NACC ILP). Same blob/layout as bench_woq_w10_nested;
  // measures whether converting the nested variant's per-lane gathers into a coalesced block load lifts
  // it off the ~150 GBps plateau toward the S4 reference.
  template <int BLK, int NACC, typename T>
  void bench_woq_w10_nested_coal(const std::string& name, size_t n, size_t k, int warmup, int iters) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % BLK != 0) {
      throw std::runtime_error("bench_woq_w10_nested_coal requires k % 32 == 0 and k % BLK == 0");
    }
    int blks = int(k) / BLK;
    int wpb = w10_spike::words_per_block(BLK);
    size_t wpc = size_t(blks) * wpb;
    size_t blob_bytes = wpc * n * sizeof(uint32_t);

    std::mt19937 rng(29u + uint32_t(n) + uint32_t(k) + uint32_t(BLK));
    std::uniform_int_distribution<int> wdist(-4, 3);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.05f);

    std::vector<int8_t> raw(k * n);
    for (auto& w : raw) w = int8_t(wdist(rng));
    std::vector<float> hostA(k);
    for (auto& a : hostA) a = adist(rng);
    std::vector<float> hostScale(size_t(blks) * n);  // [blks, n]
    for (auto& s : hostScale) s = sdist(rng);

    std::vector<float> refC(n, 0.0f);
    for (size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (size_t kk = 0; kk < k; ++kk) acc += hostA[kk] * float(raw[kk * n + j]) * hostScale[(kk / BLK) * n + j];
      refC[j] = acc;
    }

    std::vector<uint32_t> blob = w10_spike::pack(raw, int(n), int(k), BLK);
    std::vector<T> scaleNK(size_t(blks) * n);  // kernel wants [n, blks]
    for (int b = 0; b < blks; ++b)
      for (size_t j = 0; j < n; ++j) scaleNK[j * blks + b] = T(hostScale[size_t(b) * n + j]);
    std::vector<T> hostAt(k);
    for (size_t i = 0; i < k; ++i) hostAt[i] = T(hostA[i]);

    auto* dBlob = reinterpret_cast<uint32_t*>(ctx->allocate(blob_bytes));
    auto* dScale = reinterpret_cast<T*>(ctx->allocate(scaleNK.size() * sizeof(T)));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(n * sizeof(T)));
    q->memcpy(dBlob, blob.data(), blob_bytes).wait();
    q->memcpy(dScale, scaleNK.data(), scaleNK.size() * sizeof(T)).wait();
    q->memcpy(dA, hostAt.data(), k * sizeof(T)).wait();

    w10_spike::gemv_nested_coal<BLK, NACC, T>(dA, dBlob, dScale, dC, int(n), int(k), q);
    q->wait();
    std::vector<T> hostC(n);
    q->memcpy(hostC.data(), dC, n * sizeof(T)).wait();
    float max_diff = 0.0f;
    for (size_t j = 0; j < n; ++j) max_diff = std::max(max_diff, std::fabs(float(hostC[j]) - refC[j]));

    double ms = run_bench(
        [&]() { w10_spike::gemv_nested_coal<BLK, NACC, T>(dA, dBlob, dScale, dC, int(n), int(k), q); }, q, warmup, iters);
    double flops = 2.0 * double(n) * double(k);
    double tflops = flops / (ms * 1e-3) / 1e12;
    double gbps = double(blob_bytes) / (ms * 1e-3) / 1e9;
    std::cout << std::fixed << std::setprecision(4) << "[woq_w10_nested_coal][gemv_bench] " << name << " n=" << n
              << " k=" << k << " blk=" << BLK << " nacc=" << NACC << " ms=" << ms << " TFLOPS=" << tflops
              << " GBps=" << gbps << " max_diff=" << max_diff << "\n";

    ctx->deallocate(dBlob);
    ctx->deallocate(dScale);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }

  // W10 nested + VECTOR per-lane block load (padded layout). Each lane reads vec<uint32_t,VPB> of one
  // padded block; if it lowers to a transpose block load the GBps should jump toward S4. GBps is over
  // the PADDED blob bytes (blks*VPB*4*n) so it reflects the true bytes the kernel moves.
  template <int BLK, int NACC, typename T>
  void bench_woq_w10_nested_vec(const std::string& name, size_t n, size_t k, int warmup, int iters) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % BLK != 0) {
      throw std::runtime_error("bench_woq_w10_nested_vec requires k % 32 == 0 and k % BLK == 0");
    }
    int blks = int(k) / BLK;
    int vpb = w10_spike::vpb_padded(BLK);
    size_t wpc = size_t(blks) * vpb;
    size_t blob_bytes = wpc * n * sizeof(uint32_t);

    std::mt19937 rng(29u + uint32_t(n) + uint32_t(k) + uint32_t(BLK));
    std::uniform_int_distribution<int> wdist(-4, 3);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.05f);

    std::vector<int8_t> raw(k * n);
    for (auto& w : raw) w = int8_t(wdist(rng));
    std::vector<float> hostA(k);
    for (auto& a : hostA) a = adist(rng);
    std::vector<float> hostScale(size_t(blks) * n);  // [blks, n]
    for (auto& s : hostScale) s = sdist(rng);

    std::vector<float> refC(n, 0.0f);
    for (size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (size_t kk = 0; kk < k; ++kk) acc += hostA[kk] * float(raw[kk * n + j]) * hostScale[(kk / BLK) * n + j];
      refC[j] = acc;
    }

    std::vector<uint32_t> blob = w10_spike::pack_padded(raw, int(n), int(k), BLK);
    std::vector<T> scaleNK(size_t(blks) * n);  // kernel wants [n, blks]
    for (int b = 0; b < blks; ++b)
      for (size_t j = 0; j < n; ++j) scaleNK[j * blks + b] = T(hostScale[size_t(b) * n + j]);
    std::vector<T> hostAt(k);
    for (size_t i = 0; i < k; ++i) hostAt[i] = T(hostA[i]);

    auto* dBlob = reinterpret_cast<uint32_t*>(ctx->allocate(blob_bytes));
    auto* dScale = reinterpret_cast<T*>(ctx->allocate(scaleNK.size() * sizeof(T)));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(n * sizeof(T)));
    q->memcpy(dBlob, blob.data(), blob_bytes).wait();
    q->memcpy(dScale, scaleNK.data(), scaleNK.size() * sizeof(T)).wait();
    q->memcpy(dA, hostAt.data(), k * sizeof(T)).wait();

    w10_spike::gemv_nested_vec<BLK, NACC, T>(dA, dBlob, dScale, dC, int(n), int(k), q);
    q->wait();
    std::vector<T> hostC(n);
    q->memcpy(hostC.data(), dC, n * sizeof(T)).wait();
    float max_diff = 0.0f;
    for (size_t j = 0; j < n; ++j) max_diff = std::max(max_diff, std::fabs(float(hostC[j]) - refC[j]));

    double ms = run_bench(
        [&]() { w10_spike::gemv_nested_vec<BLK, NACC, T>(dA, dBlob, dScale, dC, int(n), int(k), q); }, q, warmup, iters);
    double flops = 2.0 * double(n) * double(k);
    double tflops = flops / (ms * 1e-3) / 1e12;
    double gbps = double(blob_bytes) / (ms * 1e-3) / 1e9;
    std::cout << std::fixed << std::setprecision(4) << "[woq_w10_nested_vec][gemv_bench] " << name << " n=" << n
              << " k=" << k << " blk=" << BLK << " nacc=" << NACC << " vpb=" << vpb << " ms=" << ms
              << " TFLOPS=" << tflops << " GBps=" << gbps << " max_diff=" << max_diff << "\n";

    ctx->deallocate(dBlob);
    ctx->deallocate(dScale);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }

  // W10 nested + TRANSPOSED lane-interleaved layout (the structural fix). GBps over the transposed
  // blob bytes (tiles*32*WPB*4*n); the only padding is the final partial tile, so for k a multiple of
  // 32*BLK this equals the true blob bytes and GBps is directly comparable to bench_woq_w10_nested.
  template <int BLK, int NACC, typename T>
  void bench_woq_w10_nested_trans(const std::string& name, size_t n, size_t k, int warmup, int iters) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % BLK != 0) {
      throw std::runtime_error("bench_woq_w10_nested_trans requires k % 32 == 0 and k % BLK == 0");
    }
    int blks = int(k) / BLK;
    int wpb = w10_spike::words_per_block(BLK);
    int tiles = (blks + 31) / 32;
    size_t wpc = size_t(tiles) * 32 * wpb;
    size_t blob_bytes = wpc * n * sizeof(uint32_t);

    std::mt19937 rng(29u + uint32_t(n) + uint32_t(k) + uint32_t(BLK));
    std::uniform_int_distribution<int> wdist(-4, 3);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.05f);

    std::vector<int8_t> raw(k * n);
    for (auto& w : raw) w = int8_t(wdist(rng));
    std::vector<float> hostA(k);
    for (auto& a : hostA) a = adist(rng);
    std::vector<float> hostScale(size_t(blks) * n);  // [blks, n]
    for (auto& s : hostScale) s = sdist(rng);

    std::vector<float> refC(n, 0.0f);
    for (size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (size_t kk = 0; kk < k; ++kk) acc += hostA[kk] * float(raw[kk * n + j]) * hostScale[(kk / BLK) * n + j];
      refC[j] = acc;
    }

    std::vector<uint32_t> blob = w10_spike::pack_transposed(raw, int(n), int(k), BLK);
    std::vector<T> scaleNK(size_t(blks) * n);  // kernel wants [n, blks]
    for (int b = 0; b < blks; ++b)
      for (size_t j = 0; j < n; ++j) scaleNK[j * blks + b] = T(hostScale[size_t(b) * n + j]);
    std::vector<T> hostAt(k);
    for (size_t i = 0; i < k; ++i) hostAt[i] = T(hostA[i]);

    auto* dBlob = reinterpret_cast<uint32_t*>(ctx->allocate(blob_bytes));
    auto* dScale = reinterpret_cast<T*>(ctx->allocate(scaleNK.size() * sizeof(T)));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(n * sizeof(T)));
    q->memcpy(dBlob, blob.data(), blob_bytes).wait();
    q->memcpy(dScale, scaleNK.data(), scaleNK.size() * sizeof(T)).wait();
    q->memcpy(dA, hostAt.data(), k * sizeof(T)).wait();

    w10_spike::gemv_nested_trans<BLK, NACC, T>(dA, dBlob, dScale, dC, int(n), int(k), q);
    q->wait();
    std::vector<T> hostC(n);
    q->memcpy(hostC.data(), dC, n * sizeof(T)).wait();
    float max_diff = 0.0f;
    for (size_t j = 0; j < n; ++j) max_diff = std::max(max_diff, std::fabs(float(hostC[j]) - refC[j]));

    double ms = run_bench(
        [&]() { w10_spike::gemv_nested_trans<BLK, NACC, T>(dA, dBlob, dScale, dC, int(n), int(k), q); }, q, warmup,
        iters);
    double flops = 2.0 * double(n) * double(k);
    double tflops = flops / (ms * 1e-3) / 1e12;
    double gbps = double(blob_bytes) / (ms * 1e-3) / 1e9;
    std::cout << std::fixed << std::setprecision(4) << "[woq_w10_nested_trans][gemv_bench] " << name << " n=" << n
              << " k=" << k << " blk=" << BLK << " nacc=" << NACC << " ms=" << ms << " TFLOPS=" << tflops
              << " GBps=" << gbps << " max_diff=" << max_diff << "\n";

    ctx->deallocate(dBlob);
    ctx->deallocate(dScale);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }

  // W10 transposed layout + SOFTWARE PREFETCH. Same blob/layout as bench_woq_w10_nested_trans; adds
  // joint_prefetch (the lever the ISA showed S4 has and every int3 variant lacks). GBps directly
  // comparable to the trans bench (same byte count).
  template <int BLK, int NACC, typename T>
  void bench_woq_w10_nested_trans_pf(const std::string& name, size_t n, size_t k, int warmup, int iters) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % BLK != 0) {
      throw std::runtime_error("bench_woq_w10_nested_trans_pf requires k % 32 == 0 and k % BLK == 0");
    }
    int blks = int(k) / BLK;
    int wpb = w10_spike::words_per_block(BLK);
    int tiles = (blks + 31) / 32;
    size_t wpc = size_t(tiles) * 32 * wpb;
    size_t blob_bytes = wpc * n * sizeof(uint32_t);

    std::mt19937 rng(29u + uint32_t(n) + uint32_t(k) + uint32_t(BLK));
    std::uniform_int_distribution<int> wdist(-4, 3);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.05f);

    std::vector<int8_t> raw(k * n);
    for (auto& w : raw) w = int8_t(wdist(rng));
    std::vector<float> hostA(k);
    for (auto& a : hostA) a = adist(rng);
    std::vector<float> hostScale(size_t(blks) * n);  // [blks, n]
    for (auto& s : hostScale) s = sdist(rng);

    std::vector<float> refC(n, 0.0f);
    for (size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (size_t kk = 0; kk < k; ++kk) acc += hostA[kk] * float(raw[kk * n + j]) * hostScale[(kk / BLK) * n + j];
      refC[j] = acc;
    }

    std::vector<uint32_t> blob = w10_spike::pack_transposed(raw, int(n), int(k), BLK);
    std::vector<T> scaleNK(size_t(blks) * n);  // kernel wants [n, blks]
    for (int b = 0; b < blks; ++b)
      for (size_t j = 0; j < n; ++j) scaleNK[j * blks + b] = T(hostScale[size_t(b) * n + j]);
    std::vector<T> hostAt(k);
    for (size_t i = 0; i < k; ++i) hostAt[i] = T(hostA[i]);

    auto* dBlob = reinterpret_cast<uint32_t*>(ctx->allocate(blob_bytes));
    auto* dScale = reinterpret_cast<T*>(ctx->allocate(scaleNK.size() * sizeof(T)));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(n * sizeof(T)));
    q->memcpy(dBlob, blob.data(), blob_bytes).wait();
    q->memcpy(dScale, scaleNK.data(), scaleNK.size() * sizeof(T)).wait();
    q->memcpy(dA, hostAt.data(), k * sizeof(T)).wait();

    w10_spike::gemv_nested_trans_pf<BLK, NACC, T>(dA, dBlob, dScale, dC, int(n), int(k), q);
    q->wait();
    std::vector<T> hostC(n);
    q->memcpy(hostC.data(), dC, n * sizeof(T)).wait();
    float max_diff = 0.0f;
    for (size_t j = 0; j < n; ++j) max_diff = std::max(max_diff, std::fabs(float(hostC[j]) - refC[j]));

    double ms = run_bench(
        [&]() { w10_spike::gemv_nested_trans_pf<BLK, NACC, T>(dA, dBlob, dScale, dC, int(n), int(k), q); }, q, warmup,
        iters);
    double flops = 2.0 * double(n) * double(k);
    double tflops = flops / (ms * 1e-3) / 1e12;
    double gbps = double(blob_bytes) / (ms * 1e-3) / 1e9;
    std::cout << std::fixed << std::setprecision(4) << "[woq_w10_nested_trans_pf][gemv_bench] " << name << " n=" << n
              << " k=" << k << " blk=" << BLK << " nacc=" << NACC << " ms=" << ms << " TFLOPS=" << tflops
              << " GBps=" << gbps << " max_diff=" << max_diff << "\n";

    ctx->deallocate(dBlob);
    ctx->deallocate(dScale);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }

  // S3 dense straddle, read direct-from-global + prefetch, NO SLM/barriers. Same harness shape as
  // bench_woq_w10_spike; GBps over the dense 3-bit blob bytes (kgroups*3*4*n).
  template <typename T>
  void bench_s3_direct_spike(const std::string& name, size_t n, size_t k, int blocksize, int warmup, int iters) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % blocksize != 0 || blocksize % 32 != 0) {
      throw std::runtime_error("bench_s3_direct_spike requires k % 32 == 0, k % blocksize == 0, blocksize % 32 == 0");
    }
    int blks = int(k) / blocksize;
    int kgroups = int(k) / 32;
    size_t blob_bytes = size_t(kgroups) * 3 * sizeof(uint32_t) * n;

    std::mt19937 rng(31u + uint32_t(n) + uint32_t(k) + uint32_t(blocksize));
    std::uniform_int_distribution<int> wdist(-4, 3);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.05f);

    std::vector<int8_t> raw(k * n);
    for (auto& w : raw) w = int8_t(wdist(rng));
    std::vector<float> hostA(k);
    for (auto& a : hostA) a = adist(rng);
    std::vector<float> hostScale(size_t(blks) * n);  // [blks, n]
    for (auto& s : hostScale) s = sdist(rng);

    std::vector<float> refC(n, 0.0f);
    for (size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (size_t kk = 0; kk < k; ++kk) acc += hostA[kk] * float(raw[kk * n + j]) * hostScale[(kk / blocksize) * n + j];
      refC[j] = acc;
    }

    std::vector<uint32_t> blob = s3_direct_spike::pack(raw, int(n), int(k));
    std::vector<T> scaleNK(size_t(blks) * n);  // kernel wants [n, blks]
    for (int b = 0; b < blks; ++b)
      for (size_t j = 0; j < n; ++j) scaleNK[j * blks + b] = T(hostScale[size_t(b) * n + j]);
    std::vector<T> hostAt(k);
    for (size_t i = 0; i < k; ++i) hostAt[i] = T(hostA[i]);

    auto* dBlob = reinterpret_cast<uint32_t*>(ctx->allocate(blob_bytes));
    auto* dScale = reinterpret_cast<T*>(ctx->allocate(scaleNK.size() * sizeof(T)));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(n * sizeof(T)));
    q->memcpy(dBlob, blob.data(), blob_bytes).wait();
    q->memcpy(dScale, scaleNK.data(), scaleNK.size() * sizeof(T)).wait();
    q->memcpy(dA, hostAt.data(), k * sizeof(T)).wait();

    s3_direct_spike::gemv<T>(dA, dBlob, dScale, dC, int(n), int(k), blocksize, q);
    q->wait();
    std::vector<T> hostC(n);
    q->memcpy(hostC.data(), dC, n * sizeof(T)).wait();
    float max_diff = 0.0f;
    for (size_t j = 0; j < n; ++j) max_diff = std::max(max_diff, std::fabs(float(hostC[j]) - refC[j]));

    double ms = run_bench([&]() { s3_direct_spike::gemv<T>(dA, dBlob, dScale, dC, int(n), int(k), blocksize, q); }, q,
                          warmup, iters);
    double flops = 2.0 * double(n) * double(k);
    double tflops = flops / (ms * 1e-3) / 1e12;
    double gbps = double(blob_bytes) / (ms * 1e-3) / 1e9;
    std::cout << std::fixed << std::setprecision(4) << "[woq_s3_direct][gemv_bench] " << name << " n=" << n
              << " k=" << k << " blk=" << blocksize << " ms=" << ms << " TFLOPS=" << tflops << " GBps=" << gbps
              << " max_diff=" << max_diff << "\n";

    ctx->deallocate(dBlob);
    ctx->deallocate(dScale);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }
#endif
};