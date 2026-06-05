// =====================================================================================
// SKELETON: IKblockGemmDQCore re-expressed on CuTe atoms (drop the raw joint_matrix API)
// =====================================================================================
// Goal (NOT speed — speed is already 65 TFLOP/s): delete ARK's hand-rolled
// sycl::ext::oneapi::experimental::matrix::joint_matrix{,_load_checked,_mad} calls and the
// SPV_INTEL_subgroup_matrix_multiply_accumulate extension dependency, replacing them with
// sycl-tla's *maintained* CuTe atoms. The groupwise two-level loop STAYS — there is no
// sycl-tla collective that does s8-in + groupwise block-scale (see sycl_tla_gap_viz.html),
// so we drop to the ATOM layer, not the collective layer.
//
// WHAT CHANGES (4 leaf operations) vs WHAT STAYS (the entire algorithm):
//   change  joint_matrix<a/b/acc>            -> CuTe register fragments (make_tensor)
//   change  joint_matrix_load_checked        -> cute::copy(Copy_Atom<XE_2D_U8x..>, gmem, frag)
//   change  joint_matrix_mad                 -> cute::gemm(MMA_Atom<XE_8x16x32_S32S8S8S32_TT>)
//   change  joint_matrix_prefetch (Prefetcher::next) -> cute::prefetch(CopyAtom, gmem_tile)
//   change  get_wi_data(sg,c)[i]             -> tCrC(i)        // fragment IS the per-lane regs
//   STAY    for(ib<blks) / for(ik in block)  // two-level groupwise loop — unchanged
//   STAY    sBs[in] block-scale + acc_c +=   // per-128-block int32->float rescale — unchanged
//   STAY    scaleA epilogue, bias, work-group/sub-group geometry, PrefetchDis arithmetic — unchanged
//
// NB: joint_matrix_prefetch is the SAME experimental-matrix namespace as _mad/_load_checked, so
//     "drop the raw API" includes the Prefetcher. It is a 4th call site, NOT untouched.
//
// Reference atoms/ops confirmed present in the pinned tree:
//   include/cute/arch/mma_xe.hpp:151           CUTE_DECLARE_XE_DPAS_TT(d, s8, s8, d)  // current API
//   include/cute/arch/mma_xe_legacy.hpp:390    struct XE_8x16x32_S32S8S8S32_TT         // named alias
//   include/cute/algorithm/prefetch.hpp:97     cute::prefetch(Copy_Atom, src_tensor)   // gated on
//   include/cute/arch/copy_xe_legacy_builtin.hpp:68  XeSubgroup2DBlockPrefetch<...>      // CopyOp::PREFETCH
// =====================================================================================
#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
// ARK config/param structs reused verbatim:
#include "sycl/sycl_gemm.h"   // bestla::sycl_gemm::xmx::IKblockGemmDQCfg / IKblockGemmDQParam

namespace bestla {
namespace sycl_gemm {
namespace xmx {

template <typename TP, typename CFG>
struct IKblockGemmDQCore_CuTe {
  using DT     = typename CFG::DT;      // int8_t   (A/B element fed to DPAS)
  using DT_ACC = typename CFG::DT_ACC;  // int32_t  (accumulator)
  using DQT    = typename CFG::DQT;     // half     (scale / output)
  using Param  = typename CFG::Param;

  // ---- tile constants: unchanged, just re-used to build the CuTe atom/tiling ----
  static constexpr int TM = CFG::TM;    // 8  ── matches atom M
  static constexpr int TN = CFG::TN;    // 16 ── matches atom N
  static constexpr int TK = CFG::TK;    // 32 ── matches atom K  (so 1 atom == 1 (TM,TN,TK) MAD)
  static constexpr int SGM = CFG::SGM;  // 4  ── per-subgroup register blocking in M
  static constexpr int SGN = CFG::SGN;  // 2  ── per-subgroup register blocking in N
  static constexpr int UnrollK = CFG::UnrollK;

  // =====================================================================================
  // (1) THE ATOM — replaces every `joint_matrix_mad`.
  //     XE_8x16x32_S32S8S8S32_TT: D[8x16]s32 += A[8x32]s8 * B[32x16]s8. Exactly TM,TN,TK.
  // =====================================================================================
  using MmaAtom = cute::MMA_Atom<cute::XE_8x16x32_S32S8S8S32_TT>;

  // (1b) 2D block-load copy atoms — replace `joint_matrix_load_checked`.
  //      A is row-major (use::a, layout::row_major)  -> *_LD_N
  //      B is col-major (use::b, layout::col_major)  -> *_LD_T   (transposed load)
  //      TODO(verify): exact U8 tile dims must satisfy the Xe 2D-block constraints AND cover
  //      TM*TK (A) / TK*TN (B). Start from the dims the legacy mixed-dtype bench uses for s8
  //      and adjust; the names below are placeholders to be pinned during bring-up.
  using CopyAtomA = cute::Copy_Atom<cute::XE_2D_U8x32x32_LD_N, DT>;  // TODO(verify dims)
  using CopyAtomB = cute::Copy_Atom<cute::XE_2D_U8x32x16_LD_T, DT>;  // TODO(verify dims)
  // TODO(verify prefetch): cute::prefetch(atom, t) is a no-op unless the chosen CopyOp exposes a
  //   `PREFETCH` typedef (has_prefetch gate, prefetch.hpp:89). The 2D-block U8 load atoms above
  //   must resolve to ops with a matching XeSubgroup2DBlockPrefetch<...> specialization
  //   (copy_xe_legacy_builtin.hpp) for the prefetch swap to actually emit hardware prefetches.
  //   If they don't, either pick a prefetch-capable U8 atom or keep a thin joint_matrix_prefetch
  //   shim for ONLY this op (still removes _mad/_load_checked, the bulk of the raw-API surface).

  Param param; TP mProp;
  int g_ncnt, g_mcnt, blks; /* … same as original ctor … */

  IKblockGemmDQCore_CuTe(TP Prop, CFG, Param p) : param(p), mProp(Prop) {
    // identical to IKblockGemmDQCore ctor (BM/BN counts, blks = k / blocksize, G_NROW packing)
  }
  inline sycl::nd_range<2> get_range() { /* identical */ }

  void operator() [[sycl::reqd_sub_group_size(CFG::sg_size)]] (sycl::nd_item<2> it) const {
    auto sg = it.get_sub_group();

    // -------- work-group / sub-group geometry: copy VERBATIM from the original --------
    //   g_n,g_m, sggid_col/row, sgId, g_idn,g_idm, pointers Awg_d/Bwg_d/Cwg_d,
    //   the annotated_ptr cache hints (pA,pB,pSB,pC), bounds `if (g_idn>=n||g_idm>=m) return;`
    //   ... unchanged ...
    int blocksize = param.blocksize;
    auto k = param.k;
    // (Awg_d / Bwg_d are the int8 tiles already widened by PASS-1 unpackq — same as today.)

    // =====================================================================================
    // (2) FRAGMENTS — replace the three joint_matrix<> arrays.
    //     A CuTe MMA fragment already *is* the per-work-item register tensor, so the later
    //     get_wi_data() hack disappears (see point (4)).
    // =====================================================================================
    TiledMMA tiled_mma = make_tiled_mma(MmaAtom{});      // TODO: tile by SGM×SGN (see note ‡)
    auto thr_mma = tiled_mma.get_slice(sg.get_local_id()[0]);

    // SGM×SGN register-blocked accumulators (mirror sub_c[SGM*SGN]); int32.
    auto tCrC = partition_fragment_C(tiled_mma, cute::Shape<cute::C<TM*SGM>, cute::C<TN*SGN>>{});
    auto tCrA = thr_mma.partition_fragment_A(/* A tile gA(TM*SGM, TK) */);   // mirrors sub_a[SGM]
    auto tCrB = thr_mma.partition_fragment_B(/* B tile gB(TK, TN*SGN) */);   // mirrors sub_b[SGN]

    // float spill accumulator across blocks — UNCHANGED (this is the groupwise carry).
    float acc_c[SGM * SGN * TM];
    for (int i = 0; i < SGM*SGN*TM; ++i) acc_c[i] = 0.f;

    // ---- prologue prefetch: replaces Prefetcher<CFG>::next(sg, 0, ...) ----
    // Same gmem A/B tiles, look-ahead by PrefetchDis; only the intrinsic changes.
    //   cute::prefetch(CopyAtomA{}, gA(_, 0 + TK*UnrollK*PrefetchDis));
    //   cute::prefetch(CopyAtomB{}, gB(0 + TK*UnrollK*PrefetchDis, _));
    // (the sggid_col/sggid_row work-split + the `+TM/+TN <= m/n` bounds stay as index math)
    // group_barrier — UNCHANGED.

    // =====================================================================================
    // (3) THE TWO-LEVEL GROUPWISE LOOP — structurally identical to the original.
    //     Outer: per 128-element block (groupwise scale boundary).
    //     Inner: pure int32-accumulate DPAS over the block — now via cute::gemm.
    // =====================================================================================
    for (int ib = 0; ib < blks; ib++) {
      cute::clear(tCrC);                                   // == joint_matrix_fill(sub_c, 0)

      #pragma unroll(1)
      for (int ik = ib*blocksize; ik < (ib+1)*blocksize; ik += TK*UnrollK) {
        #pragma unroll
        for (int ikk = 0; ikk < UnrollK; ikk++) {
          // ---- load s8 A/B tiles for this k-step: replaces joint_matrix_load_checked ----
          // copy(CopyAtomA{}, gA(_, ik+ikk*TK), tCrA);    // 2D block load, bounds via gmem tensor
          // copy(CopyAtomB{}, gB(ik+ikk*TK, _), tCrB);
          //
          // ---- the matmul: replaces the SGM×SGN joint_matrix_mad nest ----
          cute::gemm(tiled_mma, tCrA, tCrB, tCrC);          // int32 += s8 * s8  (pure DPAS)
        }
        // ---- in-loop prefetch: replaces Prefetcher<CFG>::next(sg, ik, ...) ----
        //   cute::prefetch(CopyAtomA{}, gA(_, ik + TK*UnrollK*PrefetchDis));
        //   cute::prefetch(CopyAtomB{}, gB(ik + TK*UnrollK*PrefetchDis, _));
        // The `if (ik + TK*UnrollK*(PrefetchDis+1) <= k)` guard stays as plain index math.
does       }
      sycl::group_barrier(sg);

      // =================================================================================
      // (4) PER-BLOCK GROUPWISE RESCALE — same math, cleaner access.
      //     OLD: get_wi_data(sg, sub_c[..]) then index wi_data_c[i].
      //     NEW: tCrC is already the per-lane register fragment — index it directly.
      // =================================================================================
      DQT sBs[SGN];
      #pragma unroll
      for (int in = 0; in < SGN; in++) {
        if (in*TN + /*g_idn*/0 + /*sgId*/0 >= param.n) break;
        sBs[in] = /* pSB[(in*TN + g_idn + sgId)*blks + ib] */ DQT(0);   // UNCHANGED indexing
      }
      #pragma unroll
      for (int im = 0; im < SGM; im++) {
        #pragma unroll
        for (int in = 0; in < SGN; in++) {
          // iterate this (im,in) accumulator sub-tile's per-lane elements
          CUTE_UNROLL
          for (int i = 0; i < TM /* elems this lane holds for the sub-tile */; ++i) {
            auto element = tCrC(/* offset(im,in,i) */);    // <-- replaces wi_data_c[i]
            acc_c[(im*TM + i)*SGN + in] += float(element) * float(sBs[in]);   // UNCHANGED
          }
        }
      }
      sycl::group_barrier(sg);
    }

    // =====================================================================================
    // (5) EPILOGUE — scaleA (per-row act scale) + bias + store. COPY VERBATIM.
    //     pC[(im*TM+imm)*ldc + sgId + in*TN] = acc_c[...] * scale2 + bBs[in];
    // =====================================================================================
  }

  auto get(syclex::properties_tag) const { return mProp; }
};

// ‡ NOTE on TiledMMA shape — the one real design decision.
//   ARK expresses register blocking by *literally looping* SGM×SGN atom calls inside one
//   subgroup, and the subgroup *grid* by sg_row×sg_col. CuTe can express both inside a
//   TiledMMA (AtomLayout = sg_row×sg_col across subgroups, PermutationMNK / value-layout =
//   SGM×SGN within a subgroup). Two valid skeletons:
//     (a) FAITHFUL: keep ARK's explicit `for(im<SGM) for(in<SGN)` nest, call cute::gemm on
//         single-atom fragments — smallest diff, easiest to validate against the UT.
//     (b) IDIOMATIC: fold SGM/SGN/sg_row/sg_col into one TiledMMA, single cute::gemm. Cleaner,
//         but you must get the partitioning to reproduce ARK's exact data layout.
//   Recommend (a) first (prove numerics vs UT_SyclInt4Dequant), then optionally refactor to (b).

}  // namespace xmx
}  // namespace sycl_gemm
}  // namespace bestla
