//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#include <array>

#include "bestla_utils.h"
#include "bestla_jit.h"

namespace bestla {
namespace gemm {
enum class CompType : uint16_t {
  // base type, too many bits if reuse BTLA_DTYPE
  tFP32 = 0,
  tBF16 = 1,
  tFP16 = 2,
  tS8 = 3,
  tU8 = 4,
  tS32 = 5,
  tS16 = 6,
  MASK_A = 0xf,
  SHIFT_A = 0,
  MASK_B = 0xf0,
  SHIFT_B = 4,
  MASK_C = 0xf00,
  SHIFT_C = 8,
  COMP_FP32 = (tFP32 << SHIFT_A) | (tFP32 << SHIFT_B) | (tFP32 << SHIFT_C),
  COMP_BF16_FP32 = (tBF16 << SHIFT_A) | (tBF16 << SHIFT_B) | (tFP32 << SHIFT_C),
  COMP_FP16_FP16 = (tFP16 << SHIFT_A) | (tFP16 << SHIFT_B) | (tFP16 << SHIFT_C),
  COMP_FP16_FP32 = (tFP16 << SHIFT_A) | (tFP16 << SHIFT_B) | (tFP32 << SHIFT_C),
  COMP_INT8_US_INT32 = (tU8 << SHIFT_A) | (tS8 << SHIFT_B) | (tS32 << SHIFT_C),
  COMP_INT8_UU_INT32 = (tU8 << SHIFT_A) | (tU8 << SHIFT_B) | (tS32 << SHIFT_C),
  COMP_INT8_SS_INT32 = (tS8 << SHIFT_A) | (tS8 << SHIFT_B) | (tS32 << SHIFT_C),
  COMP_INT8_SU_INT32 = (tS8 << SHIFT_A) | (tU8 << SHIFT_B) | (tS32 << SHIFT_C),
  COMP_INT16_SS_INT32 = (tS16 << SHIFT_A) | (tS16 << SHIFT_B) | (tS32 << SHIFT_C),
  COMP_INT8_US_FP32 = (tU8 << SHIFT_A) | (tS8 << SHIFT_B) | (tFP32 << SHIFT_C),
  COMP_INT8_UU_FP32 = (tU8 << SHIFT_A) | (tU8 << SHIFT_B) | (tFP32 << SHIFT_C),
  COMP_INT8_SS_FP32 = (tS8 << SHIFT_A) | (tS8 << SHIFT_B) | (tFP32 << SHIFT_C),
  COMP_INT8_SU_FP32 = (tS8 << SHIFT_A) | (tU8 << SHIFT_B) | (tFP32 << SHIFT_C),
};

class CompTypeHelper {
 public:
  static inline uint64_t get_mask_val(CompType raw, CompType mask, CompType shift) {
    return (static_cast<uint64_t>(raw) & static_cast<uint64_t>(mask)) >> static_cast<uint64_t>(shift);
  }

  static void parse_id(CompType id, uint64_t* vals) {
    vals[0] = get_mask_val(id, CompType::MASK_A, CompType::SHIFT_A);
    vals[1] = get_mask_val(id, CompType::MASK_B, CompType::SHIFT_B);
    vals[2] = get_mask_val(id, CompType::MASK_C, CompType::SHIFT_C);
  }

  static const char* to_str(CompType id) {
    static char tmp[128];
    uint64_t vals[3];
    parse_id(id, vals);
    sprintf(tmp, "A%d_B%d_C%d", static_cast<int>(vals[0]), static_cast<int>(vals[1]), static_cast<int>(vals[2]));
    return tmp;
  }

  static inline uint64_t get_B(CompType id) { return get_mask_val(id, CompType::MASK_B, CompType::SHIFT_B); }

  static inline BTLA_DTYPE to_dtype(CompType _id_type) {
    switch (_id_type) {
      case CompType::tBF16:
        return BTLA_DTYPE::BF16;
      case CompType::tFP16:
        return BTLA_DTYPE::F16;
      case CompType::tFP32:
        return BTLA_DTYPE::F32;
      case CompType::tS8:
        return BTLA_DTYPE::S8;
      case CompType::tU8:
        return BTLA_DTYPE::U8;
      default:
        return BTLA_DTYPE::EleBitsUndef;
    }
  }

  static inline bool is_integer(CompType id) {
    auto bt = get_B(id);
    bool flag = false;
    flag |= bt == static_cast<uint64_t>(CompType::tS8);
    flag |= bt == static_cast<uint64_t>(CompType::tU8);
    return flag;
  }
};

class CoreAttr {
 public:
  // INT64=LSB|**8bits:NTile**||**8bits:PackRow**||**16bits:CompType**||**8bits:ISA**||**24bits:reversed**|
  static uint64_t constexpr NTILE_MASK = 0xff, NTILE_SHIFT = 0, PACKROW_MASK = 0xff00, PACKROW_SHIFT = 8,
                            COMP_MASK = 0xffff0000, COMP_SHIFT = 16, ISA_MASK = 0xff00000000, ISA_SHIFT = 32;

  static inline uint64_t get_mask_val(uint64_t raw, uint64_t mask, uint64_t shift) { return (raw & mask) >> shift; }

  static constexpr uint64_t make_core_id(int NTile, int PackRow, CompType CompType, BTLA_ISA ISA) {
    return (static_cast<uint64_t>(NTile) << NTILE_SHIFT) | (static_cast<uint64_t>(PackRow) << PACKROW_SHIFT) |
           (static_cast<uint64_t>(CompType) << COMP_SHIFT) | (static_cast<uint64_t>(ISA) << ISA_SHIFT);
  }
  static void parse_id(uint64_t id, uint64_t* vals) {
    vals[0] = get_mask_val(id, NTILE_MASK, NTILE_SHIFT);
    vals[1] = get_mask_val(id, PACKROW_MASK, PACKROW_SHIFT);
    vals[2] = get_mask_val(id, COMP_MASK, COMP_SHIFT);
    vals[3] = get_mask_val(id, ISA_MASK, ISA_SHIFT);
  }

  static const char* to_str(uint64_t id) {
    static char tmp[128];
    uint64_t vals[4];
    parse_id(id, vals);
    sprintf(tmp, "N%d_PACK%d_COMP%d_ISA%d", static_cast<int>(vals[0]), static_cast<int>(vals[1]),
            static_cast<int>(vals[2]), static_cast<int>(vals[3]));
    return tmp;
  }

  static inline int get_packrow(uint64_t id) { return static_cast<int>(get_mask_val(id, PACKROW_MASK, PACKROW_SHIFT)); }

  static inline size_t get_bsize(uint64_t id) {
    auto packrow = get_packrow(id);
    return size_t(4 / packrow);
  }

  static inline BTLA_ISA get_ISA(uint64_t id) { return static_cast<BTLA_ISA>(get_mask_val(id, ISA_MASK, ISA_SHIFT)); }

  static inline CompType get_comp(uint64_t id) {
    return static_cast<CompType>(get_mask_val(id, COMP_MASK, COMP_SHIFT));
  }
};

namespace code {

template <int _NTILE, int _MTILE = 0>
class Avx2N8P1 : protected bestla::xbyak::JitAvx2 {
 public:
  static int constexpr RegLen = 8, PackRow = 1;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 1;
  static int constexpr KUNROLL = 2;
  static auto constexpr ISA = BTLA_ISA::AVX2;
  static auto constexpr COMPUTE = CompType::COMP_FP32;
  typedef float AType;
  typedef float BType;
  typedef float CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* match;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;
  Xbyak::Opmask msk_wr = k1;

  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = RegCount - ARegCount - CRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg <= RegCount);
    TmpRegCount = RegCount - TmpReg;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _ktile) {
    for (int kk = 0; kk < _ktile; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vbroadcastss(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vfmadd231ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vbroadcastss(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vfmadd231ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                          ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class Avx512fN16P1 : protected bestla::xbyak::JitAvx512f {
 public:
  static int constexpr RegLen = 16, PackRow = 1;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 1;
  static int constexpr KUNROLL = 2;
  static auto constexpr ISA = BTLA_ISA::AVX512F;
  static auto constexpr COMPUTE = CompType::COMP_FP32;
  typedef float AType;
  typedef float BType;
  typedef float CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* match;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;
  Xbyak::Opmask msk_wr = k1;

  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = RegCount - ARegCount - CRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg <= RegCount);
    TmpRegCount = RegCount - TmpReg;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _ktile) {
    for (int kk = 0; kk < _ktile; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vbroadcastss(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vfmadd231ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vbroadcastss(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vfmadd231ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                          ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class Avx512fp16N32P1 : protected bestla::xbyak::JitAvx512_fp16 {
 public:
  static int constexpr RegLen = 32, PackRow = 1;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 1;
  static int constexpr KUNROLL = 2;
  static auto constexpr ISA = BTLA_ISA::AVX512_FP16;
  static auto constexpr COMPUTE = CompType::COMP_FP16_FP16;
  typedef utils::fp16 AType;
  typedef utils::fp16 BType;
  typedef utils::fp16 CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* match;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;
  Xbyak::Opmask msk_wr = k1;

  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = RegCount - ARegCount - CRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg <= RegCount);
    TmpRegCount = RegCount - TmpReg;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _ktile) {
    for (int kk = 0; kk < _ktile; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastw(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vfmadd231ph(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vpbroadcastw(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vfmadd231ph(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                          ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class Avx512bf16N16P2 : protected bestla::xbyak::JitAvx512_bf16 {
 public:
  static int constexpr RegLen = 16, PackRow = 2;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 2;
  static int constexpr KUNROLL = 2;
  static auto constexpr ISA = BTLA_ISA::AVX512_BF16;
  static auto constexpr COMPUTE = CompType::COMP_BF16_FP32;
  typedef utils::bf16 AType;
  typedef utils::bf16 BType;
  typedef float CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* match;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;
  Xbyak::Opmask msk_wr = k1;

  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = RegCount - ARegCount - CRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg <= RegCount);
    TmpRegCount = RegCount - TmpReg;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _ktile) {
    for (int kk = 0; kk < _ktile; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vbroadcastss(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vdpbf16ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vbroadcastss(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vdpbf16ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                        ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class Avx512vnniN16P4 : protected bestla::xbyak::JitAvx512vnni {
 public:
  static int constexpr RegLen = 16, PackRow = 4;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 4;
  static int constexpr KUNROLL = 2;
  static auto constexpr ISA = BTLA_ISA::AVX512_VNNI;
  static auto constexpr COMPUTE = CompType::COMP_INT8_US_INT32;
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef int32_t CType;
  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* match;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 private:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;

 protected:
  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = RegCount - ARegCount - CRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg <= RegCount);
    TmpRegCount = RegCount - TmpReg;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _kunroll) {
    for (int kk = 0; kk < _kunroll; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vpbroadcastd(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                         ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class Avx512bwN16P4 : protected bestla::xbyak::JitAvx512bw {
 public:
  static int constexpr RegLen = 16, PackRow = 4;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr KeepRegs = 3;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - KeepRegs) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 4;
  static int constexpr KUNROLL = 2;
  static auto constexpr ISA = BTLA_ISA::AVX512BW;
  static auto constexpr COMPUTE = CompType::COMP_INT8_US_INT32;
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef int32_t CType;
  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* match;
    int cstride;
    int k;
    int n;
    int init;
    const int16_t one = 1;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 private:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;

 protected:
  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    TmpRegCount = 2;

    BRegCount = RegCount - ARegCount - CRegCount - TmpRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg + TmpRegCount <= RegCount);
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);
    vpbroadcastw(vreg_t(TmpReg + 0), ptr[parambase + OFFSET(one)]);
    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }
  void generate_fma(int _mtile, int _kunroll) {
    for (int kk = 0; kk < _kunroll; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(TmpReg + 1), vreg_t(AReg), vreg_t(BReg + i),
                       vreg_t(TmpReg + 0));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vpbroadcastd(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(TmpReg + 1), vreg_t(AReg + imm),
                         ptr[reg_matBptr + kk * BKStepSize + i * VecBytes], vreg_t(TmpReg + 0));
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <typename AT, int _NTILE, int _MTILE = 0>
class AvxvnniN8P4 : protected bestla::xbyak::JitAvxvnni {
 public:
  static int constexpr RegLen = 8, PackRow = 4;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr KeepRegs = std::is_same_v<AT, uint8_t> ? 1 : 3;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - KeepRegs) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - KeepRegs);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 4;
  static int constexpr KUNROLL = 2;
  static auto constexpr ISA = BTLA_ISA::AVX_VNNI;
  static auto constexpr COMPUTE =
      std::is_same_v<AT, uint8_t> ? CompType::COMP_INT8_US_INT32 : CompType::COMP_INT8_SS_INT32;
  using AType = AT;
  typedef int8_t BType;
  typedef int32_t CType;
  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* match;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 private:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;
  Xbyak::Opmask msk_wr = k1;

 protected:
  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    if (std::is_same_v<AT, int8_t>) {
      TmpRegCount = 2;
    }
    BRegCount = RegCount - ARegCount - CRegCount - TmpRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg + TmpRegCount <= RegCount);
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _kunroll) {
    for (int kk = 0; kk < _kunroll; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(vreg_t(AReg), ptr[reg_tmp1]);
          if constexpr (std::is_same_v<AType, int8_t>) {
            vpsignb(vreg_t(TmpReg + 1), vreg_t(AReg), vreg_t(AReg));
          }
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            if constexpr (std::is_same_v<AType, uint8_t>) {
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
            } else {
              vpsignb(vreg_t(TmpReg), vreg_t(BReg + i), vreg_t(AReg));
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(TmpReg + 1), vreg_t(TmpReg));
            }
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vpbroadcastd(vreg_t(AReg + imm), ptr[reg_tmp1]);
            if constexpr (std::is_same_v<AType, int8_t>) {
              vpsignb(vreg_t(TmpReg + 1), vreg_t(AReg + imm), vreg_t(AReg + imm));
            }
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              if constexpr (std::is_same_v<AType, uint8_t>) {
                vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                           ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
              } else {
                vmovups(vreg_t(TmpReg), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
                vpsignb(vreg_t(TmpReg), vreg_t(TmpReg), vreg_t(AReg + imm));
                vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(TmpReg + 1), vreg_t(TmpReg));
              }
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int N, int M>
using AvxvnniN8P4U8 = AvxvnniN8P4<uint8_t, N, M>;

template <int N, int M>
using AvxvnniN8P4S8 = AvxvnniN8P4<int8_t, N, M>;

template <typename AT, int _NTILE, int _MTILE = 0>
class Avx2vnniN8P4 : protected bestla::xbyak::JitAvx2 {
 public:
  static int constexpr RegLen = 8, PackRow = 4;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr KeepRegs = std::is_same_v<AT, uint8_t> ? 3 : 5;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - KeepRegs) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - KeepRegs);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 4;
  static int constexpr KUNROLL = 2;
  static auto constexpr ISA = BTLA_ISA::AVX2;
  static auto constexpr COMPUTE =
      std::is_same_v<AT, uint8_t> ? CompType::COMP_INT8_US_INT32 : CompType::COMP_INT8_SS_INT32;
  using AType = AT;
  typedef int8_t BType;
  typedef int32_t CType;
  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* match;
    int cstride;
    int k;
    int n;
    int init;
    const int16_t one = 1;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 private:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;
  Xbyak::Opmask msk_wr = k1;

 protected:
  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    if (std::is_same_v<AT, int8_t>) {
      TmpRegCount = 4;
    } else {
      TmpRegCount = 2;
    }
    BRegCount = RegCount - ARegCount - CRegCount - TmpRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg + TmpRegCount <= RegCount);
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);
    vpbroadcastw(vreg_t(TmpReg + 0), ptr[parambase + OFFSET(one)]);
    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _kunroll) {
    for (int kk = 0; kk < _kunroll; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(vreg_t(AReg), ptr[reg_tmp1]);
          if constexpr (std::is_same_v<AType, int8_t>) {
            vpsignb(vreg_t(TmpReg + 2), vreg_t(AReg), vreg_t(AReg));
          }
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            if constexpr (std::is_same_v<AType, uint8_t>) {
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(TmpReg + 1), vreg_t(AReg), vreg_t(BReg + i),
                         vreg_t(TmpReg + 0));
            } else {
              vpsignb(vreg_t(TmpReg + 3), vreg_t(BReg + i), vreg_t(AReg));
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(TmpReg + 1), vreg_t(TmpReg + 2), vreg_t(TmpReg + 3),
                         vreg_t(TmpReg + 0));
            }
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vpbroadcastd(vreg_t(AReg + imm), ptr[reg_tmp1]);
            if constexpr (std::is_same_v<AType, int8_t>) {
              vpsignb(vreg_t(TmpReg + 2), vreg_t(AReg + imm), vreg_t(AReg + imm));
            }
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              if constexpr (std::is_same_v<AType, uint8_t>) {
                vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(TmpReg + 1), vreg_t(AReg + imm),
                           ptr[reg_matBptr + kk * BKStepSize + i * VecBytes], vreg_t(TmpReg + 0));
              } else {
                vmovups(vreg_t(TmpReg + 3), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
                vpsignb(vreg_t(TmpReg + 3), vreg_t(TmpReg + 3), vreg_t(AReg + imm));
                vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(TmpReg + 1), vreg_t(TmpReg + 2), vreg_t(TmpReg + 3),
                           vreg_t(TmpReg + 0));
              }
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int N, int M>
using Avx2vnniN8P4U8 = Avx2vnniN8P4<uint8_t, N, M>;

template <int N, int M>
using Avx2vnniN8P4S8 = Avx2vnniN8P4<int8_t, N, M>;

template <int _NTILE, int _MTILE = 0>
class Amxbf16N16P2 : protected bestla::xbyak::JitAmxbf16 {
 public:
  static int constexpr RegLen = 16, PackRow = 2;
  static_assert(_NTILE % RegLen == 0);
  static_assert(_MTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? 1 : _MTILE / RegLen;
  static_assert(NRegs * MRegs + 2 <= TileCount);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs * RegLen, KTILE = 32;
  static int constexpr KUNROLL = 2;
  static auto constexpr ISA = BTLA_ISA::AMX_BF16;
  static auto constexpr COMPUTE = CompType::COMP_BF16_FP32;
  typedef utils::bf16 AType;
  typedef utils::bf16 BType;
  typedef float CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* match;
    int cstride;
    int k;
    int n;
    int init;
    void* workspace;
  };
  typedef long long (*func_t)(params*);

  int TmpRegCount = RegCount;
  int TmpReg = 0;
  int CTileCount = 0, ATileCount = 0, BTileCount = 0;
  int CTile = 0, ATile = 0, BTile = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_tmp3;
  Xbyak::Reg64 reg_ret = rax;

  void assign_regs() {
    CTileCount = NRegs * MRegs;
    auto tile_re = TileCount - CTileCount;
    if (tile_re - 1 >= NRegs) {
      BTileCount = NRegs;
      ATileCount = tile_re - BTileCount;
    } else if (tile_re - 1 >= MRegs) {
      ATileCount = MRegs;
      BTileCount = tile_re - ATileCount;
    } else {
      ATileCount = 1;
      BTileCount = tile_re - ATileCount;
    }
    CTile = 0;
    ATile = CTile + CTileCount;
    BTile = ATile + ATileCount;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_tmp3 = st.t[10];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int kunrll) {
    auto& reg_Bstride = reg_tmp1;
    mov(reg_Bstride, NTILE * 4);
    int mtiles = _mtile / RegLen;
    for (int kk = 0; kk < kunrll; kk++) {
      auto reg_Atmp = reg_tmp2;
      if (mtiles == 1) {
        reg_Atmp = reg_matAptr;
      } else {
        mov(reg_Atmp, reg_matAptr);
      }
      if (BTileCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
        }
        for (int mm = 0; mm < mtiles; mm++) {
          tileloadd(Xbyak::Tmm(ATile), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
          for (int i = 0; i < NRegs; i++) {
            tdpbf16ps(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile), Xbyak::Tmm(BTile + i));
          }
          if (mm != mtiles - 1) {
            lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
          }
        }
      } else {
        if (ATileCount == mtiles) {
          for (int mm = 0; mm < mtiles; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
            if (mm != mtiles - 1) {
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            }
          }
          for (int i = 0; i < NRegs; i++) {
            tileloaddt1(Xbyak::Tmm(BTile), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
            for (int mm = 0; mm < mtiles; mm++) {
              tdpbf16ps(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile));
            }
          }
        } else {
          for (int mm = 0; mm < mtiles; mm++) {
            tileloadd(Xbyak::Tmm(ATile), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
            for (int i = 0; i < NRegs; i++) {
              tileloaddt1(Xbyak::Tmm(BTile), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
              tdpbf16ps(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile), Xbyak::Tmm(BTile));
            }
            if (mm != mtiles - 1) {
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            }
          }
        }
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < CTileCount; i++) {
      tilezero(Xbyak::Tmm(CTile + i));
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    int mtnum = _mtile / 16;
    for (int mm = 0; mm < mtnum; mm++) {
      for (int i = 0; i < NRegs; i++) {
        tileloaddt1(Xbyak::Tmm(CTile + mm * NRegs + i), ptr[reg_matCptr + reg_cstride + i * 64]);
      }
      if (mm != mtnum - 1) {
        lea(reg_matCptr, ptr[reg_matCptr + 8 * reg_cstride]);
        lea(reg_matCptr, ptr[reg_matCptr + 8 * reg_cstride]);
      }
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, dword[parambase + OFFSET(workspace)]);
    mov(reg_tmp1, NTILE * 4);
    for (int mm = 0; mm < MRegs; mm++) {
      for (int i = 0; i < NRegs; i++) {
        tilestored(ptr[reg_tmp + reg_tmp1 + i * 64 + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * NRegs + i));
      }
    }
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    int zunroll = TmpRegCount / NRegs;
    for (int i = 0; i < _mtile; i += zunroll) {
      int m_re = utils::remainsize(i, _mtile, zunroll);
      for (int im = 0; im < m_re; im++) {
        for (int j = 0; j < NRegs; j++) {
          vmovups(vreg_t(TmpReg + im * NRegs + j), ptr[reg_tmp + j * 64 + (i + im) * NTILE * 4]);
          vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(TmpReg + im * NRegs + j));
        }
        add(reg_matCptr, reg_cstride);
      }
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class Amxfp16N16P2 : protected bestla::xbyak::JitAmxbf16 {
 public:
  static int constexpr RegLen = 16, PackRow = 2;
  static_assert(_NTILE % RegLen == 0);
  static_assert(_MTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? 1 : _MTILE / RegLen;
  static_assert(NRegs * MRegs + 2 <= TileCount);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs * RegLen, KTILE = 32;
  static int constexpr KUNROLL = 2;
  static auto constexpr ISA = BTLA_ISA::AMX_FP16;
  static auto constexpr COMPUTE = CompType::COMP_FP16_FP32;
  typedef utils::fp16 AType;
  typedef utils::fp16 BType;
  typedef float CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* match;
    int cstride;
    int k;
    int n;
    int init;
    void* workspace;
  };
  typedef long long (*func_t)(params*);

  int TmpRegCount = RegCount;
  int TmpReg = 0;
  int CTileCount = 0, ATileCount = 0, BTileCount = 0;
  int CTile = 0, ATile = 0, BTile = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_tmp3;
  Xbyak::Reg64 reg_ret = rax;

  void assign_regs() {
    CTileCount = NRegs * MRegs;
    auto tile_re = TileCount - CTileCount;
    if (tile_re - 1 >= NRegs) {
      BTileCount = NRegs;
      ATileCount = tile_re - BTileCount;
    } else if (tile_re - 1 >= MRegs) {
      ATileCount = MRegs;
      BTileCount = tile_re - ATileCount;
    } else {
      ATileCount = 1;
      BTileCount = tile_re - ATileCount;
    }
    CTile = 0;
    ATile = CTile + CTileCount;
    BTile = ATile + ATileCount;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_tmp3 = st.t[10];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int kunrll) {
    auto& reg_Bstride = reg_tmp1;
    mov(reg_Bstride, NTILE * 4);
    int mtiles = _mtile / RegLen;
    for (int kk = 0; kk < kunrll; kk++) {
      auto reg_Atmp = reg_tmp2;
      if (mtiles == 1) {
        reg_Atmp = reg_matAptr;
      } else {
        mov(reg_Atmp, reg_matAptr);
      }
      if (BTileCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
        }
        for (int mm = 0; mm < mtiles; mm++) {
          tileloadd(Xbyak::Tmm(ATile), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
          for (int i = 0; i < NRegs; i++) {
            tdpfp16ps(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile), Xbyak::Tmm(BTile + i));
          }
          if (mm != mtiles - 1) {
            lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
          }
        }
      } else {
        if (ATileCount == mtiles) {
          for (int mm = 0; mm < mtiles; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
            if (mm != mtiles - 1) {
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            }
          }
          for (int i = 0; i < NRegs; i++) {
            tileloaddt1(Xbyak::Tmm(BTile), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
            for (int mm = 0; mm < mtiles; mm++) {
              tdpfp16ps(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile));
            }
          }
        } else {
          for (int mm = 0; mm < mtiles; mm++) {
            tileloadd(Xbyak::Tmm(ATile), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
            for (int i = 0; i < NRegs; i++) {
              tileloaddt1(Xbyak::Tmm(BTile), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
              tdpfp16ps(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile), Xbyak::Tmm(BTile));
            }
            if (mm != mtiles - 1) {
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            }
          }
        }
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < CTileCount; i++) {
      tilezero(Xbyak::Tmm(CTile + i));
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    int mtnum = _mtile / 16;
    for (int mm = 0; mm < mtnum; mm++) {
      for (int i = 0; i < NRegs; i++) {
        tileloaddt1(Xbyak::Tmm(CTile + mm * NRegs + i), ptr[reg_matCptr + reg_cstride + i * 64]);
      }
      if (mm != mtnum - 1) {
        lea(reg_matCptr, ptr[reg_matCptr + 8 * reg_cstride]);
        lea(reg_matCptr, ptr[reg_matCptr + 8 * reg_cstride]);
      }
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, dword[parambase + OFFSET(workspace)]);
    mov(reg_tmp1, NTILE * 4);
    for (int mm = 0; mm < MRegs; mm++) {
      for (int i = 0; i < NRegs; i++) {
        tilestored(ptr[reg_tmp + reg_tmp1 + i * 64 + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * NRegs + i));
      }
    }
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    int zunroll = TmpRegCount / NRegs;
    for (int i = 0; i < _mtile; i += zunroll) {
      int m_re = utils::remainsize(i, _mtile, zunroll);
      for (int im = 0; im < m_re; im++) {
        for (int j = 0; j < NRegs; j++) {
          vmovups(vreg_t(TmpReg + im * NRegs + j), ptr[reg_tmp + j * 64 + (i + im) * NTILE * 4]);
          vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(TmpReg + im * NRegs + j));
        }
        add(reg_matCptr, reg_cstride);
      }
    }
    outLocalLabel();
  }
};

template <typename AT, typename BT, int _NTILE, int _MTILE = 0>
class Amxint8N16P4 : protected bestla::xbyak::JitAmxint8 {
 public:
  static int constexpr RegLen = 16, PackRow = 4;
  static_assert(_NTILE % RegLen == 0);
  static_assert(_MTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? 1 : _MTILE / RegLen;
  static_assert(NRegs * MRegs + 2 <= TileCount);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs * RegLen, KTILE = 64;
  static int constexpr KUNROLL = 2;
  static auto constexpr ISA = BTLA_ISA::AMX_INT8;
  static auto constexpr COMPUTE =
      (std::is_same_v<AT, int8_t>
           ? std::is_same_v<BT, int8_t> ? CompType::COMP_INT8_SS_INT32 : CompType::COMP_INT8_SU_INT32
       : std::is_same_v<BT, int8_t> ? CompType::COMP_INT8_US_INT32
                                    : CompType::COMP_INT8_UU_INT32);
  using AType = AT;
  using BType = BT;
  typedef int32_t CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* match;
    int cstride;
    int k;
    int n;
    int init;
    void* workspace;
  };
  typedef long long (*func_t)(params*);

  int TmpRegCount = RegCount;
  int TmpReg = 0;
  int CTileCount = 0, ATileCount = 0, BTileCount = 0;
  int CTile = 0, ATile = 0, BTile = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_tmp3;
  Xbyak::Reg64 reg_ret = rax;

  void assign_regs() {
    CTileCount = NRegs * MRegs;
    auto tile_re = TileCount - CTileCount;
    if (tile_re - 1 >= NRegs) {
      BTileCount = NRegs;
      ATileCount = tile_re - BTileCount;
    } else if (tile_re - 1 >= MRegs) {
      ATileCount = MRegs;
      BTileCount = tile_re - ATileCount;
    } else {
      ATileCount = 1;
      BTileCount = tile_re - ATileCount;
    }
    CTile = 0;
    ATile = CTile + CTileCount;
    BTile = ATile + ATileCount;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_tmp3 = st.t[10];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int kunrll) {
    auto& reg_Bstride = reg_tmp1;
    mov(reg_Bstride, NTILE * 4);
    int mtiles = _mtile / RegLen;

    for (int kk = 0; kk < kunrll; kk++) {
      auto reg_Atmp = reg_tmp2;
      if (mtiles == 1) {
        reg_Atmp = reg_matAptr;
      } else {
        mov(reg_Atmp, reg_matAptr);
      }
      if (BTileCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
        }
        for (int mm = 0; mm < mtiles; mm++) {
          tileloadd(Xbyak::Tmm(ATile), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
          for (int i = 0; i < NRegs; i++) {
            _tdpb<AT, BT>(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile), Xbyak::Tmm(BTile + i));
          }
          if (mm != mtiles - 1) {
            lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
          }
        }
      } else {
        if (ATileCount == mtiles) {
          for (int mm = 0; mm < mtiles; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
            if (mm != mtiles - 1) {
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            }
          }
          for (int i = 0; i < NRegs; i++) {
            tileloaddt1(Xbyak::Tmm(BTile), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
            for (int mm = 0; mm < mtiles; mm++) {
              _tdpb<AT, BT>(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile));
            }
          }
        } else {
          for (int mm = 0; mm < mtiles; mm++) {
            tileloadd(Xbyak::Tmm(ATile), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
            for (int i = 0; i < NRegs; i++) {
              tileloaddt1(Xbyak::Tmm(BTile), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
              _tdpb<AT, BT>(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile), Xbyak::Tmm(BTile));
            }
            if (mm != mtiles - 1) {
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            }
          }
        }
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < CTileCount; i++) {
      tilezero(Xbyak::Tmm(CTile + i));
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    int mtnum = _mtile / 16;
    for (int mm = 0; mm < mtnum; mm++) {
      for (int i = 0; i < NRegs; i++) {
        tileloaddt1(Xbyak::Tmm(CTile + mm * NRegs + i), ptr[reg_matCptr + reg_cstride + i * 64]);
      }
      if (mm != mtnum - 1) {
        lea(reg_matCptr, ptr[reg_matCptr + 8 * reg_cstride]);
        lea(reg_matCptr, ptr[reg_matCptr + 8 * reg_cstride]);
      }
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, dword[parambase + OFFSET(workspace)]);
    mov(reg_tmp1, NTILE * 4);
    for (int mm = 0; mm < MRegs; mm++) {
      for (int i = 0; i < NRegs; i++) {
        tilestored(ptr[reg_tmp + reg_tmp1 + i * 64 + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * NRegs + i));
      }
    }
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    int zunroll = TmpRegCount / NRegs;
    for (int i = 0; i < _mtile; i += zunroll) {
      int m_re = utils::remainsize(i, _mtile, zunroll);
      for (int im = 0; im < m_re; im++) {
        for (int j = 0; j < NRegs; j++) {
          vmovups(vreg_t(TmpReg + im * NRegs + j), ptr[reg_tmp + j * 64 + (i + im) * NTILE * 4]);
          vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(TmpReg + im * NRegs + j));
        }
        add(reg_matCptr, reg_cstride);
      }
    }
    outLocalLabel();
  }
};
template <int N, int M>
using Amxint8N16P4US = Amxint8N16P4<uint8_t, int8_t, N, M>;

template <int N, int M>
using Amxint8N16P4SS = Amxint8N16P4<int8_t, int8_t, N, M>;

class AmxConfigure : protected xbyak::JitAmxtile {
 public:
  typedef long long (*func_t)(tileconfig_t*);

  static void configure(int TILE_M, int TILE_N, int TILE_K, int elesize, int ANum, int BNum, int CNum) {
    static AmxConfigure code;
    tileconfig_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    configure_tiles(cfg, TILE_M, TILE_N, TILE_K, elesize, ANum, BNum, CNum);
    code.mKernel(&cfg);
  }

 protected:
  AmxConfigure() {
    generate_config(this);
    mKernel = getCode<func_t>();
  }

  func_t mKernel = nullptr;
};
}  // namespace code
template <template <int, int> class CodeT, int _NTILE, int _MTILE = 0>
class CoreCodeBase {
 public:
  using Code = CodeT<_NTILE, _MTILE>;
  using AType = typename Code::AType;
  using BType = typename Code::BType;
  using CType = typename Code::CType;
  static auto constexpr NTILE = Code::NTILE;
  static auto constexpr MTILE = Code::MTILE;
  static auto constexpr KTILE = Code::KTILE;
  static auto constexpr PACK_ROW = Code::PackRow;
  static auto constexpr COMP = Code::COMPUTE;
  static int constexpr PREFERRED_N = NTILE * 4;
  static auto constexpr ISA = Code::ISA;
  static auto constexpr ID = CoreAttr::make_core_id(NTILE, PACK_ROW, COMP, ISA);
  static void configure(int _M, int _N, int _K) { (void)(0); }

  static CoreCodeBase<CodeT, _NTILE, _MTILE>* getInstance() {
    static CoreCodeBase<CodeT, _NTILE, _MTILE> instance;
    return &instance;
  }

  std::array<Code, Code::MTILE> mCodes;

 protected:
  CoreCodeBase() {
    for (int i = 0; i < mCodes.size(); i++) {
      mCodes[i].generate_code(i + 1);
    }
  }
};

template <template <int, int> class CodeT, int _NTILE, int _MTILE = 0>
class CoreCodeBaseAMX {
 public:
  using Code = CodeT<_NTILE, _MTILE>;
  using AType = typename Code::AType;
  using BType = typename Code::BType;
  using CType = typename Code::CType;
  static auto constexpr NTILE = Code::NTILE;
  static auto constexpr MTILE = Code::MTILE;
  static auto constexpr KTILE = Code::KTILE;
  static auto constexpr PACK_ROW = Code::PackRow;
  static auto constexpr COMP = Code::COMPUTE;
  static int constexpr PREFERRED_N = NTILE * 4;
  static auto constexpr ISA = Code::ISA;
  static auto constexpr ID = CoreAttr::make_core_id(_NTILE, PACK_ROW, COMP, ISA);
  Xbyak::CodeGenerator cfgcode;

  static CoreCodeBaseAMX<CodeT, _NTILE, _MTILE>* getInstance() {
    static CoreCodeBaseAMX<CodeT, _NTILE, _MTILE> instance;
    return &instance;
  }

  std::array<Code, Code::MRegs> mCodes;

 protected:
  CoreCodeBaseAMX() {
    for (int i = 0; i < mCodes.size(); i++) {
      mCodes[i].generate_code((i + 1) * 16);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class SCoreRowNAvx2 : public CoreCodeBase<code::Avx2N8P1, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBase<code::Avx2N8P1, _NTILE, _MTILE>;
  using Code = typename Base::Code;
  static void forward(float* matA, float* matB, float* match, int _m, int _n, int _k, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      Base::getInstance()->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class SCoreRowNAvx512f : public CoreCodeBase<code::Avx512fN16P1, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBase<code::Avx512fN16P1, _NTILE, _MTILE>;
  using Code = typename Base::Code;
  static void forward(float* matA, float* matB, float* match, int _m, int _n, int _k, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      Base::getInstance()->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class HCoreRowNAvx512fp16 : public CoreCodeBase<code::Avx512fp16N32P1, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBase<code::Avx512fp16N32P1, _NTILE, _MTILE>;
  using Code = typename Base::Code;
  static void forward(utils::fp16* matA, utils::fp16* matB, utils::fp16* match, int _m, int _n, int _k, int _astride,
                      int _bstride, int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      Base::getInstance()->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class HCoreRowNAvx512bf16 : public CoreCodeBase<code::Avx512bf16N16P2, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBase<code::Avx512bf16N16P2, _NTILE, _MTILE>;
  using Code = typename Base::Code;
  static void forward(utils::bf16* matA, utils::bf16* matB, float* match, int _m, int _n, int _k, int _astride,
                      int _bstride, int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      Base::getInstance()->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class HCoreRowNAmxbf16 : public CoreCodeBaseAMX<code::Amxbf16N16P2, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBaseAMX<code::Amxbf16N16P2, _NTILE, _MTILE>;
  using Code = typename Base::Code;
  using AType = typename Code::AType;
  using BType = typename Code::BType;
  using CType = typename Code::CType;

  static void configure(int _M, int _N, int _K) {
    code::AmxConfigure::configure(_M < 16 ? _M : 16, 16, Code::KTILE, sizeof(BType),
                                  Base::getInstance()->mCodes[0].ATileCount, Base::getInstance()->mCodes[0].BTileCount,
                                  Base::getInstance()->mCodes[0].CTileCount);
  }

  static void forward(AType* matA, BType* matB, CType* match, int _m, int _n, int _k, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param =
        typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0, tmpcache};
    if (_m <= Code::MTILE) {
      int idx = utils::updiv(_m, 16) - 1;
      Base::getInstance()->mCodes[idx].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class HCoreRowNAmxfp16 : public CoreCodeBaseAMX<code::Amxfp16N16P2, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBaseAMX<code::Amxfp16N16P2, _NTILE, _MTILE>;
  using Code = typename Base::Code;
  using AType = typename Code::AType;
  using BType = typename Code::BType;
  using CType = typename Code::CType;

  static void configure(int _M, int _N, int _K) {
    code::AmxConfigure::configure(_M < 16 ? _M : 16, 16, Code::KTILE, sizeof(BType),
                                  Base::getInstance()->mCodes[0].ATileCount, Base::getInstance()->mCodes[0].BTileCount,
                                  Base::getInstance()->mCodes[0].CTileCount);
  }

  static void forward(AType* matA, BType* matB, CType* match, int _m, int _n, int _k, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param =
        typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0, tmpcache};
    if (_m <= Code::MTILE) {
      int idx = utils::updiv(_m, 16) - 1;
      Base::getInstance()->mCodes[idx].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAvx512vnni : public CoreCodeBase<code::Avx512vnniN16P4, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBase<code::Avx512vnniN16P4, _NTILE, _MTILE>;
  using Code = typename Base::Code;
  static void forward(uint8_t* matA, int8_t* matB, int32_t* match, int _m, int _n, int _k, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      Base::getInstance()->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAvxvnni : public CoreCodeBase<code::AvxvnniN8P4U8, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBase<code::AvxvnniN8P4U8, _NTILE, _MTILE>;
  using Code = typename Base::Code;

  static void forward(uint8_t* matA, int8_t* matB, int32_t* match, int _m, int _n, int _k, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      Base::getInstance()->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAvxvnniSS : public CoreCodeBase<code::AvxvnniN8P4S8, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBase<code::AvxvnniN8P4S8, _NTILE, _MTILE>;
  using Code = typename CoreCodeBase<code::AvxvnniN8P4S8, _NTILE, _MTILE>::Code;

  static void forward(int8_t* matA, int8_t* matB, int32_t* match, int _m, int _n, int _k, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      Base::getInstance()->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAvx2vnni : public CoreCodeBase<code::Avx2vnniN8P4U8, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBase<code::Avx2vnniN8P4U8, _NTILE, _MTILE>;
  using Code = typename CoreCodeBase<code::Avx2vnniN8P4U8, _NTILE, _MTILE>::Code;

  static void forward(uint8_t* matA, int8_t* matB, int32_t* match, int _m, int _n, int _k, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      Base::getInstance()->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAvx2vnniSS : public CoreCodeBase<code::Avx2vnniN8P4S8, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBase<code::Avx2vnniN8P4S8, _NTILE, _MTILE>;
  using Code = typename CoreCodeBase<code::Avx2vnniN8P4S8, _NTILE, _MTILE>::Code;

  static void forward(int8_t* matA, int8_t* matB, int32_t* match, int _m, int _n, int _k, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      Base::getInstance()->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAvx512bw : public CoreCodeBase<code::Avx512bwN16P4, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBase<code::Avx512bwN16P4, _NTILE, _MTILE>;
  using Code = typename CoreCodeBase<code::Avx512bwN16P4, _NTILE, _MTILE>::Code;

  static void forward(uint8_t* matA, int8_t* matB, int32_t* match, int _m, int _n, int _k, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      Base::getInstance()->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAmxint8 : public CoreCodeBaseAMX<code::Amxint8N16P4US, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBaseAMX<code::Amxint8N16P4US, _NTILE, _MTILE>;
  using Code = typename CoreCodeBaseAMX<code::Amxint8N16P4US, _NTILE, _MTILE>::Code;
  using AType = typename Code::AType;
  using BType = typename Code::BType;
  using CType = typename Code::CType;
  static void configure(int _M, int _N, int _K) {
    code::AmxConfigure::configure(_M < 16 ? _M : 16, 16, Code::KTILE, sizeof(BType),
                                  Base::getInstance()->mCodes[0].ATileCount, Base::getInstance()->mCodes[0].BTileCount,
                                  Base::getInstance()->mCodes[0].CTileCount);
  }

  static void forward(uint8_t* matA, int8_t* matB, int32_t* match, int _m, int _n, int _k, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param =
        typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0, tmpcache};
    if (_m <= Code::MTILE) {
      int idx = utils::updiv(_m, 16) - 1;
      Base::getInstance()->mCodes[idx].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAmxint8SS : public CoreCodeBaseAMX<code::Amxint8N16P4SS, _NTILE, _MTILE> {
 public:
  using Base = CoreCodeBaseAMX<code::Amxint8N16P4SS, _NTILE, _MTILE>;
  using Code = typename CoreCodeBaseAMX<code::Amxint8N16P4SS, _NTILE, _MTILE>::Code;
  using AType = typename Code::AType;
  using BType = typename Code::BType;
  using CType = typename Code::CType;
  static void configure(int _M, int _N, int _K) {
    code::AmxConfigure::configure(_M < 16 ? _M : 16, 16, Code::KTILE, sizeof(BType),
                                  Base::getInstance()->mCodes[0].ATileCount, Base::getInstance()->mCodes[0].BTileCount,
                                  Base::getInstance()->mCodes[0].CTileCount);
  }

  static void forward(int8_t* matA, int8_t* matB, int32_t* match, int _m, int _n, int _k, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param =
        typename Code::params{matA, _astride, matB, _bstride, match, _cstride, _k, _n, kpos == 0 ? 1 : 0, tmpcache};
    if (_m <= Code::MTILE) {
      int idx = utils::updiv(_m, 16) - 1;
      Base::getInstance()->mCodes[idx].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

}  // namespace gemm
}  // namespace bestla
