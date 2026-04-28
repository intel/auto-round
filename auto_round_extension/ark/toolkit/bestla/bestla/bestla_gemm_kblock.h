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

#include "bestla_gemm.h"

namespace bestla {
namespace gemm {
namespace code {
namespace kblock {
struct alignas(64) param_i8_i8_f32 {
  void* matA;
  int astride;
  void* matB;
  int bstride;
  float* match;
  int cstride;
  void* zpA;
  float* scaleA;
  int ldsa;
  float* scaleB;
  int* reduceB;
  int ldsb;
  int k;
  int n;
  int kblock;
  int init;
  void* workspace;
  const uint16_t one = 1;
};

typedef long long (*func_i8_i8_f32_t)(param_i8_i8_f32*);

template <BTLA_ISA ISA_T, typename AT, int _NTILE, int _MTILE = 0>
class IGemmP4I8S8F32 : public bestla::xbyak::JitCode {
 public:
  using Info = xbyak::JitInfo<ISA_T>;
  using vreg_t = typename xbyak::JitInfo<ISA_T>::vreg_t;
  using AType = AT;
  typedef int8_t BType;
  typedef float CType;
  static auto constexpr ISA = ISA_T;
  static auto constexpr COMPUTE =
      std::is_same_v<AT, uint8_t> ? CompType::COMP_INT8_US_FP32 : CompType::COMP_INT8_SS_FP32;
  static int constexpr KUNROLL = 2;
  static int constexpr RegLen = Info::VecLen32, PACK_ROW = 4;
  static int constexpr VecBytes = Info::VecBytes;
  static_assert(_NTILE % RegLen == 0);
  static_assert(!(Info::HasTile && _MTILE == 0));  // No auto resize for AMX tiles
  static int constexpr NRegs = _NTILE / RegLen;

  static int constexpr TmpRegCount = ISA == BTLA_ISA::AMX_INT8      ? 0
                                     : ISA == BTLA_ISA::AVX512BW    ? 2
                                     : ISA == BTLA_ISA::AVX512_VNNI ? 1
                                     : ISA == BTLA_ISA::AVX_VNNI    ? std::is_same_v<AT, uint8_t> ? 1 : 2
                                     : ISA == BTLA_ISA::AVX2        ? std::is_same_v<AT, uint8_t> ? 2 : 4
                                                                    : 0;
  static int constexpr ARegCount = 1;
  static int constexpr ABMinReg = ARegCount;  // B count can be 0
  static int constexpr ReserveReg = ABMinReg + TmpRegCount;
  static int constexpr MRegs = _MTILE == 0 ? (Info::RegCount - ReserveReg) / (NRegs * 2) : _MTILE;
  static int constexpr CRegCount = NRegs * MRegs;
  static int constexpr BRegCount = Info::RegCount - CRegCount * 2 - ARegCount - TmpRegCount >= NRegs ? NRegs : 0;
  static int constexpr CReg = 0, CF32Reg = CReg + CRegCount, AReg = CF32Reg + CRegCount, BReg = AReg + ARegCount,
                       TmpReg = BReg + BRegCount;
  static_assert(!(!Info::HasTile && TmpReg + TmpRegCount > Info::RegCount));
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = Info::HasTile ? 64 : 4;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);
  static int constexpr NTiles = NRegs, MTiles = MTILE / 16;
  static_assert(!(Info::HasTile && MTILE % 16 != 0));
  static int constexpr CTileCount = NTiles * MTiles;
  static int constexpr BTileCount = Info::TileCount - CTileCount - 1 >= NTiles ? NTiles : 1;
  static int constexpr ATileCount = Info::TileCount - CTileCount - BTileCount >= MTiles ? MTiles : 1;
  static_assert(!(Info::HasTile && ATileCount + BTileCount + CTileCount > Info::TileCount));
  static auto constexpr ID = CoreAttr::make_core_id(NTILE, PACK_ROW, COMPUTE, ISA);
  static int constexpr PREFERRED_N = NTILE * 4;
  static int constexpr CTile = 0, ATile = CTile + CTileCount, BTile = ATile + ATileCount;
  static int constexpr MStepPerKernel = Info::HasTile ? 16 : 1;

  using params = param_i8_i8_f32;
  func_i8_i8_f32_t mKernel = nullptr;

  void generate_code(int _mtile) {
    reset();
    bestla::xbyak::JitCode::ISA = ISA;
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_i8_i8_f32_t>();
  }

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
  Xbyak::Reg64 reg_iterkb;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_tmp3;
  Xbyak::Reg64 reg_tmp4;
  Xbyak::Reg64 reg_ret = rax;

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 13, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_iterkb = st.t[12];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_tmp3 = st.t[10];
    reg_tmp4 = st.t[11];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    load_constant();
    L(".nloop");
    init_accumulator(_mtile);
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

  void load_constant() {
    if constexpr (ISA == BTLA_ISA::AVX2) {
      if constexpr (TmpRegCount == 4) {
        vpbroadcastw(vreg_t(TmpReg + 3), ptr[parambase + OFFSET(one)]);
      } else {
        vpbroadcastw(vreg_t(TmpReg + 1), ptr[parambase + OFFSET(one)]);
      }
    } else if constexpr (ISA == BTLA_ISA::AVX512BW) {
      vpbroadcastw(vreg_t(TmpReg + 1), ptr[parambase + OFFSET(one)]);
    }
  }

  void zero_int_acc(int _mtile) {
    if constexpr (ISA != BTLA_ISA::AMX_INT8) {
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
        }
      }
    } else {
      int mtiles = _mtile / MStepPerKernel;
      static_assert(VecBytes == 64);
      for (int mm = 0; mm < mtiles; mm++) {
        for (int i = 0; i < NRegs; i++) {
          tilezero(Xbyak::Tmm(CTile + mm * NRegs + i));
        }
      }
    }
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    xor_(reg_iterkb, reg_iterkb);
    L(".kloop");
    zero_int_acc(_mtile);
    xor_(reg_tmp2, reg_tmp2);
    load32(reg_tmp3, ptr[parambase + OFFSET(kblock)]);
    mov(reg_tmp, reg_tmp3);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kbloop", T_NEAR);
    L(".unkbloop");
    generate_mma(_mtile, KUNROLL, reg_tmp1, reg_tmp4);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_tmp2, KUNROLL * KTILE);
    cmp(reg_tmp2, reg_tmp);
    jb(".unkbloop");
    cmp(reg_tmp, reg_tmp3);
    jge(".kind", T_NEAR);
    L(".kbloop");
    generate_mma(_mtile, 1, reg_tmp1, reg_tmp4);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_tmp2, 1 * KTILE);
    cmp(reg_tmp2, reg_tmp3);
    jb(".kbloop");
    L(".kind");
    add(reg_iterk, reg_tmp2);
    generate_zp_correction(_mtile);
    generate_f32_accumulate(_mtile);
    inc(reg_iterkb);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    outLocalLabel();
  }

  void generate_mma(int _mtile, int _kunroll, Xbyak::Reg64& tmp, Xbyak::Reg64& tmp1) {
    if constexpr (ISA == BTLA_ISA::AVX_VNNI) {
      generate_mma_avxvnni(_mtile, _kunroll, tmp, tmp1);
    } else if constexpr (ISA == BTLA_ISA::AVX2) {
      generate_mma_avx2vnni(_mtile, _kunroll, tmp, tmp1);
    } else if constexpr (ISA == BTLA_ISA::AVX512_VNNI) {
      generate_mma_avx512vnni(_mtile, _kunroll, tmp, tmp1);
    } else if constexpr (ISA == BTLA_ISA::AVX512BW) {
      generate_mma_avx512bw(_mtile, _kunroll, tmp, tmp1);
    } else if constexpr (ISA == BTLA_ISA::AMX_INT8) {
      generate_mma_amxint8(_mtile, _kunroll, tmp, tmp1);
    }
  }

  void init_accumulator(int _mtile) {
    if constexpr (ISA == BTLA_ISA::AMX_INT8) {
      mov(reg_tmp3, ptr[parambase + OFFSET(workspace)]);
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(AccReg + j), vreg_t(AccReg + j), vreg_t(AccReg + j));
      }
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vmovups(ptr[reg_tmp3 + j * VecBytes + i * 4 * NTILE + WorkspaceFp32Offset], vreg_t(AccReg + j));
        }
      }
    } else {
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vxor(vreg_t(CF32Reg + i * NRegs + j), vreg_t(CF32Reg + i * NRegs + j), vreg_t(CF32Reg + i * NRegs + j));
        }
      }
    }
  }

  void generate_f32_accumulate(int _mtile) {
    if constexpr (ISA == BTLA_ISA::AMX_INT8) {
      return;
    }
    load32(reg_tmp, ptr[parambase + OFFSET(ldsb)]);
    imul(reg_tmp, reg_iterkb);
    mov(reg_tmp2, ptr[parambase + OFFSET(scaleB)]);
    lea(reg_tmp2, ptr[reg_tmp2 + reg_tmp * sizeof(float)]);
    lea(reg_tmp2, ptr[reg_tmp2 + reg_itern * sizeof(float)]);

    mov(reg_tmp, ptr[parambase + OFFSET(scaleA)]);
    lea(reg_tmp, ptr[reg_tmp + reg_iterkb * sizeof(float)]);
    load32(reg_tmp1, ptr[parambase + OFFSET(ldsa)]);
    if (BRegCount == NRegs) {
      for (int i = 0; i < NRegs; i++) {
        vmovups(vreg_t(BReg + i), ptr[reg_tmp2 + i * VecBytes]);
      }
    }
    for (int mm = 0; mm < _mtile; mm++) {
      vbroadcastss(vreg_t(TmpReg), ptr[reg_tmp]);
      lea(reg_tmp, ptr[reg_tmp + reg_tmp1 * sizeof(float)]);
      for (int i = 0; i < NRegs; i++) {
        vcvtdq2ps(vreg_t(CReg + mm * NRegs + i), vreg_t(CReg + mm * NRegs + i));
        if (BRegCount == NRegs) {
          vmulps(vreg_t(AReg), vreg_t(TmpReg), vreg_t(BReg + i));
        } else {
          vmulps(vreg_t(AReg), vreg_t(TmpReg), ptr[reg_tmp2 + i * VecBytes]);
        }
        vmulps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg));
        vaddps(vreg_t(CF32Reg + mm * NRegs + i), vreg_t(CReg + mm * NRegs + i));
      }
    }
  }

  void generate_zp_correction(int _mtile) {
    if constexpr (ISA == BTLA_ISA::AMX_INT8) {
      return generate_zp_correction_amx(_mtile);
    }
    inLocalLabel();
    mov(reg_tmp, ptr[parambase + OFFSET(zpA)]);
    cmp(reg_tmp, 0);
    je(".NOZP", T_NEAR);
    lea(reg_tmp, ptr[reg_tmp + reg_iterkb * sizeof(AType)]);
    auto& reg_zpA = reg_tmp;

    load32(reg_tmp1, ptr[parambase + OFFSET(ldsb)]);
    imul(reg_tmp1, reg_iterkb);
    mov(reg_tmp2, ptr[parambase + OFFSET(reduceB)]);
    lea(reg_tmp2, ptr[reg_tmp2 + reg_tmp1 * sizeof(int32_t)]);
    lea(reg_tmp2, ptr[reg_tmp2 + reg_itern * sizeof(int32_t)]);
    auto& reg_redB = reg_tmp2;

    load32(reg_tmp3, ptr[parambase + OFFSET(ldsa)]);
    auto& reg_ldsa = reg_tmp3;
    if (BRegCount == NRegs) {
      for (int i = 0; i < NRegs; i++) {
        vmovups(vreg_t(BReg + i), ptr[reg_redB + i * VecBytes]);
      }
    }
    for (int i = 0; i < _mtile; i++) {
      vpbroadcastb(Xbyak::Xmm(AReg), ptr[reg_zpA]);
      vpmovzxbd(vreg_t(AReg), Xbyak::Xmm(AReg));
      for (int j = 0; j < NRegs; j++) {
        if (BRegCount == NRegs) {
          vpmulld(vreg_t(TmpReg), vreg_t(AReg), vreg_t(BReg + j));
        } else {
          vpmulld(vreg_t(TmpReg), vreg_t(AReg), ptr[reg_redB + j * VecBytes]);
        }
        vpsubd(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(TmpReg));
      }
      lea(reg_zpA, ptr[reg_zpA + reg_ldsa * sizeof(AType)]);
    }
    L(".NOZP");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    if (ISA == BTLA_ISA::AMX_INT8) {
      return write_back_amx(_mtile);
    }
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);

    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".add_back", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CF32Reg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    jmp(".end", T_NEAR);
    L(".add_back");
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vaddps(vreg_t(CF32Reg + i * NRegs + j), vreg_t(CF32Reg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CF32Reg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

 private:
  void generate_mma_avxvnni(int _mtile, int kunrll, Xbyak::Reg64& tmp, Xbyak::Reg64& tmp1) {
    for (int kk = 0; kk < kunrll; kk++) {
      lea(tmp, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(vreg_t(AReg), ptr[reg_tmp1]);
          if constexpr (std::is_same_v<AType, int8_t>) {
            vpsignb(vreg_t(TmpReg + 1), vreg_t(AReg), vreg_t(AReg));
          }
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            if constexpr (std::is_same_v<AType, uint8_t>) {
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg),
                         ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            } else {
              vmovups(vreg_t(TmpReg), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
              vpsignb(vreg_t(TmpReg), vreg_t(TmpReg), vreg_t(AReg));
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(TmpReg + 1), vreg_t(TmpReg));
            }
          }
        }
      } else {
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
      }
    }
  }

  void generate_mma_avx2vnni(int _mtile, int kunrll, Xbyak::Reg64& tmp, Xbyak::Reg64& tmp1) {
    for (int kk = 0; kk < kunrll; kk++) {
      lea(tmp, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(vreg_t(AReg), ptr[reg_tmp1]);
          if constexpr (std::is_same_v<AType, int8_t>) {
            vpsignb(vreg_t(TmpReg + 2), vreg_t(AReg), vreg_t(AReg));
          }
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            if constexpr (std::is_same_v<AType, uint8_t>) {
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes],
                         vreg_t(TmpReg), vreg_t(TmpReg + 1));
            } else {
              vmovups(vreg_t(TmpReg), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
              vpsignb(vreg_t(TmpReg), vreg_t(TmpReg), vreg_t(AReg));
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(TmpReg + 2), vreg_t(TmpReg), vreg_t(TmpReg + 1),
                         vreg_t(TmpReg + 3));
            }
          }
        }
      } else {
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
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i), vreg_t(TmpReg + 0),
                         vreg_t(TmpReg + 1));
            } else {
              vpsignb(vreg_t(TmpReg + 0), vreg_t(BReg + i), vreg_t(AReg));
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(TmpReg + 2), vreg_t(TmpReg + 0), vreg_t(TmpReg + 1),
                         vreg_t(TmpReg + 3));
            }
          }
        }
      }
    }
  }

  void generate_mma_avx512vnni(int _mtile, int kunrll, Xbyak::Reg64& tmp, Xbyak::Reg64& tmp1) {
    for (int kk = 0; kk < kunrll; kk++) {
      lea(tmp, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
      }
      for (int mm = 0; mm < _mtile; mm++) {
        vpbroadcastd(vreg_t(AReg), ptr[reg_tmp1]);
        add(reg_tmp1, reg_astride);
        for (int i = 0; i < NRegs; i++) {
          if (BRegCount == NRegs) {
            vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          } else {
            vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
          }
        }
      }
    }
  }

  void generate_mma_avx512bw(int _mtile, int kunrll, Xbyak::Reg64& tmp, Xbyak::Reg64& tmp1) {
    static_assert(std::is_same_v<AType, uint8_t>);
    for (int kk = 0; kk < kunrll; kk++) {
      lea(tmp, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
      }
      for (int mm = 0; mm < _mtile; mm++) {
        vpbroadcastd(vreg_t(AReg), ptr[reg_tmp1]);
        add(reg_tmp1, reg_astride);
        for (int i = 0; i < NRegs; i++) {
          if (BRegCount == NRegs) {
            vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i), vreg_t(TmpReg + 0),
                       vreg_t(TmpReg + 1));
          } else {
            vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes],
                       vreg_t(TmpReg + 0), vreg_t(TmpReg + 1));
          }
        }
      }
    }
  }

  static int constexpr WorkspaceFp32Offset = MTILE * NTILE * 4;
  static int constexpr AccReg = 0;
  static int constexpr RedBReg = AccReg + NRegs;
  static int constexpr ScaBReg = RedBReg + NRegs;
  static int constexpr ZpAReg = ScaBReg + NRegs;
  static int constexpr ScaAReg = ZpAReg + 1;
  static int constexpr TmpCReg = ScaAReg + 1;
  void generate_mma_amxint8(int _mtile, int kunrll, Xbyak::Reg64& tmpreg, Xbyak::Reg64& tmpreg2) {
    auto& reg_Bstride = tmpreg2;
    mov(reg_Bstride, NTILE * 4);
    int mtiles = _mtile / MStepPerKernel;

    for (int kk = 0; kk < kunrll; kk++) {
      auto reg_Atmp = tmpreg;
      if (mtiles == 1) {
        reg_Atmp = reg_matAptr;
      } else {
        mov(reg_Atmp, reg_matAptr);
      }
      if (BTileCount == NTiles) {
        for (int i = 0; i < NTiles; i++) {
          tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
        }
        for (int mm = 0; mm < mtiles; mm++) {
          tileloadd(Xbyak::Tmm(ATile), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
          for (int i = 0; i < NTiles; i++) {
            _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * NTiles + i), Xbyak::Tmm(ATile), Xbyak::Tmm(BTile + i));
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
          for (int i = 0; i < NTiles; i++) {
            tileloaddt1(Xbyak::Tmm(BTile), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
            for (int mm = 0; mm < mtiles; mm++) {
              _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * NTiles + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile));
            }
          }
        } else {
          for (int mm = 0; mm < mtiles; mm++) {
            tileloadd(Xbyak::Tmm(ATile), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
            for (int i = 0; i < NTiles; i++) {
              tileloaddt1(Xbyak::Tmm(BTile), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
              _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * NTiles + i), Xbyak::Tmm(ATile), Xbyak::Tmm(BTile));
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

  void generate_zp_correction_amx(int _mtile) {
    int mtiles = _mtile / MStepPerKernel;
    static_assert(VecBytes == 64);
    inLocalLabel();
    mov(reg_tmp3, ptr[parambase + OFFSET(workspace)]);
    mov(reg_tmp1, NTILE * 4);
    for (int mm = 0; mm < mtiles; mm++) {
      for (int i = 0; i < NRegs; i++) {
        tilestored(ptr[reg_tmp3 + reg_tmp1 + i * VecBytes + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * NRegs + i));
      }
    }

    load32(reg_tmp1, ptr[parambase + OFFSET(ldsb)]);
    imul(reg_tmp1, reg_iterkb);
    mov(reg_tmp2, ptr[parambase + OFFSET(scaleB)]);
    lea(reg_tmp2, ptr[reg_tmp2 + reg_tmp1 * sizeof(float)]);
    lea(reg_tmp2, ptr[reg_tmp2 + reg_itern * sizeof(float)]);
    for (int i = 0; i < NRegs; i++) {
      vmovups(vreg_t(ScaBReg + i), ptr[reg_tmp2 + i * VecBytes]);
    }

    mov(reg_tmp, ptr[parambase + OFFSET(zpA)]);
    cmp(reg_tmp, 0);
    je(".NOZP", T_NEAR);

    lea(reg_tmp, ptr[reg_tmp + reg_iterkb * sizeof(AType)]);
    auto& reg_zpA = reg_tmp;

    mov(reg_tmp2, ptr[parambase + OFFSET(reduceB)]);
    lea(reg_tmp2, ptr[reg_tmp2 + reg_tmp1 * sizeof(int32_t)]);
    lea(reg_tmp2, ptr[reg_tmp2 + reg_itern * sizeof(int32_t)]);
    for (int i = 0; i < NRegs; i++) {
      vmovups(vreg_t(RedBReg + i), ptr[reg_tmp2 + i * VecBytes]);
    }

    load32(reg_tmp1, ptr[parambase + OFFSET(ldsa)]);
    auto& reg_ldsa = reg_tmp1;
    auto& reg_sca = reg_tmp2;
    mov(reg_sca, ptr[parambase + OFFSET(scaleA)]);
    lea(reg_sca, ptr[reg_sca + reg_iterkb * sizeof(float)]);

    for (int i = 0; i < _mtile; i++) {
      vpbroadcastb(Xbyak::Xmm(ZpAReg), ptr[reg_zpA]);
      vpmovzxbd(vreg_t(ZpAReg), Xbyak::Xmm(ZpAReg));
      vbroadcastss(vreg_t(ScaAReg), ptr[reg_sca]);
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(AccReg + j), ptr[reg_tmp3 + j * VecBytes + i * 4 * NTILE]);
        vpmulld(vreg_t(TmpCReg), vreg_t(ZpAReg), vreg_t(RedBReg + j));
        vpsubd(vreg_t(AccReg + j), vreg_t(AccReg + j), vreg_t(TmpCReg));
        vmulps(vreg_t(TmpCReg), vreg_t(ScaAReg), vreg_t(ScaBReg + j));
        vcvtdq2ps(vreg_t(AccReg + j), vreg_t(AccReg + j));
        vmulps(vreg_t(AccReg + j), vreg_t(TmpCReg));
        vaddps(vreg_t(AccReg + j), vreg_t(AccReg + j),
               ptr[reg_tmp3 + j * VecBytes + i * 4 * NTILE + WorkspaceFp32Offset]);
        vmovups(ptr[reg_tmp3 + j * VecBytes + i * 4 * NTILE + WorkspaceFp32Offset], vreg_t(AccReg + j));
      }
      lea(reg_zpA, ptr[reg_zpA + reg_ldsa * sizeof(AType)]);
      lea(reg_sca, ptr[reg_sca + reg_ldsa * sizeof(float)]);
    }
    jmp(".END", T_NEAR);
    L(".NOZP");
    load32(reg_ldsa, ptr[parambase + OFFSET(ldsa)]);
    mov(reg_sca, ptr[parambase + OFFSET(scaleA)]);
    lea(reg_sca, ptr[reg_sca + reg_iterkb * sizeof(float)]);

    for (int i = 0; i < _mtile; i++) {
      vbroadcastss(vreg_t(ScaAReg), ptr[reg_sca]);
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(AccReg + j), ptr[reg_tmp3 + j * VecBytes + i * 4 * NTILE]);
        vmulps(vreg_t(TmpCReg), vreg_t(ScaAReg), vreg_t(ScaBReg + j));
        vcvtdq2ps(vreg_t(AccReg + j), vreg_t(AccReg + j));
        vmulps(vreg_t(AccReg + j), vreg_t(TmpCReg));
        vaddps(vreg_t(AccReg + j), vreg_t(AccReg + j),
               ptr[reg_tmp3 + j * VecBytes + i * 4 * NTILE + WorkspaceFp32Offset]);
        vmovups(ptr[reg_tmp3 + j * VecBytes + i * 4 * NTILE + WorkspaceFp32Offset], vreg_t(AccReg + j));
      }
      lea(reg_sca, ptr[reg_sca + reg_ldsa * sizeof(float)]);
    }
    L(".END");
    outLocalLabel();
  }

  void write_back_amx(int _mtile) {
    inLocalLabel();
    mov(reg_tmp3, ptr[parambase + OFFSET(workspace)]);
    mov(reg_matCptr, ptr[parambase + OFFSET(match)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);

    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".add_back", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(AccReg + j), ptr[reg_tmp3 + j * VecBytes + i * 4 * NTILE + WorkspaceFp32Offset]);
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(AccReg + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    jmp(".end", T_NEAR);
    L(".add_back");
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(AccReg + j), ptr[reg_tmp3 + j * VecBytes + i * 4 * NTILE + WorkspaceFp32Offset]);
        vaddps(vreg_t(AccReg + j), vreg_t(AccReg + j), ptr[reg_matCptr + j * VecBytes]);
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(AccReg + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }
};

template <int N, int M>
using AvxvnniUS = IGemmP4I8S8F32<BTLA_ISA::AVX_VNNI, uint8_t, N, M>;
template <int N, int M>
using AvxvnniSS = IGemmP4I8S8F32<BTLA_ISA::AVX_VNNI, int8_t, N, M>;
template <int N, int M>
using Avx2vnniUS = IGemmP4I8S8F32<BTLA_ISA::AVX2, uint8_t, N, M>;
template <int N, int M>
using Avx2vnniSS = IGemmP4I8S8F32<BTLA_ISA::AVX2, int8_t, N, M>;

template <int N, int M>
using Avx512vnniUS = IGemmP4I8S8F32<BTLA_ISA::AVX512_VNNI, uint8_t, N, M>;
template <int N, int M>
using Avx512bwUS = IGemmP4I8S8F32<BTLA_ISA::AVX512BW, uint8_t, N, M>;

template <int N, int M>
using Amxint8US = IGemmP4I8S8F32<BTLA_ISA::AMX_INT8, uint8_t, N, M>;
template <int N, int M>
using Amxint8SS = IGemmP4I8S8F32<BTLA_ISA::AMX_INT8, int8_t, N, M>;
}  // namespace kblock
}  // namespace code

template <template <int, int> class CodeT, int _NTILE, int _MTILE = 0>
class KblockCoreInterface {
 public:
  using Code = CodeT<_NTILE, _MTILE>;
  using AType = typename Code::AType;
  using BType = typename Code::BType;
  using CType = typename Code::CType;
  static auto constexpr NTILE = Code::NTILE;
  static auto constexpr MTILE = Code::MTILE;
  static auto constexpr KTILE = Code::KTILE;
  static auto constexpr PACK_ROW = Code::PACK_ROW;
  static auto constexpr COMP = Code::COMPUTE;
  static int constexpr PREFERRED_N = Code::PREFERRED_N;
  static auto constexpr ISA = Code::ISA;
  static auto constexpr ID = Code::ID;
  static auto constexpr MStep = Code::MStepPerKernel;
  static void configure(int _M, int _N, int _K) {
    if (ISA >= BTLA_ISA::AMX_BF16) {
      gemm::code::AmxConfigure::configure(_M < 16 ? _M : 16, 16, Code::KTILE, sizeof(BType), Code::ATileCount,
                                          Code::BTileCount, Code::CTileCount);

    } else {
      (void)(_M);
      (void)(_N);
      (void)(_K);
    }
  }

  static void forward(AType* matA, BType* matB, CType* match, AType* zpA, float* scaleA, int _ldsa, float* scaleB,
                      int32_t* reduceB, int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride,
                      int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA,    _astride, matB,    _bstride, match, _cstride, zpA,     scaleA,
                                       _ldsa,   scaleB,   reduceB, _ldsb,    _k,   _n,       _kblock, kpos == 0 ? 1 : 0,
                                       tmpcache};
    if (_m <= Code::MTILE) {
      int code_idx = utils::updiv(_m, MStep);
      getInstance()->mCodes[code_idx - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }

  static KblockCoreInterface<CodeT, _NTILE, _MTILE>* getInstance() {
    static KblockCoreInterface<CodeT, _NTILE, _MTILE> instance;
    return &instance;
  }

  std::array<Code, Code::MTILE / MStep> mCodes;

 protected:
  KblockCoreInterface() {
    for (int i = 0; i < mCodes.size(); i++) {
      mCodes[i].generate_code((i + 1) * MStep);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
using KblockGemmAvxvnni = KblockCoreInterface<code::kblock::AvxvnniUS, _NTILE, _MTILE>;

template <int _NTILE, int _MTILE = 0>
using KblockGemmAvxvnniSS = KblockCoreInterface<code::kblock::AvxvnniSS, _NTILE, _MTILE>;

template <int _NTILE, int _MTILE = 0>
using KblockGemmAvx2vnni = KblockCoreInterface<code::kblock::Avx2vnniUS, _NTILE, _MTILE>;

template <int _NTILE, int _MTILE = 0>
using KblockGemmAvx2vnniSS = KblockCoreInterface<code::kblock::Avx2vnniSS, _NTILE, _MTILE>;

template <int _NTILE, int _MTILE = 0>
using KblockGemmAvx512vnni = KblockCoreInterface<code::kblock::Avx512vnniUS, _NTILE, _MTILE>;

template <int _NTILE, int _MTILE = 0>
using KblockGemmAvx512bw = KblockCoreInterface<code::kblock::Avx512bwUS, _NTILE, _MTILE>;

template <int _NTILE, int _MTILE = 0>
using KblockGemmAmxint8 = KblockCoreInterface<code::kblock::Amxint8US, _NTILE, _MTILE>;

template <int _NTILE, int _MTILE = 0>
using KblockGemmAmxint8SS = KblockCoreInterface<code::kblock::Amxint8SS, _NTILE, _MTILE>;

}  // namespace gemm
}  // namespace bestla
