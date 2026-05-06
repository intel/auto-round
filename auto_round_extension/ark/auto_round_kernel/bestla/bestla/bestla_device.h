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
#include <map>
#include <thread>
#include <vector>
#include "bestla.h"
#include "xbyak/xbyak_util.h"
#include "bestla_utils.h"
#ifdef _WIN32
#include <windows.h>
#define FIXED_CACHE 1
#else
#include <sched.h>
#include <unistd.h>
#define FIXED_CACHE 0
#endif

#define FIXED_CACHE_SIZE ((1 << 20) - (128 << 10))

namespace bestla {

namespace device {

struct X64_ISA {
  int64_t MMX : 1;                  // 0
  int64_t SSE : 1;                  // 1
  int64_t SSE2 : 1;                 // 2
  int64_t SSE3 : 1;                 // 3
  int64_t SSSE3 : 1;                // 4
  int64_t SSE41 : 1;                // 5
  int64_t SSE42 : 1;                // 6
  int64_t AVX : 1;                  // 7
  int64_t F16C : 1;                 // 8
  int64_t FMA : 1;                  // 9
  int64_t AVX2 : 1;                 // 10
  int64_t AVX_VNNI : 1;             // 11
  int64_t AVX_VNNI_INT8 : 1;        // 12
  int64_t AVX_NE_CONVERT : 1;       // 13
  int64_t AVX_IFMA : 1;             // 14
  int64_t AVX512F : 1;              // 15
  int64_t AVX512BW : 1;             // 16
  int64_t AVX512CD : 1;             // 17
  int64_t AVX512DQ : 1;             // 18
  int64_t AVX512ER : 1;             // 19
  int64_t AVX512IFMA52 : 1;         // 20
  int64_t AVX512PF : 1;             // 21
  int64_t AVX512VL : 1;             // 22
  int64_t AVX512VPOPCNTDQ : 1;      // 23
  int64_t AVX512_4FMAPS : 1;        // 24
  int64_t AVX512_4VNNIW : 1;        // 25
  int64_t AVX512_BF16 : 1;          // 26
  int64_t AVX512_BITALG : 1;        // 27
  int64_t AVX512_VBMI : 1;          // 28
  int64_t AVX512_VBMI2 : 1;         // 29
  int64_t AVX512_VNNI : 1;          // 30
  int64_t AVX512_VP2INTERSECT : 1;  // 31
  int64_t AVX512_FP16 : 1;          // 32
  int64_t AMX_TILE : 1;             // 33
  int64_t AMX_BF16 : 1;             // 34
  int64_t AMX_INT8 : 1;             // 35
  int64_t AMX_FP16 : 1;             // 36
  int64_t AMX_COMPLEX : 1;          // 37
  int64_t reserved : (64 - 38);
};

class AVX2_Default {
 public:
  static constexpr bool MMX = 1;
  static constexpr bool SSE = 1;
  static constexpr bool SSE2 = 1;
  static constexpr bool SSE3 = 1;
  static constexpr bool SSSE3 = 1;
  static constexpr bool SSE41 = 1;
  static constexpr bool SSE42 = 1;
  static constexpr bool AVX = 1;
  static constexpr bool F16C = 1;
  static constexpr bool FMA = 1;
  static constexpr bool AVX2 = 1;
  static constexpr bool AVX_VNNI = 0;
  static constexpr bool AVX_VNNI_INT8 = 0;
  static constexpr bool AVX_NE_CONVERT = 0;
  static constexpr bool AVX_IFMA = 0;
  static constexpr bool AVX512F = 0;
  static constexpr bool AVX512BW = 0;
  static constexpr bool AVX512CD = 0;
  static constexpr bool AVX512DQ = 0;
  static constexpr bool AVX512ER = 0;
  static constexpr bool AVX512IFMA52 = 0;
  static constexpr bool AVX512PF = 0;
  static constexpr bool AVX512VL = 0;
  static constexpr bool AVX512VPOPCNTDQ = 0;
  static constexpr bool AVX512_4FMAPS = 0;
  static constexpr bool AVX512_4VNNIW = 0;
  static constexpr bool AVX512_BF16 = 0;
  static constexpr bool AVX512_BITALG = 0;
  static constexpr bool AVX512_VBMI = 0;
  static constexpr bool AVX512_VBMI2 = 0;
  static constexpr bool AVX512_VNNI = 0;
  static constexpr bool AVX512_VP2INTERSECT = 0;
  static constexpr bool AVX512_FP16 = 0;
  static constexpr bool AMX_TILE = 0;
  static constexpr bool AMX_BF16 = 0;
  static constexpr bool AMX_INT8 = 0;
  static constexpr bool AMX_FP16 = 0;
  static constexpr bool AMX_COMPLEX = 0;
};

class AVX512_VNNI_Default {
 public:
  static constexpr bool MMX = 1;
  static constexpr bool SSE = 1;
  static constexpr bool SSE2 = 1;
  static constexpr bool SSE3 = 1;
  static constexpr bool SSSE3 = 1;
  static constexpr bool SSE41 = 1;
  static constexpr bool SSE42 = 1;
  static constexpr bool AVX = 1;
  static constexpr bool F16C = 1;
  static constexpr bool FMA = 1;
  static constexpr bool AVX2 = 1;
  static constexpr bool AVX_VNNI = 0;
  static constexpr bool AVX_VNNI_INT8 = 0;
  static constexpr bool AVX_NE_CONVERT = 0;
  static constexpr bool AVX_IFMA = 0;
  static constexpr bool AVX512F = 1;
  static constexpr bool AVX512BW = 1;
  static constexpr bool AVX512CD = 1;
  static constexpr bool AVX512DQ = 1;
  static constexpr bool AVX512ER = 0;
  static constexpr bool AVX512IFMA52 = 0;
  static constexpr bool AVX512PF = 0;
  static constexpr bool AVX512VL = 1;
  static constexpr bool AVX512VPOPCNTDQ = 0;
  static constexpr bool AVX512_4FMAPS = 0;
  static constexpr bool AVX512_4VNNIW = 0;
  static constexpr bool AVX512_BF16 = 0;
  static constexpr bool AVX512_BITALG = 0;
  static constexpr bool AVX512_VBMI = 0;
  static constexpr bool AVX512_VBMI2 = 0;
  static constexpr bool AVX512_VNNI = 1;
  static constexpr bool AVX512_VP2INTERSECT = 0;
  static constexpr bool AVX512_FP16 = 0;
  static constexpr bool AMX_TILE = 0;
  static constexpr bool AMX_BF16 = 0;
  static constexpr bool AMX_INT8 = 0;
  static constexpr bool AMX_FP16 = 0;
  static constexpr bool AMX_COMPLEX = 0;
};

class SapphireRapids {
 public:
  static constexpr bool MMX = 1;
  static constexpr bool SSE = 1;
  static constexpr bool SSE2 = 1;
  static constexpr bool SSE3 = 1;
  static constexpr bool SSSE3 = 1;
  static constexpr bool SSE41 = 1;
  static constexpr bool SSE42 = 1;
  static constexpr bool AVX = 1;
  static constexpr bool F16C = 1;
  static constexpr bool FMA = 1;
  static constexpr bool AVX2 = 1;
  static constexpr bool AVX_VNNI = 0;
  static constexpr bool AVX_VNNI_INT8 = 0;
  static constexpr bool AVX_NE_CONVERT = 0;
  static constexpr bool AVX_IFMA = 0;
  static constexpr bool AVX512F = 1;
  static constexpr bool AVX512BW = 1;
  static constexpr bool AVX512CD = 1;
  static constexpr bool AVX512DQ = 1;
  static constexpr bool AVX512ER = 0;
  static constexpr bool AVX512IFMA52 = 0;
  static constexpr bool AVX512PF = 0;
  static constexpr bool AVX512VL = 1;
  static constexpr bool AVX512VPOPCNTDQ = 0;
  static constexpr bool AVX512_4FMAPS = 0;
  static constexpr bool AVX512_4VNNIW = 0;
  static constexpr bool AVX512_BF16 = 0;
  static constexpr bool AVX512_BITALG = 0;
  static constexpr bool AVX512_VBMI = 0;
  static constexpr bool AVX512_VBMI2 = 0;
  static constexpr bool AVX512_VNNI = 1;
  static constexpr bool AVX512_VP2INTERSECT = 0;
  static constexpr bool AVX512_FP16 = 0;
  static constexpr bool AMX_TILE = 1;
  static constexpr bool AMX_BF16 = 1;
  static constexpr bool AMX_INT8 = 1;
  static constexpr bool AMX_FP16 = 0;
  static constexpr bool AMX_COMPLEX = 0;
};

template <BTLA_ISA ISA_T>
class isa_base {
 public:
  static bool constexpr avx = ISA_T >= BTLA_ISA::AVX;
  static bool constexpr avx2 = ISA_T >= BTLA_ISA::AVX2;
  static bool constexpr avx512f = ISA_T >= BTLA_ISA::AVX512F;
  static bool constexpr avx512bw = ISA_T >= BTLA_ISA::AVX512BW;
  static bool constexpr avx512_vnni = ISA_T >= BTLA_ISA::AVX512_VNNI;
  static bool constexpr avx512_fp16 = ISA_T >= BTLA_ISA::AVX512_FP16;
  static bool constexpr amx_bf16 = ISA_T >= BTLA_ISA::AMX_BF16;
  static bool constexpr amx_int8 = ISA_T >= BTLA_ISA::AMX_INT8;
};

class CpuRuntimeV2 {
 public:
  CpuRuntimeV2() = default;
  inline void adjustPE(int _kernel_id, const float PE_) {
    // printf("Adjust:%d,%f\n",int(isa),PE_);
    PR[_kernel_id] = PR[_kernel_id] * PE_ * 0.7f + PR[_kernel_id] * 0.3f;
  }

  size_t L1_cache_ = 0, L2_cache_ = 0;
  int core_id_ = 0, core_sub_id_ = 0;
  static constexpr int MAX_KERNEL_NUM = 256;
  float PR[MAX_KERNEL_NUM];
};

class CpuDevice {
 public:
  inline int getThreads() { return numthreads; }
  inline int getTotalThreads() { return num_total_threads_; }
  inline int getCores() { return isHybrid() ? getPcoreNum() + getEcoreNum() : numcores; }
  inline uint32_t getL3CacheSize() { return L3Cache; }
  inline uint32_t getL2CacheSize() { return L2Cache; }
  inline uint32_t getL1CacheSize() { return L1Cache; }
  inline uint32_t getL2CacheSize_E() { return E_L2Cache; }
  inline uint32_t getL1CacheSize_E() { return E_L1Cache; }
  inline bool INTEL() { return mIntel; }
  inline bool AVX() { return mHasAVX; }
  inline bool AVX2() { return mHasAVX2; }
  inline bool AVX_VNNI() { return mHasAVX_VNNI; }
  inline bool AVX512F() { return mHasAVX512F; }
  inline bool AVX512BW() { return mHasAVX512BW; }
  inline bool AVX512_VNNI() { return mHasAVX512_VNNI; }
  inline bool AMX_INT8() { return mHasAMX_INT8; }
  inline bool AMX_BF16() { return mHasAMX_BF16; }
  inline bool AVX512_BF16() { return mHasAVX512_BF16; }
  inline bool AVX512_FP16() { return mHasAVX512_FP16; }
  inline bool AMX_FP16() { return mHasAMX_FP16; }
  inline float* const getPE() { return PE; }
  inline int getPcoreNum() { return static_cast<int>(P_core.size()); }
  inline int getEcoreNum() { return static_cast<int>(E_core.size()); }
  inline int getSMTcoreNum() { return isHybrid() ? static_cast<int>(SMT_core.size()) : getCores(); }
  inline int* getPCores() { return P_core.data(); }
  inline int* getECores() { return E_core.data(); }
  inline int* getSMTCores() { return SMT_core.data(); }
#define ADD_FLAG(isa) mHas##isa = _cpu.has(_cpu.t##isa)
  CpuDevice() {
    static Xbyak::util::Cpu _cpu;
    ADD_FLAG(AVX);
    ADD_FLAG(AVX2);
    ADD_FLAG(AVX512F);
    ADD_FLAG(AVX512BW);
    ADD_FLAG(AVX512_VNNI);
    ADD_FLAG(AVX_VNNI);
    ADD_FLAG(AMX_BF16);
    ADD_FLAG(AMX_INT8);
    ADD_FLAG(AVX512_BF16);
    ADD_FLAG(AVX512_FP16);
    ADD_FLAG(AMX_FP16);
    mIntel = _cpu.has(_cpu.tINTEL);
    numcores = _cpu.getNumCores(Xbyak::util::IntelCpuTopologyLevel::CoreLevel);
    numcores = numcores >= 48 ? numcores * 2 : numcores;
    if (mHasAMX_BF16 || mHasAMX_INT8 || mHasAMX_FP16) {
      utils::request_perm_xtile_data();
    }
    static bool p = false;
    {
      uint32_t tmp[4];
      _cpu.getCpuid(7, tmp);
      if (tmp[3] & (1U << 15)) mHybrid = true;
      if (p) printf("!!!Hybrid:%d\t%x\t%x\t%x\t%x!!!\n", mHybrid, tmp[0], tmp[1], tmp[2], tmp[3]);
    }
    num_total_threads_ = numcores * _cpu.getNumCores(Xbyak::util::IntelCpuTopologyLevel::SmtLevel);
    compute_infos_.resize(num_total_threads_);

    if (mHybrid) {
      mClient = true;
      std::vector<int> core_type(num_total_threads_), core_id(num_total_threads_), L1(num_total_threads_),
          L2(num_total_threads_);
      std::map<int, int> core_id_count;
      std::vector<CpuRuntimeV2> cores(num_total_threads_);
      {
        // classify E-core / LPE-core and  P-core / smt
        std::vector<std::thread> thdset(num_total_threads_);
        for (size_t i = 0; i < num_total_threads_; i++) {
          thdset[i] = std::thread(
              [&](int tidx) {
                core_bond(tidx);
                Xbyak::util::Cpu cpu;
                L1[tidx] = cpu.getDataCacheSize(0);
                L2[tidx] = cpu.getDataCacheSize(1);
                if (isEcore(cpu))
                  core_type[tidx] = 1;
                else
                  core_type[tidx] = 2;
                core_id[tidx] = getCoreId(cpu);
              },
              int(i));
        }
        for (size_t i = 0; i < num_total_threads_; i++) {
          thdset[i].join();
          core_id_count[core_id[i]] = core_id_count[core_id[i]] + 1;
        }
        if (p) {
          for (int i = 0; i < num_total_threads_; i++) printf("%d %d\n", core_type[i], core_id[i]);
          for (auto& kv : core_id_count) printf("%d,%d\n", kv.first, kv.second);
        }
        for (int i = 0; i < num_total_threads_; i++) {
          if (core_type[i] == 2) {
            if (core_id_count[core_id[i]] > 0) {
              P_core.push_back(i);
              core_id_count[core_id[i]] = 0;
            } else {
              SMT_core.push_back(i);
            }
          } else {
            if (core_id_count[core_id[i]] == 4) E_core.push_back(i);
          }
        }
        if (p) {
          printf("Pcore:");
          for (auto& i : P_core) printf("%d,", i);
          printf("\nEcore:");
          for (auto& i : E_core) printf("%d,", i);
          printf("\nsmt:");
          for (auto& i : SMT_core) printf("%d,", i);
          printf("\n");
        }
        mHybrid = !(E_core.empty() || P_core.empty());  // in case of bond core by external
        if (!E_core.empty()) {
          E_L1Cache = L1[E_core[0]];
          E_L2Cache = L2[E_core[0]] / 4;
        };
        if (!P_core.empty()) {
          L1Cache = L1[P_core[0]];
          L2Cache = L2[P_core[0]];
        }
      }
      numcores = static_cast<int>(P_core.size() + E_core.size());
      numthreads = static_cast<int>(P_core.size() + E_core.size() + SMT_core.size());

      {
        // set PE
        uint32_t tmp[4];
        _cpu.getCpuid(1, tmp);
        if (p) printf("!!!\t%x\t%x\t%x\t%x!!!\n", tmp[0], tmp[1], tmp[2], tmp[3]);
        const int famliy = (tmp[0] >> 8) & ((1u << 4) - 1);          // cpu.extractBit(a[0], 8, 11);
        const int extendedModel = (tmp[0] >> 16) & ((1u << 4) - 1);  // cpu.extractBit(a[0], 16, 24);
        {
          for (int i = 0; i < int(BTLA_ISA::ISA_COUNT); i++) PE[i] = 1.0f;
          // CPU identification refer to: https://en.wikichip.org/wiki/intel/cpuid
          if (famliy == 6) switch (extendedModel) {
              case 9:  // ALD
                PE[int(BTLA_ISA::AVX2)] = 3.0f;
                PE[int(BTLA_ISA::AVX_VNNI)] = 5.0f;
                PE[int(BTLA_ISA::NoSIMD)] = 3.5f;
                break;
              case 10:  // MTL
                PE[int(BTLA_ISA::AVX2)] = 2.2f;
                PE[int(BTLA_ISA::AVX_VNNI)] = 3.0f;
                PE[int(BTLA_ISA::NoSIMD)] = 3.0f;
                break;
              case 11:  // RPL
                PE[int(BTLA_ISA::AVX2)] = 1.8f;
                PE[int(BTLA_ISA::AVX_VNNI)] = 2.6f;
                PE[int(BTLA_ISA::NoSIMD)] = 3.0f;
                break;
            }
        }
      }
    } else {
      L1Cache = _cpu.getDataCacheSize(0);
      L2Cache = _cpu.getDataCacheSize(1);
      numthreads = numcores;
    }
    L3Cache = _cpu.getDataCacheSize(2);
#if FIXED_CACHE
    L2Cache = L2Cache >= FIXED_CACHE_SIZE ? FIXED_CACHE_SIZE : L2Cache;
    E_L2Cache = E_L2Cache >= FIXED_CACHE_SIZE ? FIXED_CACHE_SIZE : E_L2Cache;
#endif
  }

  static CpuDevice* getInstance() {
    static CpuDevice instance;
    return &instance;
  }

  int get_actual_usable_cores() {
#ifdef _WIN32
    return numcores;
#else
    cpu_set_t cpuset;
    // 初始化 CPU 集合
    CPU_ZERO(&cpuset);

    // 获取当前进程的亲和性掩码
    // 第一个参数为 0 表示当前进程
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
      return numcores;
    }

    // 计算集合中包含的核心数量
    int count = CPU_COUNT(&cpuset);
    return count;
#endif
  }

  void print() {
    printf(
        "AVX:%d AVX2:%d AVX512F:%d AVX512BW:%d AVX_VNNI:%d AVX512_VNNI:%d AMX_INT8:%d AMX_BF16:%d AVX512_BF16:%d "
        "AVX512_FP16:%d\n",
        mHasAVX, mHasAVX2, mHasAVX512F, mHasAVX512BW, mHasAVX_VNNI, mHasAVX512_VNNI, mHasAMX_INT8, mHasAMX_BF16,
        mHasAVX512_BF16, mHasAVX512_FP16);
  }
#undef ADD_FLAG

  static bool isEcore() {
    Xbyak::util::Cpu cpu;
    uint32_t tmp[4];
    cpu.getCpuid(0x1A, tmp);
    int core_type = (tmp[0] >> 24) & ((1u << 8) - 1);  // cpu.extractBit(a[0], 24, 31);
    switch (core_type) {
      case 32:
        // printf("Atom\n");
        return true;  // E-core or LPE-core
        break;
      case 64:
        // printf("Core\n");
        return false;  // P-core
        break;
      default:
        // printf("No hyper\n");
        return false;
        break;
    }
    return false;
  }

  int getCoreId(Xbyak::util::Cpu& cpu) {
    uint32_t tmp[4];
    cpu.getCpuidEx(0x1F, 1, tmp);  // sub-leaf 1 is core domain
    // printf("!!!%x\t%x\t%x\t%x!!!\n", tmp[0], tmp[1], tmp[2], tmp[3]);
    if (tmp[0] != 0 && tmp[1] != 0)
      return tmp[3] >> 3;  // tmp[3] is APIC
    else
      return tmp[3];
  }

  bool isEcore(Xbyak::util::Cpu& cpu) {
    uint32_t tmp[4];
    cpu.getCpuid(0x1A, tmp);
    int core_type = (tmp[0] >> 24) & ((1u << 7) - 1);  // cpu.extractBit(a[0], 24, 31);
    switch (core_type) {
      case 32:
        // printf("Atom\n");
        return true;  // E-core or LPE-core
        break;
      case 64:
        // printf("Core\n");
        return false;  // P-core
        break;
      default:
        // printf("No hyper\n");
        return false;
        break;
    }
    return false;
  }
  static void core_bond(int core) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1LL << core);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    int s = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (s != 0) printf("Bond Core ERROR:%d\n", core);
#endif
  }

  static void core_bond(std::thread& thread, int core) {
#ifdef _WIN32
    HANDLE handle = thread.native_handle();
    SetThreadAffinityMask(handle, 1LL << core);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_t pt = thread.native_handle();
    int s = pthread_setaffinity_np(pt, sizeof(cpuset), &cpuset);
    if (s != 0) printf("Bond Core ERROR:%d\n", core);
#endif
  }

  bool isHybrid() { return mHybrid; }
  bool isClient() { return mClient; }

 protected:
  std::vector<CpuRuntimeV2> compute_infos_;
  uint32_t L2Cache = 0, L1Cache = 0, L3Cache = 0;
  bool mHybrid = false, mClient = false;
  bool mHasAVX2 = false, mHasAVX_VNNI = false, mHasAVX = false, mHasAVX512_VNNI = false, mHasAMX_INT8 = false,
       mHasAMX_BF16 = false, mHasAVX512F = false, mHasAVX512BW, mHasAVX512_BF16 = false, mHasAVX512_FP16 = false,
       mHasAMX_FP16 = false, mIntel = false;
  int numcores = 0;
  int numthreads = 0, num_total_threads_ = 0;
  std::vector<int> P_core, E_core, SMT_core;
  uint32_t E_L2Cache = 0, E_L1Cache = 0;
  float PE[int(BTLA_ISA::ISA_COUNT)] = {1.f};
};

#define GetCPUDevice() auto _cd = bestla::device::CpuDevice::getInstance();

/*
1. P coers
3. P cores + P hyper
5. P cores => numanodes
6. P cores => numanodes + hyper
7. P cores => sockets (numanodes + hyper)

client only:
2. P cores + E Cores
4. P cores + P hyper + E Cores
*/
class CpuRange {
 public:
  enum class CoreType : int { Undef = 0, E = 1, P = 2, Count };
  /*inline bool AVX2() { return mHasAVX2; }
  inline bool AVX_VNNI() { return mHasAVX_VNNI; }
  inline bool AVX512F() { return mHasAVX512F; }
  inline bool AVX512BW() { return mHasAVX512BW; }
  inline bool AVX512_VNNI() { return mHasAVX512_VNNI; }
  inline bool AMX_INT8() { return mHasAMX_INT8; }
  inline bool AMX_BF16() { return mHasAMX_BF16; }
  inline bool AVX512_BF16() { return mHasAVX512_BF16; }
  inline bool AVX512_FP16() { return mHasAVX512_FP16; }
  inline bool AMX_FP16() { return mHasAMX_FP16; }
  inline float* const getPE() { return PE; }
  inline int getPcoreNum() { return static_cast<int>(P_core.size()); }
  inline int getEcoreNum() { return static_cast<int>(E_core.size()); }
  inline int getSMTcoreNum() { return isHybrid() ? static_cast<int>(SMT_core.size()) : getCores(); }
  inline int* getPCores() { return P_core.data(); }
  inline int* getECores() { return E_core.data(); }
  inline int* getSMTCores() { return SMT_core.data(); }*/

  CpuRange() {  // single thread
    cores_ = {0};
    update();
  }

  CpuRange(const std::vector<int>& _cores, bool _hyper = false) {
    cores_ = _cores;
    hyper_ = _hyper;
    update();
  }

  CpuRange(const std::vector<std::vector<int>>& _core_groups, bool _hyper = true) {
    hyper_ = _hyper;
    core_groups_ = _core_groups;
    for (size_t i = 0; i < _core_groups.size(); i++) {
      cores_.insert(cores_.end(), _core_groups[i].begin(), _core_groups[i].end());
    }
    update();
  }

  CpuRange(int pcores, int ecores, bool _hyper = false) {
    int p_num = _hyper ? pcores * 2 : pcores;
    int e_num = ecores;
    hyper_ = _hyper;
    Xbyak::util::Cpu cpu;
    auto n_core = cpu.getNumCores(Xbyak::util::SmtLevel) * cpu.getNumCores(Xbyak::util::CoreLevel);
    std::vector<int> all_cores(n_core);
    for (size_t i = 0; i < all_cores.size(); i++) {
      all_cores[i] = i;
    }
    cores_ = all_cores;
    update();
    std::vector<int> p_cores(p_num);
    for (size_t i = 0; i < p_num; i++) {
      p_cores[i] = _hyper ? i : i * 2;
    }
    core_groups_.emplace_back(p_cores);
    if (e_num) {
      std::vector<int> e_cores(e_num);
      auto e_start = core_type_count_[(int)CoreType::P];
      for (size_t i = 0; i < e_num; i++) {
        e_cores[i] = e_start + i;
      }
      core_groups_.emplace_back(e_cores);
    }
  }

  void update() {
    std::vector<std::thread> thdset(cores_.size());
    core_infos_.resize(cores_.size());
    core_types_.resize(cores_.size());
    core_ids_.resize(cores_.size());
    core_type_count_.resize((int)CoreType::Count, 0);
    for (size_t i = 0; i < thdset.size(); i++) {
      thdset[i] = std::thread(
          [&](int tidx) {
            core_bond(cores_[tidx]);
            Xbyak::util::Cpu cpu;
            core_infos_[tidx] = cpu;
            if (isEcore(cpu))
              core_types_[tidx] = CoreType::E;
            else
              core_types_[tidx] = CoreType::P;
            core_ids_[tidx] = getCoreId(cpu);
          },
          int(i));
    }
    for (size_t i = 0; i < thdset.size(); i++) {
      thdset[i].join();
    }
    for (size_t i = 0; i < core_ids_.size(); i++) {
      core_type_count_[(int)core_types_[i]] += 1;
    }
  }

#define ADD_FUNC(isa) \
  inline bool isa() { return core_infos_[0].has(core_infos_[0].t##isa); }
  inline int core_size() { return cores_.size(); }
  ADD_FUNC(AVX);
  ADD_FUNC(AVX2);
  ADD_FUNC(AVX512F);
  ADD_FUNC(AVX512BW);
  ADD_FUNC(AVX512_VNNI);
  ADD_FUNC(AVX_VNNI);
  ADD_FUNC(AMX_INT8);
  ADD_FUNC(AMX_BF16);
  ADD_FUNC(AMX_FP16);
  ADD_FUNC(AVX512_BF16);
  ADD_FUNC(AVX512_FP16);
  void print() {
    if (core_infos_.size()) {
      auto& info = core_infos_[0];
      printf(
          "AVX:%d AVX2:%d AVX512F:%d AVX512BW:%d AVX_VNNI:%d AVX512_VNNI:%d AMX_INT8:%d AMX_BF16:%d AVX512_BF16:%d "
          "AVX512_FP16:%d\n",
          AVX(), AVX2(), AVX512F(), AVX512BW(), AVX_VNNI(), AVX512_VNNI(), AMX_INT8(), AMX_BF16(), AVX512_BF16(),
          AVX512_FP16());
    }
  }
#undef ADD_FUNC

  static bool isEcore() {
    Xbyak::util::Cpu cpu;
    uint32_t tmp[4];
    cpu.getCpuid(0x1A, tmp);
    int core_type = (tmp[0] >> 24) & ((1u << 8) - 1);  // cpu.extractBit(a[0], 24, 31);
    switch (core_type) {
      case 32:
        // printf("Atom\n");
        return true;  // E-core or LPE-core
        break;
      case 64:
        // printf("Core\n");
        return false;  // P-core
        break;
      default:
        // printf("No hyper\n");
        return false;
        break;
    }
    return false;
  }

  int getCoreId(Xbyak::util::Cpu& cpu) {
    uint32_t tmp[4];
    cpu.getCpuidEx(0x1F, 1, tmp);  // sub-leaf 1 is core domain
    // printf("!!!%x\t%x\t%x\t%x!!!\n", tmp[0], tmp[1], tmp[2], tmp[3]);
    if (tmp[0] != 0 && tmp[1] != 0)
      return tmp[3] >> 3;  // tmp[3] is APIC
    else
      return tmp[3];
  }

  bool isEcore(Xbyak::util::Cpu& cpu) {
    uint32_t tmp[4];
    cpu.getCpuid(0x1A, tmp);
    int core_type = (tmp[0] >> 24) & ((1u << 7) - 1);  // cpu.extractBit(a[0], 24, 31);
    switch (core_type) {
      case 32:
        // printf("Atom\n");
        return true;  // E-core or LPE-core
        break;
      case 64:
        // printf("Core\n");
        return false;  // P-core
        break;
      default:
        // printf("No hyper\n");
        return false;
        break;
    }
    return false;
  }

  static void core_bond(int core) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1LL << core);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    int s = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (s != 0) printf("Bond Core ERROR:%d\n", core);
#endif
  }

  static void core_bond(std::thread& thread, int core) {
#ifdef _WIN32
    HANDLE handle = thread.native_handle();
    SetThreadAffinityMask(handle, 1LL << core);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_t pt = thread.native_handle();
    int s = pthread_setaffinity_np(pt, sizeof(cpuset), &cpuset);
    if (s != 0) printf("Bond Core ERROR:%d\n", core);
#endif
  }

  std::vector<std::vector<int>> get_groups() const { return core_groups_; }

 protected:
  std::vector<std::vector<int>> core_groups_;
  std::vector<int> cores_;
  std::vector<Xbyak::util::Cpu> core_infos_;
  std::vector<CoreType> core_types_;
  std::vector<int> core_ids_;
  std::vector<CpuRuntimeV2> compute_infos_;
  std::vector<int> core_type_count_;
  bool hyper_ = false;
};

#define GetCPUDevice() auto _cd = bestla::device::CpuDevice::getInstance();

class CpuRuntime {
 public:
  CpuRuntime() = default;
  static CpuRuntime& getInstance(int thread) {
    static std::map<int, CpuRuntime> instances;
    if (instances.count(thread) == 0) instances[thread] = CpuRuntime(thread);
    return instances[thread];
  }

  inline float getPE(const BTLA_ISA isa) {
    // printf("GET:%d\t%f\n",int(isa), *cur_PE);
    return PE[int(isa)] * P_core_num / E_core_num;
  }

  inline void adjustPE(const BTLA_ISA isa, const float PE_) {
    // printf("Adjust:%d,%f\n",int(isa),PE_);
    PE[int(isa)] = PE[int(isa)] * PE_ * 0.7f + PE[int(isa)] * 0.3f;
  }

  size_t mL2Cache = 0, mL1Cache = 0, mL2Cache_P = 0, mL1Cache_P = 0, mL2Cache_E = 0, mL1Cache_E = 0;
  int P_core_num = 0, E_core_num = 0;
  bool mHybrid = false;

 private:
  CpuRuntime(int thread) {
    GetCPUDevice();
    mL2Cache = _cd->getL2CacheSize();
    mL1Cache = _cd->getL1CacheSize();
    maxThreads = _cd->getThreads();
    mHybrid = false;
    if (_cd->isClient() && thread > _cd->getPcoreNum()) {
      if (thread > _cd->getPcoreNum() + _cd->getEcoreNum()) {
        mL1Cache_P = mL1Cache / 2;
        mL2Cache_P = mL2Cache / 2;
        P_core_num = _cd->getPcoreNum();
        E_core_num = _cd->getEcoreNum();
      } else {
        mL1Cache_P = mL1Cache;
        mL2Cache_P = mL2Cache;
        P_core_num = static_cast<int>(_cd->getPcoreNum());
        E_core_num = thread - P_core_num;
      }
      if (_cd->isHybrid()) {
        mL1Cache_E = _cd->getL1CacheSize_E();
        mL2Cache_E = _cd->getL2CacheSize_E();
        mHybrid = true;
        memcpy(PE, _cd->getPE(), int(BTLA_ISA::ISA_COUNT) * sizeof(float));
      }
    }
  }
  float PE[int(BTLA_ISA::ISA_COUNT)] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  int maxThreads = 0;
};
}  // namespace device
}  // namespace bestla
