#include <stdio.h>
#include "bestla_wrapper.h"
#include "bestla_gemm_kblock.h"
#include "bestla_ut.h"

namespace bestla {
using namespace utils;
namespace ut {
int constexpr TestMs = 500;
class Benchmark_Fp32Fp32 {
 public:
  Benchmark_Fp32Fp32() {
    UT_START();
    benchmark_all(1, 4096, 4096);
    benchmark_all(1024, 4096, 4096);
  }

  using AType = float;
  using BType = float;
  using CType = float;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher = wrapper::gemm::LauncherBase<Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                                 epilogue::gemm::AccumulatorWriteBackFp32>;

    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = Launcher::PrologueB::createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      Launcher::PrologueB::packWeight(n, k, {B + i * n * k, n, &packBs[i]}, UT_Threading::get());
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        utils::GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
           flops / threads);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, BTLA_DTYPE::F32, BTLA_DTYPE::F32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32),
           bestla_dtype_str(BTLA_DTYPE::F32), bestla_dtype_str(BTLA_DTYPE::F32));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch, 0);
    fill_buffer_randn(A.data(), m * k, -0.5f, 0.5f);
    fill_buffer_randn(B.data(), n * k, -0.5f, 0.5f);
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      if (_cd->AVX512F()) {
        benchmark<gemm::SCoreRowNAvx512f<48, 8>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
      }
      if (_cd->AVX2()) {
        benchmark<gemm::SCoreRowNAvx2<24, 4>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
      }
    }
  }
};
#ifdef BTLA_UT_WRAPPER
static Benchmark_Fp32Fp32 sBenchmark_Fp32Fp32;
#endif

class Benchmark_U8S8S32 {
 public:
  Benchmark_U8S8S32() {
    UT_START();
    benchmark_all(1, 4096, 4096);
    benchmark_all(1024, 4096, 4096);
  }

  using AType = uint8_t;
  using BType = int8_t;
  using CType = int;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher = wrapper::gemm::LauncherBase<Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                                 epilogue::gemm::AccumulatorWriteBackInt32>;
    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = Launcher::PrologueB::createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      Launcher::PrologueB::packWeight(n, k, {B + i * n * k, n, &packBs[i]}, UT_Threading::get());
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        utils::GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
           flops / threads);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::U8, BTLA_DTYPE::S8, BTLA_DTYPE::S32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::U8), bestla_dtype_str(BTLA_DTYPE::S8),
           bestla_dtype_str(BTLA_DTYPE::S32));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), m * k, AType(0), AType(255));
    fill_buffer_randn(B.data(), k * n, BType(-127), BType(127));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      if (_cd->AMX_INT8()) {
        benchmark<gemm::ICoreRowNAmxint8<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
        benchmark<gemm::ICoreRowNAmxint8<48, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
        benchmark<gemm::ICoreRowNAmxint8<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
      }
      if (_cd->AVX512_VNNI()) {
        benchmark<gemm::ICoreRowNAvx512vnni<48, 8>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                         threads);
      }
      if (_cd->AVX512BW()) {
        benchmark<gemm::ICoreRowNAvx512bw<48, 8>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
      }
      if (_cd->AVX_VNNI()) {
        benchmark<gemm::ICoreRowNAvxvnni<24, 4>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
      }
      if (_cd->AVX2()) {
        benchmark<gemm::ICoreRowNAvx2vnni<24, 4>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
      }
    }
  }
};
#ifdef BTLA_UT_WRAPPER
static Benchmark_U8S8S32 sBenchmark_U8S8S32;
#endif

class Benchmark_S8S8S32 {
 public:
  Benchmark_S8S8S32() {
    UT_START();
    benchmark_all(1, 4096, 4096);
    benchmark_all(1024, 4096, 4096);
  }

  using AType = int8_t;
  using BType = int8_t;
  using CType = int;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher = wrapper::gemm::LauncherBase<Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                                 epilogue::gemm::AccumulatorWriteBackInt32>;

    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = Launcher::PrologueB::createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      Launcher::PrologueB::packWeight(n, k, {B + i * n * k, n, &packBs[i]}, UT_Threading::get());
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        utils::GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
           flops / threads);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::S8, BTLA_DTYPE::S8, BTLA_DTYPE::S32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::S8), bestla_dtype_str(BTLA_DTYPE::S8),
           bestla_dtype_str(BTLA_DTYPE::S32));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), m * k, AType(0), AType(255));
    fill_buffer_randn(B.data(), k * n, BType(-127), BType(127));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(AType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      if (_cd->AVX2()) {
        benchmark<gemm::ICoreRowNAvx2vnniSS<24, 2>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                         threads);
      }
      if (_cd->AVX_VNNI()) {
        benchmark<gemm::ICoreRowNAvxvnniSS<24, 4>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                        threads);
        benchmark<gemm::ICoreRowNAvxvnniSS<24, 3>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                        threads);
      }
      if (_cd->AMX_INT8()) {
        benchmark<gemm::ICoreRowNAmxint8SS<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                         threads);
        benchmark<gemm::ICoreRowNAmxint8SS<48, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                         threads);
        benchmark<gemm::ICoreRowNAmxint8SS<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                         threads);
      }
    }
  }
};
#ifdef BTLA_UT_WRAPPER
static Benchmark_S8S8S32 sBenchmark_S8S8S32;
#endif

class Benchmark_Bf16Bf16Fp32 {
 public:
  Benchmark_Bf16Bf16Fp32() {
    UT_START();
    benchmark_all(1, 4096, 4096);
    benchmark_all(1024, 4096, 4096);
  }

  using AType = utils::bf16;
  using BType = utils::bf16;
  using CType = float;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher = wrapper::gemm::LauncherBase<Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                                 epilogue::gemm::AccumulatorWriteBackFp32>;

    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = Launcher::PrologueB::createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      Launcher::PrologueB::packWeight(n, k, {B + i * n * k, n, &packBs[i]}, UT_Threading::get());
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        utils::GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
           flops / threads);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::BF16, BTLA_DTYPE::BF16, BTLA_DTYPE::F32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::BF16),
           bestla_dtype_str(BTLA_DTYPE::BF16), bestla_dtype_str(BTLA_DTYPE::F32));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), k * m, AType(-0.5f), AType(0.5f));
    fill_buffer_randn(B.data(), k * n, BType(-0.5f), BType(0.5f));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      if (_cd->AMX_BF16()) {
        benchmark<gemm::HCoreRowNAmxbf16<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
        benchmark<gemm::HCoreRowNAmxbf16<48, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
        benchmark<gemm::HCoreRowNAmxbf16<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
      }
    }
  }
};
#ifdef BTLA_UT_WRAPPER
static Benchmark_Bf16Bf16Fp32 sBenchmark_Bf16Bf16Fp32;
#endif

class Benchmark_Fp16Fp16Fp32 {
 public:
  Benchmark_Fp16Fp16Fp32() {
    UT_START();
    benchmark_all(1, 4096, 4096);
    benchmark_all(1024, 4096, 4096);
  }

  using AType = utils::fp16;
  using BType = utils::fp16;
  using CType = float;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher = wrapper::gemm::LauncherBase<Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                                 epilogue::gemm::AccumulatorWriteBackFp32>;

    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = Launcher::PrologueB::createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      Launcher::PrologueB::packWeight(n, k, {B + i * n * k, n, &packBs[i]}, UT_Threading::get());
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
           flops / threads);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F16, BTLA_DTYPE::F16, BTLA_DTYPE::F32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F16),
           bestla_dtype_str(BTLA_DTYPE::F16), bestla_dtype_str(BTLA_DTYPE::F32));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), k * m, AType(-0.5f), AType(0.5f));
    fill_buffer_randn(B.data(), k * n, AType(-0.5f), AType(0.5f));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      if (_cd->AMX_FP16()) {
        benchmark<gemm::HCoreRowNAmxfp16<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
        benchmark<gemm::HCoreRowNAmxfp16<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
      }
    }
  }
};
#ifdef BTLA_UT_WRAPPER
static Benchmark_Fp16Fp16Fp32 sBenchmark_Fp16Fp16Fp32;
#endif

class Benchmark_Fp16Fp16Fp16 {
 public:
  Benchmark_Fp16Fp16Fp16() {
    UT_START();
    benchmark_all(1, 4096, 4096);
    benchmark_all(1024, 4096, 4096);
  }

  using AType = utils::fp16;
  using BType = utils::fp16;
  using CType = utils::fp16;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher = wrapper::gemm::LauncherBase<Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                                 epilogue::gemm::AccumulatorWriteBackFp16>;

    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = Launcher::PrologueB::createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      Launcher::PrologueB::packWeight(n, k, {B + i * n * k, n, &packBs[i]}, UT_Threading::get());
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
           flops / threads);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F16, BTLA_DTYPE::F16, BTLA_DTYPE::F16);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F16),
           bestla_dtype_str(BTLA_DTYPE::F16), bestla_dtype_str(BTLA_DTYPE::F16));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), k * m, AType(-0.5f), AType(0.5f));
    fill_buffer_randn(B.data(), k * n, AType(-0.5f), AType(0.5f));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      if (_cd->AVX512_FP16()) {
        benchmark<sAVX512_FP16, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
        benchmark<gemm::HCoreRowNAvx512fp16<64, 12>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                          threads);
      }
    }
  }
};
#ifdef BTLA_UT_WRAPPER
static Benchmark_Fp16Fp16Fp16 sBenchmark_Fp16Fp16Fp16;
#endif

class UTWOQ_CompFp32 {
 public:
  UTWOQ_CompFp32() {
    UT_START();
    ut_s4_full();
    ut_new_dtype(BTLA_DTYPE::S1_CLIP);
    ut_new_dtype(BTLA_DTYPE::S2_CLIP);
    ut_new_dtype(BTLA_DTYPE::S3_CLIP);
    ut_new_dtype(BTLA_DTYPE::S5_CLIP);
    ut_new_dtype(BTLA_DTYPE::S6_CLIP);
    ut_new_dtype(BTLA_DTYPE::S7_CLIP);
    ut_new_dtype(BTLA_DTYPE::S8);
    ut_f4();
  }

  void ut_new_dtype(BTLA_DTYPE qtype) {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(1, 4096, 4096, qtype, true);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, qtype);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, qtype, true);
  }

  void ut_s4_full() {
    BTLA_DTYPE qtype = BTLA_DTYPE::S4_CLIP;
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(1, 4096, 4096, qtype, true);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, qtype);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, qtype, true);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, qtype);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1024, 4096, 4096, qtype);
  }

  void ut_f4() {
    benchmark_all<prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::F4_BNB);
    benchmark_all<prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::F4_E2M1);
    benchmark_all<prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::F4_NF4);
    benchmark_all<prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(1024, 4096, 4096, BTLA_DTYPE::F4_BNB);
  }

  template <typename Core_T, typename LOG_T, template <class _T> class Wei, typename Scale_T>
  void benchmark(int m, int n, int k, int batch, int blocksize, float* A, float* B, float* C, float timems, int threads,
                 BTLA_DTYPE qtype, bool isasym) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher = wrapper::gemm::LauncherBase<Core_T, prologue_a::gemm::ActivationBase, Wei,
                                                 epilogue::gemm::AccumulatorWriteBackFp32>;

    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    using WType = typename Wei<Core_T>::StorageWeight;
    WType tmpB(0);
    if constexpr (std::is_same_v<Wei<Core_T>, prologue_b::gemm::WeightKBlockNInteger<Core_T>>) {
      tmpB = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, isasym);

    } else if constexpr (std::is_same_v<Wei<Core_T>, prologue_b::gemm::WeightKBlockNFloat<Core_T>>) {
      tmpB = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);
    }
    std::vector<WType> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
    }
    Launcher::PrologueB::packWeight(n, k, B, n, &packBs[0], UT_Threading::get());
    for (size_t i = 1; i < batch; i++) {
      memcpy(packBs[i].template WPtr<void>(), packBs[0].template WPtr<void>(), packBs[0].template WSize<char>());
      memcpy(packBs[i].template SPtr<void>(), packBs[0].template SPtr<void>(), packBs[0].CSize() * sizeof(Scale_T));
    }
    auto psize = (size_t)m * n * k * 2;
    int blks = k / blocksize;
    size_t nbits = utils::bestla_dtype_bits(qtype);
    auto memsize = (size_t)(n * k * nbits / 8 + n * blks * sizeof(Scale_T)) + (m * k + m * n) * sizeof(float);
    tm.start();
    while (tm.stop() < timems) {
      for (int i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k, blocksize);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {&packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    double band = double(memsize) / log.min_val / 1e6;
    int cores = std::min(threads, device::CpuDevice::getInstance()->getCores());
    printf("Threads %d Block %d %s %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, blocksize,
           corestr, log.get_log_str(), flops, flops / cores, band);
  }

  template <template <class _T> class Wei, typename Scale_T>
  void benchmark_all(int m, int n, int k, BTLA_DTYPE qtype, bool isasym = false) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, qtype, BTLA_DTYPE::F32);
    int batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s isasym:%d\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32),
           bestla_dtype_str(qtype), bestla_dtype_str(BTLA_DTYPE::F32), isasym);
    avector<float> A(size_t(m) * k * batch);
    avector<float> B(size_t(k) * n);
    avector<float> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), k * m, (-0.5f), (0.5f));
    fill_buffer_randn(B.data(), k * n, (-0.5f), (0.5f));
    for (int i = 1; i < batch; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(float));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      for (auto blocksize : {32, 128}) {
        if (_cd->AVX512F()) {
          benchmark<gemm::SCoreRowNAvx512f<48, 8>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                      C.data(), testtime, threads, qtype, isasym);
        }
        if (_cd->AVX2()) {
          benchmark<gemm::SCoreRowNAvx2<24, 4>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                   C.data(), testtime, threads, qtype, isasym);
        }
      }
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UTWOQ_CompFp32 sUTWOQ_CompFp32;
#endif

class UTWOQ_CompBf16 {
 public:
  UTWOQ_CompBf16() {
    UT_START();
    ut_s4();
    ut_s8();
    ut_f4();
  }

  void ut_s4() {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1024, 4096, 4096, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(2048, 4096, 4096, BTLA_DTYPE::S4_CLIP);
  }

  void ut_s8() {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::S8);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1024, 4096, 4096, BTLA_DTYPE::S8);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(2048, 4096, 4096, BTLA_DTYPE::S8);
  }

  void ut_f4() {
    benchmark_all<prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::F4_BNB);
    benchmark_all<prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(1024, 4096, 4096, BTLA_DTYPE::F4_BNB);
    benchmark_all<prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2048, 4096, 4096, BTLA_DTYPE::F4_BNB);
  }

  template <typename Core_T, typename LOG_T, template <class _T> class Wei, typename Scale_T>
  void benchmark(int m, int n, int k, int batch, int blocksize, float* A, float* B, float* C, float timems, int threads,
                 BTLA_DTYPE qtype) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher = wrapper::gemm::LauncherBase<Core_T, prologue_a::gemm::ActivationConverterFp32, Wei,
                                                 epilogue::gemm::AccumulatorWriteBackFp32>;

    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    using WType = typename Wei<Core_T>::StorageWeight;
    WType tmpB(0);
    if constexpr (std::is_same_v<Wei<Core_T>, prologue_b::gemm::WeightKBlockNInteger<Core_T>>) {
      tmpB = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, false);

    } else if constexpr (std::is_same_v<Wei<Core_T>, prologue_b::gemm::WeightKBlockNFloat<Core_T>>) {
      tmpB = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);
    }
    std::vector<WType> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
    }
    Launcher::PrologueB::packWeight(n, k, B, n, &packBs[0], UT_Threading::get());
    for (size_t i = 1; i < batch; i++) {
      memcpy(packBs[i].template WPtr<void>(), packBs[0].template WPtr<void>(), packBs[0].template WSize<char>());
      memcpy(packBs[i].template SPtr<void>(), packBs[0].template SPtr<void>(), packBs[0].CSize() * sizeof(Scale_T));
    }
    auto psize = (size_t)m * n * k * 2;
    int blks = k / blocksize;
    size_t nbits = utils::bestla_dtype_bits(qtype);
    auto memsize = (size_t)(n * k * nbits / 8 + n * blks * sizeof(Scale_T)) + (m * k + m * n) * sizeof(float);
    tm.start();
    while (tm.stop() < timems) {
      for (int i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {&packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel, Launcher>(args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    double band = double(memsize) / log.min_val / 1e6;
    int cores = std::min(threads, device::CpuDevice::getInstance()->getCores());
    printf("Threads %d Block %d %s %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, blocksize,
           corestr, log.get_log_str(), flops, flops / cores, band);
  }

  template <template <class _T> class Wei, typename Scale_T>
  void benchmark_all(int m, int n, int k, BTLA_DTYPE qtype) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, qtype, BTLA_DTYPE::F32);
    int batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32), bestla_dtype_str(qtype),
           bestla_dtype_str(BTLA_DTYPE::F32));
    avector<float> A(size_t(m) * k * batch);
    avector<float> B(size_t(k) * n);
    avector<float> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), k * m, (-0.5f), (0.5f));
    fill_buffer_randn(B.data(), k * n, (-0.5f), (0.5f));
    for (int i = 1; i < batch; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(float));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      for (auto blocksize : {32, 128}) {
        if (_cd->AMX_BF16()) {
          benchmark<gemm::HCoreRowNAmxbf16<64, 16>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                       C.data(), testtime, threads, qtype);
        }
      }
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UTWOQ_CompBf16 sUTWOQ_CompBf16;
#endif

class UTWOQ_CompInt8 {
 public:
  UTWOQ_CompInt8() {
    UT_START();
    ut_s4_full();
    ut_new_dtype(BTLA_DTYPE::S1_CLIP);
    ut_new_dtype(BTLA_DTYPE::S2_CLIP);
    ut_new_dtype(BTLA_DTYPE::S3_CLIP);
    ut_new_dtype(BTLA_DTYPE::S5_CLIP);
    ut_new_dtype(BTLA_DTYPE::S6_CLIP);
    ut_new_dtype(BTLA_DTYPE::S7_CLIP);
    ut_new_dtype(BTLA_DTYPE::S8);
  }
  void ut_new_dtype(BTLA_DTYPE qtype) {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(1, 4096, 4096, qtype, true);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, qtype);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, qtype, true);
  }

  void ut_s4_full() {
    BTLA_DTYPE qtype = BTLA_DTYPE::S4_CLIP;
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(1, 4096, 4096, qtype);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, qtype);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, qtype, true);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(4, 4096, 4096, qtype);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(8, 4096, 4096, qtype);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1024, 4096, 4096, qtype);
  }

  using PcWriteBack = epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue::gemm::AccumulatorWriteBackFp32>;

  template <typename Core_T, typename LOG_T, template <class _T> class Wei, typename Scale_T>
  void benchmark_pc(int m, int n, int k, int batch, int blocksize, float* A, float* B, float* C, float timems,
                    int threads, BTLA_DTYPE qtype, bool isasym) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherBase<Core_T, prologue_a::gemm::ActivationF32KBlockQuantize, Wei, PcWriteBack>;

    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    using WType = typename Wei<Core_T>::StorageWeight;
    WType tmpB = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, isasym);
    std::vector<WType> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
    }
    Launcher::PrologueB::packWeight(n, k, B, n, &packBs[0], UT_Threading::get());
    for (size_t i = 1; i < batch; i++) {
      memcpy(packBs[i].template WPtr<void>(), packBs[0].template WPtr<void>(), packBs[0].template WSize<char>());
      memcpy(packBs[i].template SPtr<void>(), packBs[0].template SPtr<void>(), packBs[0].CSize() * sizeof(Scale_T));
    }
    auto quanA = Launcher::PrologueA::createStorage(m, k, blocksize, false);
    utils::avector<int8_t> bufferA(quanA.mSize);
    quanA.assign(bufferA.data());
    auto psize = (size_t)m * n * k * 2;
    int blks = k / blocksize;
    auto nbits = utils::bestla_dtype_bits(qtype);
    auto memsize = (size_t)(n * k * nbits / 8 + n * blks * sizeof(Scale_T)) + (m * k + m * n) * sizeof(float);
    if (isasym) {
      memsize += n * blks * sizeof(int8_t);
    }
    tm.start();
    while (tm.stop() < timems) {
      for (int i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k, blocksize);
        typename Launcher::Param args{
            gp,
            {A + i * m * k, k, &quanA},
            {&packBs[i]},
            {{packBs[i].template SPtr<char>(), packBs[i].SDtype(), quanA.template SPtr<float>(),
              quanA.template ZPtr<uint8_t>(), packBs[i].template RPtr<int32_t>(), nullptr, nullptr, k},
             {C + i * m * n, n}}};
        parallel::GemmRunWithA<Parallel, Launcher>(args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    double band = double(memsize) / log.min_val / 1e6;
    int cores = std::min(threads, device::CpuDevice::getInstance()->getCores());
    printf("Threads %d Block %d %s %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, blocksize,
           corestr, log.get_log_str(), flops, flops / cores, band);
  }

  template <typename Core_T, typename LOG_T, template <class _T> class Wei, typename Scale_T>
  void benchmark(int m, int n, int k, int batch, int blocksize, float* A, float* B, float* C, float timems, int threads,
                 BTLA_DTYPE qtype, bool isasym) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerIntKBlock<Core_T>;
    using Launcher = wrapper::gemm::LauncherIntKBlock<Core_T, prologue_a::gemm::ActivationF32KBlockQuantize, Wei,
                                                      epilogue::gemm::AccumulatorWriteBackFp32>;

    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    using WType = typename Wei<Core_T>::StorageWeight;
    WType tmpB = Launcher::PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, isasym);
    std::vector<WType> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
    }
    Launcher::PrologueB::packWeight(n, k, B, n, &packBs[0], UT_Threading::get());
    for (size_t i = 1; i < batch; i++) {
      memcpy(packBs[i].template WPtr<void>(), packBs[0].template WPtr<void>(), packBs[0].template WSize<char>());
      memcpy(packBs[i].template SPtr<void>(), packBs[0].template SPtr<void>(), packBs[0].CSize() * sizeof(Scale_T));
    }
    auto quanA = Launcher::PrologueA::createStorage(m, k, blocksize, false);
    utils::avector<int8_t> bufferA(quanA.mSize);
    quanA.assign(bufferA.data());
    auto psize = (size_t)m * n * k * 2;
    int blks = k / blocksize;
    auto nbits = utils::bestla_dtype_bits(qtype);
    auto memsize = (size_t)(n * k * nbits / 8 + n * blks * sizeof(Scale_T)) + (m * k + m * n) * sizeof(float);
    if (isasym) {
      memsize += n * blks * sizeof(int8_t);
    }
    tm.start();
    while (tm.stop() < timems) {
      for (int i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k, blocksize);
        typename Launcher::Param args{gp, {A + i * m * k, k, &quanA}, {&packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRunWithA<Parallel, Launcher>(args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    double band = double(memsize) / log.min_val / 1e6;
    printf("Threads %d Block %d %s %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, blocksize,
           corestr, log.get_log_str(), flops, flops / threads, band);
  }

  template <template <class _T> class Wei, typename Scale_T>
  void benchmark_all(int m, int n, int k, BTLA_DTYPE qtype, bool isasym = false) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, qtype, BTLA_DTYPE::F32);
    int batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s Asym:%d Scale:%s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32),
           bestla_dtype_str(qtype), bestla_dtype_str(BTLA_DTYPE::F32), isasym, bestla_dtype_str(bestla_dtype<Scale_T>));
    avector<float> A(size_t(m) * k * batch);
    avector<float> B(size_t(k) * n);
    avector<float> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), k * m, (-0.5f), (0.5f));
    fill_buffer_randn(B.data(), k * n, (-0.5f), (0.5f));
    for (int i = 1; i < batch; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(float));
    }
    using LOG = timer_statistics_logger<TestMs / 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      for (auto blocksize : {k}) {
        if (_cd->AMX_INT8() && blocksize % 64 == 0) {
          benchmark_pc<gemm::ICoreRowNAmxint8SS<64, 16>, LOG, Wei, Scale_T>(
              m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype, isasym);
        }
        if (_cd->AVX512_VNNI()) {
          benchmark_pc<gemm::ICoreRowNAvx512vnni<48, 8>, LOG, Wei, Scale_T>(
              m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype, isasym);
        }
        if (_cd->AVX512BW()) {
          benchmark_pc<gemm::ICoreRowNAvx512bw<48, 8>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                          C.data(), testtime, threads, qtype, isasym);
        }
        if (_cd->AVX_VNNI()) {
          benchmark_pc<gemm::ICoreRowNAvxvnni<24, 4>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                         C.data(), testtime, threads, qtype, isasym);
        }
        if (_cd->AVX2()) {
          benchmark_pc<gemm::ICoreRowNAvx2vnni<24, 4>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                          C.data(), testtime, threads, qtype, isasym);
        }
      }
    }

    for (auto threads : threads_cfg) {
      for (auto blocksize : {32, 128}) {
        if (_cd->AMX_INT8() && blocksize % 64 == 0) {
          benchmark<gemm::KblockGemmAmxint8<48, 16>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                        C.data(), testtime, threads, qtype, isasym);
        }
        if (_cd->AVX512_VNNI()) {
          benchmark<gemm::KblockGemmAvx512vnni<48, 4>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                          C.data(), testtime, threads, qtype, isasym);
        }
        if (_cd->AVX512BW()) {
          benchmark<gemm::KblockGemmAvx512bw<48, 4>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                        C.data(), testtime, threads, qtype, isasym);
        }
        if (_cd->AVX_VNNI()) {
          benchmark<gemm::KblockGemmAvxvnni<24, 2>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                       C.data(), testtime, threads, qtype, isasym);
        }
        if (_cd->AVX2()) {
          benchmark<gemm::KblockGemmAvx2vnni<24, 2>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                        C.data(), testtime, threads, qtype, isasym);
        }
      }
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UTWOQ_CompInt8 sUTWOQ_CompInt8;
#endif
}  // namespace ut
}  // namespace bestla
int main() {
  printf("BesTLA Benchmark done\n");
  return 0;
}
