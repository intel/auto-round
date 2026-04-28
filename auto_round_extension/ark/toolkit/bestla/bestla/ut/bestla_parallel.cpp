#include "bestla_utils.h"
#include "bestla_parallel.h"
#include "bestla_device.h"
#include "bestla_gemm.h"
#include "bestla_ut.h"
#include "bestla_prologue_a.h"

namespace bestla {
using namespace utils;
namespace ut {
#ifdef _OPENMP
class UT_OMPThreading {
 public:
  UT_OMPThreading() {
    UT_START();
    GetCPUDevice();
    ut_transpose(1024, 1024, _cd->getThreads());
    ut_transpose(123, 111, _cd->getThreads());
  }

  void ut_transpose(int row, int col, int threads) {
    printf("%s %d %d %d\n", __FUNCTION__, row, col, threads);
    avector<float> src(row * col), dst(row * col), ref(row * col);
    fill_buffer_randn(src.data(), src.size(), -0.5f, 0.5f);
    int ld_src = col, ld_dst = row;
    kernel::wrapper::Transpose2D<float>::template forward<BTLA_ISA::AVX512F>(src.data(), ref.data(), row, col, col,
                                                                             row);
    parallel::Scheduler2D _para({threads, row, col, 1, 1, 0, 0});
    UT_Threading::get()->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      _para.getIndex(thdp);
      if (thdp.valid) {
        kernel::wrapper::Transpose2D<float>::template forward<BTLA_ISA::AVX512F>(
            src.data() + thdp.loc[0] * ld_src + thdp.loc[1], dst.data() + thdp.loc[0] + thdp.loc[1] * ld_dst,
            thdp.size[0], thdp.size[1], ld_src, ld_dst);
      }
    });
    buffer_error(ref.data(), dst.data(), ref.size());
  }
};
#ifdef BTLA_UT_PARALLEL
static UT_OMPThreading sUT_OMPThreading;
#endif
#endif

class UT_StdThreading {
 public:
  UT_StdThreading() {
    UT_START();
    GetCPUDevice();
    ut_transpose(1024, 1024, _cd->getThreads());
    ut_transpose(123, 111, _cd->getThreads());
  }

  void ut_transpose(int row, int col, int threads) {
    printf("%s %d %d %d\n", __FUNCTION__, row, col, threads);
    avector<float> src(row * col), dst(row * col), ref(row * col);
    fill_buffer_randn(src.data(), src.size(), -0.5f, 0.5f);
    int ld_src = col, ld_dst = row;
    kernel::wrapper::Transpose2D<float>::template forward<BTLA_ISA::AVX512F>(src.data(), ref.data(), row, col, col,
                                                                             row);
    parallel::Scheduler2D _para({threads, row, col, 1, 1, 0, 0});
    UT_Threading::get()->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      _para.getIndex(thdp);
      if (thdp.valid) {
        kernel::wrapper::Transpose2D<float>::template forward<BTLA_ISA::AVX512F>(
            src.data() + thdp.loc[0] * ld_src + thdp.loc[1], dst.data() + thdp.loc[0] + thdp.loc[1] * ld_dst,
            thdp.size[0], thdp.size[1], ld_src, ld_dst);
      }
    });
    buffer_error(ref.data(), dst.data(), ref.size());
  }
};
#ifdef BTLA_UT_PARALLEL
static UT_StdThreading sUT_StdThreading;
#endif

class UT_StdThreadingV2 {
 public:
  UT_StdThreadingV2() {
    UT_START();
    new_thread_group();
    gelu_test(4, 8);
    // gemm_test_hyper();
  }

  void gelu_test(int pcores, int ecores = 8) {
    size_t batch = 100;
    size_t count = 4 * 384 * 1024;
    utils::avector<float> src(batch * count, 1.f), dst(batch * count), ref(batch * count);
    utils::timer tm;
    tm.start();
    for (size_t i = 0; i < batch; i++) {
      bestla::kernel::wrapper::Memcpy2D::forward1_auto<float, float, BTLA_ELTWISEOP::GELU>(
          src.data() + i * count, ref.data() + i * count, 1, count, 0, 0, nullptr);
    }
    printf("Single Time:%f\n", tm.stop());
    parallel::StdGroupThreading th;
    th.set_threads(device::CpuRange(pcores, 0, false));
    for (size_t i = 0; i < batch; i++) {
      th.parallel_for(parallel::Config{count, 64}, [&](const parallel::ThreadRange& tr) {
        if (tr.size_) {
          bestla::kernel::wrapper::Memcpy2D::forward1_auto<float, float, BTLA_ELTWISEOP::GELU>(
              src.data() + tr.loc_ + i * count, dst.data() + tr.loc_ + i * count, 1, tr.size_, 0, 0, nullptr);
        }
      });
    }
    buffer_error(dst.data(), ref.data(), dst.size());

    tm.start();
    utils::timer_statistics_logger<5000> log;
    while (tm.stop() < 3000) {
      log.start();
      for (size_t i = 0; i < batch; i++) {
        th.parallel_for(parallel::Config{count, 64}, [&](const parallel::ThreadRange& tr) {
          if (tr.size_) {
            bestla::kernel::wrapper::Memcpy2D::forward1_auto<float, float, BTLA_ELTWISEOP::GELU>(
                src.data() + tr.loc_ + i * count, dst.data() + tr.loc_ + i * count, 1, tr.size_, 0, 0, nullptr);
          }
        });
      }
      log.stop();
    }
    log.record();
    printf("MT %s\n", log.get_log_str());

    th.set_threads(device::CpuRange(pcores, ecores, false));
    th.enable_profiling();

    for (size_t i = 0; i < batch; i++) {
      th.parallel_for(parallel::Config{count, 64, 0}, [&](const parallel::ThreadRange& tr) {
        if (tr.size_) {
          bestla::kernel::wrapper::Memcpy2D::forward1_auto<float, float, BTLA_ELTWISEOP::GELU>(
              src.data() + tr.loc_ + i * count, dst.data() + tr.loc_ + i * count, 1, tr.size_, 0, 0, nullptr);
        }
      });
    }
    th.disable_profiling();
    log.clear();
    tm.start();
    th.enable_profiling(1);
    while (tm.stop() < 3000) {
      log.start();
      for (size_t i = 0; i < batch; i++) {
        th.parallel_for(parallel::Config{count, 64, 0}, [&](const parallel::ThreadRange& tr) {
          if (tr.size_) {
            bestla::kernel::wrapper::Memcpy2D::forward1_auto<float, float, BTLA_ELTWISEOP::GELU>(
                src.data() + tr.loc_ + i * count, dst.data() + tr.loc_ + i * count, 1, tr.size_, 0, 0, nullptr);
          }
        });
      }
      log.stop();
    }
    log.record();
    printf("MT %s\n", log.get_log_str());
  }

  void new_thread_group() {
    device::CpuRange crange(8, 4, true);
    crange.print();
    parallel::StdGroupThreading thread;
    thread.enable_profiling();
    thread.set_threads(crange);
    auto wait_func = [&](const parallel::ThreadRange& tr) {
      utils::timer tm;
      int n = 0;
      printf("Thread %d-%d Start\n", tr.info_.group_idx_, tr.info_.thread_idx_);
      tm.start();
      while (tm.stop() < 3000) {
        n++;
      }
      printf("Thread %d-%d End %d\n", tr.info_.group_idx_, tr.info_.thread_idx_, n);
    };
    thread.parallel_for({}, wait_func);
    thread.set_threads(device::CpuRange(1, 8, true));
    thread.parallel_for({}, wait_func);
    thread.set_threads(device::CpuRange(8, 0, true));
    thread.parallel_for({}, wait_func);
  }
};
#ifdef BTLA_UT_PARALLEL
static UT_StdThreadingV2 sUT_StdThreadingV2;
#endif

class UT_Scheduler2D {
 public:
  UT_Scheduler2D() {
    UT_START();
    ut(4096, 4096, 24);
    ut(4096, 4096, 28);
    ut(4096, 4096, 48);
    ut(4096, 4096, 56);
  }

  void ut(int row, int col, int threads) {
    printf("%s %d %d %d\n", __FUNCTION__, row, col, threads);
    parallel::Scheduler2D sch;
    sch.update({threads, row, col, 1, 1, 0, 0});
    sch.print();
    parallel::ThreadProblem2D prb{threads - 1};
    sch.getIndex(prb);
    prb.print();
  }
};
#ifdef BTLA_UT_PARALLEL
static UT_Scheduler2D sUT_Scheduler2D;
#endif

class UT_SchedulerGemmBase {
 public:
  UT_SchedulerGemmBase() {
    UT_START();
    ut<gemm::ICoreRowNAmxint8<64, 16>>(2024, 11008, 4096, 48, 2048 * 1024, 32 * 1024);
    ut<gemm::ICoreRowNAmxint8<48, 16>>(2048, 4096, 4096, 48, 2048 * 1024, 32 * 1024);
    ut<gemm::ICoreRowNAmxint8SS<32, 32>>(2048, 4096, 4096, 24);
    ut<gemm::SCoreRowNAvx512f<48, 8>>(1, 4096, 4096, 24);
    ut<gemm::ICoreRowNAmxint8SS<32, 32>>(4, 4096, 4096, 48);
  }

  template <class GemmCore_T>
  void ut(int m, int n, int k, int threads, size_t l2cache = 0, size_t l1cache = 0) {
    printf("%s %d %d %d %d core:%s\n", __FUNCTION__, m, n, k, threads, gemm::CoreAttr::to_str(GemmCore_T::ID));
    parallel::gemm::SchedulerBase<GemmCore_T> sch;
    GetCPUDevice();
    utils::GemmProblem gp(1, m, n, k);
    sch.update({threads, gp, 0, 0, l2cache == 0 ? _cd->getL2CacheSize() : l2cache,
                l1cache == 0 ? _cd->getL1CacheSize() : l1cache});
    sch.print();
    parallel::gemm::ThreadProblemBase prb{sch.valid_threads() - 1};
    sch.getIndex(prb);
    prb.print();
  }
};
#ifdef BTLA_UT_PARALLEL
static UT_SchedulerGemmBase sUT_SchedulerGemmBase;
#endif

class UT_SchedulerGemmKBlockNew {
 public:
  UT_SchedulerGemmKBlockNew() {
    UT_START();
    ut<gemm::ICoreRowNAmxint8SS<48, 16>>(2011, 32000, 4096, 128, 32);
    ut<gemm::ICoreRowNAmxint8SS<64, 16>>(2048, 4096, 4096, 4096, 48);
    ut<gemm::ICoreRowNAmxint8SS<64, 16>>(2048, 4096, 4096, 128, 24);
    ut<gemm::ICoreRowNAmxint8SS<64, 16>>(2048, 4096, 4096, 4096, 24);
    ut<gemm::SCoreRowNAvx512f<48, 8>>(1, 4096, 4096, 32, 24);
    ut<gemm::SCoreRowNAvx512f<48, 8>>(1, 4096, 4096, 64, 22, 32 * 1024);
    ut<gemm::SCoreRowNAvx512f<48, 8>>(1, 4096, 4096, 128, 24);
    ut<gemm::SCoreRowNAvx512f<48, 8>>(1, 4096, 4096, 1024, 24);
    ut<gemm::ICoreRowNAmxint8SS<64, 16>>(1, 4096, 4096, 64, 24, 32 * 1024);
    ut<gemm::ICoreRowNAmxint8SS<64, 16>>(2048, 4096, 4096, 64, 24);
    ut<gemm::ICoreRowNAmxint8SS<64, 16>>(2048, 4096, 4096, 4096, 56);
    ut<gemm::ICoreRowNAmxint8SS<64, 16>>(2048, 4096, 4096, 64, 56);
    ut<gemm::ICoreRowNAmxint8SS<64, 16>>(4, 4096, 4096, 128, 48);
    ut<gemm::ICoreRowNAmxint8SS<64, 16>>(4, 4096, 3072, 64, 48);
    ut<gemm::ICoreRowNAmxint8SS<64, 16>>(2048, 4096, 3072, 3072, 48);
    ut<gemm::ICoreRowNAmxint8SS<64, 16>>(2048, 4096, 3072, 64, 56);
  }

  template <class GemmCore_T>
  void ut(int m, int n, int k, int kblock, int threads, size_t l1cache = 0) {
    printf("%s %d %d %d %d %d core:%s\n", __FUNCTION__, m, n, k, kblock, threads,
           gemm::CoreAttr::to_str(GemmCore_T::ID));
    parallel::gemm::SchedulerIntKBlock<GemmCore_T> sch;
    GetCPUDevice();
    utils::GemmProblem gp(1, m, n, k, kblock);
    sch.update({threads, gp, 0, 0, _cd->getL2CacheSize(), l1cache == 0 ? _cd->getL1CacheSize() : l1cache});
    sch.print();
    parallel::gemm::ThreadProblemBase prb{sch.valid_threads() - 1};
    sch.getIndex(prb);
    prb.print();
  }
};
#ifdef BTLA_UT_PARALLEL
static UT_SchedulerGemmKBlockNew sUT_SchedulerGemmKBlockNew;
#endif
}  // namespace ut
}  // namespace bestla
