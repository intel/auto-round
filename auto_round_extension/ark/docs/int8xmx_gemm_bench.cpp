// Standalone throughput benchmark for bestla::sycl_gemm::xmx::IKblockGemmDQCore
// (the fused int8-XMX GEMM core used by the ARK XPU woq_s4 / woq_s2 m>1 prefill path).
//
// This isolates ONLY the s8 x s8 -> s32 joint_matrix GEMM (+ per-k-block accumulator
// dequant). Inputs (int8 weight B, int8 activation A, fp scales) are pre-materialized on
// device, exactly as the real dispatcher hands them to the kernel after unpackq +
// sycl_dyn_quant_s8. So this measures the XMX compute ceiling of the prefill path WITHOUT
// the int4->int8 unpack pass or the activation-quant pass (those are separate kernels in
// the two-pass design; see docs/ark_xpu_int4_prefill_path.md).
//
// It does NOT check correctness -- the bestla UT (UT_SyclInt4Dequant / sycl_gemm.cpp)
// already does that. This is a pure best-of-N throughput probe for the roofline.
//
// Dispatcher call mirrored (wrapper/include/dnnl_wrapper.hpp:160):
//   Launcher<xmx::IKblockGemmDQCfg<T>, xmx::IKblockGemmDQCore>::run(
//       q, {a, b, c, m, n, k, /*lda*/k, /*ldb*/k, /*ldc*/n, bias, scaleA, scaleB, blocksize});
//
// Build: see build_int8xmx_bench.sh (icpx -fsycl). Run on the BMG GPU:
//   ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./int8xmx_gemm_bench
//   ROOFLINE_TOPS=197 ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./int8xmx_gemm_bench   # B60 anchor
//   ./int8xmx_gemm_bench 512 4096 11008 128                                        # single shape

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Include order mirrors bestla/bestla/sycl/sycl_wrapper.h: sycl_gemm.h references
// sycl_utils::Helper_t, so sycl_utils.h must precede it.
#include "bestla/bestla_utils.h"
#include "sycl/sycl_utils.h"
#include "sycl/sycl_device.h"
#include "sycl/sycl_gemm.h"

using bestla::sycl_gemm::Launcher;
namespace xmx = bestla::sycl_gemm::xmx;

// The kernel's matrix element type for A/B is always int8 (Cfg::DT); DQT (= T) is the
// fp output / scale type. ARK's woq path drives this with half activations, so default T=half.
using T = sycl::half;
using Cfg = xmx::IKblockGemmDQCfg<T>;

struct Shape {
  const char* name;
  int m, n, k, blocksize;
};

// Best-of-N kernel-only time (ms) via SYCL event profiling. Returns {min_ms, mean_ms}.
static std::pair<double, double> time_kernel(sycl::queue* q, const Cfg::Param& param, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    Launcher<Cfg, xmx::IKblockGemmDQCore>::run(q, param).wait();
  }
  double min_ms = 1e30, sum_ms = 0.0;
  for (int i = 0; i < iters; ++i) {
    auto e = Launcher<Cfg, xmx::IKblockGemmDQCore>::run(q, param);
    e.wait();
    auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double ms = double(t1 - t0) / 1e6;
    min_ms = std::min(min_ms, ms);
    sum_ms += ms;
  }
  return {min_ms, sum_ms / iters};
}

static void bench_one(sycl::queue* q, const Shape& s, double roofline_tops, int warmup, int iters) {
  const int m = s.m, n = s.n, k = s.k, blk = s.blocksize;
  const int blks = k / blk;  // scaleB is laid out [n][blks]

  // Device buffers, sized exactly as the kernel's input contract.
  auto* A = sycl::malloc_device<int8_t>(size_t(m) * k, *q);      // [m][k] row-major, lda=k
  auto* B = sycl::malloc_device<int8_t>(size_t(n) * k, *q);      // [n][k] (transposed weight), ldb=k
  auto* C = sycl::malloc_device<T>(size_t(m) * n, *q);           // [m][n] row-major, ldc=n
  auto* scaleA = sycl::malloc_device<T>(size_t(m), *q);          // per-row act scale
  auto* scaleB = sycl::malloc_device<T>(size_t(n) * blks, *q);   // per-(n,block) weight scale

  // Values don't affect timing; fill with something finite. memset the int8 operands,
  // splat the fp scales via a trivial kernel (works for any T).
  q->memset(A, 1, size_t(m) * k);
  q->memset(B, 1, size_t(n) * k);
  q->memset(C, 0, size_t(m) * n * sizeof(T));
  q->parallel_for(size_t(m), [=](sycl::id<1> i) { scaleA[i] = T(0.01f); });
  q->parallel_for(size_t(n) * blks, [=](sycl::id<1> i) { scaleB[i] = T(0.02f); });
  q->wait();

  // Param order == GemmParam{A,B,C,m,n,k,lda,ldb,ldc,Bias} + IGemmDQParam{scaleA,scaleB} + {blocksize}.
  Cfg::Param param{(void*)A, (void*)B, (void*)C, m, n, k, /*lda*/ k, /*ldb*/ k, /*ldc*/ n,
                   /*Bias*/ nullptr, (void*)scaleA, (void*)scaleB, blk};

  auto [min_ms, mean_ms] = time_kernel(q, param, warmup, iters);

  const double flop = 2.0 * double(m) * n * k;            // MAC = mul+add
  const double tops = flop / (min_ms * 1e-3) / 1e12;      // best-of-N -> roofline
  const double pct = tops / roofline_tops * 100.0;

  std::printf("[i8xmx_gemm] %-22s m=%-4d n=%-5d k=%-5d blk=%-3d  min=%.4f ms  mean=%.4f ms  "
              "TOPS=%6.1f  %%peak=%5.1f%%\n",
              s.name, m, n, k, blk, min_ms, mean_ms, tops, pct);

  sycl::free(A, *q);
  sycl::free(B, *q);
  sycl::free(C, *q);
  sycl::free(scaleA, *q);
  sycl::free(scaleB, *q);
}

int main(int argc, char** argv) {
  // Hardware-queried roofline for THIS node: 20 Xe-cores x 2048 INT8 MAC x 2 x 2.4 GHz =
  // 196.6 TOPS. Device reports name "Arc Pro B60" but arch intel_gpu_bmg_g21 / PCI 8086:e211.
  double roofline_tops = 197.0;
  if (const char* e = std::getenv("ROOFLINE_TOPS")) roofline_tops = std::atof(e);
  const int warmup = 5, iters = 50;

  bestla::sycl_device::SyclDevice dev(true);  // profiling queue
  auto* q = dev.getQueue();
  std::printf("Device: %s   roofline=%.0f INT8 TOPS  (override via ROOFLINE_TOPS)\n",
              dev.getName().c_str(), roofline_tops);

  std::vector<Shape> shapes;
  if (argc >= 5) {
    static char nm[64];
    std::snprintf(nm, sizeof(nm), "custom");
    shapes.push_back({nm, std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4])});
  } else {
    // Mirror the gemm shapes in ark_xpu_build_bkc.md, plus the m=1024 crossover probe.
    shapes = {
        {"m32_n4096_k4096", 32, 4096, 4096, 128},
        {"m128_n4096_k4096", 128, 4096, 4096, 128},
        {"m512_n4096_k11008", 512, 4096, 11008, 128},
        {"m1024_n4096_k11008", 1024, 4096, 11008, 128},
    };
  }

  for (const auto& s : shapes) bench_one(q, s, roofline_tops, warmup, iters);
  return 0;
}
