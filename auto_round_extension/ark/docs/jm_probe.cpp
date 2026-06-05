// jm_probe.cpp
// Minimal joint_matrix probe matching ARK IGemmDQCore's int8 M8xN16xK32 DPAS op.
// Compiled to generic SPIR-V (spir64, NO AOT) so success/failure depends purely on
// the GPU driver's runtime ability to JIT-lower the joint_matrix SPIR-V.
//
// PASS  -> prints "JOINT_MATRIX_OK <checksum>"  => driver CAN lower joint_matrix
// FAIL  -> throws "no matrix hardware on the target device, joint_matrix is not supported"
//
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// Same tile shape as ARK IGemmDQCfg: TM=8, TN=16, TK=32, sub_group=16, int8->int32
static constexpr int TM = 8;
static constexpr int TN = 16;
static constexpr int TK = 32;
static constexpr int SG = 16;

int main() {
  try {
    queue q{gpu_selector_v};
    auto dev = q.get_device();
    std::printf("device: %s\n", dev.get_info<info::device::name>().c_str());
    std::printf("driver: %s\n", dev.get_info<info::device::driver_version>().c_str());

    const int M = TM, N = TN, K = TK;
    int8_t *A = malloc_shared<int8_t>(M * K, q);
    int8_t *B = malloc_shared<int8_t>(K * N, q);
    int32_t *C = malloc_shared<int32_t>(M * N, q);
    for (int i = 0; i < M * K; i++) A[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)((i % 5) - 2);
    for (int i = 0; i < M * N; i++) C[i] = 0;

    // one work-group, one sub-group of SG lanes
    range<2> global{1, SG};
    range<2> local{1, SG};

    q.submit([&](handler &h) {
       h.parallel_for(
           nd_range<2>(global, local),
           [=](nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
             auto sg = it.get_sub_group();
             joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> sub_a;
             joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> sub_b;
             joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> sub_c;
             joint_matrix_fill(sg, sub_c, 0);
             joint_matrix_load(sg, sub_a, multi_ptr<int8_t, access::address_space::global_space>(A), K);
             joint_matrix_load(sg, sub_b, multi_ptr<int8_t, access::address_space::global_space>(B), N);
             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
             joint_matrix_store(sg, sub_c, multi_ptr<int32_t, access::address_space::global_space>(C), N,
                                layout::row_major);
           });
     }).wait();

    long checksum = 0;
    for (int i = 0; i < M * N; i++) checksum += C[i];
    std::printf("JOINT_MATRIX_OK %ld\n", checksum);
    free(A, q); free(B, q); free(C, q);
    return 0;
  } catch (sycl::exception const &e) {
    std::printf("JOINT_MATRIX_FAIL sycl::exception: %s\n", e.what());
    return 2;
  } catch (std::exception const &e) {
    std::printf("JOINT_MATRIX_FAIL std::exception: %s\n", e.what());
    return 3;
  }
}
