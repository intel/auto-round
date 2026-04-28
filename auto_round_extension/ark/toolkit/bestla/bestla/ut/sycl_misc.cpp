#include "bestla_ut.h"
#include "bestla_prologue_b.h"
#include "sycl_ut.h"
#include "sycl/sycl_device.h"
#include "sycl/sycl_utils.h"
#include "sycl/sycl_storage.h"
#include "sycl/sycl_gemm.h"
#include "sycl/sycl_prologue_b.h"
namespace bestla {
using namespace ut;
using namespace utils;
namespace sycl_ut {
class UT_SyclDevice {
 public:
  UT_SyclDevice() {
    UT_START();
    auto dev = UT_Device::get();
    dev->print();
  }
};
// static UT_SyclDevice sUT_SyclDevice;

class UT_SyclVector {
 public:
  UT_SyclVector() {
    UT_START();
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    auto svec = sycl_utils::sycl_vector<float>(1000, q);
    utils::avector<float> hsrc(1000);
    ut::fill_buffer_randn(hsrc.data(), hsrc.size(), -0.5f, 0.5f);
    q->memcpy(svec.data(), hsrc.data(), hsrc.size() * 4).wait();
    auto hdst = sycl_utils::sycl2host(svec.data(), svec.size(), q);
    ut::buffer_error(hsrc.data(), hdst.data(), hsrc.size(), 0.f);
  }
};
// static UT_SyclVector sUT_SyclVector;

}  // namespace sycl_ut
}  // namespace bestla
