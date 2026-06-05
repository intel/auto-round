// What architecture does the runtime ACTUALLY resolve for b70's device?
// This is the value extOneapiArchitectureIs() compares against the supported list.
#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;
namespace syclex = sycl::ext::oneapi::experimental;
int main() {
  queue q;
  auto d = q.get_device();
  std::cout << "name: " << d.get_info<info::device::name>() << "\n";
  try {
    auto a = d.get_info<syclex::info::device::architecture>();
    std::cout << "architecture enum int = " << static_cast<long long>(a)
              << " (hex 0x" << std::hex << static_cast<long long>(a) << std::dec << ")\n";
  } catch (const sycl::exception& e) {
    std::cout << "architecture query THREW: " << e.what() << "\n";
  }
  // direct arch comparisons
  auto cmp=[&](const char* n, syclex::architecture a){
    try { std::cout << "  is " << n << " ? " << (d.ext_oneapi_architecture_is(a)?"YES":"no") << "\n"; }
    catch(const sycl::exception&e){ std::cout << "  is " << n << " ? THREW: "<<e.what()<<"\n"; }
  };
  cmp("intel_gpu_bmg_g21", syclex::architecture::intel_gpu_bmg_g21);
  cmp("intel_gpu_bmg_g31", syclex::architecture::intel_gpu_bmg_g31);
  std::cout << "has(ext_intel_matrix) = " << (d.has(aspect::ext_intel_matrix)?"YES":"NO") << "\n";
  return 0;
}
