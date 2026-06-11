#pragma once

#include <cstdint>

namespace oneapi::mkl::rng::device {

template <typename T>
struct uniform {
  T min_;
  T max_;

  constexpr uniform(T min_value, T max_value) : min_(min_value), max_(max_value) {}
};

template <int VecSize = 4>
struct philox4x32x10 {
  std::uint64_t seed_;

  template <typename Counter>
  constexpr philox4x32x10(std::uint64_t seed, Counter const&) : seed_(seed) {}
};

template <typename T, int VecSize>
constexpr T generate(uniform<T> const& distribution, philox4x32x10<VecSize>&) {
  return distribution.min_;
}

}  // namespace oneapi::mkl::rng::device
