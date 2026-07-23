cmake_minimum_required(VERSION 3.5...3.31)
project(ark LANGUAGES CXX)

option(ARK_XPU "Build XPU kernels" OFF)
option(ARK_UT "Build XPU kernels UT" OFF)
option(ARK_RESCALE "Experimental" OFF)
option(ARK_SIMULATOR "Build only Sage FP8 kernels for the XPU simulator" OFF)

# ARK_SYCL_TLA defaults to ON when ARK_XPU is enabled
if(ARK_XPU AND NOT DEFINED ARK_SYCL_TLA)
  set(ARK_SYCL_TLA ON CACHE BOOL "Build SYCL TLA (auto-enabled with ARK_XPU)")
else()
  option(ARK_SYCL_TLA "Build SYCL TLA" OFF)
endif()

if(ARK_SIMULATOR AND (NOT ARK_XPU OR NOT ARK_SYCL_TLA))
  message(FATAL_ERROR "ARK_SIMULATOR requires ARK_XPU=ON and ARK_SYCL_TLA=ON")
endif()

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.1
)
FetchContent_MakeAvailable(pybind11)

set(libs)

if(ARK_SIMULATOR)
  message(STATUS "ARK simulator build: Sage FP8 only; CPU, BesTLA, and oneDNN are disabled")
else()
  include(FindOpenMP)
  set(BTLA_ENABLE_OPENMP ON CACHE BOOL "BesTLA enable compiling OpenMP threading")
endif()

if(ARK_XPU)
    set(BTLA_SYCL ON CACHE BOOL "BesTLA with SYCL")
    set(PY_NAME auto_round_kernel_xpu)
    set(ARK_TYPE ARK_XPU)
    find_package(IntelSYCL REQUIRED)
    list(APPEND libs IntelSYCL::SYCL_CXX)
else()
    set(BTLA_SYCL OFF CACHE BOOL "BesTLA without SYCL")
    set(PY_NAME auto_round_kernel_cpu)
    set(ARK_TYPE ARK_CPU)
endif()

if(NOT ARK_SIMULATOR)
  add_subdirectory(bestla)
  if(MSVC)
      target_compile_options(bestla INTERFACE /w)
  else()
      target_compile_options(bestla INTERFACE -w)
  endif()
  list(APPEND libs bestla)
endif()

if(ARK_SIMULATOR)
  set(SRCS ark_simulator.cpp sdpa_fp8.cpp)
else()
  file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
  list(FILTER SRCS EXCLUDE REGEX "/(ark_simulator|sdpa_fp8)\\.cpp$")
endif()
set(SDPA_GENERATED_SRCS)
set(SDPA_KERNEL_DECLARATIONS)

include_directories(wrapper/include)

# Build flash_attn_wrapper as a separate static library with sycl-tla flags
if(ARK_XPU AND ARK_SYCL_TLA)
  # Fetch sycl-tla headers (header-only, don't build)
  set(SYCL_TLA_GIT_REPOSITORY
      "https://github.com/intel-innersource/libraries.ai.cutlass.internal.git"
      CACHE STRING "sycl-tla git repository")
  set(SYCL_TLA_GIT_TAG
      "bf8db07f06f8b82b4d4ecd4122df36695e6e0a32"
      CACHE STRING "sycl-tla git tag/commit")

  FetchContent_Declare(
    sycl_tla_latest
    GIT_REPOSITORY ${SYCL_TLA_GIT_REPOSITORY}
    GIT_TAG ${SYCL_TLA_GIT_TAG}
  )

  # Map target to device name for -Xs flag
  set(DPCPP_SYCL_TARGET "intel_gpu_bmg_g21" CACHE STRING "SYCL target (intel_gpu_pvc, intel_gpu_bmg_g21)")
  if(DPCPP_SYCL_TARGET STREQUAL "intel_gpu_bmg_g21" OR DPCPP_SYCL_TARGET STREQUAL "bmg")
    set(SYCL_DEVICE_NAME "bmg_g21")
  elseif(DPCPP_SYCL_TARGET STREQUAL "intel_gpu_pvc" OR DPCPP_SYCL_TARGET STREQUAL "pvc")
    set(SYCL_DEVICE_NAME "pvc")
  else()
    set(SYCL_DEVICE_NAME "${DPCPP_SYCL_TARGET}")
  endif()

  # Link flags for sycl-tla (applied to final module)
  set(SYCL_TLA_LINK_FLAGS
    -fsycl
    -fsycl-targets=spir64
    "-Xs" "-device ${SYCL_DEVICE_NAME}"
    -Xspirv-translator
    "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")

  set(CUTLASS_ENABLE_SYCL ON)
  set(CUTLASS_SYCL_PROFILING_ENABLED ON)
  if(ARK_SIMULATOR)
    set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Use sycl-tla headers without building tools or oneMKL" FORCE)
  endif()
  set(CUTLASS_ENABLE_BENCHMARKS OFF)
  set(CUTLASS_ENABLE_EXAMPLES OFF)
  set(CUTLASS_ENABLE_TESTS OFF)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(CUTLASS_ENABLE_LIBRARY OFF)
  set(CUTLASS_ENABLE_TOOLS OFF)
  set(CUTLASS_ENABLE_GDC_FOR_SM100_DEFAULT
      OFF
      CACHE BOOL "DISABLE CUDA")
  FetchContent_MakeAvailable(sycl_tla_latest)

  set(_sycl_tla_include_dirs
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/applications
    ${CUTLASS_DIR}/tools/util/include
    ${CUTLASS_DIR}/examples/common
    ${CUTLASS_DIR}/examples/06_bmg_flash_attention
    ${CUTLASS_DIR}/examples/12_xe20_moe_gemm_cute_interface
  )
  include(${CMAKE_CURRENT_LIST_DIR}/sdpa_generation.cmake)
endif()

pybind11_add_module(${PY_NAME} ${SRCS} ${HEADERS} ${SDPA_GENERATED_SRCS})
target_compile_features(${PY_NAME} PRIVATE cxx_std_17)
target_compile_definitions(${PY_NAME} PRIVATE PY_NAME=${PY_NAME} ${ARK_TYPE}=1)
if(ARK_SIMULATOR)
  target_compile_definitions(${PY_NAME} PRIVATE ARK_SIMULATOR=1)
  target_include_directories(${PY_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bestla)
endif()
if(ARK_RESCALE)
    target_compile_definitions(${PY_NAME} PRIVATE ARK_RESCALE=1)
endif()
if(ARK_XPU AND ARK_SYCL_TLA)
  target_compile_definitions(${PY_NAME} PRIVATE ARK_SYCL_TLA=1 CUTLASS_ENABLE_SYCL=1 SYCL_INTEL_TARGET=1)
  target_include_directories(${PY_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/include)
  target_include_directories(${PY_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated/sdpa)
  # Use SYSTEM include directories for sycl-tla to suppress warnings/errors from third-party headers
  # (e.g., std::common_type specialization issues in traits.hpp)
  foreach(_inc_dir IN LISTS _sycl_tla_include_dirs)
    if(EXISTS "${_inc_dir}")
      target_include_directories(${PY_NAME} SYSTEM PRIVATE "${_inc_dir}")
    endif()
  endforeach()
  target_compile_options(${PY_NAME} PRIVATE -fsycl -fno-sycl-instrument-device-code)
  target_link_options(${PY_NAME} PRIVATE ${SYCL_TLA_LINK_FLAGS})
endif()
target_link_libraries(${PY_NAME} PRIVATE ${libs})

if(ARK_UT)
    set(TEST_NAME test_${ARK_TYPE})
    set(TEST_SRCS wrapper/test/test_main.cpp)
    if(ARK_XPU AND ARK_SYCL_TLA)
      list(APPEND TEST_SRCS sdpa.cpp)
      list(APPEND TEST_SRCS ${SDPA_GENERATED_SRCS})
    endif()
    add_executable(${TEST_NAME} ${TEST_SRCS})
    target_compile_features(${TEST_NAME} PRIVATE cxx_std_17)
    target_compile_definitions(${TEST_NAME} PRIVATE ${ARK_TYPE}=1)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/include)
    if(ARK_XPU AND ARK_SYCL_TLA)
      target_compile_definitions(${TEST_NAME} PRIVATE ARK_SYCL_TLA=1 CUTLASS_ENABLE_SYCL=1 SYCL_INTEL_TARGET=1)
      target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated/sdpa)
      foreach(_inc_dir IN LISTS _sycl_tla_include_dirs)
        if(EXISTS "${_inc_dir}")
          target_include_directories(${TEST_NAME} SYSTEM PRIVATE "${_inc_dir}")
        endif()
      endforeach()
      target_compile_options(${TEST_NAME} PRIVATE -fsycl -fno-sycl-instrument-device-code)
      target_link_options(${TEST_NAME} PRIVATE ${SYCL_TLA_LINK_FLAGS})
    endif()
    target_link_libraries(${TEST_NAME} PRIVATE ${libs})
endif()
