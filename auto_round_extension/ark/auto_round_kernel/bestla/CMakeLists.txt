cmake_minimum_required(VERSION 3.12)
project(bestla LANGUAGES CXX VERSION 0.1.0)

if(BTLA_SYCL)
  include(cmake/sycl.cmake)
endif()
include(cmake/FindSIMD.cmake)

file(GLOB headers ${PROJECT_NAME}/*.h ${PROJECT_NAME}/*.hpp)

option(BTLA_ENABLE_OPENMP "Compile OpenMP thread pool if OMP can be found" ON)
option(BTLA_SYCL "Compile OpenMP thread pool if OMP can be found" OFF)

option(BTLA_UT_ALL "Enable all unit tests" OFF)
option(BTLA_UT_DEBUG "Enable debug unit tests" OFF)
option(BTLA_UT_EPILOGUE "Enable unit test for epilogue" OFF)
option(BTLA_UT_PROLOGUE_A "Enable unit test for activation prologue" OFF)
option(BTLA_UT_PROLOGUE_B "Enable unit test for weight prologue" OFF)
option(BTLA_UT_GEMM "Enable unit test for micro gemm kernels" OFF)
option(BTLA_UT_WRAPPER "Enable unit test for parallel gemms" OFF)
option(BTLA_UT_PARALLEL "Enable unit test for parallel set" OFF)
option(BTLA_UT_KERNEL_JIT "Enable unit test for jit kernels" OFF)
option(BTLA_UT_KERNEL_INTRIN "Enable unit test for intrinsic kernels" OFF)
option(BTLA_UT_KERNEL_WRAPPER "Enable unit test for runtime ISA kernels" OFF)
option(BTLA_UT_NOASAN "Disable sanitize" OFF)
option(BTLA_UT_BENCHMARK "Benchmark ON may take a long time to finish all tests" OFF)
option(BTLA_UT_OPENMP "Use OpenMP for UT tests" ON)

include(FetchContent)
FetchContent_Declare(
    xbyak
    GIT_REPOSITORY https://github.com/herumi/xbyak.git
    GIT_TAG v7.06
)
FetchContent_MakeAvailable(xbyak)

add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE xbyak)
add_library(neural_speed::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(
	${PROJECT_NAME} INTERFACE
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
	"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

function(add_isa_def ARG)
	if(${${ARG}_FOUND})
    target_compile_definitions(${PROJECT_NAME} INTERFACE BTLA_${ARG}_FOUND=1)
  else()
    target_compile_definitions(${PROJECT_NAME} INTERFACE BTLA_${ARG}_FOUND=0)
	endif()
endfunction()

foreach (ISA ${ISA_SET})
  add_isa_def(${ISA})
endforeach()

set(sycl_headers)
set(sycl_libs)
if(BTLA_SYCL)
  set(BTLA_UT_NOASAN ON) # hang issue with santizer
  file(GLOB sycl_headers ${PROJECT_NAME}/sycl/*.h ${PROJECT_NAME}/sycl/*.hpp)
  target_compile_definitions(${PROJECT_NAME} INTERFACE BTLA_SYCL)
  list(APPEND sycl_libs IntelSYCL::SYCL_CXX)
  target_compile_options(${PROJECT_NAME} INTERFACE -march=native)
  target_link_libraries(${PROJECT_NAME} INTERFACE ${sycl_libs})
  # add_link_options(-fsycl-targets=spir64 -Xsycl-target-backend "-options -ze-opt-large-register-file")
endif(BTLA_SYCL)

if(BTLA_ENABLE_OPENMP)
  include(FindOpenMP)
  message(STATUS "BesTLA enable OpenMP ThreadPool")
  target_compile_definitions(${PROJECT_NAME} INTERFACE BTLA_USE_OPENMP)
  target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX)
endif(BTLA_ENABLE_OPENMP)

if(WIN32)
	target_compile_definitions(${PROJECT_NAME} INTERFACE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif(WIN32)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

if(BTLA_UT_ALL)
  set(BTLA_UT_EPILOGUE ON)
  set(BTLA_UT_PROLOGUE_A ON)
  set(BTLA_UT_PROLOGUE_B ON)
  set(BTLA_UT_GEMM ON)
  set(BTLA_UT_WRAPPER ON)
  set(BTLA_UT_PARALLEL ON)
  set(BTLA_UT_KERNEL_JIT ON)
  set(BTLA_UT_KERNEL_INTRIN ON)
  set(BTLA_UT_KERNEL_WRAPPER ON)
endif(BTLA_UT_ALL)

set(UT_BUILD FALSE)
if(BTLA_UT_DEBUG OR BTLA_UT_PROLOGUE_A OR BTLA_UT_PROLOGUE_B OR BTLA_UT_EPILOGUE OR BTLA_UT_GEMM
OR BTLA_UT_WRAPPER OR BTLA_UT_PARALLEL OR BTLA_UT_KERNEL_JIT OR BTLA_UT_KERNEL_INTRIN
OR BTLA_UT_KERNEL_WRAPPER)
  set(UT_BUILD TRUE)
endif()

function(add_ut_flag UT_OPTION)
	if(${${UT_OPTION}})
	  # target_compile_definitions(${PROJECT_NAME}_ut PRIVATE ${UT_OPTION})
    add_compile_definitions(${UT_OPTION})
	endif()
endfunction()

set(benchmark_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/ut/bestla_benchmark.cpp)
list(APPEND benchmark_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/ut/sycl_benchmark.cpp)
# Flash attention benchmarks are in separate files to avoid header conflicts
#list(APPEND benchmark_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/ut/sycl_tla_flash_attn_prefill_bench.cpp)
#list(APPEND benchmark_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/ut/sycl_tla_flash_attn_decode_bench.cpp)
# MOE GEMM benchmark
#list(APPEND benchmark_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/ut/sycl_tla_moe_bench.cpp)


if(UT_BUILD)
	file(GLOB srcs ${PROJECT_NAME}/ut/*.cc ${PROJECT_NAME}/ut/*.cpp) #compile everything even run parts of UTs
  file(GLOB sycl_srcs ${PROJECT_NAME}/ut/sycl*)
  if(NOT BTLA_SYCL)
    list(REMOVE_ITEM srcs ${sycl_srcs})
  endif()
  list(REMOVE_ITEM srcs ${benchmark_srcs})
	file(GLOB ut_headers ${PROJECT_NAME}/ut/*.h)
  include_directories(${PROJECT_NAME})
	add_executable(${PROJECT_NAME}_ut ${srcs} ${headers} ${sycl_headers} ${ut_headers})
  target_compile_options(${PROJECT_NAME}_ut PRIVATE -w)
  if(BTLA_UT_OPENMP)
    target_compile_definitions(${PROJECT_NAME} INTERFACE BTLA_USE_OPENMP)
    target_link_libraries(${PROJECT_NAME}_ut PRIVATE OpenMP::OpenMP_CXX)
  endif()
	if(NOT WIN32)
		if(NOT BTLA_UT_NOASAN)
		  target_compile_options(${PROJECT_NAME}_ut PRIVATE -fsanitize=address)
		  target_link_options(${PROJECT_NAME}_ut PRIVATE -fsanitize=address)
		endif()
		target_link_options(${PROJECT_NAME}_ut PRIVATE -lpthread)
  else()
    target_link_options(${PROJECT_NAME}_ut PUBLIC /STACK:5242880)
	endif()

	add_ut_flag(BTLA_UT_DEBUG)
	add_ut_flag(BTLA_UT_EPILOGUE)
	add_ut_flag(BTLA_UT_PROLOGUE_A)
	add_ut_flag(BTLA_UT_PROLOGUE_B)
	add_ut_flag(BTLA_UT_GEMM)
	add_ut_flag(BTLA_UT_PARALLEL)
	add_ut_flag(BTLA_UT_WRAPPER)
	add_ut_flag(BTLA_UT_KERNEL_INTRIN)
	add_ut_flag(BTLA_UT_KERNEL_JIT)
	add_ut_flag(BTLA_UT_KERNEL_WRAPPER)
  if(BTLA_SYCL)
    # add_compile_definitions(BTLA_UT_SYCL)
  endif()
	target_link_libraries(${PROJECT_NAME}_ut PRIVATE ${PROJECT_NAME} ${sycl_libs})
endif(UT_BUILD)

if(BTLA_UT_BENCHMARK)
  file(GLOB ut_headers ${PROJECT_NAME}/ut/*.h)
  include_directories(${PROJECT_NAME})
  if(NOT BTLA_SYCL)
    list(REMOVE_ITEM benchmark_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/ut/sycl_benchmark.cpp)
  endif()
	add_executable(${PROJECT_NAME}_benchmark ${benchmark_srcs} ${headers} ${ut_headers})
  if(BTLA_UT_OPENMP)
    include(FindOpenMP)
    target_compile_definitions(${PROJECT_NAME} INTERFACE BTLA_USE_OPENMP)
    target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(NOT WIN32)
		target_link_options(${PROJECT_NAME}_benchmark PRIVATE -lpthread)
  else()
    target_link_options(${PROJECT_NAME}_benchmark PUBLIC /STACK:5242880)
	endif()
  target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE ${PROJECT_NAME} ${sycl_libs} dnnl)
  target_compile_options(${PROJECT_NAME}_benchmark PRIVATE -w)
  # Add SYCL target for Intel GPU with XMX/2D block IO support (required for sycl-tla flash attention)
  if(BTLA_SYCL AND ARK_SYCL_TLA)
    # Header-only consumption of sycl-tla (do NOT build sycl-tla as a subproject).
    set(SYCL_TLA_GIT_REPOSITORY "https://github.com/intel/sycl-tla.git" CACHE STRING "sycl-tla git repository")
    set(SYCL_TLA_GIT_TAG "main" CACHE STRING "sycl-tla git tag/commit")

    FetchContent_Declare(
      sycl_tla
      GIT_REPOSITORY ${SYCL_TLA_GIT_REPOSITORY}
      GIT_TAG ${SYCL_TLA_GIT_TAG}
    )
    FetchContent_GetProperties(sycl_tla)
    if(NOT sycl_tla_POPULATED)
      FetchContent_Populate(sycl_tla)
    endif()

    set(_sycl_tla_include_dirs
      ${sycl_tla_SOURCE_DIR}/include
      ${sycl_tla_SOURCE_DIR}/applications
      ${sycl_tla_SOURCE_DIR}/tools/util/include
      ${sycl_tla_SOURCE_DIR}/examples/common
      ${sycl_tla_SOURCE_DIR}/examples/06_bmg_flash_attention
      ${sycl_tla_SOURCE_DIR}/examples/12_xe20_moe_gemm_cute_interface
    )
    foreach(_inc_dir IN LISTS _sycl_tla_include_dirs)
      if(EXISTS "${_inc_dir}")
        target_include_directories(${PROJECT_NAME}_benchmark PRIVATE "${_inc_dir}")
      endif()
    endforeach()

    # AOT compile target for Intel GPUs
    # Use intel_gpu_pvc for Data Center GPU Max series, or intel_gpu_bmg_g21 for Battlemage
    set(DPCPP_SYCL_TARGET "intel_gpu_bmg_g21" CACHE STRING "SYCL target (intel_gpu_pvc, intel_gpu_bmg_g21)")
    
    # Map target to device name for -Xs flag
    if(DPCPP_SYCL_TARGET STREQUAL "intel_gpu_bmg_g21" OR DPCPP_SYCL_TARGET STREQUAL "bmg")
      set(SYCL_DEVICE_NAME "bmg_g21")
    elseif(DPCPP_SYCL_TARGET STREQUAL "intel_gpu_pvc" OR DPCPP_SYCL_TARGET STREQUAL "pvc")
      set(SYCL_DEVICE_NAME "pvc")
    else()
      set(SYCL_DEVICE_NAME "${DPCPP_SYCL_TARGET}")
    endif()
    
    target_compile_definitions(${PROJECT_NAME}_benchmark PRIVATE ARK_SYCL_TLA=1 CUTLASS_ENABLE_SYCL=1 SYCL_INTEL_TARGET=1)
    # Compile flags (no AOT, JIT at runtime)
    target_compile_options(${PROJECT_NAME}_benchmark PRIVATE 
      -fsycl
      -fno-sycl-instrument-device-code)
    # Link flags: use spir64 (JIT) with device hint and enable required SPIR-V extensions
    target_link_options(${PROJECT_NAME}_benchmark PRIVATE 
      -fsycl
      -fsycl-targets=spir64
      "-Xs" "-device ${SYCL_DEVICE_NAME}"
      -Xspirv-translator
      "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")
  endif()
endif(BTLA_UT_BENCHMARK)
