include(CheckCXXSourceCompiles)

# "avx2", "fma", "f16c"
set(AVX2_CODE "
    #include <immintrin.h>
    int main()
    {
      __m128i raw_data = _mm_set1_epi32(1); // sse2
      __m256i ymm0 = _mm256_cvtepu8_epi16(raw_data); // avx2
      __m256 ymm1 = _mm256_castsi256_ps(ymm0);
      ymm1 = _mm256_fmadd_ps(ymm1, ymm1, ymm1); // fma
      __m128i xmm = _mm256_cvtps_ph(ymm1, _MM_FROUND_TO_NEAREST_INT); // f16c
      return 0;
    }
")

set(AVX2_FLAGS "-mavx2 -mfma -mf16c")

# "avx512f", "avx512bw", "avx512vl", "avx512dq"
set(AVX512_CODE "
    #include <immintrin.h>
    int main()
    {
        __m512i a = _mm512_set1_epi32(1); // avx512f
        a = _mm512_abs_epi16(a); // avx512bw
        __m512 b = _mm512_castsi512_ps(a);
        b = _mm512_and_ps(b, b); // avx512dq
        __m256 c = _mm512_castps512_ps256(b);
        __m256i d = _mm256_castps_si256(c);
        d = _mm256_abs_epi64(d);
        return 0;
    }
")

set(AVX512_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq")

# "avx512_vnni"
set(AVX512_VNNI_CODE "
    #include <immintrin.h>
    int main()
    {
        __m512i a = _mm512_set1_epi32(1); // avx512f
        a = _mm512_dpbusds_epi32(a, a, a); // avx512_vnni
        return 0;
    }
")
set(AVX512_VNNI_FLAGS "-mavx512f -mavx512vnni")

# "avx512_vnni"
set(AVX_VNNI_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256i a = _mm256_set1_epi32(1); // avx
        a = _mm256_dpbusd_avx_epi32(a, a, a); // avx_vnni
        return 0;
    }
")
set(AVX_VNNI_FLAGS "-mavx -mavxvnni")

# "amx_tile", "amx_bf16"
set(AMX_BF16_CODE "
    #include <immintrin.h>
    int main()
    {
        _tile_dpbf16ps(0, 1, 2); // amx_bf16
        return 0;
    }
")
set(AMX_BF16_FLAGS " ")

# "amx_tile", "amx_int8"
set(AMX_INT8_CODE "
    #include <immintrin.h>
    int main()
    {
        _tile_dpbusd(0, 1, 2); // amx_bf16
        return 0;
    }
")
set(AMX_INT8_FLAGS " ")

# "amx_tile", "amx_fp16"
set(AMX_FP16_CODE "
    #include <immintrin.h>
    int main()
    {
        _tile_dpfp16ps(0, 1, 2); // amx_bf16
        return 0;
    }
")
set(AMX_FP16_FLAGS " ")

# "avx512_fp16"
set(AVX512_FP16_CODE "
    #include <immintrin.h>
    int main()
    {
        char tmp[512];
        __m256h a = _mm256_loadu_ph(tmp); // avx512vl + avx512fp16
        __m512 b = _mm512_cvtxph_ps(a); // avx512fp16
        a = _mm512_cvtxps_ph(b); // avx512fp16
        _mm256_storeu_ph(tmp,a);
        return 0;
    }
")
set(AVX512_FP16_FLAGS "-mavx512f -mavx512vl -mavx512fp16")

# "avx512_bf16"
set(AVX512_BF16_CODE "
    #include <immintrin.h>
    int main()
    {
        __m512 a = _mm512_set1_ps(1.f); // avx512f
        __m256bh b = _mm512_cvtneps_pbh(a); // avx512_bf16 + AVX512F
        return 0;
    }
")
set(AVX512_BF16_FLAGS "-mavx512f -mavx512bf16")

macro(check_isa type)
    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    if (NOT ${type}_FOUND)
        set(CMAKE_REQUIRED_FLAGS ${${type}_FLAGS})
        check_cxx_source_compiles("${${type}_CODE}" HAS_${type})
        if (HAS_${type})
            set(${type}_FOUND TRUE CACHE BOOL "${type} support")
            set(${type}_FLAGS "${${type}_FLAGS}" CACHE STRING "${type} flags")
        endif()
    endif()
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    if (NOT ${type}_FOUND)
        set(${type}_FOUND FALSE CACHE BOOL "${type} support")
        set(${type}_FLAGS "" CACHE STRING "${type} flags")
    endif()

    mark_as_advanced(${type}_FOUND ${type}_FLAGS)
endmacro()

set(ISA_SET AVX2 AVX512 AVX512_VNNI AVX_VNNI AVX512_BF16 AVX512_FP16 AMX_BF16 AMX_INT8 AMX_FP16)
foreach (ISA ${ISA_SET})
  check_isa(${ISA})
endforeach()


