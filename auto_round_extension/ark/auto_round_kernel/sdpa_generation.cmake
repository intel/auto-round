file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated/sdpa)
set(SDPA_DECLARATIONS_FILE ${CMAKE_CURRENT_BINARY_DIR}/generated/sdpa/sdpa_kernel_declarations.hpp)

function(generate_sdpa_instantiation mode dtype_name dtype_type head_dim variant_name causal_value has_mask_value
         is_varlen_value)
  set(function_name launch_${mode}_kernel_${dtype_name}_${head_dim}_${variant_name})
  set(SDPA_FUNCTION_NAME ${function_name})
  set(SDPA_TEMPLATE_NAME launch_${mode}_kernel_${head_dim})
  set(SDPA_CAUSAL ${causal_value})
  set(SDPA_HAS_MASK ${has_mask_value})
  set(SDPA_IS_VARLEN ${is_varlen_value})
  set(SDPA_ELEMENT_TYPE ${dtype_type})
  string(APPEND SDPA_KERNEL_DECLARATIONS "int ${function_name}(Options const& options);\n")
  set(SDPA_KERNEL_DECLARATIONS ${SDPA_KERNEL_DECLARATIONS} PARENT_SCOPE)
  set(_generated_src
    ${CMAKE_CURRENT_BINARY_DIR}/generated/sdpa/${mode}_${dtype_name}_${head_dim}_${variant_name}.cpp)
  configure_file(${CMAKE_CURRENT_LIST_DIR}/sdpa_kernel_instantiation.cpp.in ${_generated_src} @ONLY)
  list(APPEND SDPA_GENERATED_SRCS ${_generated_src})
  set(SDPA_GENERATED_SRCS ${SDPA_GENERATED_SRCS} PARENT_SCOPE)
endfunction()

foreach(mode IN ITEMS prefill decode)
  foreach(dtype_name IN ITEMS f16 bf16)
    if(dtype_name STREQUAL "f16")
      set(dtype_type "cute::half_t")
    else()
      set(dtype_type "cute::bfloat16_t")
    endif()

    foreach(head_dim IN ITEMS 128 64 96 192)
      foreach(varlen_name IN ITEMS normal varlen)
        if(varlen_name STREQUAL "varlen")
          if(mode STREQUAL "decode")
            continue()
          endif()
          set(is_varlen_value true)
          set(varlen_suffix "_varlen")
        else()
          set(is_varlen_value false)
          set(varlen_suffix "")
        endif()

        foreach(causal_name IN ITEMS causal noncausal)
          if(causal_name STREQUAL "causal")
            set(causal_value true)
            set(mask_names unmasked)
          else()
            set(causal_value false)
            set(mask_names masked unmasked)
          endif()

          foreach(mask_name IN LISTS mask_names)
            if(mask_name STREQUAL "masked")
              set(has_mask_value true)
            else()
              set(has_mask_value false)
            endif()

            set(variant_name ${causal_name})
            if(causal_name STREQUAL "noncausal")
              set(variant_name noncausal_${mask_name})
            endif()
            string(APPEND variant_name ${varlen_suffix})
            generate_sdpa_instantiation(${mode} ${dtype_name} "${dtype_type}" ${head_dim} ${variant_name}
                                         ${causal_value} ${has_mask_value} ${is_varlen_value})
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

configure_file(${CMAKE_CURRENT_LIST_DIR}/sdpa_kernel_declarations.hpp.in ${SDPA_DECLARATIONS_FILE} @ONLY)

# ========================================================================
# Sparse attention instantiations
#
# The sparse families (INT8 qks8 "sage" and native BF16/FP16 "sdpa") used to
# instantiate every causal/mask combination of every launcher inside a single
# translation unit, which peaked around 18 GiB of compiler RSS per TU. Emit one
# translation unit per (family, shape, dtype, causal, mask) combination instead,
# mirroring the dense SDPA split above.
# ========================================================================
set(SDPA_SPARSE_DECLARATIONS_FILE
  ${CMAKE_CURRENT_BINARY_DIR}/generated/sdpa/sdpa_sparse_kernel_declarations.hpp)

function(generate_sdpa_sparse_instantiation family shape launcher el_q el_k el_v dtype_name causal_value
         has_mask_value)
  if(causal_value)
    set(variant_name causal)
  elseif(has_mask_value)
    set(variant_name noncausal_masked)
  else()
    set(variant_name noncausal_unmasked)
  endif()
  set(function_name launch_${family}_sparse_${shape}_${dtype_name}_${variant_name})
  set(SDPA_SPARSE_FUNCTION_NAME ${function_name})
  set(SDPA_SPARSE_TEMPLATE_NAME ${launcher})
  set(SDPA_SPARSE_CAUSAL ${causal_value})
  set(SDPA_SPARSE_HAS_MASK ${has_mask_value})
  set(SDPA_SPARSE_EL_Q ${el_q})
  set(SDPA_SPARSE_EL_K ${el_k})
  set(SDPA_SPARSE_EL_V ${el_v})
  string(APPEND SDPA_SPARSE_KERNEL_DECLARATIONS "int ${function_name}(Options const& options);\n")
  set(SDPA_SPARSE_KERNEL_DECLARATIONS ${SDPA_SPARSE_KERNEL_DECLARATIONS} PARENT_SCOPE)
  set(_generated_src
    ${CMAKE_CURRENT_BINARY_DIR}/generated/sdpa/sparse_${family}_${shape}_${dtype_name}_${variant_name}.cpp)
  configure_file(${CMAKE_CURRENT_LIST_DIR}/sdpa_sparse_kernel_instantiation.cpp.in ${_generated_src} @ONLY)
  list(APPEND SDPA_GENERATED_SRCS ${_generated_src})
  set(SDPA_GENERATED_SRCS ${SDPA_GENERATED_SRCS} PARENT_SCOPE)
endfunction()

# Expands the causal / noncausal-masked / noncausal-unmasked variants for one
# (family, shape, dtype) triple. Causal never needs a mask, matching the
# runtime behaviour of the launchers this replaces.
function(generate_sdpa_sparse_family family shape launcher el_q el_k el_v dtype_name)
  foreach(causal_name IN ITEMS causal noncausal)
    if(causal_name STREQUAL "causal")
      set(causal_value true)
      set(mask_names unmasked)
    else()
      set(causal_value false)
      set(mask_names masked unmasked)
    endif()

    foreach(mask_name IN LISTS mask_names)
      if(mask_name STREQUAL "masked")
        set(has_mask_value true)
      else()
        set(has_mask_value false)
      endif()

      generate_sdpa_sparse_instantiation(${family} ${shape} ${launcher} "${el_q}" "${el_k}" "${el_v}"
                                          ${dtype_name} ${causal_value} ${has_mask_value})
    endforeach()
  endforeach()

  # generate_sdpa_sparse_instantiation() writes through PARENT_SCOPE, which here
  # is this function's scope. Re-export the accumulators so the caller sees them.
  set(SDPA_GENERATED_SRCS ${SDPA_GENERATED_SRCS} PARENT_SCOPE)
  set(SDPA_SPARSE_KERNEL_DECLARATIONS ${SDPA_SPARSE_KERNEL_DECLARATIONS} PARENT_SCOPE)
endfunction()

# Native BF16/FP16 sparse SDPA (sdpa_sparse_sdpa.cpp).
foreach(shape IN ITEMS 128 128_qtile64 64)
  if(shape STREQUAL "128")
    set(launcher launch_sparse_sdpa_prefill_kernel_128)
  elseif(shape STREQUAL "128_qtile64")
    set(launcher launch_sparse_sdpa_prefill_kernel_128_qtile64)
  else()
    set(launcher launch_sparse_sdpa_prefill_kernel_64)
  endif()

  foreach(dtype_name IN ITEMS bf16 f16)
    if(dtype_name STREQUAL "f16")
      set(element_type "cute::half_t")
    else()
      set(element_type "cute::bfloat16_t")
    endif()
    generate_sdpa_sparse_family(sdpa ${shape} ${launcher} "${element_type}" "${element_type}" "${element_type}"
                                 ${dtype_name})
  endforeach()
endforeach()

# INT8 qks8 sparse SAGE (sdpa_sparse.cpp): int8 Q/K with half or bf16 P*V.
foreach(shape IN ITEMS 128 128_qtile64 64)
  if(shape STREQUAL "128")
    set(launcher launch_sparse_sage_prefill_kernel_128)
  elseif(shape STREQUAL "128_qtile64")
    set(launcher launch_sparse_sage_prefill_kernel_128_qtile64)
  else()
    set(launcher launch_sparse_sage_prefill_kernel_64)
  endif()

  foreach(dtype_name IN ITEMS i8half i8bf16)
    if(dtype_name STREQUAL "i8half")
      set(pv_type "cute::half_t")
    else()
      set(pv_type "cute::bfloat16_t")
    endif()
    generate_sdpa_sparse_family(sage ${shape} ${launcher} "cute::int8_t" "cute::int8_t" "${pv_type}" ${dtype_name})
  endforeach()
endforeach()

configure_file(${CMAKE_CURRENT_LIST_DIR}/sdpa_sparse_kernel_declarations.hpp.in ${SDPA_SPARSE_DECLARATIONS_FILE} @ONLY)