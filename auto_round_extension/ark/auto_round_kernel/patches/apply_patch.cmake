# Idempotently apply a git patch to a fetched dependency source tree.
#
# Usage:
#   cmake -DGIT_EXECUTABLE=<git> -DPATCH_FILE=<abs path> -P apply_patch.cmake
#
# The script is safe to run multiple times: if the patch is already applied
# (detected via `git apply --reverse --check`), it does nothing instead of
# failing with "patch does not apply".

if(NOT GIT_EXECUTABLE)
  set(GIT_EXECUTABLE "git")
endif()

if(NOT PATCH_FILE)
  message(FATAL_ERROR "apply_patch.cmake: PATCH_FILE must be provided")
endif()

# If the patch reverse-applies cleanly, it is already present -> nothing to do.
execute_process(
  COMMAND ${GIT_EXECUTABLE} apply --reverse --check --ignore-whitespace "${PATCH_FILE}"
  RESULT_VARIABLE _already_applied
  OUTPUT_QUIET
  ERROR_QUIET
)

if(_already_applied EQUAL 0)
  message(STATUS "apply_patch.cmake: patch already applied, skipping: ${PATCH_FILE}")
  return()
endif()

execute_process(
  COMMAND ${GIT_EXECUTABLE} apply --ignore-whitespace "${PATCH_FILE}"
  RESULT_VARIABLE _apply_result
)

if(NOT _apply_result EQUAL 0)
  message(FATAL_ERROR "apply_patch.cmake: failed to apply patch: ${PATCH_FILE}")
endif()

message(STATUS "apply_patch.cmake: applied patch: ${PATCH_FILE}")
