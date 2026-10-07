cmake_minimum_required(VERSION 3.14)

file(REMOVE "${REPORT}")
execute_process(COMMAND ${TEST_COMMAND}
                RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
message("${stdout}${stderr}")

if (EXPECT_EXIT STREQUAL "nonzero")
    if (NOT "${result}" MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "Expected an error exit, got ${result}")
    endif ()
elseif (NOT "${result}" STREQUAL "${EXPECT_EXIT}")
    message(FATAL_ERROR "Exit status ${result}; expected ${EXPECT_EXIT}")
endif ()

set(output "${stdout}${stderr}")
foreach(expected IN LISTS EXPECT_OUTPUT)
    string(FIND "${output}" "${expected}" position)
    if (position EQUAL -1)
        message(FATAL_ERROR "Missing '${expected}' in output:\n${output}")
    endif ()
endforeach()

if (REPORT AND NOT EXISTS "${REPORT}")
    message(FATAL_ERROR "TeaLeaf did not create report: ${REPORT}")
endif ()
