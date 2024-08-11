cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()

project(tealeaf VERSION 1.0 LANGUAGES C CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

## Flags for debugging only, enable for development (ASan only works on few models)
set(SANITIZE OFF)
if (SANITIZE)
    set(CXX_EXTRA_FLAGS -fsanitize=address -fsanitize=undefined -g)
    set(CXX_EXTRA_LINK_FLAGS -fsanitize=address -fsanitize=undefined -g)
endif ()


# the final executable name
set(EXE_NAME tealeaf)


## select default build type if not given
if (NOT CMAKE_BUILD_TYPE)
    message("No CMAKE_BUILD_TYPE specified, defaulting to 'Release'")
    set(CMAKE_BUILD_TYPE "Release")
endif ()

string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
if ((NOT BUILD_TYPE STREQUAL RELEASE) AND (NOT BUILD_TYPE STREQUAL DEBUG))
    message(FATAL_ERROR "Only Release or Debug is supported, got `${CMAKE_BUILD_TYPE}`")
endif ()

# setup some defaults flags for everything
set(DEFAULT_DEBUG_FLAGS -Wall -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -O2)
set(DEFAULT_RELEASE_FLAGS -Wall -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -O3)

macro(hint_flag FLAG DESCRIPTION)
    if (NOT DEFINED ${FLAG})
        message(STATUS "${FLAG}: ${DESCRIPTION}")
    else ()
        # i.e. `-DFOO="-a -b"` becomes CMake's semicolon separated list `FOO=`-a;-b`
        separate_arguments(${FLAG})
    endif ()
endmacro()

# hint common extra flag options for all models if they are not set
hint_flag(CXX_EXTRA_FLAGS "
        Appends to common compile flags. These will be appended at link phase as well.
        To use separate flags at link phase, set `CXX_EXTRA_LINK_FLAGS`")
hint_flag(CXX_EXTRA_LINK_FLAGS "
        Appends to link flags which appear *before* the objects.
        Do not use this for linking libraries, as the link line is order-dependent")
hint_flag(CXX_EXTRA_LIBRARIES "
        Append to link flags which appear *after* the objects.
        Use this for linking extra libraries (e.g `-lmylib`, or simply `mylib`)")
hint_flag(CXX_EXTRA_LINKER_FLAGS "
        Append to linker flags (i.e GCC's `-Wl` or equivalent)")

# copy CXX_EXTRA_FLAGS <- CXX_EXTRA_LINK_FLAGS
if ((DEFINED CXX_EXTRA_FLAGS) AND (NOT DEFINED CXX_EXTRA_LINK_FLAGS))
    set(CXX_EXTRA_LINK_FLAGS ${CXX_EXTRA_FLAGS})
endif ()


option(USE_TBB "Enable oneTBB library for *supported* models. Enabling this on models that
                don't explicitly link against TBB is a no-op, see description of your selected
                model on how this is used." OFF)

if (USE_TBB)
    FetchContent_Declare(
            TBB
            GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
            GIT_TAG v2021.13.0
    )
    # Don't fail builds on waring (TBB has -Wall while not being free of warnings from unused symbols...)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(TBB_STRICT OFF)
    # Not using FetchContent_MakeAvailable (CMake>= 3.14) because we need EXCLUDE_FROM_ALL
    FetchContent_GetProperties(TBB)
    if (NOT TBB_POPULATED)
        FetchContent_Populate(TBB)
        add_subdirectory(${tbb_SOURCE_DIR} ${tbb_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
endif ()

option(USE_ONEDPL "Enable oneDPL library for *supported* models. Enabling this on models that
                   don't explicitly link against DPL is a no-op, see description of your selected
                   model on how this is used." OFF)

if (USE_ONEDPL)
    FetchContent_Declare(
            oneDPL
            GIT_REPOSITORY https://github.com/oneapi-src/oneDPL.git
            GIT_TAG oneDPL-2022.6.0-rc1
    )
    string(TOLOWER ${USE_ONEDPL} ONEDPL_BACKEND)
    # XXX oneDPL looks for omp instead of openmp, which mismatches(!) with ONEDPL_PAR_BACKEND if using find_package
    if (ONEDPL_BACKEND STREQUAL "openmp")
        set(ONEDPL_BACKEND omp)
    endif ()
    # Not using FetchContent_MakeAvailable (CMake>= 3.14) because we need EXCLUDE_FROM_ALL
    FetchContent_GetProperties(oneDPL)
    if (NOT oneDPL_POPULATED)
        FetchContent_Populate(oneDPL)
        if (USE_TBB)
            macro(find_package NAME)
                if ("${NAME}" STREQUAL "TBB")
                    message(STATUS "Discarding oneDPL's call to find_package(${NAME} ${ARGN})")
                else ()
                    _find_package(${NAME} ${ARGN})
                endif ()
            endmacro()
        endif ()
        add_subdirectory(${onedpl_SOURCE_DIR} ${onedpl_BINARY_DIR} EXCLUDE_FROM_ALL)

        # Fixup oneDPL's omission on setting DPCPP definitions.
        # We do this after the creation of the oneDPL target.
        if (ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
            target_compile_definitions(oneDPL INTERFACE ONEDPL_USE_DPCPP_BACKEND=1)
        endif ()

    endif ()
endif ()


# include our macros
include(cmake/register_models.cmake)

register_flag_optional(ENABLE_MPI "Enables MPI support at compile time, set MPI_HOME (e.g -DMPI_HOME=/usr/lib64/openmpi/) if not on PATH" OFF)
register_flag_optional(ENABLE_PROFILING "Enables kernel profiler, this may introduce synchronisation overhead for some models." OFF)

if ("${MODEL}" STREQUAL "omp-target")
    set(MODEL omp)
    set(ALTERNATIVE_BIN_NAME "omp-target-tealeaf")
endif ()

set(MODEL_SRC
        cg.cpp
        cheby.cpp
        jacobi.cpp
        kernel_initialise.cpp
        local_halos.cpp
        pack_halos.cpp
        ppcg.cpp
        solver_methods.cpp
)

# register out models <model_name> <preprocessor_def_name> <source files...>

register_model(serial USE_SERIAL ${MODEL_SRC})
register_model(omp USE_OMP ${MODEL_SRC} diffuse_overload.cpp)
register_model(kokkos USE_KOKKOS ${MODEL_SRC})
register_model(cuda USE_CUDA ${MODEL_SRC})
register_model(hip USE_HIP ${MODEL_SRC})
register_model(std-indices USE_STD ${MODEL_SRC})
register_model(sycl-acc USE_SYCL_ACC ${MODEL_SRC})
register_model(sycl-usm USE_SYCL_USM ${MODEL_SRC})
register_model(tbb USE_TBB ${MODEL_SRC})

if ("${MODEL}" STREQUAL omp-target)
    set(BIN_NAME omp-target-tealeaf)
endif ()


#register_model(acc ACC fasten.hpp)  # TODO
# defining RAJA collides with the RAJA namespace so USE_RAJA
#register_model(raja USE_RAJA fasten.hpp) # TODO
#register_model(tbb TBB fasten.hpp) # TODO
#register_model(thrust THRUST fasten.hpp) # TODO

set(USAGE ON CACHE BOOL "Whether to print all custom flags for the selected model")

message(STATUS "Available models:  ${REGISTERED_MODELS}")
if (NOT DEFINED MODEL)
    message(FATAL_ERROR "MODEL is unspecified, pick one from the available models")
else ()
    string(TOLOWER ${MODEL} MODEL)
    message(STATUS "Selected model  :  ${MODEL}")
endif ()

# load the $MODEL.cmake file and setup the correct IMPL_* based on $MODEL
load_model(${MODEL})

if (USAGE)
    # print the usage of the registered flag options
    registered_flags_action(print RESULT)
    message(STATUS "${RESULT}")
endif ()
# check required/set default for all registered flag options
registered_flags_action(check RESULT)
message(STATUS "${RESULT}")

list(APPEND IMPL_SOURCES
        driver/main.cpp
        driver/comms.cpp
        driver/chunk.cpp
        driver/shared.cpp
        driver/diffuse.cpp
        driver/profiler.cpp
        driver/settings.cpp
        driver/initialise.cpp
        driver/parse_config.cpp

        driver/cg_driver.cpp
        driver/ppcg_driver.cpp
        driver/cheby_driver.cpp
        driver/jacobi_driver.cpp
        driver/eigenvalue_driver.cpp
        driver/halo_update_driver.cpp
        driver/remote_halo_driver.cpp
        driver/store_energy_driver.cpp
        driver/field_summary_driver.cpp
        driver/set_chunk_data_driver.cpp
        driver/solve_finished_driver.cpp
        driver/set_chunk_state_driver.cpp
        driver/kernel_initialise_driver.cpp

        driver/mpi_shim.cpp
)

# run model specific setup, i.e append build flags, etc
setup()

if (ALTERNATIVE_BIN_NAME)
    set(BIN_NAME "${ALTERNATIVE_BIN_NAME}")
endif ()

# CMake insists that -O2 (or equivalent) is the universally accepted optimisation level
# we remove that here and use our own <BUILD_TYPE>_FLAGS
if (CMAKE_CXX_FLAGS_${BUILD_TYPE})
    wipe_gcc_style_optimisation_flags(CMAKE_CXX_FLAGS_${BUILD_TYPE})
endif ()

message(STATUS "Default ${CMAKE_BUILD_TYPE} flags are `${DEFAULT_${BUILD_TYPE}_FLAGS}`, set ${BUILD_TYPE}_FLAGS to override (CXX_EXTRA_* flags are not affected)")


# setup common build flag defaults if there are no overrides
if (NOT DEFINED ${BUILD_TYPE}_FLAGS)
    set(ACTUAL_${BUILD_TYPE}_FLAGS ${DEFAULT_${BUILD_TYPE}_FLAGS})
else ()
    set(ACTUAL_${BUILD_TYPE}_FLAGS ${${BUILD_TYPE}_FLAGS})
endif ()

if (NOT ENABLE_MPI)
    list(APPEND IMPL_DEFINITIONS NO_MPI)
else ()
    find_package(MPI REQUIRED)
    list(APPEND LINK_LIBRARIES MPI::MPI_C)
endif ()
if (ENABLE_PROFILING)
    list(APPEND IMPL_DEFINITIONS ENABLE_PROFILING)
endif ()

message(STATUS "CXX vendor  : ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "Platform    : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Sources     : ${IMPL_SOURCES}")
message(STATUS "Libraries   : ${LINK_LIBRARIES} ${CXX_EXTRA_LIBRARIES}")

message(STATUS "CXX Flags   : ${CMAKE_CXX_FLAGS_${BUILD_TYPE}} ${ACTUAL_${BUILD_TYPE}_CXX_FLAGS} ${CXX_EXTRA_FLAGS}
    CXX flags derived from (CMake + (Override ? Override : Default) + Extras), where:
        CMake                    = `${CMAKE_CXX_FLAGS_${BUILD_TYPE}}`
        Default                  = `${DEFAULT_${BUILD_TYPE}_FLAGS}`
        Override (${BUILD_TYPE}_FLAGS) = `${${BUILD_TYPE}_FLAGS}`
        Extras (CXX_EXTRA_FLAGS) = `${CXX_EXTRA_FLAGS}`")
message(STATUS "CXX Link Flags  : ${LINK_FLAGS} ${CXX_EXTRA_LINK_FLAGS}")
message(STATUS "CXX Linker Flags: ${CMAKE_EXE_LINKER_FLAGS} ${CXX_EXTRA_LINKER_FLAGS} ")

message(STATUS "Defs        : ${IMPL_DEFINITIONS}")
message(STATUS "Executable  : ${BIN_NAME}")

# below we have all the usual CMake target setup steps


include_directories(${CMAKE_BINARY_DIR}/generated)


add_executable(${EXE_NAME} ${IMPL_SOURCES})
target_link_libraries(${EXE_NAME} PUBLIC ${LINK_LIBRARIES} m)
target_compile_definitions(${EXE_NAME} PUBLIC ${IMPL_DEFINITIONS})
target_include_directories(${EXE_NAME} PUBLIC driver)


if (CXX_EXTRA_LIBRARIES)
    target_link_libraries(${EXE_NAME} PUBLIC ${CXX_EXTRA_LIBRARIES})
endif ()

target_compile_options(${EXE_NAME} PUBLIC "$<$<CONFIG:Release>:${ACTUAL_RELEASE_FLAGS};${CXX_EXTRA_FLAGS}>")
target_compile_options(${EXE_NAME} PUBLIC "$<$<CONFIG:Debug>:${ACTUAL_DEBUG_FLAGS};${CXX_EXTRA_FLAGS}>")

target_link_options(${EXE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:LINKER:${CXX_EXTRA_LINKER_FLAGS}>)
target_link_options(${EXE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${LINK_FLAGS};${CXX_EXTRA_LINK_FLAGS}>)


# some models require the target to be already specified so they can finish their setup here
# this only happens if the model.cmake definition contains the `setup_target` macro
if (COMMAND setup_target)
    setup_target(${EXE_NAME})
endif ()

target_compile_definitions(${EXE_NAME} PRIVATE)

#if ((CMAKE_GENERATOR MATCHES "Unix Makefiles") OR (CMAKE_GENERATOR MATCHES "Ninja"))
#    add_custom_target(extract_compile_commands ALL
#            COMMAND ${CMAKE_COMMAND}
#            -DCOMMAND_FILE=${CMAKE_BINARY_DIR}/compile_commands.json
#            -DPRE_CONFIGURE_FILE= "${CMAKE_SOURCE_DIR}/src/meta_build.h.in"
#            -DPOST_CONFIGURE_FILE= "${CMAKE_BINARY_DIR}/generated/meta_build.h"
#            -DSRC_DIR_PREFIX= "${CMAKE_SOURCE_DIR}/src"
#            -DOUT_DIR_PREFIX= "CMakeFiles/${EXE_NAME}.dir"
#            -P ${CMAKE_SOURCE_DIR}/cmake/extract_compile_command.cmake
#            )
#    add_dependencies(${EXE_NAME} extract_compile_commands)
#else ()
#    set(COMPILE_COMMANDS COMPILE_FLAGS)
#endif ()

set_target_properties(${EXE_NAME} PROPERTIES OUTPUT_NAME "${BIN_NAME}")

install(TARGETS ${EXE_NAME} DESTINATION bin)