
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection, this is used for host compilation"
        "c++")

register_flag_required(CMAKE_CUDA_COMPILER
        "Path to the CUDA nvcc compiler")

# XXX we may want to drop this eventually and use CMAKE_CUDA_ARCHITECTURES directly
register_flag_optional(CUDA_ARCH
        "Nvidia architecture, this will be append the compile line via `-arch=` (e.g `sm_70`) for nvcc or --cuda-arch for clang++"
        "sm_86")


register_flag_optional(CUDA_CLANG_DRIVER
        "Disable any nvcc-specific flags so that setting CMAKE_CUDA_COMPILER to clang++ can compile successfully"
        "OFF")

register_flag_optional(ACPP_PCUDA_DRIVER
        "Compile CUDA syntax through AdaptiveCpp's portable CUDA mode"
        "OFF")

register_flag_optional(CUDA_EXTRA_FLAGS
        "Additional CUDA flags passed to the CUDA compiler, this is appended after `CUDA_CXX_FLAGS`"
        "")

register_flag_optional(MANAGED_ALLOC "Use UVM (cudaMallocManaged) instead of the device-only allocation (cudaMalloc)"
        "OFF")

register_flag_optional(SYNC_ALL_KERNELS
        "Fully synchronise all kernels after launch, this also enables synchronous error checking with line and file name"
        "OFF")


macro(setup)

    # Keep the explicit -arch flag below without also letting CMake add one.
    if (POLICY CMP0104)
        cmake_policy(SET CMP0104 NEW)
        set(CMAKE_CUDA_ARCHITECTURES OFF)
    endif ()

    set(CMAKE_CXX_STANDARD 17)

    if (CUDA_CLANG_DRIVER)
        if (CMAKE_VERSION VERSION_LESS "3.18.0")
            message(FATAL_ERROR "Using clang driver for CUDA is only supported for CMake >= 3.18")
        endif ()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --cuda-gpu-arch=${CUDA_ARCH} ${CUDA_EXTRA_FLAGS}")
        enable_language(CUDA)
    elseif (ACPP_PCUDA_DRIVER)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --acpp-pcuda --acpp-pcuda-chevron-launch ${CUDA_EXTRA_FLAGS}")
    else ()
        # add -forward-unknown-to-host-compiler for compatibility reasons
        # add -std=c++17 manually as older CMake seems to omit this (source gets treated as C otherwise)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -forward-unknown-to-host-compiler -extended-lambda -use_fast_math -restrict -arch=${CUDA_ARCH} ${CUDA_EXTRA_FLAGS}")
        enable_language(CUDA)
    endif ()


    # CMake defaults to -O2 for CUDA at Release, let's wipe that and use the global RELEASE_FLAG
    # appended later
    if (NOT ACPP_PCUDA_DRIVER)
        wipe_gcc_style_optimisation_flags(CMAKE_CUDA_FLAGS_${BUILD_TYPE})
    endif ()

    if (MANAGED_ALLOC)
        register_definitions(CLOVER_MANAGED_ALLOC)
    endif ()

    if (SYNC_ALL_KERNELS)
        register_definitions(CLOVER_SYNC_ALL_KERNELS)
    endif ()

    message(STATUS "NVCC flags: ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")
endmacro()

macro(setup_target NAME)
    # Treat everything as CUDA source
    get_target_property(PROJECT_SRC "${NAME}" SOURCES)
    foreach (SRC ${PROJECT_SRC})
        if (NOT ACPP_PCUDA_DRIVER)
            set_source_files_properties("${SRC}" PROPERTIES LANGUAGE CUDA)
        endif ()
    endforeach ()
endmacro()
