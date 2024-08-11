#pragma once

#include "hip/hip_runtime.h"

#include "shared.h"
#ifndef BLOCK_SIZE
  #define BLOCK_SIZE 1024 // XXX anything less than 512 would break reduction
#endif

#ifdef CLOVER_MANAGED_ALLOC
  #define CLOVER_MEMCPY_KIND_D2H (hipMemcpyDefault)
  #define CLOVER_MEMCPY_KIND_H2D (hipMemcpyDefault)
#else
  #define CLOVER_MEMCPY_KIND_D2H (hipMemcpyDeviceToHost)
  #define CLOVER_MEMCPY_KIND_H2D (hipMemcpyHostToDevice)
#endif

__device__ inline double SUM(double a, double b) { return a + b; }

template <typename T, int offset> class reduce {
public:
  __device__ inline static void run(T *array, T *out, T (*func)(T, T)) {
    __syncthreads(); // don't optimise for sub-warp, always sync
    // only continue if it's in the lower half
    if (threadIdx.x < offset) {
      array[threadIdx.x] = func(array[threadIdx.x], array[threadIdx.x + offset]);
      reduce<T, offset / 2>::run(array, out, func);
    }
  }
};

template <typename T> class reduce<T, 0> {
public:
  __device__ inline static void run(T *array, T *out, T (*)(T, T)) { out[blockIdx.x] = array[0]; }
};

inline void check_errors(int line_num, const char *file) {
  if (auto result = hipDeviceSynchronize(); result != hipSuccess) {
    die(line_num, file, "Error in %s - return code %d (%s)\n", file, result, hipGetErrorName(result));
  }
}

void sum_reduce_buffer(double *buffer, double *result, int len);

#define KERNELS_START(pad)                                                 \
  START_PROFILING(settings.kernel_profile);                                \
  int x_inner = chunk->x - (pad);                                          \
  int y_inner = chunk->y - (pad);                                          \
  int num_blocks = ceil((double)(x_inner * y_inner) / double(BLOCK_SIZE)); \
  do {                                                                     \
  } while (false)
#ifdef CLOVER_SYNC_ALL_KERNELS
  #define KERNELS_END()               \
    check_errors(__LINE__, __FILE__); \
    STOP_PROFILING(settings.kernel_profile, __func__);
#else
  #define KERNELS_END() STOP_PROFILING(settings.kernel_profile, __func__)
#endif
