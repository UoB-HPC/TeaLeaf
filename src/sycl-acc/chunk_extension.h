#pragma once

#if __has_include(<sycl/sycl.hpp>)
  #include <sycl/sycl.hpp>
#else
  #include <CL/sycl.hpp>
#endif

using FieldBufferType = sycl::buffer<double, 1> *;
using StagingBufferType = sycl::buffer<double, 1> *;

struct ChunkExtension {
  sycl::queue *device_queue;
};
