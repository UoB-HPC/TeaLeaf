#pragma once

#if __has_include(<sycl/sycl.hpp>)
  #include <sycl/sycl.hpp>
#else
  #include <CL/sycl.hpp>
#endif

using namespace sycl;

using SyclBuffer = double *;

template <typename T, typename BinaryOp> inline auto reduction_shim(T *b, T init, BinaryOp f) {
#if defined(__HIPSYCL__) || defined(__OPENSYCL__)
  return sycl::reduction(b, init, f);
#else
  return sycl::reduction(b, init, f, sycl::property::reduction::initialize_to_identity());
#endif
}
