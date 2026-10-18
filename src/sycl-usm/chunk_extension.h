#pragma once

#if __has_include(<sycl/sycl.hpp>)
  #include <sycl/sycl.hpp>
#else
  #include <CL/sycl.hpp>
#endif

using FieldBufferType = double *;
using StagingBufferType = double *;

struct Summary {
  double vol = 0.0;
  double mass = 0.0;
  double ie = 0.0;
  double temp = 0.0;
  [[nodiscard]] constexpr Summary operator+(const Summary &that) const { //
    return {vol + that.vol, mass + that.mass, ie + that.ie, temp + that.temp};
  }
};

struct ChunkExtension {
  sycl::queue *device_queue;
  double *reduction_cg_rro;
  double *reduction_cg_pw;
  double *reduction_cg_rrn;
  double *reduction_jacobi_error;
  double *reduction_norm;
  Summary *reduction_field_summary;
};
