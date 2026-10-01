#include "chunk.h"
#include "shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Initialises the Jacobi solver
void jacobi_init(const int x,             //
                 const int y,             //
                 const size_t halo_depth, //
                 const int coefficient,   //
                 const double rx,         //
                 const double ry,         //
                 SyclBuffer &u,           //
                 SyclBuffer &u0,          //
                 SyclBuffer &density,     //
                 SyclBuffer &energy,      //
                 SyclBuffer &kx,          //
                 SyclBuffer &ky,          //
                 queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class jacobi_init>(range<1>(x * y), [=](id<1> idx) {
          const auto kk = idx[0] % x;
          const auto jj = idx[0] / x;
          if (kk > 0 && kk < size_t(x - 1) && jj > 0 && jj < size_t(y - 1)) {
            u0[idx[0]] = energy[idx[0]] * density[idx[0]];
            u[idx[0]] = u0[idx[0]];
          }
          if (jj >= halo_depth && jj < size_t(y - 1) && kk >= halo_depth && kk < size_t(x - 1)) {
            double densityCentre = (coefficient == CONDUCTIVITY) ? density[idx[0]] : 1.0 / density[idx[0]];
            double densityLeft = (coefficient == CONDUCTIVITY) ? density[idx[0] - 1] : 1.0 / density[idx[0] - 1];
            double densityDown = (coefficient == CONDUCTIVITY) ? density[idx[0] - x] : 1.0 / density[idx[0] - x];

            kx[idx[0]] = rx * (densityLeft + densityCentre) / (2.0 * densityLeft * densityCentre);
            ky[idx[0]] = ry * (densityDown + densityCentre) / (2.0 * densityDown * densityCentre);
          }
        });
      })
      .wait_and_throw();
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Main Jacobi solver method.
void jacobi_iterate(const int x,             //
                    const int y,             //
                    const size_t halo_depth, //
                    SyclBuffer &u,           //
                    SyclBuffer &u0,          //
                    SyclBuffer &r,           //
                    SyclBuffer &kx,          //
                    SyclBuffer &ky,          //
                    SyclBuffer &error_temp,  //
                    double *error,           //
                    queue &device_queue) {
  auto event = device_queue.submit([&](handler &h) {
    h.parallel_for<class jacobi_iterate>(                         //
        range<1>(x * y),                                          //
        reduction_shim(error_temp, *error, sycl::plus<double>()), //
        [=](item<1> item, auto &acc) {
          const auto kk = item[0] % x;
          const auto jj = item[0] / x;
          if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
            u[item[0]] = (u0[item[0]] + (kx[item[0] + 1] * r[item[0] + 1] + kx[item[0]] * r[item[0] - 1]) +
                          (ky[item[0] + x] * r[item[0] + x] + ky[item[0]] * r[item[0] - x])) /
                         (1.0 + (kx[item[0]] + kx[item[0] + 1]) + (ky[item[0]] + ky[item[0] + x]));
            acc += ::fabs((u[item[0]] - r[item[0]])); // fabs is float version of abs
          }
        });
  });
  device_queue.copy(error_temp, error, 1, event).wait_and_throw();
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Copies u into r
void jacobi_copy_u(const int x,   //
                   const int y,   //
                   SyclBuffer &r, //
                   SyclBuffer &u, //
                   queue &device_queue) {
  device_queue.submit([&](handler &h) { h.parallel_for<class jacobi_copy_u>(range<1>(x * y), [=](id<1> idx) { r[idx[0]] = u[idx[0]]; }); });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Jacobi solver kernels
void run_jacobi_init(Chunk *chunk, Settings &settings, double rx, double ry) {
  START_PROFILING(settings.kernel_profile);

  jacobi_init(chunk->x, chunk->y, settings.halo_depth, settings.coefficient, rx, ry, (chunk->u), (chunk->u0), (chunk->density),
              (chunk->energy), (chunk->kx), (chunk->ky), *(chunk->ext->device_queue));

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_jacobi_iterate(Chunk *chunk, Settings &settings, double *error) {
  START_PROFILING(settings.kernel_profile);

  jacobi_copy_u(chunk->x, chunk->y, (chunk->r), (chunk->u), *(chunk->ext->device_queue));

  jacobi_iterate(chunk->x, chunk->y, settings.halo_depth, (chunk->u), (chunk->u0), (chunk->r), (chunk->kx), (chunk->ky),
                 (chunk->ext->reduction_jacobi_error), error, *(chunk->ext->device_queue));

  STOP_PROFILING(settings.kernel_profile, __func__);
}
