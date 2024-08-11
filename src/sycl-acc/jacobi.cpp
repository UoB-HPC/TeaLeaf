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
                 SyclBuffer &uBuff,       //
                 SyclBuffer &u0Buff,      //
                 SyclBuffer &densityBuff, //
                 SyclBuffer &energyBuff,  //
                 SyclBuffer &kxBuff,      //
                 SyclBuffer &kyBuff,      //
                 queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto u = uBuff.get_access<access::mode::write>(h);
    auto u0 = u0Buff.get_access<access::mode::read_write>(h);
    auto density = densityBuff.get_access<access::mode::read>(h);
    auto energy = energyBuff.get_access<access::mode::read>(h);
    auto kx = kxBuff.get_access<access::mode::write>(h);
    auto ky = kyBuff.get_access<access::mode::write>(h);
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
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Main Jacobi solver method.
void jacobi_iterate(const int x,             //
                    const int y,             //
                    const size_t halo_depth, //
                    SyclBuffer &uBuff,       //
                    SyclBuffer &u0Buff,      //
                    SyclBuffer &rBuff,       //
                    SyclBuffer &kxBuff,      //
                    SyclBuffer &kyBuff,      //
                    double *error,           //
                    queue &device_queue) {
  buffer<double, 1> error_temp{range<1>{1}};
  device_queue.submit([&](handler &h) {
    auto r = rBuff.get_access<access::mode::read>(h);
    auto u = uBuff.get_access<access::mode::read_write>(h);
    auto u0 = u0Buff.get_access<access::mode::read>(h);
    auto kx = kxBuff.get_access<access::mode::read>(h);
    auto ky = kyBuff.get_access<access::mode::read>(h);

    h.parallel_for<class jacobi_iterate>(                        //
        range<1>(x * y),                                         //
        reduction_shim(error_temp, h, {}, sycl::plus<double>()), //
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
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
  *error = error_temp.get_host_access()[0];
}

// Copies u into r
void jacobi_copy_u(const int x,       //
                   const int y,       //
                   SyclBuffer &rBuff, //
                   SyclBuffer &uBuff, //
                   queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto r = rBuff.get_access<access::mode::write>(h);
    auto u = uBuff.get_access<access::mode::read>(h);
    h.parallel_for<class jacobi_copy_u>(range<1>(x * y), [=](id<1> idx) { r[idx[0]] = u[idx[0]]; });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Jacobi solver kernels
void run_jacobi_init(Chunk *chunk, Settings &settings, double rx, double ry) {
  START_PROFILING(settings.kernel_profile);

  jacobi_init(chunk->x, chunk->y, settings.halo_depth, settings.coefficient, rx, ry, *(chunk->u), *(chunk->u0), *(chunk->density),
              *(chunk->energy), *(chunk->kx), *(chunk->ky), *(chunk->ext->device_queue));

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_jacobi_iterate(Chunk *chunk, Settings &settings, double *error) {
  START_PROFILING(settings.kernel_profile);

  jacobi_copy_u(chunk->x, chunk->y, *(chunk->r), *(chunk->u), *(chunk->ext->device_queue));

  jacobi_iterate(chunk->x, chunk->y, settings.halo_depth, *(chunk->u), *(chunk->u0), *(chunk->r), *(chunk->kx), *(chunk->ky), error,
                 *(chunk->ext->device_queue));

  STOP_PROFILING(settings.kernel_profile, __func__);
}