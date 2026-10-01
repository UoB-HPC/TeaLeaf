#include "chunk.h"
#include "shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Initialises p,r,u,w
void cg_init_u(const int x,           //
               const int y,           //
               const int coefficient, //
               SyclBuffer &p,         //
               SyclBuffer &r,         //
               SyclBuffer &u,         //
               SyclBuffer &w,         //
               SyclBuffer &density,   //
               SyclBuffer &energy, queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class cg_init_u>(range<1>(x * y), [=](id<1> idx) {
          const auto kk = idx[0] % x;
          const auto jj = idx[0] / x;
          p[idx[0]] = 0.0;
          r[idx[0]] = 0.0;
          u[idx[0]] = energy[idx[0]] * density[idx[0]];
          if (jj > 0 && jj < size_t(y - 1) && kk > 0 && kk < size_t(x - 1)) {
            w[idx[0]] = (coefficient == CONDUCTIVITY) ? density[idx[0]] : 1.0 / density[idx[0]];
          }
        });
      })
      .wait_and_throw();
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Initialises kx,ky
void cg_init_k(const int x,             //
               const int y,             //
               const size_t halo_depth, //
               SyclBuffer &w,           //
               SyclBuffer &kx,          //
               SyclBuffer &ky,          //
               const double rx,         //
               const double ry, queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class cg_init_k>(range<1>(x * y), [=](id<1> idx) {
          const auto kk = idx[0] % x;
          const auto jj = idx[0] / x;
          if (jj >= halo_depth && jj < size_t(y - 1) && kk >= halo_depth && kk < size_t(x - 1)) {
            kx[idx[0]] = rx * (w[idx[0] - 1] + w[idx[0]]) / (2.0 * w[idx[0] - 1] * w[idx[0]]);
            ky[idx[0]] = ry * (w[idx[0] - x] + w[idx[0]]) / (2.0 * w[idx[0] - x] * w[idx[0]]);
          }
        });
      })
      .wait_and_throw();
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Initialises w,r,p and calculates rro
void cg_init_others(const int x,             //
                    const int y,             //
                    const size_t halo_depth, //
                    SyclBuffer &kx,          //
                    SyclBuffer &ky,          //
                    SyclBuffer &p,           //
                    SyclBuffer &r,           //
                    SyclBuffer &u,           //
                    SyclBuffer &w,           //
                    SyclBuffer &rro_temp,    //
                    double *rro,             //
                    queue &device_queue) {
  auto event = device_queue.submit([&](handler &h) {
    h.parallel_for<class cg_init_others>(                     //
        range<1>(x * y),                                      //
        reduction_shim(rro_temp, *rro, sycl::plus<double>()), //
        [=](item<1> item, auto &acc) {
          const auto kk = item[0] % x;
          const auto jj = item[0] / x;
          if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
            auto index = item[0]; // smvp uses kx and ky and index
            const double smvp = tealeaf_SMVP(u);
            w[item[0]] = smvp;
            r[item[0]] = u[item[0]] - w[item[0]];
            p[item[0]] = r[item[0]];
            acc += r[item[0]] * p[item[0]];
          }
        });
  });
  device_queue.copy(rro_temp, rro, 1, event).wait_and_throw();
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Calculates the value for w
void cg_calc_w(const int x,             //
               const int y,             //
               const size_t halo_depth, //
               SyclBuffer &w,           //
               SyclBuffer &p,           //
               SyclBuffer &kx,          //
               SyclBuffer &ky,          //
               SyclBuffer &pw_temp,     //
               double *pw,              //
               queue &device_queue) {
  auto event = device_queue.submit([&](handler &h) {
    h.parallel_for<class cg_calc_w>(                        //
        range<1>(x * y),                                    //
        reduction_shim(pw_temp, *pw, sycl::plus<double>()), //
        [=](item<1> item, auto &acc) {
          const auto kk = item[0] % x;
          const auto jj = item[0] / x;
          if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
            // smvp uses kx and ky and index
            int index = item[0];
            const double smvp = tealeaf_SMVP(p);
            w[item[0]] = smvp;
            acc += w[item[0]] * p[item[0]];
          }
        });
  });
  device_queue.copy(pw_temp, pw, 1, event).wait_and_throw();
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Calculates the value of u and r
void cg_calc_ur(const int x,             //
                const int y,             //
                const size_t halo_depth, //
                SyclBuffer &u,           //
                SyclBuffer &r,           //
                SyclBuffer &p,           //
                SyclBuffer &w,           //
                SyclBuffer &rrn_temp,    //
                const double alpha,      //
                double *rrn,             //
                queue &device_queue) {
  auto event = device_queue.submit([&](handler &h) {
    h.parallel_for<class cg_calc_ur>(                         //
        range<1>(x * y),                                      //
        reduction_shim(rrn_temp, *rrn, sycl::plus<double>()), //
        [=](item<1> item, auto &acc) {
          const auto kk = item[0] % x;
          const auto jj = item[0] / x;
          if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
            u[item[0]] += alpha * p[item[0]];
            r[item[0]] -= alpha * w[item[0]];
            acc += r[item[0]] * r[item[0]];
          }
        });
  });
  device_queue.copy(rrn_temp, rrn, 1, event).wait_and_throw();
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Calculates a value for p
void cg_calc_p(const int x,             //
               const int y,             //
               const size_t halo_depth, //
               const double beta,       //
               SyclBuffer &p,           //
               SyclBuffer &r,           //
               queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class cg_calc_p>(range<1>(x * y), [=](id<1> idx) {
          const auto kk = idx[0] % x;
          const auto jj = idx[0] / x;
          if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
            p[idx[0]] = beta * p[idx[0]] + r[idx[0]];
          }
        });
      })
      .wait_and_throw(); // this is followed by local_halo, so we must synchronise here
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// CG solver kernels
void run_cg_init(Chunk *chunk, Settings &settings, double rx, double ry, double *rro) {
  START_PROFILING(settings.kernel_profile);

  cg_init_u(chunk->x, chunk->y, settings.coefficient, (chunk->p), (chunk->r), (chunk->u), (chunk->w), (chunk->density), (chunk->energy),
            *(chunk->ext->device_queue));

  cg_init_k(chunk->x, chunk->y, settings.halo_depth, (chunk->w), (chunk->kx), (chunk->ky), rx, ry, *(chunk->ext->device_queue));

  cg_init_others(chunk->x, chunk->y, settings.halo_depth, (chunk->kx), (chunk->ky), (chunk->p), (chunk->r), (chunk->u), (chunk->w),
                 (chunk->ext->reduction_cg_rro), rro, *(chunk->ext->device_queue));

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_cg_calc_w(Chunk *chunk, Settings &settings, double *pw) {
  START_PROFILING(settings.kernel_profile);

  cg_calc_w(chunk->x, chunk->y, settings.halo_depth, (chunk->w), (chunk->p), (chunk->kx), (chunk->ky), (chunk->ext->reduction_cg_pw), pw,
            *(chunk->ext->device_queue));
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_cg_calc_ur(Chunk *chunk, Settings &settings, double alpha, double *rrn) {
  START_PROFILING(settings.kernel_profile);
  cg_calc_ur(chunk->x, chunk->y, settings.halo_depth, (chunk->u), (chunk->r), (chunk->p), (chunk->w), (chunk->ext->reduction_cg_rrn), alpha,
             rrn, *(chunk->ext->device_queue));
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_cg_calc_p(Chunk *chunk, Settings &settings, double beta) {
  START_PROFILING(settings.kernel_profile);
  cg_calc_p(chunk->x, chunk->y, settings.halo_depth, beta, (chunk->p), (chunk->r), *(chunk->ext->device_queue));
  STOP_PROFILING(settings.kernel_profile, __func__);
}
