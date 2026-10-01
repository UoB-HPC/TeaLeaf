#include "chunk.h"
#include "shared.h"
#include "tbb_shared.h"
/*
 *		CHEBYSHEV SOLVER KERNEL
 */

// Calculates the new value for u.
void cheby_calc_u(const int x,          //
                  const int y,          //
                  const int halo_depth, //
                  double *u,            //
                  const double *p) {
  Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
  tbb::parallel_for(
      0, range.sizeXY(),
      [&](int i) {
        const int index = range.restore(i, x);
        u[index] += p[index];
      },
      partitioner);
}

// Initialises the Chebyshev solver
void cheby_init(const int x,          //
                const int y,          //
                const int halo_depth, //
                const double theta,   //
                double *u,            //
                const double *u0,     //
                double *p,            //
                double *r,            //
                double *w,            //
                const double *kx,     //
                const double *ky) {
  Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
  tbb::parallel_for(
      0, range.sizeXY(),
      [&](int i) {
        const int index = range.restore(i, x);
        const double smvp = tealeaf_SMVP(u);
        w[index] = smvp;
        r[index] = u0[index] - w[index];
        p[index] = r[index] / theta;
      },
      partitioner);

  cheby_calc_u(x, y, halo_depth, u, p);
}

// The main chebyshev iteration
void cheby_iterate(const int x,          //
                   const int y,          //
                   const int halo_depth, //
                   double alpha,         //
                   double beta,          //
                   double *u,            //
                   const double *u0,     //
                   double *p,            //
                   double *r,            //
                   double *w,            //
                   const double *kx,     //
                   const double *ky) {
  Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
  tbb::parallel_for(
      0, range.sizeXY(),
      [&](int i) {
        const int index = range.restore(i, x);
        const double smvp = tealeaf_SMVP(u);
        w[index] = smvp;
        r[index] = u0[index] - w[index];
        p[index] = alpha * p[index] + beta * r[index];
      },
      partitioner);

  cheby_calc_u(x, y, halo_depth, u, p);
}

// Chebyshev solver kernels
void run_cheby_init(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);
  cheby_init(chunk->x, chunk->y, settings.halo_depth, chunk->theta, chunk->u, chunk->u0, chunk->p, chunk->r, chunk->w, chunk->kx,
             chunk->ky);
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_cheby_iterate(Chunk *chunk, Settings &settings, double alpha, double beta) {
  START_PROFILING(settings.kernel_profile);
  cheby_iterate(chunk->x, chunk->y, settings.halo_depth, alpha, beta, chunk->u, chunk->u0, chunk->p, chunk->r, chunk->w, chunk->kx,
                chunk->ky);
  STOP_PROFILING(settings.kernel_profile, __func__);
}
