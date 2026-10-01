#include "chunk.h"
#include "shared.h"
#include "tbb_shared.h"
/*
 *		PPCG SOLVER KERNEL
 */

// Initialises the PPCG solver
void ppcg_init(const int x,          //
               const int y,          //
               const int halo_depth, //
               double theta,         //
               const double *r,      //
               double *sd) {
  Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
  tbb::parallel_for(
      0, range.sizeXY(),
      [&](int i) {
        const int index = range.restore(i, x);
        sd[index] = r[index] / theta;
      },
      partitioner);
}

// The PPCG inner iteration
void ppcg_inner_iteration(const int x,          //
                          const int y,          //
                          const int halo_depth, //
                          double alpha,         //
                          double beta,          //
                          double *u,            //
                          double *r,            //
                          const double *kx,     //
                          const double *ky,     //
                          double *sd) {

  Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
  tbb::parallel_for(
      0, range.sizeXY(),
      [&](int i) {
        const int index = range.restore(i, x);
        const double smvp = tealeaf_SMVP(sd);
        r[index] -= smvp;
        u[index] += sd[index];
      },
      partitioner);

  tbb::parallel_for(
      0, range.sizeXY(),
      [&](int i) {
        const int index = range.restore(i, x);
        sd[index] = alpha * sd[index] + beta * r[index];
      },
      partitioner);
}

// PPCG solver kernels
void run_ppcg_init(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);
  ppcg_init(chunk->x, chunk->y, settings.halo_depth, chunk->theta, chunk->r, chunk->sd);
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_ppcg_inner_iteration(Chunk *chunk, Settings &settings, double alpha, double beta) {
  START_PROFILING(settings.kernel_profile);
  ppcg_inner_iteration(chunk->x, chunk->y, settings.halo_depth, alpha, beta, chunk->u, chunk->r, chunk->kx, chunk->ky, chunk->sd);
  STOP_PROFILING(settings.kernel_profile, __func__);
}
