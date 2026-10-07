#include "chunk.h"
#include "kokkos_shared.hpp"
#include "shared.h"

// Initialises p,r,u,w
void cg_init_u(const int x, const int y, const int coefficient, KView &p, KView &r, KView &u, KView &w, KView &density, KView &energy) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        const int kk = index % x;
        const int jj = index / x;

        p(index) = 0.0;
        r(index) = 0.0;
        u(index) = energy(index) * density(index);
        if (jj > 0 && jj < y - 1 && kk > 0 && kk < x - 1) {
          w(index) = (coefficient == CONDUCTIVITY) ? density(index) : 1.0 / density(index);
        }
      });
}

// Initialises kx,ky
void cg_init_k(const int x, const int y, const int halo_depth, KView &w, KView &kx, KView &ky, const double rx, const double ry) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        const int kk = index % x;
        const int jj = index / x;
        if (jj >= halo_depth && jj < y - 1 && kk >= halo_depth && kk < x - 1) {
          kx(index) = rx * (w(index - 1) + w(index)) / (2.0 * w(index - 1) * w(index));
          ky(index) = ry * (w(index - x) + w(index)) / (2.0 * w(index - x) * w(index));
        }
      });
}

// Initialises w,r,p and calculates rro
void cg_init_others(const int x, const int y, const int halo_depth, KView &kx, KView &ky, KView &p, KView &r, KView &u, KView &w,
                    double *rro) {
  double rro_sum = 0.0;
  Kokkos::parallel_reduce(
      x * y,
      KOKKOS_LAMBDA(const int index, double &rro_temp) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          const double smvp = tealeaf_SMVP(u);
          w(index) = smvp;
          r(index) = u(index) - w(index);
          p(index) = r(index);
          rro_temp += r(index) * p(index);
        }
      },
      rro_sum);
  *rro += rro_sum;
}

// Calculates the value for w
void cg_calc_w(const int x, const int y, const int halo_depth, KView &w, KView &p, KView &kx, KView &ky, double *pw) {

  double pw_sum = 0.0;
  Kokkos::parallel_reduce(
      x * y,
      KOKKOS_LAMBDA(const int &index, double &pw_temp) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          const double smvp = tealeaf_SMVP(p);
          w(index) = smvp;
          pw_temp += w(index) * p(index);
        }
      },
      pw_sum);
  *pw += pw_sum;
}

// Calculates the value of u and r
void cg_calc_ur(const int x, const int y, const int halo_depth, KView &u, KView &r, KView &p, KView &w, const double alpha, double *rrn) {
  double rrn_sum = 0.0;
  Kokkos::parallel_reduce(
      x * y,
      KOKKOS_LAMBDA(const int &index, double &rrn_temp) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          u(index) += alpha * p(index);
          r(index) -= alpha * w(index);
          rrn_temp += r(index) * r(index);
        }
      },
      rrn_sum);
  *rrn += rrn_sum;
}

// Calculates a value for p
void cg_calc_p(const int x, const int y, const int halo_depth, const double beta, KView &p, KView &r) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int &index) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          p(index) = beta * p(index) + r(index);
        }
      });
}

// CG solver kernels
void run_cg_init(Chunk *chunk, Settings &settings, double rx, double ry, double *rro) {
  START_PROFILING(settings.kernel_profile);
  cg_init_u(chunk->x, chunk->y, settings.coefficient, *chunk->p, *chunk->r, *chunk->u, *chunk->w, *chunk->density, *chunk->energy);
  cg_init_k(chunk->x, chunk->y, settings.halo_depth, *chunk->w, *chunk->kx, *chunk->ky, rx, ry);
  cg_init_others(chunk->x, chunk->y, settings.halo_depth, *chunk->kx, *chunk->ky, *chunk->p, *chunk->r, *chunk->u, *chunk->w, rro);
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_cg_calc_w(Chunk *chunk, Settings &settings, double *pw) {
  START_PROFILING(settings.kernel_profile);

  cg_calc_w(chunk->x, chunk->y, settings.halo_depth, *chunk->w, *chunk->p, *chunk->kx, *chunk->ky, pw);

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_cg_calc_ur(Chunk *chunk, Settings &settings, double alpha, double *rrn) {
  START_PROFILING(settings.kernel_profile);

  cg_calc_ur(chunk->x, chunk->y, settings.halo_depth, *chunk->u, *chunk->r, *chunk->p, *chunk->w, alpha, rrn);

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_cg_calc_p(Chunk *chunk, Settings &settings, double beta) {
  START_PROFILING(settings.kernel_profile);

  cg_calc_p(chunk->x, chunk->y, settings.halo_depth, beta, *chunk->p, *chunk->r);

  STOP_PROFILING(settings.kernel_profile, __func__);
}
