#include "chunk.h"
#include "kokkos_shared.hpp"
#include "shared.h"

// Copies energy0 into energy1.
void store_energy(const int x, const int y, KView &energy, KView &energy0) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) { energy(index) = energy0(index); });
}

// Copies the inner u into u0.
void copy_u(const int x, const int y, const int halo_depth, KView &u, KView &u0) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          u0(index) = u(index);
        }
      });
}

// Calculates the residual r.
void calculate_residual(const int x, const int y, const int halo_depth, KView &u, KView &u0, KView &r, KView &kx, KView &ky) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          const double smvp = tealeaf_SMVP(u);
          r(index) = u0(index) - smvp;
        }
      });
}

// Calculates the 2 norm of the provided buffer.
void calculate_2norm(const int x, const int y, const int halo_depth, KView &buffer, double *norm) {
  double norm_sum = 0.0;
  Kokkos::parallel_reduce(
      x * y,
      KOKKOS_LAMBDA(const int index, double &norm_temp) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          norm_temp += buffer(index) * buffer(index);
        }
      },
      norm_sum);
  *norm += norm_sum;
}

// Finalises the energy field.
void finalise(const int x, const int y, const int halo_depth, KView &u, KView &density, KView &energy) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          energy(index) = u(index) / density(index);
        }
      });
}

void run_store_energy(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);
  store_energy(chunk->x, chunk->y, *chunk->energy, *chunk->energy0);
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_field_summary(Chunk *chunk, Settings &settings, double *vol, double *mass, double *ie, double *temp) {
  START_PROFILING(settings.kernel_profile);
  int x = chunk->x;
  int y = chunk->y;
  int halo_depth = settings.halo_depth;
  auto &u = *chunk->u;
  auto &density = *chunk->density;
  auto &energy0 = *chunk->energy0;
  auto &volume = *chunk->volume;

  double vol_sum = 0.0, mass_sum = 0.0, ie_sum = 0.0, temp_sum = 0.0;
  Kokkos::parallel_reduce(
      chunk->x * chunk->y,
      KOKKOS_LAMBDA(const int index, double &vol, double &mass, double &ie, double &temp) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          const double cellVol = volume[index];
          const double cellMass = cellVol * density[index];
          vol += cellVol;
          mass += cellMass;
          ie += cellMass * energy0[index];
          temp += cellMass * u[index];
        }
      },
      vol_sum, mass_sum, ie_sum, temp_sum);
  *vol += vol_sum;
  *mass += mass_sum;
  *ie += ie_sum;
  *temp += temp_sum;
  STOP_PROFILING(settings.kernel_profile, __func__);
}

// Shared solver kernels
void run_copy_u(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);

  copy_u(chunk->x, chunk->y, settings.halo_depth, *chunk->u, *chunk->u0);

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_calculate_residual(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);

  calculate_residual(chunk->x, chunk->y, settings.halo_depth, *chunk->u, *chunk->u0, *chunk->r, *chunk->kx, *chunk->ky);

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_calculate_2norm(Chunk *chunk, Settings &settings, FieldBufferType buffer, double *norm) {
  START_PROFILING(settings.kernel_profile);

  calculate_2norm(chunk->x, chunk->y, settings.halo_depth, *buffer, norm);

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_finalise(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);

  finalise(chunk->x, chunk->y, settings.halo_depth, *chunk->u, *chunk->density, *chunk->energy);

  STOP_PROFILING(settings.kernel_profile, __func__);
}
