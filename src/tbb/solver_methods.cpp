#include "chunk.h"
#include "shared.h"
#include "tbb_shared.h"

/*
 *		SHARED SOLVER METHODS
 */

struct Summary {
  double vol;
  double mass;
  double ie;
  double temp;
  [[nodiscard]] constexpr Summary operator+(const Summary &that) const { //
    return {vol + that.vol, mass + that.mass, ie + that.ie, temp + that.temp};
  }
};

// The field summary kernel
void field_summary(const int x,           //
                   const int y,           //
                   const int halo_depth,  //
                   const double *volume,  //
                   const double *density, //
                   const double *energy0, //
                   const double *u,       //
                   double *volOut,        //
                   double *massOut,       //
                   double *ieOut,         //
                   double *tempOut) {

  Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
  auto summary = tbb::parallel_reduce(
      tbb::blocked_range{0, range.sizeXY()}, Summary{},
      [&](const auto &br, Summary acc) {
        for (int i = br.begin(); i < br.end(); ++i) {
          const int index = range.restore(i, x);
          const double cellVol = volume[index];
          const double cellMass = cellVol * density[index];
          acc = acc + Summary{.vol = cellVol, .mass = cellMass, .ie = cellMass * energy0[index], .temp = cellMass * u[index]};
        }
        return acc;
      },
      std::plus<>(), partitioner);

  *volOut += summary.vol;
  *ieOut += summary.ie;
  *tempOut += summary.temp;
  *massOut += summary.mass;
}

// Store original energy state
void store_energy(int x, int y, const double *energy0, double *energy) {
  tbb::parallel_for(0, x * y, [&](int ii) { energy[ii] = energy0[ii]; }, partitioner);
}

// Copies the current u into u0
void copy_u(const int x,          //
            const int y,          //
            const int halo_depth, //
            double *u0,           //
            const double *u) {

  tbb::parallel_for(
      halo_depth, y - halo_depth,
      [&](int jj) {
        for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
          const int index = kk + jj * x;
          u0[index] = u[index];
        }
      },
      partitioner);
}

// Calculates the current value of r
void calculate_residual(const int x,          //
                        const int y,          //
                        const int halo_depth, //
                        const double *u,      //
                        const double *u0,     //
                        double *r,            //
                        const double *kx,     //
                        const double *ky) {

  tbb::parallel_for(
      halo_depth, y - halo_depth,
      [&](int jj) {
        for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
          const int index = kk + jj * x;
          const double smvp = tealeaf_SMVP(u);
          r[index] = u0[index] - smvp;
        }
      },
      partitioner);
}

// Calculates the 2 norm of a given buffer
void calculate_2norm(const int x,          //
                     const int y,          //
                     const int halo_depth, //
                     const double *buffer, //
                     double *norm) {
  *norm += tbb::parallel_reduce(
      tbb::blocked_range{halo_depth, y - halo_depth}, 0.0,
      [&](const auto &br, double acc) {
        for (int i = br.begin(); i < br.end(); ++i) {
          for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
            const int index = kk + i * x;
            acc += buffer[index] * buffer[index];
          }
        }
        return acc;
      },
      std::plus<>(), partitioner);
}

// Finalises the solution
void finalise(const int x,           //
              const int y,           //
              const int halo_depth,  //
              double *energy,        //
              const double *density, //
              const double *u) {

  tbb::parallel_for(
      halo_depth, y - halo_depth,
      [&](int jj) {
        for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
          const int index = kk + jj * x;
          energy[index] = u[index] / density[index];
        }
      },
      partitioner);
}

void run_store_energy(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);
  store_energy(chunk->x, chunk->y, chunk->energy0, chunk->energy);
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_field_summary(Chunk *chunk, Settings &settings, double *vol, double *mass, double *ie, double *temp) {
  START_PROFILING(settings.kernel_profile);
  field_summary(chunk->x, chunk->y, settings.halo_depth, chunk->volume, chunk->density, chunk->energy0, chunk->u, vol, mass, ie, temp);
  STOP_PROFILING(settings.kernel_profile, __func__);
}

// Shared solver kernels
void run_copy_u(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);
  copy_u(chunk->x, chunk->y, settings.halo_depth, chunk->u0, chunk->u);
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_calculate_residual(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);
  calculate_residual(chunk->x, chunk->y, settings.halo_depth, chunk->u, chunk->u0, chunk->r, chunk->kx, chunk->ky);
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_calculate_2norm(Chunk *chunk, Settings &settings, double *buffer, double *norm) {
  START_PROFILING(settings.kernel_profile);
  calculate_2norm(chunk->x, chunk->y, settings.halo_depth, buffer, norm);
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_finalise(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);
  finalise(chunk->x, chunk->y, settings.halo_depth, chunk->energy, chunk->density, chunk->u);
  STOP_PROFILING(settings.kernel_profile, __func__);
}
