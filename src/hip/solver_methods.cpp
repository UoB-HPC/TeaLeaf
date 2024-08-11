#include "hip/hip_runtime.h"

#include "chunk.h"
#include "cuknl_shared.h"
#include "shared.h"

__global__ void sum_reduce(const int n, double *buffer) {
  __shared__ double buffer_shared[BLOCK_SIZE];

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  buffer_shared[threadIdx.x] = (gid < n) ? buffer[gid] : 0.0;

  reduce<double, BLOCK_SIZE / 2>::run(buffer_shared, buffer, SUM);
}

__global__ void field_summary(const int x_inner, const int y_inner, const int halo_depth, const double *volume, const double *density,
                              const double *energy0, const double *u, double *vol_out, double *mass_out, double *ie_out, double *temp_out) {
  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  const int lid = threadIdx.x;

  __shared__ double vol_shared[BLOCK_SIZE];
  __shared__ double mass_shared[BLOCK_SIZE];
  __shared__ double ie_shared[BLOCK_SIZE];
  __shared__ double temp_shared[BLOCK_SIZE];

  vol_shared[lid] = 0.0;
  mass_shared[lid] = 0.0;
  ie_shared[lid] = 0.0;
  temp_shared[lid] = 0.0;

  if (gid < x_inner * y_inner) {
    const int x = x_inner + 2 * halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner;
    const int off0 = halo_depth * (x + 1);
    const int index = off0 + col + row * x;

    double cell_vol = volume[index];
    double cell_mass = cell_vol * density[index];
    vol_shared[lid] = cell_vol;
    mass_shared[lid] = cell_mass;
    ie_shared[lid] = cell_mass * energy0[index];
    temp_shared[lid] = cell_mass * u[index];
  }

  __syncthreads();

#pragma unroll
  for (int ii = BLOCK_SIZE / 2; ii > 0; ii /= 2) {
    if (lid < ii) {
      vol_shared[lid] += vol_shared[lid + ii];
      mass_shared[lid] += mass_shared[lid + ii];
      ie_shared[lid] += ie_shared[lid + ii];
      temp_shared[lid] += temp_shared[lid + ii];
    }

    __syncthreads();
  }

  vol_out[blockIdx.x] = vol_shared[0];
  mass_out[blockIdx.x] = mass_shared[0];
  ie_out[blockIdx.x] = ie_shared[0];
  temp_out[blockIdx.x] = temp_shared[0];
}

// Store original energy state
__global__ void store_energy(int x_inner, int y_inner, const double *energy0, double *energy) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= x_inner * y_inner) return;

  energy[gid] = energy0[gid];
}

void sum_reduce_buffer(double *buffer, double *result, int len) {
  while (len > 1) {
    int num_blocks = ceil(len / (double)BLOCK_SIZE);
    sum_reduce<<<num_blocks, BLOCK_SIZE>>>(len, buffer);
    len = num_blocks;
  }
  if (auto r = hipMemcpy(result, buffer, sizeof(double), CLOVER_MEMCPY_KIND_D2H); r != hipSuccess) {
    die(__LINE__, __FILE__, "hhipMemcpy failed - return code %d (%s)\n", r, hipGetErrorName(r));
  }
  check_errors(__LINE__, __FILE__);
}

__global__ void copy_u(const int x_inner, const int y_inner, const int halo_depth, const double *src, double *dest) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= x_inner * y_inner) return;

  const int x = x_inner + 2 * halo_depth;
  const int col = gid % x_inner;
  const int row = gid / x_inner;
  const int off0 = halo_depth * (x + 1);
  const int index = off0 + col + row * x;

  dest[index] = src[index];
}

__global__ void calculate_residual(const int x_inner, const int y_inner, const int halo_depth, const double *u, const double *u0,
                                   const double *kx, const double *ky, double *r) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= x_inner * y_inner) return;

  const int x = x_inner + 2 * halo_depth;
  const int col = gid % x_inner;
  const int row = gid / x_inner;
  const int off0 = halo_depth * (x + 1);
  const int index = off0 + col + row * x;

  const double smvp = tealeaf_SMVP(u);
  r[index] = u0[index] - smvp;
}

__global__ void calculate_2norm(const int x_inner, const int y_inner, const int halo_depth, const double *src, double *norm) {
  __shared__ double norm_shared[BLOCK_SIZE];
  norm_shared[threadIdx.x] = 0.0;

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;

  if (gid >= x_inner * y_inner) return;

  const int x = x_inner + 2 * halo_depth;
  const int col = gid % x_inner;
  const int row = gid / x_inner;
  const int off0 = halo_depth * (x + 1);
  const int index = off0 + col + row * x;

  norm_shared[threadIdx.x] = src[index] * src[index];

  reduce<double, BLOCK_SIZE / 2>::run(norm_shared, norm, SUM);
}

__global__ void finalise(const int x_inner, const int y_inner, const int halo_depth, const double *density, const double *u,
                         double *energy) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= x_inner * y_inner) return;

  const int x = x_inner + 2 * halo_depth;
  const int col = gid % x_inner;
  const int row = gid / x_inner;
  const int off0 = halo_depth * (x + 1);
  const int index = off0 + col + row * x;

  energy[index] = u[index] / density[index];
}

void run_store_energy(Chunk *chunk, Settings &settings) {
  KERNELS_START(0);
  store_energy<<<num_blocks, BLOCK_SIZE>>>(x_inner, y_inner, chunk->energy0, chunk->energy);
  KERNELS_END();
}

void run_field_summary(Chunk *chunk, Settings &settings, double *vol, double *mass, double *ie, double *temp) {
  KERNELS_START(2 * settings.halo_depth);
  field_summary<<<num_blocks, BLOCK_SIZE>>>(x_inner, y_inner, settings.halo_depth, chunk->volume, chunk->density, chunk->energy0, chunk->u,
                                            chunk->ext->d_reduce_buffer, chunk->ext->d_reduce_buffer2, chunk->ext->d_reduce_buffer3,
                                            chunk->ext->d_reduce_buffer4);

  sum_reduce_buffer(chunk->ext->d_reduce_buffer, vol, num_blocks);
  sum_reduce_buffer(chunk->ext->d_reduce_buffer2, mass, num_blocks);
  sum_reduce_buffer(chunk->ext->d_reduce_buffer3, ie, num_blocks);
  sum_reduce_buffer(chunk->ext->d_reduce_buffer4, temp, num_blocks);
  KERNELS_END();
}

// Shared solver kernels
void run_copy_u(Chunk *chunk, Settings &settings) {
  KERNELS_START(2 * settings.halo_depth);
  copy_u<<<num_blocks, BLOCK_SIZE>>>(x_inner, y_inner, settings.halo_depth, chunk->u, chunk->u0);
  KERNELS_END();
}

void run_calculate_residual(Chunk *chunk, Settings &settings) {
  KERNELS_START(2 * settings.halo_depth);
  calculate_residual<<<num_blocks, BLOCK_SIZE>>>(x_inner, y_inner, settings.halo_depth, chunk->u, chunk->u0, chunk->kx, chunk->ky,
                                                 chunk->r);
  KERNELS_END();
}

void run_calculate_2norm(Chunk *chunk, Settings &settings, double *buffer, double *norm) {
  KERNELS_START(2 * settings.halo_depth);
  calculate_2norm<<<num_blocks, BLOCK_SIZE>>>(x_inner, y_inner, settings.halo_depth, buffer, chunk->ext->d_reduce_buffer);
  sum_reduce_buffer(chunk->ext->d_reduce_buffer, norm, num_blocks);
  KERNELS_END();
}

void run_finalise(Chunk *chunk, Settings &settings) {
  KERNELS_START(2 * settings.halo_depth);
  finalise<<<num_blocks, BLOCK_SIZE>>>(x_inner, y_inner, settings.halo_depth, chunk->density, chunk->u, chunk->energy);
  KERNELS_END();
}
