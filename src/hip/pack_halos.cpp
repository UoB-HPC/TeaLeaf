#include "hip/hip_runtime.h"

#include "chunk.h"
#include "comms.h"
#include "cuknl_shared.h"

__global__ void pack_left(const int x, const int y, const int depth, const int halo_depth, const double *field, double *buffer,
                          int buffer_offset) {
  const int y_inner = y - 2 * halo_depth;

  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= y_inner * depth) return;

  const int lines = gid / depth;
  const int offset = halo_depth + lines * (x - depth);
  buffer[gid + buffer_offset] = field[offset + gid];
}

__global__ void pack_right(const int x, const int y, const int depth, const int halo_depth, const double *field, double *buffer,
                           int buffer_offset) {
  const int y_inner = y - 2 * halo_depth;

  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= y_inner * depth) return;

  const int lines = gid / depth;
  const int offset = x - halo_depth - depth + lines * (x - depth);
  buffer[gid + buffer_offset] = field[offset + gid];
}

__global__ void unpack_left(const int x, const int y, const int depth, const int halo_depth, double *field, const double *buffer,
                            int buffer_offset) {
  const int y_inner = y - 2 * halo_depth;

  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= y_inner * depth) return;

  const int lines = gid / depth;
  const int offset = halo_depth - depth + lines * (x - depth);
  field[offset + gid] = buffer[gid + buffer_offset];
}

__global__ void unpack_right(const int x, const int y, const int depth, const int halo_depth, double *field, const double *buffer,
                             int buffer_offset) {
  const int y_inner = y - 2 * halo_depth;

  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= y_inner * depth) return;

  const int lines = gid / depth;
  const int offset = x - halo_depth + lines * (x - depth);
  field[offset + gid] = buffer[gid + buffer_offset];
}

__global__ void pack_top(const int x, const int y, const int depth, const int halo_depth, const double *field, double *buffer,
                         int buffer_offset) {
  const int x_inner = x - 2 * halo_depth;

  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= x_inner * depth) return;

  const int lines = gid / x_inner;
  const int offset = x * (y - halo_depth - depth) + lines * 2 * halo_depth;
  buffer[gid + buffer_offset] = field[offset + gid];
}

__global__ void pack_bottom(const int x, const int y, const int depth, const int halo_depth, const double *field, double *buffer,
                            int buffer_offset) {
  const int x_inner = x - 2 * halo_depth;

  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= x_inner * depth) return;

  const int lines = gid / x_inner;
  const int offset = x * halo_depth + lines * 2 * halo_depth;
  buffer[gid + buffer_offset] = field[offset + gid];
}

__global__ void unpack_top(const int x, const int y, const int depth, const int halo_depth, double *field, const double *buffer,
                           int buffer_offset) {
  const int x_inner = x - 2 * halo_depth;

  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= x_inner * depth) return;

  const int lines = gid / x_inner;
  const int offset = x * (y - halo_depth) + lines * 2 * halo_depth;
  field[offset + gid] = buffer[gid + buffer_offset];
}

__global__ void unpack_bottom(const int x, const int y, const int depth, const int halo_depth, double *field, const double *buffer,
                              int buffer_offset) {
  const int x_inner = x - 2 * halo_depth;

  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= x_inner * depth) return;

  const int lines = gid / x_inner;
  const int offset = x * (halo_depth - depth) + lines * 2 * halo_depth;
  field[offset + gid] = buffer[gid + buffer_offset];
}

// Either packs or unpacks data from/to buffers.
void pack_or_unpack(Chunk *chunk, Settings &settings, int depth, int face, bool pack, double *field, double *buffer, int offset) {
  const int x_inner = chunk->x - 2 * settings.halo_depth;
  const int y_inner = chunk->y - 2 * settings.halo_depth;
  switch (face) {
    case CHUNK_LEFT: {
      int num_blocks = std::ceil((y_inner * depth) / double(BLOCK_SIZE));
      if (pack) pack_left<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      else
        unpack_left<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      break;
    }
    case CHUNK_RIGHT: {
      int num_blocks = std::ceil((y_inner * depth) / double(BLOCK_SIZE));
      if (pack) pack_right<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      else
        unpack_right<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      break;
    }
    case CHUNK_TOP: {
      int num_blocks = std::ceil((x_inner * depth) / double(BLOCK_SIZE));
      if (pack) pack_top<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      else
        unpack_top<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      break;
    }
    case CHUNK_BOTTOM: {
      int num_blocks = std::ceil((x_inner * depth) / double(BLOCK_SIZE));
      if (pack) pack_bottom<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      else
        unpack_bottom<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      break;
    }
    default: die(__LINE__, __FILE__, "Incorrect face provided: %d.\n", face);
  }
}

void run_pack_or_unpack(Chunk *chunk, Settings &settings, int depth, int face, bool pack, FieldBufferType field, FieldBufferType buffer,
                        int offset) {
  START_PROFILING(settings.kernel_profile);
  pack_or_unpack(chunk, settings, depth, face, pack, field, buffer, offset);
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_send_recv_halo(Chunk *, Settings &settings,                                                            //
                        FieldBufferType src_send_buffer, FieldBufferType src_recv_buffer,                       //
                        StagingBufferType dest_staging_send_buffer, StagingBufferType dest_staging_recv_buffer, //
                        int buffer_len, int neighbour,                                                          //
                        int send_tag, int recv_tag,                                                             //
                        MPI_Request *send_request, MPI_Request *recv_request) {
  START_PROFILING(settings.kernel_profile);
  if (settings.staging_buffer) {

    if (auto result = hipMemcpy(dest_staging_send_buffer, src_send_buffer, buffer_len * sizeof(double), CLOVER_MEMCPY_KIND_D2H);
        result != hipSuccess) {
      die(__LINE__, __FILE__, "hipMemcpy failed - return code %d (%s)\n", result, hipGetErrorName(result));
    }
    if (auto result = hipMemcpy(dest_staging_recv_buffer, src_recv_buffer, buffer_len * sizeof(double), CLOVER_MEMCPY_KIND_D2H);
        result != hipSuccess) {
      die(__LINE__, __FILE__, "hipMemcpy failed - return code %d (%s)\n", result, hipGetErrorName(result));
    }
    send_recv_message(settings,                                           //
                      dest_staging_send_buffer, dest_staging_recv_buffer, //
                      buffer_len, neighbour, send_tag, recv_tag, send_request, recv_request);
  } else {
    check_errors(__LINE__, __FILE__);
    send_recv_message(settings, src_send_buffer, src_recv_buffer, buffer_len, neighbour, send_tag, recv_tag, send_request, recv_request);
  }
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_before_waitall_halo(Chunk *, Settings &) {}

void run_restore_recv_halo(Chunk *, Settings &settings, //
                           FieldBufferType dest_recv_buffer, StagingBufferType src_staging_recv_buffer, int buffer_len) {
  START_PROFILING(settings.kernel_profile);
  if (settings.staging_buffer) {
    if (auto result = hipMemcpy(dest_recv_buffer, src_staging_recv_buffer, buffer_len * sizeof(double), CLOVER_MEMCPY_KIND_H2D);
        result != hipSuccess) {
      die(__LINE__, __FILE__, "hipMemcpy failed - return code %d (%s)\n", result, hipGetErrorName(result));
    }
  }
  STOP_PROFILING(settings.kernel_profile, __func__);
}
