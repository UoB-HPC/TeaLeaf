#include "chunk.h"
#include "comms.h"
#include "shared.h"

// Packs left data into buffer.
void pack_left(const int x, const int y, const int depth, const int halo_depth, const double *field, double *buffer, int offset,
               bool is_offload) {
#ifdef OMP_TARGET
  #pragma omp target teams distribute parallel for simd collapse(2) if (is_offload)
#else
  #pragma omp parallel for
#endif
  for (int jj = halo_depth; jj < y - halo_depth; ++jj) {
    for (int kk = halo_depth; kk < halo_depth + depth; ++kk) {
      int bufIndex = (kk - halo_depth) + (jj - halo_depth) * depth;
      buffer[bufIndex + offset] = field[jj * x + kk];
    }
  }
}

// Packs right data into buffer.
void pack_right(const int x, const int y, const int depth, const int halo_depth, const double *field, double *buffer, int offset,
                bool is_offload) {
#ifdef OMP_TARGET
  #pragma omp target teams distribute parallel for simd collapse(2) if (is_offload)
#else
  #pragma omp parallel for
#endif
  for (int jj = halo_depth; jj < y - halo_depth; ++jj) {
    for (int kk = x - halo_depth - depth; kk < x - halo_depth; ++kk) {
      int bufIndex = (kk - (x - halo_depth - depth)) + (jj - halo_depth) * depth;
      buffer[bufIndex + offset] = field[jj * x + kk];
    }
  }
}

// Packs top data into buffer.
void pack_top(const int x, const int y, const int depth, const int halo_depth, const double *field, double *buffer, int offset,
              bool is_offload) {
  const int x_inner = x - 2 * halo_depth;

#ifdef OMP_TARGET
  #pragma omp target teams distribute parallel for simd collapse(2) if (is_offload) // map(from : buffer[ : depth * x_inner])
#else
  #pragma omp parallel for
#endif
  for (int jj = y - halo_depth - depth; jj < y - halo_depth; ++jj) {
    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
      int bufIndex = (kk - halo_depth) + (jj - (y - halo_depth - depth)) * x_inner;
      buffer[bufIndex + offset] = field[jj * x + kk];
    }
  }
}

// Packs bottom data into buffer.
void pack_bottom(const int x, const int y, const int depth, const int halo_depth, const double *field, double *buffer, int offset,
                 bool is_offload) {
  const int x_inner = x - 2 * halo_depth;

#ifdef OMP_TARGET
  #pragma omp target teams distribute parallel for simd collapse(2) if (is_offload) // map(from : buffer[ : depth * x_inner])
#else
  #pragma omp parallel for
#endif
  for (int jj = halo_depth; jj < halo_depth + depth; ++jj) {
    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
      int bufIndex = (kk - halo_depth) + (jj - halo_depth) * x_inner;
      buffer[bufIndex + offset] = field[jj * x + kk];
    }
  }
}

// Unpacks left data from buffer.
void unpack_left(const int x, const int y, const int depth, const int halo_depth, double *field, const double *buffer, int offset,
                 bool is_offload) {
#ifdef OMP_TARGET
  #pragma omp target teams distribute parallel for simd collapse(2) if (is_offload)
#else
  #pragma omp parallel for
#endif
  for (int jj = halo_depth; jj < y - halo_depth; ++jj) {
    for (int kk = halo_depth - depth; kk < halo_depth; ++kk) {
      int bufIndex = (kk - (halo_depth - depth)) + (jj - halo_depth) * depth;
      field[jj * x + kk] = buffer[bufIndex + offset];
    }
  }
}

// Unpacks right data from buffer.
void unpack_right(const int x, const int y, const int depth, const int halo_depth, double *field, const double *buffer, int offset,
                  bool is_offload) {
#ifdef OMP_TARGET
  #pragma omp target teams distribute parallel for simd collapse(2) if (is_offload)
#else
  #pragma omp parallel for
#endif
  for (int jj = halo_depth; jj < y - halo_depth; ++jj) {
    for (int kk = x - halo_depth; kk < x - halo_depth + depth; ++kk) {
      int bufIndex = (kk - (x - halo_depth)) + (jj - halo_depth) * depth;
      field[jj * x + kk] = buffer[bufIndex + offset];
    }
  }
}

// Unpacks top data from buffer.
void unpack_top(const int x, const int y, const int depth, const int halo_depth, double *field, const double *buffer, int offset,
                bool is_offload) {
  const int x_inner = x - 2 * halo_depth;
#ifdef OMP_TARGET
  #pragma omp target teams distribute parallel for simd collapse(2) if (is_offload) // map(to : buffer[ : depth * x_inner])
#else
  #pragma omp parallel for
#endif
  for (int jj = y - halo_depth; jj < y - halo_depth + depth; ++jj) {
    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
      int bufIndex = (kk - halo_depth) + (jj - (y - halo_depth)) * x_inner;
      field[jj * x + kk] = buffer[bufIndex + offset];
    }
  }
}

// Unpacks bottom data from buffer.
void unpack_bottom(const int x, const int y, const int depth, const int halo_depth, double *field, const double *buffer, int offset,
                   bool is_offload) {
  const int x_inner = x - 2 * halo_depth;
#ifdef OMP_TARGET
  #pragma omp target teams distribute parallel for simd collapse(2) if (is_offload) // map(to : buffer[ : depth * x_inner])
#else
  #pragma omp parallel for
#endif
  for (int jj = halo_depth - depth; jj < halo_depth; ++jj) {
    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
      int bufIndex = (kk - halo_depth) + (jj - (halo_depth - depth)) * x_inner;
      field[jj * x + kk] = buffer[bufIndex + offset];
    }
  }
}

// Either packs or unpacks data from/to buffers.
void pack_or_unpack(const int x, const int y, const int depth, const int halo_depth, const int face, bool pack, double *field,
                    double *buffer, int offset, bool is_offload) {
  switch (face) {
    case CHUNK_LEFT:
      if (pack) pack_left(x, y, depth, halo_depth, field, buffer, offset, is_offload);
      else
        unpack_left(x, y, depth, halo_depth, field, buffer, offset, is_offload);
      break;
    case CHUNK_RIGHT:
      if (pack) pack_right(x, y, depth, halo_depth, field, buffer, offset, is_offload);
      else
        unpack_right(x, y, depth, halo_depth, field, buffer, offset, is_offload);
      break;
    case CHUNK_TOP:
      if (pack) pack_top(x, y, depth, halo_depth, field, buffer, offset, is_offload);
      else
        unpack_top(x, y, depth, halo_depth, field, buffer, offset, is_offload);
      break;
    case CHUNK_BOTTOM:
      if (pack) pack_bottom(x, y, depth, halo_depth, field, buffer, offset, is_offload);
      else
        unpack_bottom(x, y, depth, halo_depth, field, buffer, offset, is_offload);
      break;
    default: die(__LINE__, __FILE__, "Incorrect face provided: %d.\n", face);
  }
}

void run_pack_or_unpack(Chunk *chunk, Settings &settings, int depth, int face, bool pack, FieldBufferType field, FieldBufferType buffer,
                        int offset) {
  START_PROFILING(settings.kernel_profile);
  pack_or_unpack(chunk->x, chunk->y, depth, settings.halo_depth, face, pack, field, buffer, offset, settings.is_offload);
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_send_recv_halo(Chunk *chunk, Settings &settings,                                 //
                        FieldBufferType src_send_buffer, FieldBufferType src_recv_buffer, //
                        StagingBufferType, StagingBufferType,                             //
                        int buffer_len, int neighbour,                                    //
                        int send_tag, int recv_tag,                                       //
                        MPI_Request *send_request, MPI_Request *recv_request) {
  START_PROFILING(settings.kernel_profile);
  // We can't shorten this boolean expression thanks to a bug AOMP: https://github.com/ROCm-Developer-Tools/aomp/issues/601
#ifdef OMP_TARGET
  if (!settings.is_offload) {
    send_recv_message(settings, src_send_buffer, src_recv_buffer, buffer_len, neighbour, send_tag, recv_tag, send_request, recv_request);
  } else {
  #pragma omp target update if (settings.staging_buffer) from(src_send_buffer[ : buffer_len])
  #pragma omp target data if (!settings.staging_buffer) use_device_ptr(src_send_buffer, src_recv_buffer)
    send_recv_message(settings, src_send_buffer, src_recv_buffer, buffer_len, neighbour, send_tag, recv_tag, send_request, recv_request);
  }
#else
  send_recv_message(settings, src_send_buffer, src_recv_buffer, buffer_len, neighbour, send_tag, recv_tag, send_request, recv_request);
#endif

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_before_waitall_halo(Chunk *, Settings &) {}

void run_restore_recv_halo(Chunk *, Settings &settings, //
                           FieldBufferType dest_recv_buffer, StagingBufferType, int buffer_len) {
  START_PROFILING(settings.kernel_profile);
#ifdef OMP_TARGET
  #pragma omp target update if (settings.staging_buffer) to(dest_recv_buffer[ : buffer_len])
  {}
#endif
  STOP_PROFILING(settings.kernel_profile, __func__);
}
