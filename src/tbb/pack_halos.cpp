#include "chunk.h"
#include "comms.h"
#include "shared.h"
#include "tbb_shared.h"

// Packs top data into buffer.
void pack_top(const int x, const int y, const int depth, const int halo_depth, const double *field, double *buffer, int buffer_offset) {
  tbb::parallel_for(
      0, x * depth,
      [&](const int index) {
        const int offset = x * (y - halo_depth - depth);
        buffer[index + buffer_offset] = field[offset + index];
      },
      partitioner);
}

// Packs bottom data into buffer.
void pack_bottom(const int x, const int y, const int depth, const int halo_depth, const double *field, double *buffer, int buffer_offset) {
  tbb::parallel_for(
      0, x * depth,
      [&](const int index) {
        const int offset = x * halo_depth;
        buffer[index + buffer_offset] = field[offset + index];
      },
      partitioner);
}

// Packs left data into buffer.
void pack_left(const int x, const int y, const int depth, const int halo_depth, const double *field, double *buffer, int buffer_offset) {
  tbb::parallel_for(
      0, y * depth,
      [&](const int index) {
        const int lines = index / depth;
        const int offset = halo_depth + lines * (x - depth);
        buffer[index + buffer_offset] = field[offset + index];
      },
      partitioner);
}

// Packs right data into buffer.
void pack_right(const int x, const int y, const int depth, const int halo_depth, const double *field, double *buffer, int buffer_offset) {
  tbb::parallel_for(
      0, y * depth,
      [&](const int index) {
        const int lines = index / depth;
        const int offset = x - halo_depth - depth + lines * (x - depth);
        buffer[index + buffer_offset] = field[offset + index];
      },
      partitioner);
}

// Unpacks top data from buffer.
void unpack_top(const int x, const int y, const int depth, const int halo_depth, double *field, const double *buffer, int buffer_offset) {
  tbb::parallel_for(
      0, x * depth,
      [&](const int index) {
        const int offset = x * (y - halo_depth);
        field[offset + index] = buffer[index + buffer_offset];
      },
      partitioner);
}

// Unpacks bottom data from buffer.
void unpack_bottom(const int x, const int y, const int depth, const int halo_depth, double *field, const double *buffer,
                   int buffer_offset) {
  tbb::parallel_for(
      0, x * depth,
      [&](const int index) {
        const int offset = x * (halo_depth - depth);
        field[offset + index] = buffer[index + buffer_offset];
      },
      partitioner);
}

// Unpacks left data from buffer.
void unpack_left(const int x, const int y, const int depth, const int halo_depth, double *field, const double *buffer, int buffer_offset) {
  tbb::parallel_for(
      0, y * depth,
      [&](const int index) {
        const int lines = index / depth;
        const int offset = halo_depth - depth + lines * (x - depth);
        field[offset + index] = buffer[index + buffer_offset];
      },
      partitioner);
}

// Unpacks right data from buffer.
void unpack_right(const int x, const int y, const int depth, const int halo_depth, double *field, const double *buffer, int buffer_offset) {
  tbb::parallel_for(
      0, y * depth,
      [&](const int index) {
        const int lines = index / depth;
        const int offset = x - halo_depth + lines * (x - depth);
        field[offset + index] = buffer[index + buffer_offset];
      },
      partitioner);
}

// Either packs or unpacks data from/to buffers.
void run_pack_or_unpack(Chunk *chunk, Settings &settings, int depth, int face, bool pack, FieldBufferType field, FieldBufferType buffer,
                        int offset) {
  START_PROFILING(settings.kernel_profile);
  switch (face) {
    case CHUNK_LEFT:
      if (pack) pack_left(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      else
        unpack_left(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      break;
    case CHUNK_RIGHT:
      if (pack) pack_right(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      else
        unpack_right(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      break;
    case CHUNK_TOP:
      if (pack) pack_top(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      else
        unpack_top(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      break;
    case CHUNK_BOTTOM:
      if (pack) pack_bottom(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      else
        unpack_bottom(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset);
      break;
    default: die(__LINE__, __FILE__, "Incorrect face provided: %d.\n", face);
  }
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_send_recv_halo(Chunk *, Settings &settings, FieldBufferType src_send_buffer, FieldBufferType src_recv_buffer, StagingBufferType,
                        StagingBufferType, int buffer_len, int neighbour, int send_tag, int recv_tag, MPI_Request *send_request,
                        MPI_Request *recv_request) {
  // Host/USM model, no-op for staging buffers here
  send_recv_message(settings, src_send_buffer, src_recv_buffer, buffer_len, neighbour, send_tag, recv_tag, send_request, recv_request);
}

void run_before_waitall_halo(Chunk *, Settings &) {}
void run_restore_recv_halo(Chunk *, Settings &, FieldBufferType, StagingBufferType, int) {
  // Host/USM model, no-op for staging buffers here
}
