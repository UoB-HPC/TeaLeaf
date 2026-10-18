#include "chunk.h"
#include "comms.h"
#include "shared.h"
#include "sycl_shared.hpp"

using namespace sycl;

// Packs the top halo buffer(s)
void pack_top(const int x,          //
              const int y,          //
              const int depth,      //
              const int halo_depth, //
              SyclBuffer &field,    //
              SyclBuffer &buffer,   //
              int buffer_offset, queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class pack_top>(range<1>(x * depth), [=](id<1> idx) {
          const int offset = x * (y - halo_depth - depth);
          buffer[idx[0] + buffer_offset] = field[offset + idx[0]];
        });
      })
      .wait_and_throw();
}

// Packs the bottom halo buffer(s)
void pack_bottom(const int x,          //
                 const int y,          //
                 const int depth,      //
                 const int halo_depth, //
                 SyclBuffer &field,    //
                 SyclBuffer &buffer,   //
                 int buffer_offset, queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class pack_bottom>(range<1>(x * depth), [=](id<1> idx) {
          const int offset = x * halo_depth;
          buffer[idx[0] + buffer_offset] = field[offset + idx[0]];
        });
      })
      .wait_and_throw();
}

// Packs the left halo buffer(s)
void pack_left(const int x,          //
               const int y,          //
               const int depth,      //
               const int halo_depth, //
               SyclBuffer &field,    //
               SyclBuffer &buffer,   //
               int buffer_offset, queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class pack_left>(range<1>(y * depth), [=](id<1> idx) {
          const auto lines = idx[0] / depth;
          const auto offset = halo_depth + lines * (x - depth);
          buffer[idx[0] + buffer_offset] = field[offset + idx[0]];
        });
      })
      .wait_and_throw();
}

// Packs the right halo buffer(s)
void pack_right(const int x,          //
                const int y,          //
                const int depth,      //
                const int halo_depth, //
                SyclBuffer &field,    //
                SyclBuffer &buffer,   //
                int buffer_offset, queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class pack_right>(range<1>(y * depth), [=](id<1> idx) {
          const auto lines = idx[0] / depth;
          const auto offset = x - halo_depth - depth + lines * (x - depth);
          buffer[idx[0] + buffer_offset] = field[offset + idx[0]];
        });
      })
      .wait_and_throw();
}

// Unpacks the top halo buffer(s)
void unpack_top(const int x,          //
                const int y,          //
                const int depth,      //
                const int halo_depth, //
                SyclBuffer &field,    //
                SyclBuffer &buffer,   //
                int buffer_offset, queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class unpack_top>(range<1>(x * depth), [=](id<1> idx) {
          const int offset = x * (y - halo_depth);
          field[offset + idx[0]] = buffer[idx[0] + buffer_offset];
        });
      })
      .wait_and_throw();
}

// Unpacks the bottom halo buffer(s)
void unpack_bottom(const int x,          //
                   const int y,          //
                   const int depth,      //
                   const int halo_depth, //
                   SyclBuffer &field,    //
                   SyclBuffer &buffer,   //
                   int buffer_offset, queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class unpack_bottom>(range<1>(x * depth), [=](id<1> idx) {
          const int offset = x * (halo_depth - depth);
          field[offset + idx[0]] = buffer[idx[0] + buffer_offset];
        });
      })
      .wait_and_throw();
}

// Unpacks the left halo buffer(s)
void unpack_left(const int x,          //
                 const int y,          //
                 const int depth,      //
                 const int halo_depth, //
                 SyclBuffer &field,    //
                 SyclBuffer &buffer,   //
                 int buffer_offset, queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class unpack_left>(range<1>(y * depth), [=](id<1> idx) {
          const auto lines = idx[0] / depth;
          const auto offset = halo_depth - depth + lines * (x - depth);
          field[offset + idx[0]] = buffer[idx[0] + buffer_offset];
        });
      })
      .wait_and_throw();
}

// Unpacks the right halo buffer(s)
void unpack_right(const int x,          //
                  const int y,          //
                  const int depth,      //
                  const int halo_depth, //
                  SyclBuffer &field,    //
                  SyclBuffer &buffer,   //
                  int buffer_offset, queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class unpack_right>(range<1>(y * depth), [=](id<1> idx) {
          const auto lines = idx[0] / depth;
          const auto offset = x - halo_depth + lines * (x - depth);
          field[offset + idx[0]] = buffer[idx[0] + buffer_offset];
        });
      })
      .wait_and_throw();
}

void run_pack_or_unpack(Chunk *chunk, Settings &settings, int depth, int face, bool pack, FieldBufferType field, FieldBufferType buffer,
                        int offset) {
  START_PROFILING(settings.kernel_profile);
  switch (face) {
    case CHUNK_LEFT:
      if (pack) pack_left(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset, *chunk->ext->device_queue);
      else
        unpack_left(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset, *chunk->ext->device_queue);
      break;
    case CHUNK_RIGHT:
      if (pack) pack_right(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset, *chunk->ext->device_queue);
      else
        unpack_right(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset, *chunk->ext->device_queue);
      break;
    case CHUNK_TOP:
      if (pack) pack_top(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset, *chunk->ext->device_queue);
      else
        unpack_top(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset, *chunk->ext->device_queue);
      break;
    case CHUNK_BOTTOM:
      if (pack) pack_bottom(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset, *chunk->ext->device_queue);
      else
        unpack_bottom(chunk->x, chunk->y, depth, settings.halo_depth, field, buffer, offset, *chunk->ext->device_queue);
      break;
    default: die(__LINE__, __FILE__, "Incorrect face provided: %d.\n", face);
  }
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_send_recv_halo(Chunk *chunk, Settings &settings,                                 //
                        FieldBufferType src_send_buffer, FieldBufferType src_recv_buffer, //
                        StagingBufferType, StagingBufferType,                             //
                        int buffer_len, int neighbour,                                    //
                        int send_tag, int recv_tag,                                       //
                        MPI_Request *send_request, MPI_Request *recv_request) {

#ifdef USE_HOSTTASK
  chunk->ext->device_queue->submit([&](sycl::handler &h) {
    h.host_task([=, &settings]() {
      send_recv_message(settings,        //
                        src_send_buffer, //
                        src_recv_buffer, //
                        buffer_len, neighbour, send_tag, recv_tag, send_request, recv_request);
    });
  });
#else
  chunk->ext->device_queue->wait_and_throw();
  send_recv_message(settings,        //
                    src_send_buffer, //
                    src_recv_buffer, //
                    buffer_len, neighbour, send_tag, recv_tag, send_request, recv_request);
#endif
}

void run_before_waitall_halo(Chunk *chunk, Settings &) { chunk->ext->device_queue->wait_and_throw(); }
void run_restore_recv_halo(Chunk *, Settings &, FieldBufferType, StagingBufferType, int) {}
