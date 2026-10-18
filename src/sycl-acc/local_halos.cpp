#include "chunk.h"
#include "shared.h"
#include "sycl_shared.hpp"

using namespace sycl;

// Updates the local left halo region(s)
void update_left(const int x,            //
                 const int y,            //
                 const int halo_depth,   //
                 const int depth,        //
                 SyclBuffer &bufferBuff, //
                 queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::read_write>(h);
    h.parallel_for<class update_left>(range<1>(y * depth), [=](id<1> idx) {
      const auto flip = idx[0] % depth;
      const auto lines = idx[0] / depth;
      const auto offset = lines * (x - depth);
      const auto to_index = offset + halo_depth - depth + idx[0];
      const auto from_index = to_index + 2 * (depth - flip) - 1;
      buffer[to_index] = buffer[from_index];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Updates the local right halo region(s)
void update_right(const int x,            //
                  const int y,            //
                  const int halo_depth,   //
                  const int depth,        //
                  SyclBuffer &bufferBuff, //
                  queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::read_write>(h);
    h.parallel_for<class update_right>(range<1>(y * depth), [=](id<1> idx) {
      const auto flip = idx[0] % depth;
      const auto lines = idx[0] / depth;
      const auto offset = x - halo_depth + lines * (x - depth);
      const auto to_index = offset + idx[0];
      const auto from_index = to_index - (1 + flip * 2);
      buffer[to_index] = buffer[from_index];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Updates the local top halo region(s)
void update_top(const int x,            //
                const int y,            //
                const int halo_depth,   //
                const int depth,        //
                SyclBuffer &bufferBuff, //
                queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::read_write>(h);
    h.parallel_for<class update_top>(range<1>(x * depth), [=](id<1> idx) {
      const auto lines = idx[0] / x;
      const auto offset = x * (y - halo_depth);
      const auto to_index = offset + idx[0];
      const auto from_index = to_index - (1 + lines * 2) * x;
      buffer[to_index] = buffer[from_index];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Updates the local bottom halo region(s)
void update_bottom(const int x,            //
                   const int y,            //
                   const int halo_depth,   //
                   const int depth,        //
                   SyclBuffer &bufferBuff, //
                   queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::read_write>(h);
    h.parallel_for<class update_bottom>(range<1>(x * depth), [=](id<1> idx) {
      const auto lines = idx[0] / x;
      const auto offset = x * halo_depth;
      const auto from_index = offset + idx[0];
      const auto to_index = from_index - (1 + lines * 2) * x;
      buffer[to_index] = buffer[from_index];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Updates faces in turn.
void update_face(const int x, const int y, const int halo_depth, const int *chunk_neighbours, const int depth, SyclBuffer &buffer,
                 queue &queue) {
  if (chunk_neighbours[CHUNK_LEFT] == EXTERNAL_FACE) {
    update_left(x, y, halo_depth, depth, buffer, queue);
  }
  if (chunk_neighbours[CHUNK_RIGHT] == EXTERNAL_FACE) {
    update_right(x, y, halo_depth, depth, buffer, queue);
  }
  if (chunk_neighbours[CHUNK_TOP] == EXTERNAL_FACE) {
    update_top(x, y, halo_depth, depth, buffer, queue);
  }
  if (chunk_neighbours[CHUNK_BOTTOM] == EXTERNAL_FACE) {
    update_bottom(x, y, halo_depth, depth, buffer, queue);
  }
}

// The kernel for updating halos locally
void local_halos(int x, int y, int depth, int halo_depth, const int *chunk_neighbours, const bool *fields_to_exchange, SyclBuffer &density,
                 SyclBuffer &energy0, SyclBuffer &energy, SyclBuffer &u, SyclBuffer &p, SyclBuffer &sd, queue &queue) {
  if (fields_to_exchange[FIELD_DENSITY]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, density, queue);
  }
  if (fields_to_exchange[FIELD_P]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, p, queue);
  }
  if (fields_to_exchange[FIELD_ENERGY0]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, energy0, queue);
  }
  if (fields_to_exchange[FIELD_ENERGY1]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, energy, queue);
  }
  if (fields_to_exchange[FIELD_U]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, u, queue);
  }
  if (fields_to_exchange[FIELD_SD]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, sd, queue);
  }
}

// Solver-wide kernels
void run_local_halos(Chunk *chunk, Settings &settings, int depth) {
  START_PROFILING(settings.kernel_profile);
  local_halos(chunk->x, chunk->y, depth, settings.halo_depth, chunk->neighbours, settings.fields_to_exchange, *chunk->density,
              *chunk->energy0, *chunk->energy, *chunk->u, *chunk->p, *chunk->sd, *chunk->ext->device_queue);
  STOP_PROFILING(settings.kernel_profile, __func__);
}