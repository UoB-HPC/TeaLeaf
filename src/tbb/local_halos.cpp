#include "chunk.h"
#include "shared.h"
#include "tbb_shared.h"

/*
 * 		LOCAL HALOS KERNEL
 */

// Update left halo.
void update_left(const int x,          //
                 const int y,          //
                 const int halo_depth, //
                 const int depth,      //
                 double *buffer) {
  Range2d range(0, halo_depth, depth, y - halo_depth);
  tbb::parallel_for(0, range.sizeXY(), [&](int i) {
    const auto kk = (i / range.sizeY()) + range.fromX;
    const auto jj = (i % range.sizeY()) + range.fromY;
    int base = jj * x;
    buffer[base + (halo_depth - kk - 1)] = buffer[base + (halo_depth + kk)];
  });

  //  ranged<int> it(halo_depth, y - halo_depth);
  //  std::for_each(it.begin(), it.end(), [=](int jj) {
  //    for (int kk = 0; kk < depth; ++kk) {
  //      int base = jj * x;
  //      buffer[base + (halo_depth - kk - 1)] = buffer[base + (halo_depth + kk)];
  //    }
  //  });
}

// Update right halo.
void update_right(const int x,          //
                  const int y,          //
                  const int halo_depth, //
                  const int depth,      //
                  double *buffer) {
  Range2d range(0, halo_depth, depth, y - halo_depth);
  tbb::parallel_for(
      0, range.sizeXY(),
      [&](int i) {
        const auto kk = (i / range.sizeY()) + range.fromX;
        const auto jj = (i % range.sizeY()) + range.fromY;
        int base = jj * x;
        buffer[base + (x - halo_depth + kk)] = buffer[base + (x - halo_depth - 1 - kk)];
      },
      partitioner);

  //  ranged<int> it(halo_depth, y - halo_depth);
  //  std::for_each(it.begin(), it.end(), [=](int jj) {
  //    for (int kk = 0; kk < depth; ++kk) {
  //      int base = jj * x;
  //      buffer[base + (x - halo_depth + kk)] = buffer[base + (x - halo_depth - 1 - kk)];
  //    }
  //  });
}

// Update top halo.
void update_top(const int x,          //
                const int y,          //
                const int halo_depth, //
                const int depth,      //
                double *buffer) {

  Range2d range(halo_depth, 0, x - halo_depth, depth);
  tbb::parallel_for(
      0, range.sizeXY(),
      [&](int i) {
        const auto kk = (i / range.sizeY()) + range.fromX;
        const auto jj = (i % range.sizeY()) + range.fromY;
        int base = kk;
        buffer[base + (y - halo_depth + jj) * x] = buffer[base + (y - halo_depth - 1 - jj) * x];
      },
      partitioner);

  //  ranged<int> it(halo_depth, x - halo_depth);
  //  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int kk) {
  //    for (int jj = 0; jj < depth; ++jj) {
  //      int base = kk;
  //      buffer[base + (y - halo_depth + jj) * x] = buffer[base + (y - halo_depth - 1 - jj) * x];
  //    }
  //  });
}

// Updates bottom halo.
void update_bottom(const int x,          //
                   const int y,          //
                   const int halo_depth, //
                   const int depth,      //
                   double *buffer) {
  Range2d range(halo_depth, 0, x - halo_depth, depth);
  tbb::parallel_for(
      0, range.sizeXY(),
      [&](int i) {
        const auto kk = (i / range.sizeY()) + range.fromX;
        const auto jj = (i % range.sizeY()) + range.fromY;
        int base = kk;
        buffer[base + (halo_depth - jj - 1) * x] = buffer[base + (halo_depth + jj) * x];
      },
      partitioner);

  //  ranged<int> it(halo_depth, x - halo_depth);
  //  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int kk) {
  //    for (int jj = 0; jj < depth; ++jj) {
  //      int base = kk;
  //      buffer[base + (halo_depth - jj - 1) * x] = buffer[base + (halo_depth + jj) * x];
  //    }
  //  });
}

// Updates faces in turn.
void update_face(const int x, const int y, const int halo_depth, const int *chunk_neighbours, const int depth, double *buffer) {
  if (chunk_neighbours[CHUNK_LEFT] == EXTERNAL_FACE) {
    update_left(x, y, halo_depth, depth, buffer);
  }
  if (chunk_neighbours[CHUNK_RIGHT] == EXTERNAL_FACE) {
    update_right(x, y, halo_depth, depth, buffer);
  }
  if (chunk_neighbours[CHUNK_TOP] == EXTERNAL_FACE) {
    update_top(x, y, halo_depth, depth, buffer);
  }
  if (chunk_neighbours[CHUNK_BOTTOM] == EXTERNAL_FACE) {
    update_bottom(x, y, halo_depth, depth, buffer);
  }
}

// The kernel for updating halos locally
void local_halos(int x, int y, int depth, int halo_depth, const int *chunk_neighbours, const bool *fields_to_exchange, double *density,
                 double *energy0, double *energy, double *u, double *p, double *sd) {
  if (fields_to_exchange[FIELD_DENSITY]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, density);
  }
  if (fields_to_exchange[FIELD_P]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, p);
  }
  if (fields_to_exchange[FIELD_ENERGY0]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, energy0);
  }
  if (fields_to_exchange[FIELD_ENERGY1]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, energy);
  }
  if (fields_to_exchange[FIELD_U]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, u);
  }
  if (fields_to_exchange[FIELD_SD]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, sd);
  }
}

// Solver-wide kernels
void run_local_halos(Chunk *chunk, Settings &settings, int depth) {
  START_PROFILING(settings.kernel_profile);
  local_halos(chunk->x, chunk->y, depth, settings.halo_depth, chunk->neighbours, settings.fields_to_exchange, chunk->density,
              chunk->energy0, chunk->energy, chunk->u, chunk->p, chunk->sd);
  STOP_PROFILING(settings.kernel_profile, __func__);
}