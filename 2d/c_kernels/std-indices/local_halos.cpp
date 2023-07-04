#include <cstdlib>
#include <numeric>

#include "../../shared.h"
#include "dpl_shim.h"
#include "ranged.h"
#include "std_shared.h"

/*
 * 		LOCAL HALOS KERNEL
 */

void update_left(const int x, const int y, const int halo_depth, const int depth, double *buffer, bool is_offload);
void update_right(const int x, const int y, const int halo_depth, const int depth, double *buffer, bool is_offload);
void update_top(const int x, const int y, const int halo_depth, const int depth, double *buffer, bool is_offload);
void update_bottom(const int x, const int y, const int halo_depth, const int depth, double *buffer, bool is_offload);
void update_face(const int x, const int y, const int halo_depth, const int *chunk_neighbours, const int depth, double *buffer,
                 bool is_offload);

typedef void (*update_kernel)(int, double *);

// The kernel for updating halos locally
void local_halos(const int x,                    //
                 const int y,                    //
                 const int depth,                //
                 const int halo_depth,           //
                 const int *chunk_neighbours,    //
                 const bool *fields_to_exchange, //
                 double *density,                //
                 double *energy0,                //
                 double *energy,                 //
                 double *u,                      //
                 double *p,                      //
                 double *sd,                     //
                 bool is_offload) {
#define LAUNCH_UPDATE(index, buffer)                                                                                                       \
  if (fields_to_exchange[index]) {                                                                                                         \
    update_face(x, y, halo_depth, chunk_neighbours, depth, buffer, is_offload);                                                            \
  }

  LAUNCH_UPDATE(FIELD_DENSITY, density);
  LAUNCH_UPDATE(FIELD_P, p);
  LAUNCH_UPDATE(FIELD_ENERGY0, energy0);
  LAUNCH_UPDATE(FIELD_ENERGY1, energy);
  LAUNCH_UPDATE(FIELD_U, u);
  LAUNCH_UPDATE(FIELD_SD, sd);
#undef LAUNCH_UPDATE
}

// Updates faces in turn.
void update_face(const int x,                 //
                 const int y,                 //
                 const int halo_depth,        //
                 const int *chunk_neighbours, //
                 const int depth,             //
                 double *buffer,              //
                 bool is_offload) {
#define UPDATE_FACE(face, updateKernel)                                                                                                    \
  if (chunk_neighbours[face] == EXTERNAL_FACE) {                                                                                           \
    updateKernel(x, y, halo_depth, depth, buffer, is_offload);                                                                             \
  }

  UPDATE_FACE(CHUNK_LEFT, update_left);
  UPDATE_FACE(CHUNK_RIGHT, update_right);
  UPDATE_FACE(CHUNK_TOP, update_top);
  UPDATE_FACE(CHUNK_BOTTOM, update_bottom);
}

// Update left halo.
void update_left(const int x,          //
                 const int y,          //
                 const int halo_depth, //
                 const int depth,      //
                 double *buffer,       //
                 bool is_offload) {
    Range2d range(0, halo_depth, depth, y - halo_depth);
    ranged<int> it(0, range.sizeXY());
    std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int i) {
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
                  double *buffer,       //
                  bool is_offload) {
    Range2d range(0, halo_depth, depth, y - halo_depth);
    ranged<int> it(0, range.sizeXY());
    std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int i) {
      const auto kk = (i / range.sizeY()) + range.fromX;
      const auto jj = (i % range.sizeY()) + range.fromY;
      int base = jj * x;
      buffer[base + (x - halo_depth + kk)] = buffer[base + (x - halo_depth - 1 - kk)];
    });

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
                double *buffer,       //
                bool is_offload) {

    Range2d range(halo_depth, 0, x - halo_depth, depth);
    ranged<int> it(0, range.sizeXY());
    std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int i) {
      const auto kk = (i / range.sizeY()) + range.fromX;
      const auto jj = (i % range.sizeY()) + range.fromY;
      int base = kk;
      buffer[base + (y - halo_depth + jj) * x] = buffer[base + (y - halo_depth - 1 - jj) * x];
    });

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
                   double *buffer,       //
                   bool is_offload) {
    Range2d range(halo_depth, 0, x - halo_depth, depth);
    ranged<int> it(0, range.sizeXY());
    std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int i) {
      const auto kk = (i / range.sizeY()) + range.fromX;
      const auto jj = (i % range.sizeY()) + range.fromY;
      int base = kk;
      buffer[base + (halo_depth - jj - 1) * x] = buffer[base + (halo_depth + jj) * x];
    });

//  ranged<int> it(halo_depth, x - halo_depth);
//  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int kk) {
//    for (int jj = 0; jj < depth; ++jj) {
//      int base = kk;
//      buffer[base + (halo_depth - jj - 1) * x] = buffer[base + (halo_depth + jj) * x];
//    }
//  });
}