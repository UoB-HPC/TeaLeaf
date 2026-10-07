#include "chunk.h"
#include "kokkos_shared.hpp"
#include "settings.h"
#include "shared.h"

// Initialises the vertices
void set_chunk_data_vertices(const int x, const int y, const int halo_depth, KView &vertex_x, KView &vertex_y, const double x_min,
                             const double y_min, const double dx, const double dy) {
  Kokkos::parallel_for(
      tealeaf_MAX(x, y) + 1, KOKKOS_LAMBDA(const int index) {
        if (index < x + 1) {
          vertex_x(index) = x_min + dx * (index - halo_depth);
        }

        if (index < y + 1) {
          vertex_y(index) = y_min + dy * (index - halo_depth);
        }
      });
}

// Sets all of the cell data for a chunk
void set_chunk_data(const int x, const int y, const int halo_depth, KView &vertex_x, KView &vertex_y, KView &cell_x, KView &cell_y,
                    KView &volume, KView &x_area, KView &y_area, const double x_min, const double y_min, const double dx, const double dy) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        if (index < x) {
          cell_x(index) = 0.5 * (vertex_x(index) + vertex_x(index + 1));
        }

        if (index < y) {
          cell_y(index) = 0.5 * (vertex_y(index) + vertex_y(index + 1));
        }

        if (index < x * y) {
          volume(index) = dx * dy;
          x_area(index) = dy;
          y_area(index) = dx;
        }
      });
}

// Sets the initial state for the chunk
void set_chunk_initial_state(const int x, const int y, double default_energy, double default_density, KView &energy0, KView &density) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        energy0(index) = default_energy;
        density(index) = default_density;
      });
}

// Sets all of the additional states in order
void set_chunk_state(const int x, const int y, const int halo_depth, State state, KView &energy0, KView &density, KView &u, KView &cell_x,
                     KView &cell_y, KView &vertex_x, KView &vertex_y) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        const int kk = index % x;
        const int jj = index / x;

        bool applyState = false;

        if (state.geometry == Geometry::RECTANGULAR) // Rectangular state
        {
          applyState = (vertex_x(kk + 1) >= state.x_min && vertex_x(kk) < state.x_max && vertex_y(jj + 1) >= state.y_min &&
                        vertex_y(jj) < state.y_max);
        } else if (state.geometry == Geometry::CIRCULAR) // Circular state
        {
          double radius = Kokkos::sqrt((cell_x(kk) - state.x_min) * (cell_x(kk) - state.x_min) +
                                       (cell_y(jj) - state.y_min) * (cell_y(jj) - state.y_min));

          applyState = (radius <= state.radius);
        } else if (state.geometry == Geometry::POINT) // Point state
        {
          applyState = (vertex_x(kk) == state.x_min && vertex_y(jj) == state.y_min);
        }

        // Check if state applies at this vertex, and apply
        if (applyState) {
          energy0(index) = state.energy;
          density(index) = state.density;
        }

        if (kk > 0 && kk < x - 1 && jj > 0 && jj < y - 1) {
          u(index) = energy0(index) * density(index);
        }
      });
}

void run_set_chunk_data(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);

  double x_min = settings.grid_x_min + settings.dx * (double)chunk->left;
  double y_min = settings.grid_y_min + settings.dy * (double)chunk->bottom;

  set_chunk_data_vertices(chunk->x, chunk->y, settings.halo_depth, *chunk->vertex_x, *chunk->vertex_y, x_min, y_min, settings.dx,
                          settings.dy);

  set_chunk_data(chunk->x, chunk->y, settings.halo_depth, *chunk->vertex_x, *chunk->vertex_y, *chunk->cell_x, *chunk->cell_y,
                 *chunk->volume, *chunk->x_area, *chunk->y_area, x_min, y_min, settings.dx, settings.dy);

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_set_chunk_state(Chunk *chunk, Settings &settings, State *states) {
  START_PROFILING(settings.kernel_profile);

  set_chunk_initial_state(chunk->x, chunk->y, states[0].energy, states[0].density, *chunk->energy0, *chunk->density);

  for (int ii = 1; ii < settings.num_states; ++ii) {
    set_chunk_state(chunk->x, chunk->y, settings.halo_depth, states[ii], *chunk->energy0, *chunk->density, *chunk->u, *chunk->cell_x,
                    *chunk->cell_y, *chunk->vertex_x, *chunk->vertex_y);
  }

  STOP_PROFILING(settings.kernel_profile, __func__);
}

// Allocates, and zeroes an individual buffer
void allocate_buffer(double **a, int x, int y) {
  *a = (double *)malloc(sizeof(double) * x * y);

  if (*a == nullptr) {
    die(__LINE__, __FILE__, "Error allocating buffer %s\n");
  }

#pragma omp parallel for
  for (int jj = 0; jj < y; ++jj) {
    for (int kk = 0; kk < x; ++kk) {
      const int index = kk + jj * x;
      (*a)[index] = 0.0;
    }
  }
}

void run_model_info(Settings &settings) {
  settings.model_name = "Kokkos" + std::to_string(KOKKOS_VERSION / 10000) + "." + std::to_string(KOKKOS_VERSION / 100 % 100) + "." +
                        std::to_string(KOKKOS_VERSION % 100);
  settings.model_kind = ModelKind::Offload;
}

static int live_chunks = 0;

void run_kernel_initialise(Chunk *chunk, Settings &settings, int comms_lr_len, int comms_tb_len) {

  if (live_chunks++ == 0) Kokkos::initialize();

  print_and_log(settings, " - Backend space: %s\n", typeid(Kokkos::DefaultExecutionSpace).name());
  print_and_log(settings, " - Backend host space: %s\n", typeid(Kokkos::DefaultHostExecutionSpace).name());

  chunk->staging_left_send = new KView::HostMirror{};
  chunk->staging_left_recv = new KView::HostMirror{};
  chunk->staging_right_send = new KView::HostMirror{};
  chunk->staging_right_recv = new KView::HostMirror{};
  chunk->staging_top_send = new KView::HostMirror{};
  chunk->staging_top_recv = new KView::HostMirror{};
  chunk->staging_bottom_send = new KView::HostMirror{};
  chunk->staging_bottom_recv = new KView::HostMirror{};

  chunk->density0 = new KView(Kokkos::ViewAllocateWithoutInitializing("density0"), chunk->x * chunk->y);
  chunk->density = new KView(Kokkos::ViewAllocateWithoutInitializing("density"), chunk->x * chunk->y);
  chunk->energy0 = new KView(Kokkos::ViewAllocateWithoutInitializing("energy0"), chunk->x * chunk->y);
  chunk->energy = new KView(Kokkos::ViewAllocateWithoutInitializing("energy"), chunk->x * chunk->y);
  chunk->u = new KView(Kokkos::ViewAllocateWithoutInitializing("u"), chunk->x * chunk->y);
  chunk->u0 = new KView(Kokkos::ViewAllocateWithoutInitializing("u0"), chunk->x * chunk->y);
  chunk->p = new KView(Kokkos::ViewAllocateWithoutInitializing("p"), chunk->x * chunk->y);
  chunk->r = new KView(Kokkos::ViewAllocateWithoutInitializing("r"), chunk->x * chunk->y);
  chunk->mi = new KView(Kokkos::ViewAllocateWithoutInitializing("mi"), chunk->x * chunk->y);
  chunk->w = new KView(Kokkos::ViewAllocateWithoutInitializing("w"), chunk->x * chunk->y);
  chunk->kx = new KView(Kokkos::ViewAllocateWithoutInitializing("kx"), chunk->x * chunk->y);
  chunk->ky = new KView(Kokkos::ViewAllocateWithoutInitializing("ky"), chunk->x * chunk->y);
  chunk->sd = new KView(Kokkos::ViewAllocateWithoutInitializing("sd"), chunk->x * chunk->y);
  chunk->volume = new KView(Kokkos::ViewAllocateWithoutInitializing("volume"), chunk->x * chunk->y);
  chunk->x_area = new KView(Kokkos::ViewAllocateWithoutInitializing("x_area"), (chunk->x + 1) * chunk->y);
  chunk->y_area = new KView(Kokkos::ViewAllocateWithoutInitializing("y_area"), chunk->x * (chunk->y + 1));
  chunk->cell_x = new KView(Kokkos::ViewAllocateWithoutInitializing("cell_x"), chunk->x);
  chunk->cell_y = new KView(Kokkos::ViewAllocateWithoutInitializing("cell_y"), chunk->y);
  chunk->cell_dx = new KView(Kokkos::ViewAllocateWithoutInitializing("cell_dx"), chunk->x);
  chunk->cell_dy = new KView(Kokkos::ViewAllocateWithoutInitializing("cell_dy"), chunk->y);
  chunk->vertex_dx = new KView(Kokkos::ViewAllocateWithoutInitializing("vertex_dx"), (chunk->x + 1));
  chunk->vertex_dy = new KView(Kokkos::ViewAllocateWithoutInitializing("vertex_dy"), (chunk->y + 1));
  chunk->vertex_x = new KView(Kokkos::ViewAllocateWithoutInitializing("vertex_x"), (chunk->x + 1));
  chunk->vertex_y = new KView(Kokkos::ViewAllocateWithoutInitializing("vertex_y"), (chunk->y + 1));

  chunk->left_send = new KView(Kokkos::ViewAllocateWithoutInitializing("left_send"), comms_lr_len);
  chunk->left_recv = new KView(Kokkos::ViewAllocateWithoutInitializing("left_recv"), comms_lr_len);
  chunk->right_send = new KView(Kokkos::ViewAllocateWithoutInitializing("right_send"), comms_lr_len);
  chunk->right_recv = new KView(Kokkos::ViewAllocateWithoutInitializing("right_recv"), comms_lr_len);
  chunk->top_send = new KView(Kokkos::ViewAllocateWithoutInitializing("top_send"), comms_tb_len);
  chunk->top_recv = new KView(Kokkos::ViewAllocateWithoutInitializing("top_recv"), comms_tb_len);
  chunk->bottom_send = new KView(Kokkos::ViewAllocateWithoutInitializing("bottom_send"), comms_tb_len);
  chunk->bottom_recv = new KView(Kokkos::ViewAllocateWithoutInitializing("bottom_recv"), comms_tb_len);

  allocate_buffer(&(chunk->cg_alphas), settings.max_iters, 1);
  allocate_buffer(&(chunk->cg_betas), settings.max_iters, 1);
  allocate_buffer(&(chunk->cheby_alphas), settings.max_iters, 1);
  allocate_buffer(&(chunk->cheby_betas), settings.max_iters, 1);
}

void run_kernel_finalise(Chunk *chunk, Settings &) {
  free(chunk->cg_alphas);
  free(chunk->cg_betas);
  free(chunk->cheby_alphas);
  free(chunk->cheby_betas);

  if (--live_chunks == 0) Kokkos::finalize();
}
