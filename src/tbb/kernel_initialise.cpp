#include "kernel_interface.h"
#include "tbb_shared.h"

// Initialisation kernels
void run_set_chunk_data(Chunk *chunk, Settings &settings) {
  double x_min = settings.grid_x_min + settings.dx * (double)chunk->left;
  double y_min = settings.grid_y_min + settings.dy * (double)chunk->bottom;
  double dx = settings.dx;
  double dy = settings.dy;
  double halo_depth = settings.halo_depth;

  tbb::parallel_for(
      0, chunk->x + 1, [&, vertex_x = chunk->vertex_x](const int ii) { vertex_x[ii] = x_min + dx * (ii - halo_depth); }, partitioner);

  tbb::parallel_for(
      0, chunk->y + 1, [&, vertex_y = chunk->vertex_y](const int ii) { vertex_y[ii] = y_min + dy * (ii - halo_depth); }, partitioner);

  tbb::parallel_for(
      0, chunk->x * chunk->y,
      [&, x = chunk->x, y = chunk->y,                          //
       vertex_x = chunk->vertex_x, vertex_y = chunk->vertex_y, //
       cell_x = chunk->cell_x, cell_y = chunk->cell_y,         //
       x_area = chunk->x_area, y_area = chunk->y_area,         //
       volume = chunk->volume](const int ii) {
        if (ii < x) {
          cell_x[ii] = 0.5 * (vertex_x[ii] + vertex_x[ii + 1]);
        }

        if (ii < y) {
          cell_y[ii] = 0.5 * (vertex_y[ii] + vertex_y[ii + 1]);
        }
        //    if (ii <  x *  y) {
        volume[ii] = dx * dy;
        x_area[ii] = dy;
        y_area[ii] = dx;
        //    }
      },
      partitioner);
}

void run_set_chunk_state(Chunk *chunk, Settings &settings, State *states) {
  // Set the initial state
  {
    double defaultEnergy = states[0].energy;
    double defaultDensity = states[0].density;
    tbb::parallel_for(
        0, chunk->x * chunk->y,
        [&, density = chunk->density, energy0 = chunk->energy0](int index) {
          energy0[index] = defaultEnergy;
          density[index] = defaultDensity;
        },
        partitioner);
  }

  // Apply all of the states in turn
  for (int ss = 1; ss < settings.num_states; ++ss) {
    State &state = states[ss];
    tbb::parallel_for(
        0, chunk->x * chunk->y,
        [&, x = chunk->x, y = chunk->y,                          //
         vertex_x = chunk->vertex_x, vertex_y = chunk->vertex_y, //
         cell_x = chunk->cell_x, cell_y = chunk->cell_y,         //
         density = chunk->density, energy0 = chunk->energy0, u = chunk->u](int ii) {
          const int kk = ii % x;
          const int jj = ii / x;

          bool applyState = false;

          if (state.geometry == Geometry::RECTANGULAR) { // Rectangular state

            applyState = (vertex_x[kk + 1] >= state.x_min && vertex_x[kk] < state.x_max && //
                          vertex_y[jj + 1] >= state.y_min && vertex_y[jj] < state.y_max);
          } else if (state.geometry == Geometry::CIRCULAR) { // Circular state

            double radius = std::sqrt((cell_x[kk] - state.x_min) * (cell_x[kk] - state.x_min) + //
                                      (cell_y[jj] - state.y_min) * (cell_y[jj] - state.y_min));

            applyState = (radius <= state.radius);
          } else if (state.geometry == Geometry::POINT) { // Point state
            applyState = (vertex_x[kk] == state.x_min && vertex_y[jj] == state.y_min);
          }

          // Check if state applies at this vertex, and apply
          if (applyState) {
            energy0[ii] = state.energy;
            density[ii] = state.density;
          }

          if (kk > 0 && kk < x - 1 && jj > 0 && jj < y - 1) {
            u[ii] = energy0[ii] * density[ii];
          }
        },
        partitioner);
  }
}

// Allocates, and zeroes and individual buffer
static inline void allocate_buffer(double **a, int x, int y) {
  *a = alloc_raw<double>(x * y);
  if (!*a) {
    die(__LINE__, __FILE__, "Error allocating buffer %s\n");
  }

  tbb::parallel_for(0, x * y, [&](auto i) { (*a)[i] = 0.0; }, partitioner);
}

void run_model_info(Settings &settings) {
  settings.model_name = "TBB";
  settings.model_kind = ModelKind::Unified;
}

void run_kernel_initialise(Chunk *chunk, Settings &settings, int comms_lr_len, int comms_tb_len) {

  if (settings.device_selector) {
    print_and_log(settings, "# Device selection is unsupported for this model, ignoring selector `%s`\n", settings.device_selector);
  }

  allocate_buffer(&chunk->density0, chunk->x, chunk->y);
  allocate_buffer(&chunk->density, chunk->x, chunk->y);
  allocate_buffer(&chunk->energy0, chunk->x, chunk->y);
  allocate_buffer(&chunk->energy, chunk->x, chunk->y);
  allocate_buffer(&chunk->u, chunk->x, chunk->y);
  allocate_buffer(&chunk->u0, chunk->x, chunk->y);
  allocate_buffer(&chunk->p, chunk->x, chunk->y);
  allocate_buffer(&chunk->r, chunk->x, chunk->y);
  allocate_buffer(&chunk->mi, chunk->x, chunk->y);
  allocate_buffer(&chunk->w, chunk->x, chunk->y);
  allocate_buffer(&chunk->kx, chunk->x, chunk->y);
  allocate_buffer(&chunk->ky, chunk->x, chunk->y);
  allocate_buffer(&chunk->sd, chunk->x, chunk->y);
  allocate_buffer(&chunk->volume, chunk->x, chunk->y);
  allocate_buffer(&chunk->x_area, chunk->x + 1, chunk->y);
  allocate_buffer(&chunk->y_area, chunk->x, chunk->y + 1);
  allocate_buffer(&chunk->cell_x, chunk->x, 1);
  allocate_buffer(&chunk->cell_y, 1, chunk->y);
  allocate_buffer(&chunk->cell_dx, chunk->x, 1);
  allocate_buffer(&chunk->cell_dy, 1, chunk->y);
  allocate_buffer(&chunk->vertex_dx, chunk->x + 1, 1);
  allocate_buffer(&chunk->vertex_dy, 1, chunk->y + 1);
  allocate_buffer(&chunk->vertex_x, chunk->x + 1, 1);
  allocate_buffer(&chunk->vertex_y, 1, chunk->y + 1);
  allocate_buffer(&chunk->cg_alphas, settings.max_iters, 1);
  allocate_buffer(&chunk->cg_betas, settings.max_iters, 1);
  allocate_buffer(&chunk->cheby_alphas, settings.max_iters, 1);
  allocate_buffer(&chunk->cheby_betas, settings.max_iters, 1);

  allocate_buffer(&chunk->left_send, comms_lr_len, 1);
  allocate_buffer(&chunk->left_recv, comms_lr_len, 1);
  allocate_buffer(&chunk->right_send, comms_lr_len, 1);
  allocate_buffer(&chunk->right_recv, comms_lr_len, 1);
  allocate_buffer(&chunk->top_send, comms_tb_len, 1);
  allocate_buffer(&chunk->top_recv, comms_tb_len, 1);
  allocate_buffer(&chunk->bottom_send, comms_tb_len, 1);
  allocate_buffer(&chunk->bottom_recv, comms_tb_len, 1);
}

void run_kernel_finalise(Chunk *chunk, Settings &) {
  dealloc_raw(chunk->density0);
  dealloc_raw(chunk->density);
  dealloc_raw(chunk->energy0);
  dealloc_raw(chunk->energy);
  dealloc_raw(chunk->u);
  dealloc_raw(chunk->u0);
  dealloc_raw(chunk->p);
  dealloc_raw(chunk->r);
  dealloc_raw(chunk->mi);
  dealloc_raw(chunk->w);
  dealloc_raw(chunk->kx);
  dealloc_raw(chunk->ky);
  dealloc_raw(chunk->sd);
  dealloc_raw(chunk->volume);
  dealloc_raw(chunk->x_area);
  dealloc_raw(chunk->y_area);
  dealloc_raw(chunk->cell_x);
  dealloc_raw(chunk->cell_y);
  dealloc_raw(chunk->cell_dx);
  dealloc_raw(chunk->cell_dy);
  dealloc_raw(chunk->vertex_dx);
  dealloc_raw(chunk->vertex_dy);
  dealloc_raw(chunk->vertex_x);
  dealloc_raw(chunk->vertex_y);
  dealloc_raw(chunk->cg_alphas);
  dealloc_raw(chunk->cg_betas);
  dealloc_raw(chunk->cheby_alphas);
  dealloc_raw(chunk->cheby_betas);

  dealloc_raw(chunk->left_send);
  dealloc_raw(chunk->left_recv);
  dealloc_raw(chunk->right_send);
  dealloc_raw(chunk->right_recv);
  dealloc_raw(chunk->top_send);
  dealloc_raw(chunk->top_recv);
  dealloc_raw(chunk->bottom_send);
  dealloc_raw(chunk->bottom_recv);
}
