#include "chunk.h"
#include "settings.h"
#include "shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Allocates, and zeroes an individual buffer
void allocate_buffer(double **a, int x, int y) {
  //*a = (double*)malloc(sizeof(double)*x*y);
  *a = new double[x * y];

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

// Sets the initial state for the chunk
void set_chunk_initial_state(const int x,             //
                             const int y,             //
                             double default_energy,   //
                             double default_density,  //
                             SyclBuffer &energy0Buff, //
                             SyclBuffer &densityBuff, //
                             queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto density = densityBuff.get_access<access::mode::discard_write>(h);
    auto energy0 = energy0Buff.get_access<access::mode::discard_write>(h);

    h.parallel_for<class set_chunk_initial_state>(range<1>(x * y), [=](id<1> idx) {
      energy0[idx[0]] = default_energy;
      density[idx[0]] = default_density;
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Sets all of the additional states in order
void set_chunk_state(const int x,              //
                     const int y,              //
                     const int halo_depth,     //
                     State state,              //
                     SyclBuffer &energy0Buff,  //
                     SyclBuffer &densityBuff,  //
                     SyclBuffer &uBuff,        //
                     SyclBuffer &cell_xBuff,   //
                     SyclBuffer &cell_yBuff,   //
                     SyclBuffer &vertex_xBuff, //
                     SyclBuffer &vertex_yBuff, //
                     queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto energy0 = energy0Buff.get_access<access::mode::read_write>(h);
    auto density = densityBuff.get_access<access::mode::read_write>(h);
    auto u = uBuff.get_access<access::mode::write>(h);
    auto cell_x = cell_xBuff.get_access<access::mode::read>(h);
    auto cell_y = cell_yBuff.get_access<access::mode::read>(h);
    auto vertex_x = vertex_xBuff.get_access<access::mode::read>(h);
    auto vertex_y = vertex_yBuff.get_access<access::mode::read>(h);
    h.parallel_for<class set_chunk_state>(range<1>(x * y), [=](id<1> idx) {
      const auto kk = idx[0] % x;
      const auto jj = idx[0] / x;
      bool applyState = false;

      if (state.geometry == Geometry::RECTANGULAR) // Rectangular state
      {
        applyState = (vertex_x[kk + 1] >= state.x_min && vertex_x[kk] < state.x_max && vertex_y[jj + 1] >= state.y_min &&
                      vertex_y[jj] < state.y_max);
      } else if (state.geometry == Geometry::CIRCULAR) // Circular state
      {
        double radius =
            sycl::sqrt((cell_x[kk] - state.x_min) * (cell_x[kk] - state.x_min) + (cell_y[jj] - state.y_min) * (cell_y[jj] - state.y_min));

        applyState = (radius <= state.radius);
      } else if (state.geometry == Geometry::POINT) // Point state
      {
        applyState = (vertex_x[kk] == state.x_min && vertex_y[jj] == state.y_min);
      }

      // Check if state applies at this vertex, and apply
      if (applyState) {
        energy0[idx[0]] = state.energy;
        density[idx[0]] = state.density;
      }

      if (kk > 0 && kk < size_t(x - 1) && jj > 0 && jj < size_t(y - 1)) {
        u[idx[0]] = energy0[idx[0]] * density[idx[0]];
      }
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Initialises the vertices
void set_chunk_data_vertices(const int x,              //
                             const int y,              //
                             const int halo_depth,     //
                             SyclBuffer &vertex_xBuff, //
                             SyclBuffer &vertex_yBuff, //
                             const double x_min,       //
                             const double y_min,
                             const double dx, //
                             const double dy, //
                             queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto vertex_x = vertex_xBuff.get_access<access::mode::discard_write>(h);
    auto vertex_y = vertex_yBuff.get_access<access::mode::discard_write>(h);
    h.parallel_for<class set_chunk_data_vertices>(range<1>(tealeaf_MAX(x, y) + 1), [=](id<1> idx) {
      if (idx[0] < size_t(x + 1)) {
        vertex_x[idx[0]] = x_min + dx * static_cast<double>(static_cast<int>(idx[0]) - halo_depth);
      }
      if (idx[0] < size_t(y + 1)) {
        vertex_y[idx[0]] = y_min + dy * static_cast<double>(static_cast<int>(idx[0]) - halo_depth);
      }
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Sets all of the cell data for a chunk
void set_chunk_data(const int x,              //
                    const int y,              //
                    const int halo_depth,     //
                    SyclBuffer &vertex_xBuff, //
                    SyclBuffer &vertex_yBuff, //
                    SyclBuffer &cell_xBuff,   //
                    SyclBuffer &cell_yBuff,   //
                    SyclBuffer &volumeBuff,   //
                    SyclBuffer &x_areaBuff,   //
                    SyclBuffer &y_areaBuff,   //
                    const double x_min,       //
                    const double y_min,       //
                    const double dx,          //
                    const double dy,          //
                    queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto vertex_x = vertex_xBuff.get_access<access::mode::read>(h);
    auto vertex_y = vertex_yBuff.get_access<access::mode::read>(h);
    auto volume = volumeBuff.get_access<access::mode::write>(h);
    auto x_area = x_areaBuff.get_access<access::mode::write>(h);
    auto y_area = y_areaBuff.get_access<access::mode::write>(h);
    auto cell_y = cell_yBuff.get_access<access::mode::write>(h);
    auto cell_x = cell_xBuff.get_access<access::mode::write>(h);

    h.parallel_for<class set_chunk_data>(range<1>(x * y), [=](id<1> idx) {
      if (idx[0] < size_t(x)) {
        cell_x[idx[0]] = 0.5 * (vertex_x[idx[0]] + vertex_x[idx[0] + 1]);
      }
      if (idx[0] < size_t(y)) {
        cell_y[idx[0]] = 0.5 * (vertex_y[idx[0]] + vertex_y[idx[0] + 1]);
      }

      if (idx[0] < size_t(x * y)) {
        volume[idx[0]] = dx * dy;
        x_area[idx[0]] = dy;
        y_area[idx[0]] = dx;
      }
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

void run_set_chunk_data(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);

  double x_min = settings.grid_x_min + settings.dx * (double)chunk->left;
  double y_min = settings.grid_y_min + settings.dy * (double)chunk->bottom;
  set_chunk_data_vertices(chunk->x, chunk->y, settings.halo_depth, *(chunk->vertex_x), *(chunk->vertex_y), x_min, y_min, settings.dx,
                          settings.dy, *(chunk->ext->device_queue));

  set_chunk_data(chunk->x, chunk->y, settings.halo_depth, *(chunk->vertex_x), *(chunk->vertex_y), *(chunk->cell_x), *(chunk->cell_y),
                 *(chunk->volume), *(chunk->x_area), *(chunk->y_area), x_min, y_min, settings.dx, settings.dy, *(chunk->ext->device_queue));

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_set_chunk_state(Chunk *chunk, Settings &settings, State *states) {
  START_PROFILING(settings.kernel_profile);

  set_chunk_initial_state(chunk->x, chunk->y, states[0].energy, states[0].density, *(chunk->energy0), *(chunk->density),
                          *(chunk->ext->device_queue));

  for (int ii = 1; ii < settings.num_states; ++ii) {
    set_chunk_state(chunk->x, chunk->y, settings.halo_depth, states[ii], *(chunk->energy0), *(chunk->density), *(chunk->u),
                    *(chunk->cell_x), *(chunk->cell_y), *(chunk->vertex_x), *(chunk->vertex_y), *(chunk->ext->device_queue));
  }

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_model_info(Settings &settings) {
  settings.model_name = "SYCL (accessors)";
  settings.model_kind = ModelKind::Offload;
}

void run_kernel_initialise(Chunk *chunk, Settings &settings, int comms_lr_len, int comms_tb_len) {
  auto selector = !settings.device_selector ? "0" : std::string(settings.device_selector);
  auto devices = sycl::device::get_devices();

  print_and_log(settings, "Devices:\n");
  if (devices.empty()) {
    die(__LINE__, __FILE__, "sycl::device::get_devices() returned 0 devices.");
  }
  for (size_t i = 0; i < devices.size(); ++i) {
    print_and_log(settings, " %d: %s\n", i, devices[i].get_info<info::device::name>().c_str());
  }

  device selected;
  try {
    selected = devices.at(std::stoul(selector));
  } catch (const std::exception &e) {
    print_and_log(settings, "# Unable to parse/select device index `%s`:%s\n", selector.c_str(), e.what());
    print_and_log(settings, "# Attempting to match device with substring `%s`\n", selector.c_str());

    auto matching = std::find_if(devices.begin(), devices.end(), [selector](const device &device) {
      return device.get_info<info::device::name>().find(selector) != std::string::npos;
    });
    if (matching != devices.end()) {
      selected = *matching;
      print_and_log(settings, "# Using first device matching substring `%s`\n", selector.c_str());
    } else if (devices.size() == 1) {
      print_and_log(settings, "# No matching device but there's only one device, will be using that anyway\n");
    } else {
      die(__LINE__, __FILE__, "# No matching devices for `%s`\n", selector.c_str());
    }
  }

  chunk->ext->device_queue = new queue(selected);
  print_and_log(settings, " - SYCL device: %s\n", chunk->ext->device_queue->get_device().get_info<info::device::name>().c_str());

  chunk->density0 = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->density = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->energy0 = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->energy = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->u = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->u0 = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->p = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->r = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->mi = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->w = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->kx = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->ky = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->sd = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->volume = new SyclBuffer{range<1>{(size_t)chunk->x * chunk->y}};
  chunk->x_area = new SyclBuffer{range<1>{(size_t)(chunk->x + 1) * chunk->y}};
  chunk->y_area = new SyclBuffer{range<1>{(size_t)chunk->x * (chunk->y + 1)}};
  chunk->cell_x = new SyclBuffer{range<1>{(size_t)chunk->x}};
  chunk->cell_y = new SyclBuffer{range<1>{(size_t)chunk->y}};
  chunk->cell_dx = new SyclBuffer{range<1>{(size_t)chunk->x}};
  chunk->cell_dy = new SyclBuffer{range<1>{(size_t)chunk->y}};
  chunk->vertex_dx = new SyclBuffer{range<1>{(size_t)(chunk->x + 1)}};
  chunk->vertex_dy = new SyclBuffer{range<1>{(size_t)(chunk->y + 1)}};
  chunk->vertex_x = new SyclBuffer{range<1>{(size_t)(chunk->x + 1)}};
  chunk->vertex_y = new SyclBuffer{range<1>{(size_t)(chunk->y + 1)}};

  chunk->left_send = new SyclBuffer{range<1>(size_t(comms_lr_len))};
  chunk->left_recv = new SyclBuffer{range<1>(size_t(comms_lr_len))};
  chunk->right_send = new SyclBuffer{range<1>(size_t(comms_lr_len))};
  chunk->right_recv = new SyclBuffer{range<1>(size_t(comms_lr_len))};
  chunk->top_send = new SyclBuffer{range<1>(size_t(comms_tb_len))};
  chunk->top_recv = new SyclBuffer{range<1>(size_t(comms_tb_len))};
  chunk->bottom_send = new SyclBuffer{range<1>(size_t(comms_tb_len))};
  chunk->bottom_recv = new SyclBuffer{range<1>(size_t(comms_tb_len))};

  allocate_buffer(&(chunk->cg_alphas), settings.max_iters, 1);
  allocate_buffer(&(chunk->cg_betas), settings.max_iters, 1);
  allocate_buffer(&(chunk->cheby_alphas), settings.max_iters, 1);
  allocate_buffer(&(chunk->cheby_betas), settings.max_iters, 1);
}

void run_kernel_finalise(Chunk *chunk, Settings &) {
  delete[] chunk->cg_alphas;
  delete[] chunk->cg_betas;
  delete[] chunk->cheby_alphas;
  delete[] chunk->cheby_betas;

  delete chunk->density0;
  delete chunk->density;
  delete chunk->energy0;
  delete chunk->energy;
  delete chunk->u;
  delete chunk->u0;
  delete chunk->p;
  delete chunk->r;
  delete chunk->mi;
  delete chunk->w;
  delete chunk->kx;
  delete chunk->ky;
  delete chunk->sd;
  delete chunk->volume;
  delete chunk->x_area;
  delete chunk->y_area;
  delete chunk->cell_x;
  delete chunk->cell_y;
  delete chunk->cell_dx;
  delete chunk->cell_dy;
  delete chunk->vertex_dx;
  delete chunk->vertex_dy;
  delete chunk->vertex_x;
  delete chunk->vertex_y;

  delete chunk->left_send;
  delete chunk->left_recv;
  delete chunk->right_send;
  delete chunk->right_recv;
  delete chunk->top_send;
  delete chunk->top_recv;
  delete chunk->bottom_send;
  delete chunk->bottom_recv;

  delete chunk->ext->device_queue;
}
