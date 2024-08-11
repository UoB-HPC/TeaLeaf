#include "hip/hip_runtime.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "chunk.h"
#include "cuknl_shared.h"
#include "shared.h"

// Allocates, and zeroes and individual buffer
void allocate_device_buffer(double **a, int x, int y) {
#ifdef CLOVER_MANAGED_ALLOC
  if (auto result = hipMallocManaged(a, x * y * sizeof(double)); result != hipSuccess) {
    die(__LINE__, __FILE__, "hipMallocManaged failed - return code %d (%s)\n", result, hipGetErrorName(result));
  }
#else
  if (auto result = hipMalloc(a, x * y * sizeof(double)); result != hipSuccess) {
    die(__LINE__, __FILE__, "hipMalloc failed - return code %d (%s)\n", result, hipGetErrorName(result));
  }
#endif
  check_errors(__LINE__, __FILE__);
  if (auto result = hipMemset(*a, 0, x * y * sizeof(double)); result != hipSuccess) {
    die(__LINE__, __FILE__, "hipMemset failed - return code %d (%s)\n", result, hipGetErrorName(result));
  }
  check_errors(__LINE__, __FILE__);
}

void allocate_host_buffer(double **a, int x, int y) {
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
  settings.model_name = "HIP";
#ifdef CLOVER_MANAGED_ALLOC
  settings.model_kind = ModelKind::Unified;
#else
  settings.model_kind = ModelKind::Offload;
#endif
}

void run_kernel_initialise(Chunk *chunk, Settings &settings, int comms_lr_len, int comms_tb_len) {
  int count;
  if (auto result = hipGetDeviceCount(&count); result != hipSuccess) {
    die(__LINE__, __FILE__, "hipGetDeviceCount failed - return code %d (%s)\n", result, hipGetErrorName(result));
  }
  std::vector<std::pair<int, std::string>> devices(count);
  for (int i = 0; i < count; ++i) {
    hipDeviceProp_t props{};
    if (auto result = hipGetDeviceProperties(&props, i); result != hipSuccess) {
      die(__LINE__, __FILE__, "hipGetDeviceProperties failed - return code %d (%s)\n", result, hipGetErrorName(result));
    }
    devices[i] = {i, std::string(props.name)};
  }

  print_and_log(settings, "Devices:\n");
  if (count == 0) {
    print_and_log(settings, "# WARNING: hipGetDeviceCount returned 0 devices.\n");
  }
  for (auto &d : devices) {
    print_and_log(settings, " %d: %s\n", d.first, d.second.c_str());
  }

  auto selector = !settings.device_selector ? "0" : std::string(settings.device_selector);
  int selected = 0;
  try {
    selected = std::stoi(selector);
  } catch (const std::exception &e) {
    print_and_log(settings, "# Unable to parse/select device index `%s`: %s\n", selector.c_str(), e.what());
    print_and_log(settings, "# Attempting to match device with substring  `%s`\n", selector.c_str());

    auto matching = std::find_if(devices.begin(), devices.end(),
                                 [selector](const auto &device) { return device.second.find(selector) != std::string::npos; });
    if (matching != devices.end()) {
      selected = matching->first;
      print_and_log(settings, "# Using first device matching substring `%s`\n", selector.c_str());
    } else if (devices.size() == 1)
      print_and_log(settings, "# No matching device but there's only one device, will be using that anyway\n");
    else {
      die(__LINE__, __FILE__, "# No matching devices for `%s`\n", selector.c_str());
    }
  }

  int result = hipSetDevice(selected);
  if (result != hipSuccess) {
    die(__LINE__, __FILE__, "Could not allocate CUDA device %d.\n", selected);
  }

  hipDeviceProp_t properties{};
  if (auto result = hipGetDeviceProperties(&properties, selected); result != hipSuccess) {
    die(__LINE__, __FILE__, "hipGetDeviceProperties failed - return code %d (%s)\n", result, hipGetErrorName(result));
  }
  print_and_log(settings, "# Rank %d using %s device id %d\n", settings.rank, properties.name, selected);

  chunk->staging_left_send = static_cast<double *>(std::malloc(sizeof(double) * comms_lr_len));
  chunk->staging_left_recv = static_cast<double *>(std::malloc(sizeof(double) * comms_lr_len));
  chunk->staging_right_send = static_cast<double *>(std::malloc(sizeof(double) * comms_lr_len));
  chunk->staging_right_recv = static_cast<double *>(std::malloc(sizeof(double) * comms_lr_len));

  chunk->staging_top_send = static_cast<double *>(std::malloc(sizeof(double) * comms_tb_len));
  chunk->staging_top_recv = static_cast<double *>(std::malloc(sizeof(double) * comms_tb_len));
  chunk->staging_bottom_send = static_cast<double *>(std::malloc(sizeof(double) * comms_tb_len));
  chunk->staging_bottom_recv = static_cast<double *>(std::malloc(sizeof(double) * comms_tb_len));

  allocate_device_buffer(&chunk->density0, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->density, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->energy0, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->energy, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->u, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->u0, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->p, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->r, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->mi, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->w, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->kx, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->ky, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->sd, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->volume, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->x_area, chunk->x + 1, chunk->y);
  allocate_device_buffer(&chunk->y_area, chunk->x, chunk->y + 1);
  allocate_device_buffer(&chunk->cell_x, chunk->x, 1);
  allocate_device_buffer(&chunk->cell_y, 1, chunk->y);
  allocate_device_buffer(&chunk->cell_dx, chunk->x, 1);
  allocate_device_buffer(&chunk->cell_dy, 1, chunk->y);
  allocate_device_buffer(&chunk->vertex_dx, chunk->x + 1, 1);
  allocate_device_buffer(&chunk->vertex_dy, 1, chunk->y + 1);
  allocate_device_buffer(&chunk->vertex_x, chunk->x + 1, 1);
  allocate_device_buffer(&chunk->vertex_y, 1, chunk->y + 1);
  allocate_device_buffer(&chunk->ext->d_reduce_buffer, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->ext->d_reduce_buffer2, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->ext->d_reduce_buffer3, chunk->x, chunk->y);
  allocate_device_buffer(&chunk->ext->d_reduce_buffer4, chunk->x, chunk->y);

  allocate_device_buffer(&chunk->left_send, comms_lr_len, 1);
  allocate_device_buffer(&chunk->left_recv, comms_lr_len, 1);
  allocate_device_buffer(&chunk->right_send, comms_lr_len, 1);
  allocate_device_buffer(&chunk->right_recv, comms_lr_len, 1);
  allocate_device_buffer(&chunk->top_send, comms_tb_len, 1);
  allocate_device_buffer(&chunk->top_recv, comms_tb_len, 1);
  allocate_device_buffer(&chunk->bottom_send, comms_tb_len, 1);
  allocate_device_buffer(&chunk->bottom_recv, comms_tb_len, 1);

  allocate_host_buffer(&(chunk->cg_alphas), settings.max_iters, 1);
  allocate_host_buffer(&(chunk->cg_betas), settings.max_iters, 1);
  allocate_host_buffer(&(chunk->cheby_alphas), settings.max_iters, 1);
  allocate_host_buffer(&(chunk->cheby_betas), settings.max_iters, 1);
}

__global__ void set_chunk_data_vertices(int x, int y, int halo_depth, double dx, double dy, double x_min, double y_min, double *vertex_x,
                                        double *vertex_y, double *vertex_dx, double *vertex_dy) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < x + 1) {
    vertex_x[gid] = x_min + dx * (gid - halo_depth);
    vertex_dx[gid] = dx;
  }

  if (gid < y + 1) {
    vertex_y[gid] = y_min + dy * (gid - halo_depth);
    vertex_dy[gid] = dy;
  }
}

// Extended kernel for the chunk initialisation
__global__ void set_chunk_data(int x, int y, double dx, double dy, double *cell_x, double *cell_y, double *cell_dx, double *cell_dy,
                               const double *vertex_x, const double *vertex_y, double *volume, double *x_area, double *y_area) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < x) {
    cell_x[gid] = 0.5 * (vertex_x[gid] + vertex_x[gid + 1]);
    cell_dx[gid] = dx;
  }

  if (gid < y) {
    cell_y[gid] = 0.5 * (vertex_y[gid] + vertex_y[gid + 1]);
    cell_dy[gid] = dy;
  }

  if (gid < x * y) {
    volume[gid] = dx * dy;
  }

  if (gid < (x + 1) * y) {
    x_area[gid] = dy;
  }

  if (gid < x * (y + 1)) {
    y_area[gid] = dx;
  }
}

__global__ void set_chunk_initial_state(const int x, const int y, const double default_energy, const double default_density,
                                        double *energy0, double *density) {
  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= x * y) return;

  energy0[gid] = default_energy;
  density[gid] = default_density;
}

__global__ void set_chunk_state(const int x, const int y, const double *vertex_x, const double *vertex_y, const double *cell_x,
                                const double *cell_y, double *density, double *energy0, double *u, State state) {
  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  const int x_loc = gid % x;
  const int y_loc = gid / x;
  int apply_state = 0;

  if (gid < x * y) {
    if (state.geometry == Geometry::RECTANGULAR) {
      apply_state = (vertex_x[x_loc + 1] >= state.x_min && vertex_x[x_loc] < state.x_max && vertex_y[y_loc + 1] >= state.y_min &&
                     vertex_y[y_loc] < state.y_max);
    } else if (state.geometry == Geometry::CIRCULAR) {
      double radius = sqrt((cell_x[x_loc] - state.x_min) * (cell_x[x_loc] - state.x_min) +
                           (cell_y[y_loc] - state.y_min) * (cell_y[y_loc] - state.y_min));

      apply_state = (radius <= state.radius);
    } else if (state.geometry == Geometry::POINT) {
      apply_state = (vertex_x[x_loc] == state.x_min && vertex_y[y_loc] == state.y_min);
    }

    // Check if state applies at this vertex, and apply
    if (apply_state) {
      energy0[gid] = state.energy;
      density[gid] = state.density;
    }
  }

  if (x_loc > 0 && x_loc < x - 1 && y_loc > 0 && y_loc < y - 1) {
    u[gid] = energy0[gid] * density[gid];
  }
}

void run_set_chunk_data(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);
  double x_min = settings.grid_x_min + settings.dx * (double)chunk->left;
  double y_min = settings.grid_y_min + settings.dy * (double)chunk->bottom;
  int num_threads = 1 + std::max(chunk->x, chunk->y);
  int num_blocks = ceil((double)num_threads / (double)BLOCK_SIZE);
  set_chunk_data_vertices<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, settings.halo_depth, settings.dx, settings.dy, x_min, y_min,
                                                      chunk->vertex_x, chunk->vertex_y, chunk->vertex_dx, chunk->vertex_dy);
  num_blocks = ceil((double)(chunk->x * chunk->y) / (double)BLOCK_SIZE);
  set_chunk_data<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, settings.dx, settings.dy, chunk->cell_x, chunk->cell_y, chunk->cell_dx,
                                             chunk->cell_dy, chunk->vertex_x, chunk->vertex_y, chunk->volume, chunk->x_area, chunk->y_area);
  KERNELS_END();
}

void run_set_chunk_state(Chunk *chunk, Settings &settings, State *states) {
  KERNELS_START(0);
  set_chunk_initial_state<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, states[0].energy, states[0].density, chunk->energy0,
                                                      chunk->density);
  for (int ii = 1; ii < settings.num_states; ++ii) {
    set_chunk_state<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, chunk->vertex_x, chunk->vertex_y, chunk->cell_x, chunk->cell_y,
                                                chunk->density, chunk->energy0, chunk->u, states[ii]);
  }
  KERNELS_END();
}

void run_kernel_finalise(Chunk *chunk, Settings &) {
  std::free(chunk->cg_alphas);
  std::free(chunk->cg_betas);
  std::free(chunk->cheby_alphas);
  std::free(chunk->cheby_betas);
}
