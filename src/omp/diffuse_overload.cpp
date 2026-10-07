#ifdef OMP_TARGET

  #include "application.h"
  #include "drivers.h"

double solve(Chunk *chunks, Settings &settings, int tt, double *wallclock_prev);

void enter_chunk_data(Chunk &chunk, const Settings &settings) {
  int n = chunk.x * chunk.y;
  double *r = chunk.r;
  double *sd = chunk.sd;
  double *kx = chunk.kx;
  double *ky = chunk.ky;
  double *w = chunk.w;
  double *p = chunk.p;
  double *cheby_alphas = chunk.cheby_alphas;
  double *cheby_betas = chunk.cheby_betas;
  double *cg_alphas = chunk.cg_alphas;
  double *cg_betas = chunk.cg_betas;
  double *energy = chunk.energy;
  double *density = chunk.density;
  double *energy0 = chunk.energy0;
  double *density0 = chunk.density0;
  double *u = chunk.u;
  double *u0 = chunk.u0;
  double *left_send = chunk.left_send;
  double *left_recv = chunk.left_recv;
  double *right_send = chunk.right_send;
  double *right_recv = chunk.right_recv;
  double *top_send = chunk.top_send;
  double *top_recv = chunk.top_recv;
  double *bottom_send = chunk.bottom_send;
  double *bottom_recv = chunk.bottom_recv;
  int lr_len = chunk.y * settings.halo_depth * NUM_FIELDS;
  int tb_len = chunk.x * settings.halo_depth * NUM_FIELDS;

  #pragma omp target enter data map(to : r[ : n], sd[ : n], kx[ : n], ky[ : n], w[ : n], p[ : n], cheby_alphas[ : settings.max_iters], \
                                        cheby_betas[ : settings.max_iters], cg_alphas[ : settings.max_iters],                          \
                                        cg_betas[ : settings.max_iters])                                                               \
      map(to : density[ : n], energy[ : n], density0[ : n], energy0[ : n], u[ : n], u0[ : n]),                                         \
      map(alloc : left_send[ : lr_len], left_recv[ : lr_len], right_send[ : lr_len], right_recv[ : lr_len], top_send[ : tb_len],       \
              top_recv[ : tb_len], bottom_send[ : tb_len], bottom_recv[ : tb_len])
}

void exit_chunk_data(Chunk &chunk, const Settings &settings) {
  int n = chunk.x * chunk.y;
  double *r = chunk.r;
  double *sd = chunk.sd;
  double *kx = chunk.kx;
  double *ky = chunk.ky;
  double *w = chunk.w;
  double *p = chunk.p;
  double *cheby_alphas = chunk.cheby_alphas;
  double *cheby_betas = chunk.cheby_betas;
  double *cg_alphas = chunk.cg_alphas;
  double *cg_betas = chunk.cg_betas;
  double *energy = chunk.energy;
  double *density = chunk.density;
  double *energy0 = chunk.energy0;
  double *density0 = chunk.density0;
  double *u = chunk.u;
  double *u0 = chunk.u0;
  double *left_send = chunk.left_send;
  double *left_recv = chunk.left_recv;
  double *right_send = chunk.right_send;
  double *right_recv = chunk.right_recv;
  double *top_send = chunk.top_send;
  double *top_recv = chunk.top_recv;
  double *bottom_send = chunk.bottom_send;
  double *bottom_recv = chunk.bottom_recv;
  int lr_len = chunk.y * settings.halo_depth * NUM_FIELDS;
  int tb_len = chunk.x * settings.halo_depth * NUM_FIELDS;

  #pragma omp target exit data map(from : density[ : n], energy[ : n], density0[ : n], energy0[ : n], u[ : n], u0[ : n])                   \
      map(delete : r[ : n], sd[ : n], kx[ : n], ky[ : n], w[ : n], p[ : n], cheby_alphas[ : settings.max_iters],                           \
              cheby_betas[ : settings.max_iters], cg_alphas[ : settings.max_iters], cg_betas[ : settings.max_iters], left_send[ : lr_len], \
              left_recv[ : lr_len], right_send[ : lr_len], right_recv[ : lr_len], top_send[ : tb_len], top_recv[ : tb_len],                \
              bottom_send[ : tb_len], bottom_recv[ : tb_len])
}

// An implementation specific overload of the main timestep loop
bool diffuse_overload(Chunk *chunks, Settings &settings) {
  print_and_log(settings, "This implementation overloads the diffuse function.\n");

  settings.is_offload = true;
  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc)
    enter_chunk_data(chunks[cc], settings);

  double wallclock_prev = 0.0;
  double time = 0.0;
  for (int tt = 0; tt < settings.end_step; ++tt) {
    time += solve(chunks, settings, tt, &wallclock_prev);
    settings.completed_steps = tt + 1;
    if (time + 1.0e-16 > settings.end_time) break;
  }

  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc)
    exit_chunk_data(chunks[cc], settings);

  settings.is_offload = false;

  return field_summary_driver(chunks, settings, true);
}

#endif
