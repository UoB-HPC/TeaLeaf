#ifdef OMP_TARGET

  #include "application.h"
  #include "drivers.h"

void solve(Chunk *chunks, Settings &settings, int tt, const double *wallclock_prev);

// An implementation specific overload of the main timestep loop
bool diffuse_overload(Chunk *chunks, Settings &settings) {
  int n = chunks->x * chunks->y;

  print_and_log(settings, "This implementation overloads the diffuse function.\n");

  // Currently have to place all structure enclose pointers
  // into local variables for OMP 4.0 to accept them in mapping clauses
  double *r = chunks->r;
  double *sd = chunks->sd;
  double *kx = chunks->kx;
  double *ky = chunks->ky;
  double *w = chunks->w;
  double *p = chunks->p;
  double *cheby_alphas = chunks->cheby_alphas;
  double *cheby_betas = chunks->cheby_betas;
  double *cg_alphas = chunks->cg_alphas;
  double *cg_betas = chunks->cg_betas;
  double *energy = chunks->energy;
  double *density = chunks->density;
  double *energy0 = chunks->energy0;
  double *density0 = chunks->density0;
  double *u = chunks->u;
  double *u0 = chunks->u0;

  double *left_send = chunks->left_send;
  double *left_recv = chunks->left_recv;
  double *right_send = chunks->right_send;
  double *right_recv = chunks->right_recv;
  double *top_send = chunks->top_send;
  double *top_recv = chunks->top_recv;
  double *bottom_send = chunks->bottom_send;
  double *bottom_recv = chunks->bottom_recv;

  settings.is_offload = true;

  int lr_len = chunks->y * settings.halo_depth * NUM_FIELDS;
  int tb_len = chunks->x * settings.halo_depth * NUM_FIELDS;

  #pragma omp target enter data map(to : r[ : n], sd[ : n], kx[ : n], ky[ : n], w[ : n], p[ : n], cheby_alphas[ : settings.max_iters], \
                                        cheby_betas[ : settings.max_iters], cg_alphas[ : settings.max_iters],                          \
                                        cg_betas[ : settings.max_iters])                                                               \
      map(to : density[ : n], energy[ : n], density0[ : n], energy0[ : n], u[ : n], u0[ : n]),                                         \
      map(alloc : left_send[ : lr_len], left_recv[ : lr_len], right_send[ : lr_len], right_recv[ : lr_len], top_send[ : tb_len],       \
              top_recv[ : tb_len], bottom_send[ : tb_len], bottom_recv[ : tb_len])

  double wallclock_prev = 0.0;
  for (int tt = 0; tt < settings.end_step; ++tt) {
    solve(chunks, settings, tt, &wallclock_prev);
  }

  #pragma omp target exit data map(from : density[ : n], energy[ : n], density0[ : n], energy0[ : n], u[ : n], u0[ : n])

  settings.is_offload = false;

  return field_summary_driver(chunks, settings, true);
}

#endif