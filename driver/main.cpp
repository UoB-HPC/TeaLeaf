#include <optional>

#include "application.h"
#include "chunk.h"
#include "comms.h"
#include "drivers.h"
#include "shared.h"

void settings_overload(Settings &settings, int argc, char **argv) {
  for (int aa = 1; aa < argc; ++aa) {
    // Overload the solver
    if (tealeaf_strmatch(argv[aa], "-solver") || tealeaf_strmatch(argv[aa], "--solver") || tealeaf_strmatch(argv[aa], "-s")) {
      if (aa + 1 == argc) break;
      if (tealeaf_strmatch(argv[aa + 1], "cg")) settings.solver = Solver::CG_SOLVER;
      if (tealeaf_strmatch(argv[aa + 1], "cheby")) settings.solver = Solver::CHEBY_SOLVER;
      if (tealeaf_strmatch(argv[aa + 1], "ppcg")) settings.solver = Solver::PPCG_SOLVER;
      if (tealeaf_strmatch(argv[aa + 1], "jacobi")) settings.solver = Solver::JACOBI_SOLVER;
    } else if (tealeaf_strmatch(argv[aa], "-x")) {
      if (aa + 1 == argc) break;
      settings.grid_x_cells = std::atoi(argv[aa]);
    } else if (tealeaf_strmatch(argv[aa], "-y")) {
      if (aa + 1 == argc) break;
      settings.grid_y_cells = std::atoi(argv[aa]);
    } else if (tealeaf_strmatch(argv[aa], "--staging-buffer")) {
      if (aa + 1 == argc) break;
      if (tealeaf_strmatch(argv[aa + 1], "true")) settings.staging_buffer_preference = StagingBuffer::ENABLE;
      if (tealeaf_strmatch(argv[aa + 1], "false")) settings.staging_buffer_preference = StagingBuffer::DISABLE;
      if (tealeaf_strmatch(argv[aa + 1], "auto ")) settings.staging_buffer_preference = StagingBuffer::AUTO;
    } else if (tealeaf_strmatch(argv[aa], "-d") || tealeaf_strmatch(argv[aa], "--device")) {
      if (aa + 1 == argc) break;
      settings.device_selector = argv[aa + 1];
    } else if (tealeaf_strmatch(argv[aa], "--problems") || tealeaf_strmatch(argv[aa], "-p")) {
      if (aa + 1 == argc) break;
      settings.test_problem_filename = argv[aa + 1];
    } else if (tealeaf_strmatch(argv[aa], "--in") || tealeaf_strmatch(argv[aa], "-i") || tealeaf_strmatch(argv[aa], "--file") ||
               tealeaf_strmatch(argv[aa], "-f")) {
      if (aa + 1 == argc) break;
      settings.tea_in_filename = argv[aa + 1];
    } else if (tealeaf_strmatch(argv[aa], "--out") || tealeaf_strmatch(argv[aa], "-o")) {
      if (aa + 1 == argc) break;
      settings.tea_out_filename = argv[aa + 1];
    } else if (tealeaf_strmatch(argv[aa], "-help") || tealeaf_strmatch(argv[aa], "--help") || tealeaf_strmatch(argv[aa], "-h")) {
      print_and_log(settings, "tealeaf <options>\n");
      print_and_log(settings, "options:\n");
      print_and_log(settings, "\t-solver, --solver, -s:\n");
      print_and_log(settings, "\t\tCan be 'cg', 'cheby', 'ppcg', or 'jacobi'\n");
      print_and_log(settings, "\t-p, --problems:\n");
      print_and_log(settings, "\t\tProblems file path'\n");
      print_and_log(settings, "\t-i, --in, -f, --file:\n");
      print_and_log(settings, "\t\tInput deck file path'\n");
      print_and_log(settings, "\t-o, --out:\n");
      print_and_log(settings, "\t\tOutput file path'\n");
      print_and_log(settings, "\t--staging-buffer:\n");
      print_and_log(settings, "\t\tIf true, use a host staging buffer for device-host MPI halo exchange.'\n");
      print_and_log(settings, "\t\tIf false, use device pointers directly for MPI halo exchange.'\n");
      print_and_log(settings, "\t\tDefaults to auto which elides the buffer if a device-aware (i.e CUDA-aware) is used.'\n");
      print_and_log(settings, "\t\tThis option is no-op for CPU-only models.'\n");
      print_and_log(settings, "\t\tSetting this to false on an MPI that is not device-aware may cause a segfault.'\n");
      finalise_comms();
      std::exit(EXIT_SUCCESS);
    }
  }
}

int main(int argc, char **argv) {
  // Immediately initialise MPI
  initialise_comms(argc, argv);

  barrier();

  // Create the settings wrapper
  Settings settings;
  set_default_settings(settings);
  settings_overload(settings, argc, argv);

  // Fill in rank information
  initialise_ranks(settings);
  initialise_log(settings);

  barrier();

#ifdef ENABLE_PROFILING
  bool profiling = true;
#else
  bool profiling = false;
#endif

#ifdef NO_MPI
  bool mpi_enabled = false;
#else
  bool mpi_enabled = true;
#endif

#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  std::optional<bool> mpi_cuda_aware_header = true;
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
  std::optional<bool> mpi_cuda_aware_header = false;
#else
  std::optional<bool> mpi_cuda_aware_header = {};
#endif

#if defined(MPIX_CUDA_AWARE_SUPPORT)
  std::optional<bool> mpi_cuda_aware_runtime = MPIX_Query_cuda_support() != 0;
#else
  std::optional<bool> mpi_cuda_aware_runtime = {};
#endif

#if defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM
  std::optional<bool> mpi_hip_aware_header = true;
#elif defined(OMPI_HAVE_MPI_EXT_ROCM) && !OMPI_HAVE_MPI_EXT_ROCM
  std::optional<bool> mpi_hip_aware_header = false;
#else
  std::optional<bool> mpi_hip_aware_header = {};
#endif
#if defined(OMPI_HAVE_MPI_EXT_ROCM)
  std::optional<bool> mpi_hip_aware_runtime = MPIX_Query_rocm_support() != 0;
#else
  std::optional<bool> mpi_hip_aware_runtime = {};
#endif

  initialise_model_info(settings);
  State *states{};
  read_config(settings, &states);

  switch (settings.staging_buffer_preference) {
    case StagingBuffer::ENABLE: settings.staging_buffer = true; break;
    case StagingBuffer::DISABLE: settings.staging_buffer = false; break;
    case StagingBuffer::AUTO:
      settings.staging_buffer = !(mpi_cuda_aware_header.value_or(false) && mpi_cuda_aware_runtime.value_or(false));
      break;
  }

  std::string execution_kind;
  switch (settings.model_kind) {
    case ModelKind::Host: execution_kind = "Host"; break;
    case ModelKind::Offload: execution_kind = "Offload"; break;
    case ModelKind::Unified: execution_kind = "Unified"; break;
  }

  print_and_log(settings, "TeaLeaf:\n");
  print_and_log(settings, " - Ver.:     %s\n", TEALEAF_VERSION);
  print_and_log(settings, " - Deck:     %s\n", settings.tea_in_filename);
  print_and_log(settings, " - Out:      %s\n", settings.tea_out_filename);
  print_and_log(settings, " - Problem:  %s\n", settings.test_problem_filename);
  print_and_log(settings, " - Solver:   %s\n", settings.solver_name);
  print_and_log(settings, " - Profiler: %s\n", profiling ? "true" : "false");
  print_and_log(settings, "Model:\n");
  print_and_log(settings, " - Name:      %s\n", settings.model_name.c_str());
  print_and_log(settings, " - Execution: %s\n", execution_kind.c_str());

  // Perform initialisation steps
  Chunk *chunks{};
  initialise_application(&chunks, settings, states);

  print_and_log(settings, "MPI:\n");
  print_and_log(settings, " - Enabled:     %s\n", mpi_enabled ? "true" : "false");
  print_and_log(settings, " - Total ranks: %d\n", settings.num_ranks);
  print_and_log(settings, " - Header device-awareness (CUDA-awareness):  %s\n",
                (mpi_cuda_aware_header ? (*mpi_cuda_aware_header ? "true" : "false") : "unknown"));
  print_and_log(settings, " - Runtime device-awareness (CUDA-awareness): %s\n",
                (mpi_cuda_aware_runtime ? (*mpi_cuda_aware_runtime ? "true" : "false") : "unknown"));
  print_and_log(settings, " - Header device-awareness (HIP-awareness):   %s\n",
                (mpi_hip_aware_header ? (*mpi_hip_aware_header ? "true" : "false") : "unknown"));
  print_and_log(settings, " - Runtime device-awareness (HIP-awareness):  %s\n",
                (mpi_hip_aware_runtime ? (*mpi_hip_aware_runtime ? "true" : "false") : "unknown"));
  print_and_log(settings, " - Host-Device halo exchange staging buffer:  %s\n", (settings.staging_buffer ? "true" : "false"));

  long chunk_comms_total_x = 0, chunk_comms_total_y = 0;
  for (int i = 0; i < settings.num_chunks_per_rank; ++i) {
    chunk_comms_total_x += chunks[i].x * settings.halo_depth * NUM_FIELDS;
    chunk_comms_total_y += chunks[i].y * settings.halo_depth * NUM_FIELDS;
  }
  long global_chunks_total_x = 0, global_chunks_total_y = 0;
  MPI_Reduce(&chunk_comms_total_x, &global_chunks_total_x, 1, MPI_LONG, MPI_SUM, MASTER, MPI_COMM_WORLD);
  MPI_Reduce(&chunk_comms_total_y, &global_chunks_total_y, 1, MPI_LONG, MPI_SUM, MASTER, MPI_COMM_WORLD);
  print_and_log(settings, " - X buffer elements: %ld\n", global_chunks_total_x);
  print_and_log(settings, " - Y buffer elements: %ld\n", global_chunks_total_y);
  print_and_log(settings, " - X buffer size:     %ld KB\n", chunk_comms_total_x * sizeof(double) / 1000);
  print_and_log(settings, " - Y buffer size:     %ld KB\n", chunk_comms_total_y * sizeof(double) / 1000);

  print_and_log(settings, "# ---- \n");
  print_and_log(settings, "Output: |+1\n");

  // Perform the solve using default or overloaded diffuse
#ifndef DIFFUSE_OVERLOAD
  bool valid = diffuse(chunks, settings);
#else
  bool valid = diffuse_overload(chunks, settings);
#endif

  // Print the kernel-level profiling results
  if (settings.rank == MASTER) {
    PRINT_PROFILING_RESULTS(settings.kernel_profile);
  }

  print_and_log(settings, "Result:\n");
  print_and_log(settings, " - Problem: %dx%d@%d\n", settings.grid_x_cells, settings.grid_y_cells, settings.end_step);
  print_and_log(settings, " - Outcome: %s\n", (!valid ? "FAILED" : "PASSED"));

  // Finalise the kernel
  kernel_finalise_driver(chunks, settings);

  // Finalise each individual chunk
  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc) {
    finalise_chunk(&(chunks[cc]));
    std::free(&(chunks[cc]));
  }

  profiler_finalise(&settings.kernel_profile);
  profiler_finalise(&settings.application_profile);
  profiler_finalise(&settings.wallclock_profile);

  // Finalise the application
  finalise_comms();

  return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
