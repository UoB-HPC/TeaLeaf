#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

#include "application.h"
#include "chunk.h"
#include "comms.h"
#include "drivers.h"
#include "shared.h"

namespace {
void print_help() {
  std::puts("Usage: tealeaf [OPTIONS]\n\n"
            "Options:\n"
            "  -h, --help                         Print this message\n"
            "  -s, --solver <cg|cheby|ppcg|jacobi>\n"
            "                                     Select the linear solver\n"
            "  -x <CELLS>                         Override the x cell count\n"
            "  -y <CELLS>                         Override the y cell count\n"
            "  -d, --device <INDEX|NAME>          Select an accelerator device\n"
            "  -p, --problems <FILE>              Read expected solutions from FILE\n"
            "  -i, -f, --in, --file <FILE>        Read the input deck from FILE\n"
            "  -o, --out <FILE>                   Write the report to FILE\n"
            "      --staging-buffer <true|false|auto>\n"
            "                                     Control host staging for device-aware MPI");
}

char *read_argument(int &index, int argc, char **argv) {
  if (index + 1 < argc) return argv[++index];
  std::fprintf(stderr, "%s requires a value\n", argv[index]);
  print_help();
  finalise_comms();
  std::exit(EXIT_FAILURE);
}

int read_positive_integer(int &index, int argc, char **argv) {
  const char *value = read_argument(index, argc, argv);
  int parsed = 0;
  const auto result = std::from_chars(value, value + std::strlen(value), parsed);
  if (result.ec == std::errc{} && result.ptr == value + std::strlen(value) && parsed > 0) return parsed;
  std::fprintf(stderr, "%s requires a positive integer, got: %s\n", argv[index - 1], value);
  finalise_comms();
  std::exit(EXIT_FAILURE);
}

void replace_string(char *&destination, const char *value) {
  const std::size_t length = std::strlen(value) + 1;
  auto *replacement = static_cast<char *>(std::malloc(length));
  if (!replacement) {
    std::fputs("Unable to allocate command-line option\n", stderr);
    finalise_comms();
    std::exit(EXIT_FAILURE);
  }
  std::memcpy(replacement, value, length);
  std::free(destination);
  destination = replacement;
}
} // namespace

void settings_overload(Settings &settings, int argc, char **argv) {
  for (int aa = 1; aa < argc; ++aa) {
    // Overload the solver
    if (tealeaf_strmatch(argv[aa], "-solver") || tealeaf_strmatch(argv[aa], "--solver") || tealeaf_strmatch(argv[aa], "-s")) {
      const char *value = read_argument(aa, argc, argv);
      if (tealeaf_strmatch(value, "cg")) {
        settings.solver = Solver::CG_SOLVER;
        std::strcpy(settings.solver_name, "CG");
      } else if (tealeaf_strmatch(value, "cheby")) {
        settings.solver = Solver::CHEBY_SOLVER;
        std::strcpy(settings.solver_name, "Chebyshev");
      } else if (tealeaf_strmatch(value, "ppcg")) {
        settings.solver = Solver::PPCG_SOLVER;
        std::strcpy(settings.solver_name, "PPCG");
      } else if (tealeaf_strmatch(value, "jacobi")) {
        settings.solver = Solver::JACOBI_SOLVER;
        std::strcpy(settings.solver_name, "Jacobi");
      } else {
        std::fprintf(stderr, "Unknown solver: %s\n", value);
        finalise_comms();
        std::exit(EXIT_FAILURE);
      }
    } else if (tealeaf_strmatch(argv[aa], "-x")) {
      settings.grid_x_cells = read_positive_integer(aa, argc, argv);
    } else if (tealeaf_strmatch(argv[aa], "-y")) {
      settings.grid_y_cells = read_positive_integer(aa, argc, argv);
    } else if (tealeaf_strmatch(argv[aa], "--staging-buffer")) {
      const char *value = read_argument(aa, argc, argv);
      if (tealeaf_strmatch(value, "true")) settings.staging_buffer_preference = StagingBuffer::ENABLE;
      else if (tealeaf_strmatch(value, "false"))
        settings.staging_buffer_preference = StagingBuffer::DISABLE;
      else if (tealeaf_strmatch(value, "auto"))
        settings.staging_buffer_preference = StagingBuffer::AUTO;
      else {
        std::fprintf(stderr, "Unknown staging-buffer value: %s\n", value);
        finalise_comms();
        std::exit(EXIT_FAILURE);
      }
    } else if (tealeaf_strmatch(argv[aa], "-d") || tealeaf_strmatch(argv[aa], "--device")) {
      settings.device_selector = read_argument(aa, argc, argv);
    } else if (tealeaf_strmatch(argv[aa], "--problems") || tealeaf_strmatch(argv[aa], "-p")) {
      replace_string(settings.test_problem_filename, read_argument(aa, argc, argv));
    } else if (tealeaf_strmatch(argv[aa], "--in") || tealeaf_strmatch(argv[aa], "-i") || tealeaf_strmatch(argv[aa], "--file") ||
               tealeaf_strmatch(argv[aa], "-f")) {
      replace_string(settings.tea_in_filename, read_argument(aa, argc, argv));
    } else if (tealeaf_strmatch(argv[aa], "--out") || tealeaf_strmatch(argv[aa], "-o")) {
      replace_string(settings.tea_out_filename, read_argument(aa, argc, argv));
    } else if (tealeaf_strmatch(argv[aa], "-help") || tealeaf_strmatch(argv[aa], "--help") || tealeaf_strmatch(argv[aa], "-h")) {
      print_help();
      finalise_comms();
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", argv[aa]);
      print_help();
      finalise_comms();
      std::exit(EXIT_FAILURE);
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

  initialise_model_info(settings);
  State *states{};
  read_config(settings, &states);
  // Reapply options whose values intentionally override the input deck.
  settings_overload(settings, argc, argv);
  settings.dx = (settings.grid_x_max - settings.grid_x_min) / settings.grid_x_cells;
  settings.dy = (settings.grid_y_max - settings.grid_y_min) / settings.grid_y_cells;
  if (settings.ppcg_inner_steps == -1 && settings.solver == Solver::PPCG_SOLVER) {
    const double cells = static_cast<double>(settings.grid_x_cells) * settings.grid_y_cells;
    settings.ppcg_inner_steps = 4 * static_cast<int>(std::sqrt(std::sqrt(cells)));
  }

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
  print_and_log(settings, " - Problem: %dx%d@%d\n", settings.grid_x_cells, settings.grid_y_cells, settings.completed_steps);
  print_and_log(settings, " - Outcome: %s\n", (!valid ? "FAILED" : "PASSED"));

  // Finalise the kernel
  kernel_finalise_driver(chunks, settings);

  // Finalise each individual chunk
  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc) {
    finalise_chunk(&(chunks[cc]));
  }
  std::free(chunks);
  std::free(states);

  profiler_finalise(&settings.kernel_profile);
  profiler_finalise(&settings.application_profile);
  profiler_finalise(&settings.wallclock_profile);

  if (settings.tea_out_fp) std::fclose(settings.tea_out_fp);
  std::free(settings.fields_to_exchange);
  std::free(settings.solver_name);
  std::free(settings.tea_in_filename);
  std::free(settings.tea_out_filename);
  std::free(settings.test_problem_filename);

  // Finalise the application
  finalise_comms();

  return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
