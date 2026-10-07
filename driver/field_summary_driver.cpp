#include "chunk.h"
#include "comms.h"
#include "kernel_interface.h"

bool get_checking_value(Settings &settings, double *checking_value);

// Invokes the set chunk data kernel
bool field_summary_driver(Chunk *chunks, Settings &settings, bool is_solve_finished) {
  double vol = 0.0;
  double ie = 0.0;
  double temp = 0.0;
  double mass = 0.0;

  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc) {
    if (settings.kernel_language == Kernel_Language::C) {
      run_field_summary(&(chunks[cc]), settings, &vol, &mass, &ie, &temp);
    } else if (settings.kernel_language == Kernel_Language::FORTRAN) {
    }
  }

  // Bring all of the results to the master
  sum_over_ranks(settings, &vol);
  sum_over_ranks(settings, &mass);
  sum_over_ranks(settings, &ie);
  sum_over_ranks(settings, &temp);

  if (settings.rank == MASTER && settings.check_result && is_solve_finished) {
    print_and_log(settings, "\n Checking results...\n");

    double checking_value = 1.0;
    if (!get_checking_value(settings, &checking_value)) return false;

    print_and_log(settings, " Expected %.15e\n", checking_value);
    print_and_log(settings, " Actual   %.15e\n", temp);

    double qa_diff = fabs(100.0 * (temp / checking_value) - 100.0);
    if (qa_diff < 0.001 && !std::isnan(temp)) {
      print_and_log(settings, " This run PASSED (Difference is within %.8lf%%)\n", qa_diff);
      return true;
    } else {
      print_and_log(settings, " This run FAILED (Difference is within %.8lf%%)\n", qa_diff);
      return false;
    }
  }
  // only master needs to return validation failure if we see one
  return true;
}

// Fetches the checking value from the test problems file
bool get_checking_value(Settings &settings, double *checking_value) {
  FILE *test_problem_file = std::fopen(settings.test_problem_filename, "r");

  if (!test_problem_file) {
    print_and_log(settings, "\n WARNING: Could not open the test problem file: %s, expected value will be invalid.\n",
                  settings.test_problem_filename);
    return false;
  }

  size_t len = 0;
  char *line = nullptr;

  // Get the number of states present in the config file
  while (getline(&line, &len, test_problem_file) != EOF) {
    int x = 0;
    int y = 0;
    int num_steps = 0;
    double candidate = 0.0;

    if (std::sscanf(line, "%d %d %d %lf", &x, &y, &num_steps, &candidate) != 4) continue;

    // Found the problem in the file
    if (x == settings.grid_x_cells && y == settings.grid_y_cells && num_steps == settings.completed_steps) {
      *checking_value = candidate;
      std::free(line);
      std::fclose(test_problem_file);
      return true;
    }
  }

  *checking_value = 1.0;
  print_and_log(settings, "\n WARNING: Problem was not found in the test problems file, expected value will be invalid.\n");
  std::free(line);
  std::fclose(test_problem_file);
  return false;
}
