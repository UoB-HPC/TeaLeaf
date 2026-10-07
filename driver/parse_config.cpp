#include "application.h"
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int read_states(FILE *tea_in, Settings &settings, State **states);
void read_settings(FILE *tea_in, Settings &settings);
void read_value(const char *line, const char *word, char *value);
bool starts_with(const char *word, const char *line);
bool starts_get_double(const char *key, const char *line, char *word, double *value);
bool starts_get_int(const char *key, const char *line, char *word, int *value);
double parse_double(const char *key, const char *word);

// Read configuration file
void read_config(Settings &settings, State **states) {
  // Open the configuration file
  FILE *tea_in = fopen(settings.tea_in_filename, "r");
  if (!tea_in) {
    die(__LINE__, __FILE__, "Could not open input file %s\n", settings.tea_in_filename);
  }

  // Read all of the settings from the config
  read_settings(tea_in, settings);
  rewind(tea_in);

  // Read in the states
  settings.num_states = read_states(tea_in, settings, states);

  fclose(tea_in);

  print_to_log(settings, "Solution Parameters:\n");
  print_to_log(settings, "\tdt_init = %f\n", settings.dt_init);
  print_to_log(settings, "\tend_time = %f\n", settings.end_time);
  print_to_log(settings, "\tend_step = %d\n", settings.end_step);
  print_to_log(settings, "\tgrid_x_min = %f\n", settings.grid_x_min);
  print_to_log(settings, "\tgrid_y_min = %f\n", settings.grid_y_min);
  print_to_log(settings, "\tgrid_x_max = %f\n", settings.grid_x_max);
  print_to_log(settings, "\tgrid_y_max = %f\n", settings.grid_y_max);
  print_to_log(settings, "\tgrid_x_cells = %d\n", settings.grid_x_cells);
  print_to_log(settings, "\tgrid_y_cells = %d\n", settings.grid_y_cells);
  print_to_log(settings, "\tpresteps = %d\n", settings.presteps);
  print_to_log(settings, "\tppcg_inner_steps = %d\n", settings.ppcg_inner_steps);
  print_to_log(settings, "\teps_lim = %f\n", settings.eps_lim);
  print_to_log(settings, "\tmax_iters = %d\n", settings.max_iters);
  print_to_log(settings, "\teps = %f\n", settings.eps);
  print_to_log(settings, "\thalo_depth = %d\n", settings.halo_depth);
  print_to_log(settings, "\tcheck_result = %d\n", settings.check_result);
  print_to_log(settings, "\tcoefficient = %d\n", settings.coefficient);
  print_to_log(settings, "\tnum_chunks_per_rank = %d\n", settings.num_chunks_per_rank);
  print_to_log(settings, "\tsummary_frequency = %d\n", settings.summary_frequency);

  for (int ss = 0; ss < settings.num_states; ++ss) {
    print_to_log(settings, "\t\nstate %d\n", ss);
    print_to_log(settings, "\tdensity = %.12E\n", (*states)[ss].density);
    print_to_log(settings, "\tenergy= %.12E\n", (*states)[ss].energy);
    if (ss > 0) {
      print_to_log(settings, "\tx_min = %.12E\n", (*states)[ss].x_min);
      print_to_log(settings, "\ty_min = %.12E\n", (*states)[ss].y_min);
      print_to_log(settings, "\tx_max = %.12E\n", (*states)[ss].x_max);
      print_to_log(settings, "\ty_max = %.12E\n", (*states)[ss].y_max);
      print_to_log(settings, "\tradius = %.12E\n", (*states)[ss].radius);
      print_to_log(settings, "\tgeometry = %d\n", (*states)[ss].geometry);
    }
  }
}

// Read all settings from the configuration file
void read_settings(FILE *tea_in, Settings &settings) {
  size_t len = 0;
  char *line = nullptr;

  // Get the number of states present in the config file
  while (getline(&line, &len, tea_in) != EOF) {

    std::vector<char> word(len);

    // Parse the key-value pairs
    if (starts_get_double("initial_timestep", line, word.data(), &settings.dt_init)) continue;
    if (starts_get_double("end_time", line, word.data(), &settings.end_time)) continue;
    if (starts_get_int("end_step", line, word.data(), &settings.end_step)) continue;
    if (starts_get_double("xmin", line, word.data(), &settings.grid_x_min)) continue;
    if (starts_get_double("ymin", line, word.data(), &settings.grid_y_min)) continue;
    if (starts_get_double("xmax", line, word.data(), &settings.grid_x_max)) continue;
    if (starts_get_double("ymax", line, word.data(), &settings.grid_y_max)) continue;
    if (settings.grid_x_cells == DEF_GRID_X_CELLS && starts_get_int("x_cells", line, word.data(), &settings.grid_x_cells)) continue;
    if (settings.grid_y_cells == DEF_GRID_Y_CELLS && starts_get_int("y_cells", line, word.data(), &settings.grid_y_cells)) continue;
    if (starts_get_int("summary_frequency", line, word.data(), &settings.summary_frequency)) continue;
    if (starts_get_int("tl_ch_cg_presteps", line, word.data(), &settings.presteps) ||
        starts_get_int("presteps", line, word.data(), &settings.presteps))
      continue;
    if (starts_get_int("tl_ppcg_inner_steps", line, word.data(), &settings.ppcg_inner_steps) ||
        starts_get_int("ppcg_inner_steps", line, word.data(), &settings.ppcg_inner_steps))
      continue;
    if (starts_get_double("tl_ch_cg_epslim", line, word.data(), &settings.eps_lim) ||
        starts_get_double("epslim", line, word.data(), &settings.eps_lim))
      continue;
    if (starts_get_int("tl_max_iters", line, word.data(), &settings.max_iters) ||
        starts_get_int("max_iters", line, word.data(), &settings.max_iters))
      continue;
    if (starts_get_double("tl_eps", line, word.data(), &settings.eps) || starts_get_double("eps", line, word.data(), &settings.eps))
      continue;
    if (starts_get_int("tiles_per_task", line, word.data(), &settings.num_chunks_per_rank) ||
        starts_get_int("num_chunks_per_rank", line, word.data(), &settings.num_chunks_per_rank))
      continue;
    if (starts_get_int("halo_depth", line, word.data(), &settings.halo_depth)) continue;
    if (starts_get_int("test_problem", line, word.data(), &settings.test_problem)) {
      settings.check_result = settings.test_problem > 0;
      continue;
    }

    // Parse the switches
    if (starts_with("check_result", line)) {
      settings.check_result = true;
      continue;
    }
    if (starts_with("tl_ch_cg_errswitch", line) || starts_with("errswitch", line)) {
      settings.error_switch = true;
      continue;
    }
    if (starts_with("use_fortran_kernels", line)) {
      die(__LINE__, __FILE__, "Fortran kernels are not available in this C++ port.\n");
    }
    if (starts_with("use_c_kernels", line)) {
      settings.kernel_language = Kernel_Language::C;
      continue;
    }
    if (starts_with("tl_use_jacobi", line) || starts_with("use_jacobi", line)) {
      settings.solver = Solver::JACOBI_SOLVER;
      strcpy(settings.solver_name, "Jacobi");
      continue;
    }
    if (starts_with("tl_use_cg", line) || starts_with("use_cg", line)) {
      settings.solver = Solver::CG_SOLVER;
      strcpy(settings.solver_name, "CG");
      continue;
    }
    if (starts_with("tl_use_chebyshev", line) || starts_with("use_chebyshev", line)) {
      settings.solver = Solver::CHEBY_SOLVER;
      strcpy(settings.solver_name, "Chebyshev");
      continue;
    }
    if (starts_with("tl_use_ppcg", line) || starts_with("use_ppcg", line)) {
      settings.solver = Solver::PPCG_SOLVER;
      strcpy(settings.solver_name, "PPCG");
      continue;
    }
    if (starts_with("tl_coefficient_density", line) || starts_with("coefficient_density", line)) {
      settings.coefficient = CONDUCTIVITY;
      continue;
    }
    if (starts_with("tl_coefficient_inverse_density", line) || starts_with("coefficient_inverse_density", line)) {
      settings.coefficient = RECIP_CONDUCTIVITY;
      continue;
    }
    if (starts_with("tl_check_result", line) || starts_with("tl_preconditioner_type", line) || starts_with("preconditioner_on", line) ||
        starts_with("tiles_per_problem", line) || starts_with("sub_tiles_per_tile", line) || starts_with("reflective_boundary", line) ||
        starts_with("visit_frequency", line) || starts_with("verbose_on", line)) {
      die(__LINE__, __FILE__, "Unsupported UK-MAC input option: %s", line);
    }
  }
  std::free(line);

  if (settings.grid_x_cells <= 0 || settings.grid_y_cells <= 0) die(__LINE__, __FILE__, "Cell counts must be positive.\n");
  if (settings.dt_init <= 0.0) die(__LINE__, __FILE__, "The initial timestep must be positive.\n");
  if (settings.end_step < 0 || settings.end_time < 0.0) die(__LINE__, __FILE__, "End limits cannot be negative.\n");
  if (settings.summary_frequency < 0) die(__LINE__, __FILE__, "Summary frequency cannot be negative.\n");
  if (settings.num_chunks_per_rank <= 0) die(__LINE__, __FILE__, "Chunks per rank must be positive.\n");
  if (settings.halo_depth <= 0) die(__LINE__, __FILE__, "Halo depth must be positive.\n");

  // Set the cell widths now
  settings.dx = (settings.grid_x_max - settings.grid_x_min) / (double)settings.grid_x_cells;
  settings.dy = (settings.grid_y_max - settings.grid_y_min) / (double)settings.grid_y_cells;
}

// Read all of the states from the configuration file
int read_states(FILE *tea_in, Settings &settings, State **states) {
  size_t len = 0;
  char *line = nullptr;
  int num_states = 0;

  // First find the number of states
  while (getline(&line, &len, tea_in) != EOF) {
    int state_num = 0;
    std::vector<char> word(len);

    if (starts_get_int("state", line, word.data(), &state_num)) {
      num_states = tealeaf_MAX(num_states, state_num);
    }
  }

  rewind(tea_in);

  if (num_states < 1) die(__LINE__, __FILE__, "No states are defined.\n");

  // Pre-initialise the set of states
  *states = (State *)malloc(sizeof(State) * num_states);
  for (int ss = 0; ss < num_states; ++ss) {
    (*states)[ss].defined = false;
  }

  // If a state boundary falls exactly on a cell boundary
  // then round off can cause the state to be put one cell
  // further than expected. This is compiler/system dependent.
  // To avoid this, a state boundary is reduced/increased by a
  // 100th of a cell width so it lies well within the intended
  // cell. Because a cell is either full or empty of a specified
  // state, this small modification to the state extents does
  // not change the answer.
  while (getline(&line, &len, tea_in) != EOF) {
    int state_num = 0;
    std::vector<char> word(len);

    // State found
    if (starts_get_int("state", line, word.data(), &state_num)) {
      if (state_num < 1 || state_num > num_states) die(__LINE__, __FILE__, "Invalid state number %d.\n", state_num);
      State *state = &((*states)[state_num - 1]);

      if (state->defined) {
        die(__LINE__, __FILE__, "State number %d defined twice.\n", state_num);
      }

      read_value(line, "density", word.data());
      state->density = parse_double("density", word.data());
      read_value(line, "energy", word.data());
      state->energy = parse_double("energy", word.data());

      // State 1 is the default state so geometry irrelevant
      if (state_num > 1) {
        read_value(line, "xmin", word.data());
        state->x_min = parse_double("xmin", word.data()) + settings.dx / 100.0;
        read_value(line, "ymin", word.data());
        state->y_min = parse_double("ymin", word.data()) + settings.dy / 100.0;
        read_value(line, "xmax", word.data());
        state->x_max = parse_double("xmax", word.data()) - settings.dx / 100.0;
        read_value(line, "ymax", word.data());
        state->y_max = parse_double("ymax", word.data()) - settings.dy / 100.0;

        read_value(line, "geometry", word.data());

        if (tealeaf_strmatch(word.data(), "rectangle")) {
          state->geometry = Geometry::RECTANGULAR;
        } else if (tealeaf_strmatch(word.data(), "circle") || tealeaf_strmatch(word.data(), "circular")) {
          state->geometry = Geometry::CIRCULAR;

          read_value(line, "radius", word.data());
          state->radius = parse_double("radius", word.data());
        } else if (tealeaf_strmatch(word.data(), "point")) {
          state->geometry = Geometry::POINT;
        } else {
          die(__LINE__, __FILE__, "Unknown geometry: %s\n", word.data());
        }
      }

      state->defined = true;
    }
  }

  std::free(line);

  for (int ss = 0; ss < num_states; ++ss) {
    if (!(*states)[ss].defined) die(__LINE__, __FILE__, "State number %d is not defined.\n", ss + 1);
  }

  return num_states;
}

// Checks line starts with word
bool starts_with(const char *word, const char *line) {
  int num_matched = 0;
  int word_len = std::strlen(word);

  for (int ll = 0; ll < (int)strlen(line); ++ll) {
    // Skip leading spaces
    if (!num_matched && std::isspace(static_cast<unsigned char>(line[ll]))) {
      continue;
    }

    // Match the word
    if (line[ll] != word[num_matched]) {
      return false;
    } else if (++num_matched == word_len) {
      const char next = line[ll + 1];
      return next == '\0' || next == '=' || std::isspace(static_cast<unsigned char>(next));
    }
  }

  return false;
}

// Parses key-value pairs for state in configuration file
void read_value(const char *line, const char *word, char *value) {
  const std::size_t word_len = std::strlen(word);
  const char *cursor = line;
  while ((cursor = std::strstr(cursor, word))) {
    const bool starts_token = cursor == line || std::isspace(static_cast<unsigned char>(cursor[-1]));
    const char after_word = cursor[word_len];
    const bool ends_token = after_word == '=' || std::isspace(static_cast<unsigned char>(after_word));
    if (starts_token && ends_token) {
      cursor += word_len;
      while (*cursor == '=' || std::isspace(static_cast<unsigned char>(*cursor)))
        ++cursor;
      if (*cursor && std::sscanf(cursor, "%s", value) == 1) return;
      break;
    }
    cursor += word_len;
  }

  die(__LINE__, __FILE__, "Failed to find a value for key '%s'\n", word);
}

// Gets key value pair by checking that the line starts with key and getting value
bool starts_get_int(const char *key, const char *line, char *word, int *value) {
  if (starts_with(key, line)) {
    read_value(line, key, word);
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(word, &end, 10);
    if (errno || !end || *end || parsed < INT_MIN || parsed > INT_MAX) {
      die(__LINE__, __FILE__, "Invalid integer for '%s': %s\n", key, word);
    }
    *value = static_cast<int>(parsed);
    return true;
  }

  return false;
}

// Gets key value pair by checking that the line starts with key and getting value
bool starts_get_double(const char *key, const char *line, char *word, double *value) {
  if (starts_with(key, line)) {
    read_value(line, key, word);
    *value = parse_double(key, word);
    return true;
  }

  return false;
}

double parse_double(const char *key, const char *word) {
  char *end = nullptr;
  errno = 0;
  const double parsed = std::strtod(word, &end);
  if (errno || !end || *end || !std::isfinite(parsed)) {
    die(__LINE__, __FILE__, "Invalid real number for '%s': %s\n", key, word);
  }
  return parsed;
}
