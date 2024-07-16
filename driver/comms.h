#pragma once

#ifndef NO_MPI
  // XXX OpenMPI pulls in CXX headers which we don't link against, prevent that:
  #define OMPI_SKIP_MPICXX
  #include <mpi.h>
  #if __has_include("mpi-ext.h") // C23, but everyone supports this already
    #include "mpi-ext.h"         // for CUDA-aware MPI checks
  #endif
#else
  #include "mpi_shim.h"
#endif

#include "chunk.h"
#include "settings.h"

void barrier();
void abort_comms();
void finalise_comms();
void initialise_comms(int argc, char **argv);
void initialise_ranks(Settings &settings);
void sum_over_ranks(Settings &settings, double *a);
void min_over_ranks(Settings &settings, double *a);
void wait_for_requests(Settings &settings, int num_requests, MPI_Request *requests);
void send_recv_message(Settings &settings, const double *send_buffer, double *recv_buffer, int buffer_len, int neighbour, int send_tag,
                       int recv_tag, MPI_Request *send_request, MPI_Request *recv_request);