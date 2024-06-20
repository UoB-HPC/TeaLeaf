TeaLeaf
====

A C++based implementation of the TeaLeaf heat conduction mini-app.
This implementation of TeaLeaf replicates the functionality of the reference version of
TeaLeaf (<https://github.com/UK-MAC/TeaLeaf_ref>).

This implementation has support for building with and without MPI.
When MPI is enabled, all models will adjust accordingly for asynchronous MPI send/recv.

This implementation supersedes out past porting efforts:

- <https://github.com/UoB-HPC/TeaLeaf-Kokkos>
- <https://github.com/UoB-HPC/TeaLeaf-OpenMP4>
- <https://github.com/UoB-HPC/TeaLeaf_sycl>

## Programming Models

TeaLeaf is currently implemented in the following parallel programming models, listed in no
particular order:

- CUDA
- HIP
- OpenMP
- OpenMP target
- C++ Parallel STL (StdPar)
- Kokkos >= 4
- SYCL and SYCL 2020

Planned:

- OpenACC
- RAJA
- TBB
- Thrust (via CUDA or HIP)

## Building

Drivers, compiler and software applicable to whichever implementation you would like to build
against is required.

### CMake

The project supports building with CMake >= 3.13.0, which can be installed without root via
the [official script](https://cmake.org/download/).

Each implementation (programming model) is built as follows:

```shell
$ cd TeaLeaf

# configure the build, build type defaults to Release
# The -DMODEL flag is required
$ cmake -Bbuild -H. -DMODEL=<model> -DENABLE_MPI=ON <model specific flags prefixed with -D...>

# compile
$ cmake --build build

# run executables in ./build
$ ./build/<model>-tealeaf
```

The `MODEL` option selects one implementation of TeaLeaf to build.
The source for each model's implementations are located in `./src/<model>`.

## File Input

The contents of `tea.in` defines the geometric and run time information, apart from task and thread
counts.

A complete list of options is given below, where `<R>` shows the option takes a real number as an
argument. Similarly `<I>` is an integer argument.

- `initial_timestep <R>` - Set the initial time step for TeaLeaf. This time step stays constant through the entire simulation.
The default value is
- `end_time <R>` - Sets the end time for the simulation. When the simulation time is greater than this number the
simulation will stop.
- `end_step <I>` - Sets the end step for the simulation. When the simulation step is equal to this then simulation will
stop. In the event that both `end_time` and `end_step` are set, the simulation will terminate on whichever completes first.

- `xmin <R>`,`xmax <R>`,`ymin <R>`,`ymax <R>` - Set the size of the computational domain. The default domain size is a 10cm
square.
- `x_cells <I>`, `y_cells <I>` - et the cell count for each coordinate direction. The default is 10 cells in
each direction.

The geometric information and initial conditions are set using the following keywords with three
possible variations. Note that state 1 is always the ambient material and any geometry information
is ignored. Areas not covered by other defined states receive the energy and density of state 1.

- `state <I> density <R> energy <R> geometry rectangle xmin <R> ymin <R> xmax <R> ymax <R>` - Defines a rectangular region of the domain with the specified energy and density.

- `state <I> density <R> energy <R> geometry circle xmin <R> ymin <R> radius <R>` - Defines a circular region of the domain with the specified energy and density.

- `state <I> density <R> energy <R> geometry point xmin <R> ymin <R>` - Defines a cell in the domain with the specified energy and density. Note that the generator is simple and the defined state completely fills a cell with which it intersects. In the case of over lapping regions, the last state takes priority. Hence a circular region will have a stepped interface and a point data will fill the cell it lies in with its defined energy and density.

- `visit_frequency <I>` - This is the step frequency of visualisations dumps. The files produced are text base VTK files and
are easily viewed in an application such as ViSit. The default is to output no graphical data. Note
that the overhead of output is high, so should not be invoked when performance benchmarking is being
carried out.
- `summary_frequency <I>` - This is the step frequency of summary dumps. This requires a global reduction and associated
synchronisation, so performance will be slightly affected as the frequency is increased. The default
is for a summary dump to be produced every 10 steps and at the end of the simulation.
- `tl_ch_cg_presteps  <I>` - This option specifies the number of Conjugate Gradient iterations completed before the Chebyshev
method is started. This is necessary to provide approximate minimum and maximum eigen values to
start the Chebyshev method. The default value is 30.
- `tl_ppcg_inner_steps <I>` - Number of inner steps to run when using the PPCG solver. The default value is 10.
- `tl_ch_cg_errswitch` - If enabled alongside Chebshev/PPCG solver, switch when a certain error is reached instead of when a
certain number of steps is reached. The default for this is off.
- `tl_ch_cg_epslim` - Default error to switch from CG to Chebyshev when using Chebyshev solver with the tl_cg_ch_errswitch
option enabled. The default value is 1e-5.

- `tl_check_result` - After the solver reaches convergence, calculate ||b-Ax|| to make sure the solver has actually
converged. The default for this option is off.

- `tl_preconditioner_type` - This keyword invokes the pre-conditioner. Options are:
  - `none` - No preconditioner.
  - `jac_diag` - Diagonal Jacobi preconditioner. Typically reduces condition number by around 5% but
  may not reduce time to solution
  - `jac_block` - Block Jacobi preconditioner (with a currently hardcoded block size of 4). Typically
  reduces the condition number by around 50% but may not reduce time to solution
- `tl_use_jacobi` - This keyword selects the Jacobi method to solve the linear system. Note that this a very slowly converging method compared to other options. This is the default method is no method is explicitly selected.
- `tl_use_cg` - This keyword selects the Conjugate Gradient method to solve the linear system.
- `tl_use_ppcg` - This keyword selects the Conjugate Gradient method to solve the linear system.
- `tl_use_chebyshev` - This keyword selects the Chebyshev method to solve the linear system.
- `profiler_on` - This option does not currently work. Instead compile with the `-DENABLE_PROFILING` flag being passed
to the OPTIONS parameter specified to the make command.
- `verbose_on` - The option prints out extra information such as residual per iteration of a solve.
- `tl_max_iters <I>` - This option provides an upper limit of the number of iterations used for the linear solve in a step.
If this limit is reached, then the solution vector at this iteration is used as the solution, even
if the convergence criteria has not been met. For this reason, care should be taken in the
comparison of the performance of a slowly converging method, such as Jacobi, as the convergence
criteria may not have been met for some of the steps. The default value is 1000.

- `tl_eps <R>` - This option sets the convergence criteria for the selected solver. It uses a least squares measure
of the residual. The default value is 1.0e-10.
- `tl_coefficient_density` - This option uses the density as the conduction coefficient. This is the default option.
- `tl_coefficient_inverrse_density` - This option uses the inverse density as the conduction coefficient.
- `test_problem <I>` - This keyword selects a standard test with a "known" solution. Test problem 1 is automatically
generated if the tea.in file does not exist. Test problems 2-5 are shipped in the TeaLeaf
repository. Note that the known solution for an iterative solver is not an analytic solution but is
the solution for a single core simulation with IEEE options enabled with the Intel compiler and a
strict convergence of 1.0e-15. The difference to the expected solution is reported at the end of the
simulation in the tea.out file. There is no default value for this option.

# Licence

```
Crown Copyright 2014 AWE.

Copyright (c) 2016-24 Tom Deakin, Simon McIntosh-Smith.
Copyright (c) 2024 Wei-Chen Lin
Copyright (c) 2020 Will Hawkins
Copyright (c) 2016 Matthew Martineau 


TeaLeaf is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the 
Free Software Foundation, either version 3 of the License, or (at your option) 
any later version.

TeaLeaf is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
details.

You should have received a copy of the GNU General Public License along with
TeaLeaf. If not, see http://www.gnu.org/licenses/.
 ```
