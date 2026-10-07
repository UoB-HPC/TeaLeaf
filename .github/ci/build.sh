#!/usr/bin/env bash
set -euo pipefail

if [[ -n ${CI_ENV_SCRIPT:-} ]]; then
  set +u # Vendor environment scripts may reference unset variables.
  # shellcheck disable=SC1090
  source "$CI_ENV_SCRIPT"
  set -u
fi

read -r -a model_flags <<< "$CI_CMAKE_ARGS"
mpi_flags=(-DENABLE_MPI=OFF)
if [[ -n ${CI_MPI_CC:-} ]]; then
  mpi_flags=(-DENABLE_MPI=ON "-DMPI_C_COMPILER=$CI_MPI_CC"
    "-DMPI_CXX_COMPILER=$CI_MPI_CXX" "-DMPIEXEC_EXECUTABLE=$CI_MPI_EXEC")
  export OMPI_CC="$CC" OMPI_CXX="$CXX" MPICH_CC="$CC" MPICH_CXX="$CXX"
  "$CI_MPI_EXEC" --version
fi

cmake -S . -B "$CI_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release "-DBUILD_TESTING=$BUILD_TESTING" -DUSAGE=OFF \
  "-DCMAKE_C_COMPILER=$CC" "-DCMAKE_CXX_COMPILER=$CXX" \
  "-DMODEL=$CI_MODEL" "${mpi_flags[@]}" "${model_flags[@]}"
cmake --build "$CI_BUILD_DIR"
if [[ $BUILD_TESTING == ON ]]; then
  ctest --test-dir "$CI_BUILD_DIR" --output-on-failure --no-tests=error
fi
