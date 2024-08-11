#!/usr/bin/env bash

set -eu

export NVHPC_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/24.7"
export CUDA_DIR="$NVHPC_DIR/cuda/"
export PATH=$NVHPC_DIR/compilers/bin/:${PATH:-}
export KOKKOS_DIR="/home/tom/Downloads/kokkos-4.0.01/"

export HIPSYCL_DEBUG_LEVEL=1

export CPU_RANKS=16
export GPU_RANKS=3

VERBOSE="ON"
PROBLEM="tea.in"

declare -A enabled_models=(
  ["serial"]=true
  ["omp"]=true
  ["omp-target"]=true
  ["kokkos"]=true
  ["cuda"]=true
  ["hip"]=true
  ["std-indices"]=true
  ["sycl-acc"]=true
  ["sycl-usm"]=true
)

function test_mpi() {
  if [[ "${enabled_models[$3]}" != true ]]; then
    echo "# Skipping model $3"
    return
  fi
  rm -rf "build_$3" # /CMakeCache.txt
  echo "$2" "${@:4}"
  cmake -DCMAKE_BUILD_TYPE=Release -S. -B "build_$3" -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DENABLE_MPI=ON -DMODEL="$3" "${@:4}" # &>/dev/null
  cmake --build "build_$3"                                                                                                           # &>/dev/null
  export OMP_PLACES=cores
  export OMP_PROC_BIND=true
  export OMP_NUM_THREADS=1
  export OMP_TARGET_OFFLOAD=MANDATORY
  # | grep -i -e "This run" -e "Timestep *" #
  export ASAN_OPTIONS=detect_leaks=0
  mkdir -p out

  which mpirun
  #  ldd build_$3/*-tealeaf
  mpirun -np "$1" -bind-to core -map-by core sh -c "build_$3/*-tealeaf --device $2 --file $PROBLEM --staging-buffer auto"
  # --mca pml ucx -x UCX_TLS=rc,sm,cuda_copy,gdr_copy,cuda_ipc
  # konsole -e gdb -ex run --args
}

function test_nompi() {
  if [[ "${enabled_models[$2]}" != true ]]; then
    echo "# Skipping model $2"
    return
  fi
  rm -rf "build_$2" # /CMakeCache.txt
  echo "$2" "${@:3}"
  cmake -DCMAKE_BUILD_TYPE=Release -S. -B "build_$2" -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DENABLE_MPI=OFF -DMODEL="$2" "${@:3}" # &>/dev/null
  cmake --build "build_$2"                                                                                                            # &>/dev/null
  export OMP_TARGET_OFFLOAD=MANDATORY
  #  export OMP_PLACES=cores
  #  export OMP_PROC_BIND=true
  export OMP_NUM_THREADS=$(nproc)
  # | grep -i -e "This run" -e "Timestep *" #
  export ASAN_OPTIONS=detect_leaks=0
  mkdir -p out
  build_$2/*-tealeaf --device "$1" --file "$PROBLEM" --profile true --staging-buffer auto
}

( 
  (
    :
    test_nompi "0" serial -DCXX_EXTRA_FLAGS="-Ofast" -DCMAKE_CXX_COMPILER=g++
    test_nompi "0" serial -DCXX_EXTRA_FLAGS="-Ofast" -DCMAKE_CXX_COMPILER=clang++
  )
  (
    :
    test_nompi "0" std-indices -DCMAKE_CXX_COMPILER=g++ -DCXX_EXTRA_FLAGS="-Ofast" -DUSE_TBB=ON
    test_nompi "0" std-indices -DCMAKE_CXX_COMPILER=g++ -DCXX_EXTRA_FLAGS="-Ofast" -DUSE_ONEDPL=OPENMP
    test_nompi "0" std-indices -DCMAKE_CXX_COMPILER=clang++ -DCXX_EXTRA_FLAGS="-Ofast" -DUSE_TBB=ON
    test_nompi "0" std-indices -DCMAKE_CXX_COMPILER=clang++ -DCXX_EXTRA_FLAGS="-Ofast" -DUSE_ONEDPL=OPENMP
    test_nompi "0" std-indices -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast;-stdpar;-target=multicore"
    test_nompi "0" std-indices -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast;-stdpar;-gpu=cc61"
  )
  (
    :
    test_nompi "0" omp -DCMAKE_CXX_COMPILER=g++ -DCXX_EXTRA_FLAGS="-Ofast"
    test_nompi "0" omp -DCMAKE_CXX_COMPILER=clang++ # clang fails with -Ofast, solutions if off by 0.008%
  )
  (
    :
    test_nompi "0" omp-target -DCMAKE_CXX_COMPILER=clang++ -DOFFLOAD=NVIDIA:sm_60 -DCXX_EXTRA_FLAGS="-Ofast;--cuda-path=$CUDA_DIR"
    test_nompi "0" omp-target -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast" -DOFFLOAD=ON -DOFFLOAD_FLAGS="-mp;-gpu=cc60"
  )
  (
    :
    test_nompi "0" kokkos -DCMAKE_CXX_COMPILER=g++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN3=ON -DCXX_EXTRA_FLAGS="-Ofast"
    test_nompi "0" kokkos -DCMAKE_CXX_COMPILER=clang++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN3=ON -DCXX_EXTRA_FLAGS="-Ofast"
    test_nompi "0" kokkos -DCMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL61=ON
    test_nompi "0" kokkos -DCMAKE_CXX_COMPILER=nvc++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL61=ON
  )
  (
    :
    test_nompi "0" cuda -DCXX_EXTRA_FLAGS="-O3" -DCMAKE_CUDA_COMPILER=nvcc -DCUDA_ARCH=sm_60
    test_nompi "0" cuda -DMANAGED_ALLOC=ON -DCXX_EXTRA_FLAGS="-O3" -DCMAKE_CUDA_COMPILER=nvcc -DCUDA_ARCH=sm_60
    test_nompi "0" cuda -DSYNC_ALL_KERNELS=ON -DCXX_EXTRA_FLAGS="-O3" -DCMAKE_CUDA_COMPILER=nvcc -DCUDA_ARCH=sm_60
  )
  (
    :
    test_nompi "0" hip -DCXX_EXTRA_FLAGS="--offload-arch=gfx1036" -DCMAKE_CXX_COMPILER=hipcc
    test_nompi "0" hip -DMANAGED_ALLOC=ON -DCXX_EXTRA_FLAGS="--offload-arch=gfx1036" -DCMAKE_CXX_COMPILER=hipcc
    test_nompi "0" hip -DSYNC_ALL_KERNELS=ON -DCXX_EXTRA_FLAGS="--offload-arch=gfx1036" -DCMAKE_CXX_COMPILER=hipcc
  )
  (
    :
    export HIPSYCL_TARGETS="omp.library-only"
    test_nompi "OpenMP" sycl-acc -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    test_nompi "OpenMP" sycl-usm -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    export HIPSYCL_TARGETS="cuda:sm_60"
    test_nompi "NVIDIA" sycl-acc -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    test_nompi "NVIDIA" sycl-usm -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    export HSA_OVERRIDE_GFX_VERSION=10.1.2 #  https://github.com/RadeonOpenCompute/ROCm/issues/1756#issuecomment-1160386571
    export HIPSYCL_TARGETS="hip:gfx1012"
    test_nompi "Radeon" sycl-acc -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    test_nompi "Radeon" sycl-usm -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF #?-DENABLE_PROFILING=ON
  )
  (
    :
    set +eu
    module load mpi
    source /opt/intel/oneapi/tbb/2021.13/env/vars.sh
    source /opt/intel/oneapi/compiler/2024.2/env/vars.sh
    set -eu
    export DPCPP_CPU_NUM_CUS=1
    export DPCPP_CPU_SCHEDULE=static
    test_nompi "Ryzen" sycl-acc -DSYCL_COMPILER=ONEAPI-ICPX -DUSE_HOSTTASK=ON
    test_nompi "Ryzen" sycl-acc -DSYCL_COMPILER=ONEAPI-ICPX -DUSE_HOSTTASK=OFF
    test_nompi "Ryzen" sycl-usm -DSYCL_COMPILER=ONEAPI-ICPX -DUSE_HOSTTASK=ON
    test_nompi "Ryzen" sycl-usm -DSYCL_COMPILER=ONEAPI-ICPX -DUSE_HOSTTASK=OFF
  )
  (
    :
    set +eu
    module load mpi
    source /opt/intel/oneapi/tbb/2021.13/env/vars.sh
    source /opt/intel/oneapi/compiler/2024.2/env/vars.sh --include-intel-llvm
    set -eu
    cuda_sycl_flags="-fsycl-targets=nvptx64-nvidia-cuda;-Xsycl-target-backend;--cuda-gpu-arch=sm_60;--cuda-path=$CUDA_DIR"
    test_nompi "0" std-indices -DCMAKE_CXX_COMPILER=clang++ -DUSE_ONEDPL=DPCPP -DCXX_EXTRA_FLAGS="-fsycl;$cuda_sycl_flags"
    test_nompi "NVIDIA" sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=ON -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"
    test_nompi "NVIDIA" sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=OFF -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"
    test_nompi "NVIDIA" sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=ON -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"
    test_nompi "NVIDIA" sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=OFF -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"

    export LD_LIBRARY_PATH=/opt/rocm-5.4.3/lib:${LD_LIBRARY_PATH:-}
    hip_sycl_flags="-fsycl;-fsycl-targets=amdgcn-amd-amdhsa;-Xsycl-target-backend;--offload-arch=gfx1012"
    test_nompi "0" std-indices -DCMAKE_CXX_COMPILER=clang++ -DUSE_ONEDPL=DPCPP -DCXX_EXTRA_FLAGS="-fsycl;$hip_sycl_flags"
    test_nompi "Radeon" sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=ON -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags"
    test_nompi "Radeon" sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=OFF -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags"
    test_nompi "Radeon" sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=ON -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags"  #?-DENABLE_PROFILING=ON
    test_nompi "Radeon" sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=OFF -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags" #?-DENABLE_PROFILING=ON
  )
  wait
)
#exit 0
(
  source /etc/profile.d/modules.sh
  module load mpi
  test_mpi $CPU_RANKS "0" serial -DCMAKE_CXX_COMPILER=g++
  test_mpi $CPU_RANKS "0" serial -DCMAKE_CXX_COMPILER=clang++
  (
    :
    test_mpi $CPU_RANKS "0" std-indices -DCMAKE_CXX_COMPILER=g++ -DUSE_TBB=ON
    test_mpi $CPU_RANKS "0" std-indices -DCMAKE_CXX_COMPILER=g++ -DUSE_ONEDPL=OPENMP
    test_mpi $CPU_RANKS "0" std-indices -DCMAKE_CXX_COMPILER=clang++ -DUSE_TBB=ON
    test_mpi $CPU_RANKS "0" std-indices -DCMAKE_CXX_COMPILER=clang++ -DUSE_ONEDPL=OPENMP
    test_mpi $CPU_RANKS "0" std-indices -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast;-stdpar;-target=multicore"
    test_mpi $GPU_RANKS "0" std-indices -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast;-stdpar;-gpu=cc61;--restrict"
  )
  (
    :
    test_mpi $CPU_RANKS "0" omp -DCMAKE_CXX_COMPILER=g++
    test_mpi $CPU_RANKS "0" omp -DCMAKE_CXX_COMPILER=clang++
  )
  (
    :
    test_mpi $GPU_RANKS "0" omp-target -DCMAKE_CXX_COMPILER=clang++ -DOFFLOAD=NVIDIA:sm_60 -DCXX_EXTRA_FLAGS="-Ofast;--cuda-path=$CUDA_DIR"
    test_mpi $GPU_RANKS "0" omp-target -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast" -DOFFLOAD=ON -DOFFLOAD_FLAGS="-mp;-gpu=cc60"
  )
  (
    :
    test_mpi $CPU_RANKS "0" kokkos -DCMAKE_CXX_COMPILER=g++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN3=ON
    test_mpi $CPU_RANKS "0" kokkos -DCMAKE_CXX_COMPILER=clang++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN3=ON
    test_mpi $GPU_RANKS "0" kokkos -DCMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL61=ON
    test_mpi $GPU_RANKS "0" kokkos -DCMAKE_CXX_COMPILER=nvc++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL61=ON
  )
  (
    :
    test_mpi $GPU_RANKS "0" cuda -DMANAGED_ALLOC=ON -DCXX_EXTRA_FLAGS="-O3" -DCMAKE_CUDA_COMPILER=nvcc -DCUDA_ARCH=sm_60
    test_mpi $GPU_RANKS "0" cuda -DMANAGED_ALLOC=ON -DSYNC_ALL_KERNELS=ON -DCXX_EXTRA_FLAGS="-O3" -DCMAKE_CUDA_COMPILER=nvcc -DCUDA_ARCH=sm_60
  )
  (
    :
    test_mpi $GPU_RANKS "0" hip -DMANAGED_ALLOC=ON -DCXX_EXTRA_FLAGS="-O3" -DCMAKE_CXX_COMPILER=hipcc # fails validation at > O1 lol
    test_mpi $GPU_RANKS "0" hip -DMANAGED_ALLOC=ON -DSYNC_ALL_KERNELS=ON -DCXX_EXTRA_FLAGS="-O3" -DCMAKE_CXX_COMPILER=hipcc
  )
  (
    :
    set +eu
    module load mpi
    source /opt/intel/oneapi/tbb/2021.13/env/vars.sh
    source /opt/intel/oneapi/compiler/2024.2/env/vars.sh
    set -eu
    export DPCPP_CPU_NUM_CUS=1
    export DPCPP_CPU_SCHEDULE=static
    test_mpi $CPU_RANKS "Ryzen" sycl-acc -DSYCL_COMPILER=ONEAPI-ICPX -DUSE_HOSTTASK=ON
    test_mpi $CPU_RANKS "Ryzen" sycl-acc -DSYCL_COMPILER=ONEAPI-ICPX -DUSE_HOSTTASK=OFF
    test_mpi $CPU_RANKS "Ryzen" sycl-usm -DSYCL_COMPILER=ONEAPI-ICPX -DUSE_HOSTTASK=ON
    test_mpi $CPU_RANKS "Ryzen" sycl-usm -DSYCL_COMPILER=ONEAPI-ICPX -DUSE_HOSTTASK=OFF
  )
  (
    :
    export HIPSYCL_TARGETS="omp.library-only"
    test_mpi $CPU_RANKS "OpenMP" sycl-acc -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    test_mpi $CPU_RANKS "OpenMP" sycl-usm -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    export HIPSYCL_TARGETS="cuda:sm_60"
    test_mpi $GPU_RANKS "NVIDIA" sycl-acc -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    test_mpi $GPU_RANKS "NVIDIA" sycl-usm -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    export HSA_OVERRIDE_GFX_VERSION=10.1.2 #  https://github.com/RadeonOpenCompute/ROCm/issues/1756#issuecomment-1160386571
    export HIPSYCL_TARGETS="hip:gfx1012"
    test_mpi $GPU_RANKS "Radeon" sycl-acc -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    test_mpi $GPU_RANKS "Radeon" sycl-usm -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF #?-DENABLE_PROFILING=ON
  )
  (
    :
    set +eu
    module load mpi
    source /opt/intel/oneapi/tbb/2021.13/env/vars.sh
    source /opt/intel/oneapi/compiler/2024.2/env/vars.sh --include-intel-llvm
    set -eu
    cuda_sycl_flags="-fsycl-targets=nvptx64-nvidia-cuda;-Xsycl-target-backend;--cuda-gpu-arch=sm_60;--cuda-path=$CUDA_DIR"
    test_mpi $GPU_RANKS "0" std-indices -DCMAKE_CXX_COMPILER=clang++ -DUSE_ONEDPL=DPCPP -DCXX_EXTRA_FLAGS="-fsycl;$cuda_sycl_flags"
    test_mpi $GPU_RANKS "NVIDIA" sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=ON -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"
    test_mpi $GPU_RANKS "NVIDIA" sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=OFF -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"
    test_mpi $GPU_RANKS "NVIDIA" sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=ON -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"
    test_mpi $GPU_RANKS "NVIDIA" sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=OFF -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"

    export LD_LIBRARY_PATH=/opt/rocm-5.4.3/lib:${LD_LIBRARY_PATH:-}
    hip_sycl_flags="-fsycl;-fsycl-targets=amdgcn-amd-amdhsa;-Xsycl-target-backend;--offload-arch=gfx1012"
    test_mpi $GPU_RANKS "0" std-indices -DCMAKE_CXX_COMPILER=clang++ -DUSE_ONEDPL=DPCPP -DCXX_EXTRA_FLAGS="-fsycl;$hip_sycl_flags"
    test_mpi $GPU_RANKS "Radeon" sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=ON -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags"
    test_mpi $GPU_RANKS "Radeon" sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=OFF -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags"
    test_mpi $GPU_RANKS "Radeon" sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=ON -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags"  #?-DENABLE_PROFILING=ON
    test_mpi $GPU_RANKS "Radeon" sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=OFF -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags" #?-DENABLE_PROFILING=ON
  )
  wait
)

# exit 0

### CUDA-aware MPI ###
(
  export MPI_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/comm_libs/openmpi/openmpi-3.1.5/
  export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/comm_libs/openmpi/openmpi-3.1.5/bin/:${PATH:-}"
  (
    :
    test_mpi $GPU_RANKS "0" std-indices -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast;-stdpar;-gpu=cc61;--restrict"
    test_mpi $GPU_RANKS "0" omp-target -DCMAKE_CXX_COMPILER=clang++ -DOFFLOAD=NVIDIA:sm_60 -DCXX_EXTRA_FLAGS="-Ofast;--cuda-path=$CUDA_DIR"
    test_mpi $GPU_RANKS "0" omp-target -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast" -DOFFLOAD=ON -DOFFLOAD_FLAGS="-mp;-gpu=cc60"
    test_mpi $GPU_RANKS "0" kokkos -DCMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL61=ON
    test_mpi $GPU_RANKS "0" kokkos -DCMAKE_CXX_COMPILER=nvc++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL61=ON
    test_mpi $GPU_RANKS "0" cuda -DCXX_EXTRA_FLAGS="-Ofast" -DCMAKE_CUDA_COMPILER=nvcc -DCUDA_ARCH=sm_60

    #    test_mpi $GPU_RANKS "0" hip  -DCXX_EXTRA_FLAGS="-Ofast" -DCMAKE_CXX_COMPILER=/usr/lib/aomp_17.0-1/bin/hipcc # doesn't work with NVHPC'S OpenMPI, segfaults at runtime

  )
  (
    :
    export HIPSYCL_TARGETS="cuda:sm_60"
    test_mpi $GPU_RANKS "NVIDIA" sycl-acc -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    test_mpi $GPU_RANKS "NVIDIA" sycl-usm -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    export HSA_OVERRIDE_GFX_VERSION=10.1.2 #  https://github.com/RadeonOpenCompute/ROCm/issues/1756#issuecomment-1160386571
    export HIPSYCL_TARGETS="hip:gfx1012"
    test_mpi $GPU_RANKS "Radeon" sycl-acc -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF
    test_mpi $GPU_RANKS "Radeon" sycl-usm -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR="/opt/hipsycl/e7ec6ed" -DUSE_HOSTTASK=OFF #?-DENABLE_PROFILING=ON
  )
  exit 0
  (
    :
    set +eu
    source /opt/intel/oneapi/tbb/2021.13/env/vars.sh
    source /opt/intel/oneapi/compiler/2024.2/env/vars.sh --include-intel-llvm
    set -eu
    cuda_sycl_flags="-fsycl-targets=nvptx64-nvidia-cuda;-Xsycl-target-backend;--cuda-gpu-arch=sm_60;--cuda-path=$CUDA_DIR"
    test_mpi $GPU_RANKS "0" std-indices -DCMAKE_CXX_COMPILER=clang++ -DUSE_ONEDPL=DPCPP -DCXX_EXTRA_FLAGS="-fsycl;$cuda_sycl_flags"

    test_mpi $GPU_RANKS "NVIDIA" sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=ON -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"
    test_mpi $GPU_RANKS "NVIDIA" sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=ON -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"

    export LD_LIBRARY_PATH=/opt/rocm-5.4.3/lib:${LD_LIBRARY_PATH:-}
    hip_sycl_flags="-fsycl;-fsycl-targets=amdgcn-amd-amdhsa;-Xsycl-target-backend;--offload-arch=gfx1012"
    test_mpi $GPU_RANKS "0" std-indices -DCMAKE_CXX_COMPILER=clang++ -DUSE_ONEDPL=DPCPP -DCXX_EXTRA_FLAGS="-fsycl;$hip_sycl_flags;-O1" # again, needs -O1

    test_mpi $GPU_RANKS "Radeon" sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=ON -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags"
    test_mpi $GPU_RANKS "Radeon" sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DUSE_HOSTTASK=ON -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2024.2/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags" #?-DENABLE_PROFILING=ON
  )

)
echo "All done!"
