---
parent: VERTEX-CFD v1.0 User Guide
title: Installation
nav_order: 1
---

# Installation
VERTEX-CFD supports both CPU and GPU solvers. Depending on the CPU/GPU supports, Trilinos needs to be compiled with the proper configurations as VERTEX-CFD relies on the Trilinos installation. For the VERTEX-CFD installation, Trilinos should be available in the system. Installation instructions for both Trilinos and VERTEX-CFD are available. If Trilinos is already built and ready, you can skip to VERTEX-CFD installation instructions. All of the scripts given in this file is also available in `docs/scripts` directory for users convenience. 

## Trilinos Installation
Trilinos supports both CPU and GPU if configured accordingly. For the installation with CPU support only, we have spack based procedure which is easier to follow. For GPU, as of now, Trilinos needs to be built manually. For both options we suggest using a compute node instead of login node as this will be a demanding process.

### Trilinos-CPU Installation
First of all, create a directory and get Spack.

```
mkdir trilinos
cd trilinos/
git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git
```

Next, create a file named `spack-trilinos16.yaml` which should contain the following configuration for Spack. Pay special attention to `EDIT` marks locations where edits may need to be made depending on machine. Modules already available in your machine can be loaded using the `packages` section. 

```
spack:
  concretizer:
    unify: when_possible

  config:
    build_stage:
    - $spack/var/spack/spack-stage/build-$arch-$date/
    misc_cache: $spack/.cache
    build_jobs: 16 # EDIT: change the number with the number of available cores.
    install_tree:
      root: $spack/opt/spack/
      projections:
        all: install-{os}-{target}-{compiler.name}-{compiler.version}/{name}-{version}
    keep_stage: true
    verify_ssl: true
    checksum: true
    dirty: false
    build_language: C
    ccache: false
    db_lock_timeout: 120
    package_lock_timeout: null
    shared_linking: rpath
    allow_sgid: true
    locks: true
    suppress_gpg_warnings: false
    connect_timeout: 10

  compilers:
  - compiler: # EDIT: change below with your compiler of choice
      spec: gcc@13.2.0
      paths:
        cc: /software/dev_tools/swtree/cs400/gcc/13.2.0/centos7.5_gnu8.5.0/bin/gcc
        cxx: /software/dev_tools/swtree/cs400/gcc/13.2.0/centos7.5_gnu8.5.0/bin/g++
        f77: /software/dev_tools/swtree/cs400/gcc/13.2.0/centos7.5_gnu8.5.0/bin/gfortran
        fc: /software/dev_tools/swtree/cs400/gcc/13.2.0/centos7.5_gnu8.5.0/bin/gfortran
      flags: {}
      operating_system: centos7 # EDIT: update according to `spack arch`: just the OS version
      target: x86_64
      modules: # EDIT: update with a list of modules you want to load by default
      - jobutils/1.0
      - StdEnv
      - python/3.11.5
      - openmpi/4.1.0

  packages:
    all:
      target: [broadwell] # EDIT: Change this according to `spack arch`
      compiler: [gcc@13.2.0] # EDIT: Change according to your compiler of choice
      providers: # EDIT: Change versions below according to the default in your machine
        mpi: [openmpi]
        blas: [intel-oneapi-mkl]
        lapack: [intel-oneapi-mkl]
        pkgconfig: [pkg-config]
      variants: +mpi

  specs:
  - ninja
  - boost
  - intel-oneapi-mkl
  - googletest
  - trilinos@16.0.0 +chaco+exodus+hdf5+intrepid2+kokkos+mpi+nox+openmp+panzer+phalanx+shards+shared+stk+tempus+zoltan2 build_type=RelWithDebInfo
```

Now, source the Spack setup script and create an environment named vertex. 

```
source spack/share/spack/setup-env.sh
spack env create vertex spack-trilinos16.yaml
spack env activate vertex
spack spec
```

 `spack spec` will show the configuration of what is to be installed, make sure the configuration matches what was requested in the configuration file. Next, install Trilinos and its dependencies.

```
spack install
```

If the session is ended for any reason, the user might need to reactivate the vertex spack env using `spack env activate vertex`. If all compiles without error, the modules are ready to use. `spack location -i trilinos` will return the location of the Trilinos install. The modules can be setup using

```
spack module tcl refresh
```
Trilinos can be called by loading the module as shown below. The folder names may be different depending on your machine configuration. We suggest finding the current folders instead of copying and pasting. As an example, on CADES in ORNL, they are located in:

```
module use /<SPACK INSTALL LOCATION>/share/spack/modules/linux-centos7-haswell/
module load trilinos/16.0.0-gcc-13.2.0-qzebvtj
```

### Trilinos-GPU Installation
VERTEX-CFD-GPU installation is not, as of now, supported by the Spack. Hence, it will be required to compile Trilinos with the GPU support manually. For the Trilinos-GPU installation, first step is to clone Trilinos from the GitHub repo and checkout to `Trilinos-16-0-0` as follows:

```
git clone https://github.com/trilinos/Trilinos.git
cd Trilinos
git checkout trilinos-release-16-0-branch
```

Once you checked out to the correct version, you can create a new folder for the building Trilinos as follows:

```
mkdir build
cd build
```

After that, modules required to build Trilinos needs to be loaded. For that, create a file with `trilinos-cuda-env.sh` name. Although the procedure for the module loading changes depend on the machine, as an example, the content of `trilinos-cuda-env.sh` in NERSC Perlmutter is as follows and you can adjust it according to your machine configurations:

```
#!/bin/bash

module load PrgEnv-gnu/8.5.0
module load gcc-native/12.3
module load cudatoolkit/12.4
module load craype-accel-nvidia80
module load cray-libsci/23.12.5
module load craype/2.7.30
module load cray-mpich/8.1.28
module load cray-hdf5-parallel/1.12.2.9
module load cray-netcdf-hdf5parallel/4.9.0.9
module load cray-parallel-netcdf/1.12.3.9
module load cmake/3.30.2

module load spack
spack env activate cuda
spack load metis
spack load parmetis

export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export OMP_STACKSIZE=128M
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDATOOLKIT_VERSION_STRING=${CRAY_CUDATOOLKIT_VERSION#*_}

export CRAY_ACCEL_TARGET="nvidia80"

export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export TPETRA_ASSUME_GPU_AWARE_MPI=0
```

Once this is ready, create a configuration script with `trilinos-cuda-config.sh` name and the following content:

```
#!/bin/bash
# This is a sample Trilinos configuration script for Albany on perlmutter

source trilinos-cuda-env.sh

# Cleanup old cmake files
rm -rf CMake*

# Set Trilinos build path
BUILD_DIR=`pwd`

# EDIT: Change this based on your NVCC WRAPPER location
NVCC_WRAPPER=<PATH TO NVCC WRAPPER DIRECTORY>

# EDIT: Change with path to Trilinos source directory
TRILINOS_SOURCE_DIR=<PATH TO TRILINOS SOURCE DIRECTORY>

# EDIT: Change with path to boost
BOOST_DIR=<PATH TO BOOST DIRECTORY>

TRILINOS_INSTALL_DIR=<DESIRED TRILINOS INSTALLATION DIRECTORY>

export CRAYPE_LINK_TYPE=dynamic
export CRAY_CPU_TARGET=x86-64

cmake \
  -D CMAKE_C_COMPILER=cc \
  -D CMAKE_CXX_COMPILER=${NVCC_WRAPPER} \
  -D CMAKE_Fortran_COMPILER=ftn \
\
  -D CMAKE_INSTALL_PREFIX:PATH=${TRILINOS_INSTALL_DIR} \
  -D CMAKE_BUILD_TYPE:STRING=RELEASE \
\
  -D BUILD_SHARED_LIBS=ON \
  -D Trilinos_ENABLE_ALL_PACKAGES=ON \
  -D Trilinos_ENABLE_SECONDARY_TESTED_CODE=ON \
  -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
  -D Trilinos_ENABLE_PyTrilinos=OFF \
  -D Trilinos_SHOW_DEPRECATED_WARNINGS=OFF \
  -D TPL_ENABLE_MPI=ON \
  -D TPL_Netcdf_PARALLEL=FALSE \
\
  -D TPL_ENABLE_HDF5:STRING=ON \
  -D HDF5_INCLUDE_DIRS:PATH=${HDF5_DIR}/include \
  -D HDF5_LIBRARY_DIRS:PATH=${HDF5_DIR}/lib\
\
  -D TPL_ENABLE_Netcdf=ON \
  -D Netcdf_INCLUDE_DIRS:PATH=${NETCDF_DIR}/include \
  -D Netcdf_LIBRARY_DIRS:PATH=${NETCDF_DIR}/lib \
\
  -D Kokkos_ENABLE_CUDA:BOOL=ON \
  -D Kokkos_ENABLE_CUDA_LAMBDA:BOOL=ON \
  -D Kokkos_ENABLE_CUDA_UVM:BOOL=OFF \
  -D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC:BOOL=OFF \
\
  -D TPL_ENABLE_Boost:BOOL=ON \
  -D TPL_ENABLE_BoostLib=ON \
  -D Boost_INCLUDE_DIRS:PATH=${BOOST_DIR}/include \
  -D Boost_LIBRARY_DIRS:PATH=${BOOST_DIR}/lib \
\
  -D TPL_ENABLE_Krino=OFF \
  -D TPL_ENABLE_Matio=OFF \
  -D TPL_ENABLE_CGNS=OFF \
  -D TPL_ENABLE_METIS=ON \
  -D TPL_ENABLE_ParMETIS=ON \
  -D Trilinos_ENABLE_Stokhos:BOOL=OFF \
\
  -D TPL_ENABLE_SuperLU:BOOL=OFF \
  -D ML_ENABLE_SuperLU:BOOL=OFF \
  -D TPL_ENABLE_BLAS:BOOL=ON \
  -D TPL_BLAS_LIBRARIES:STRING="${CRAY_PE_LIBSCI_PREFIX_DIR}/lib/libsci_gnu.so" \
  -D TPL_ENABLE_LAPACK:BOOL=ON \
  -D TPL_LAPACK_LIBRARIES:STRING="${CRAY_PE_LIBSCI_PREFIX_DIR}/lib/libsci_gnu.so" \
\
${TRILINOS_SOURCE_DIR}
```

As an example, in NERSC Perlmutter system, for `NVCC_WRAPPER`, we used the wrapper available in Sandia National Lab's GitHub page `https://github.com/sandialabs/Albany/blob/master/doc/LandIce/machines/perlmutter/nvcc_wrapper_a100`. After that, run the `trilinos-cuda-config.sh` script as:

```
./trilinos-cuda-config.sh
```

Once the configuration is completed, you can build and install Trilinos-16-0-0-GPU as follows:

```
make -j install
```

After the building and installation, Trilinos-GPU should be ready for VERTEX-CFD installation.

## VERTEX-CFD Installation
Once the Trilinos is installed in your system and ready to use, VERTEX-CFD can be installed. First of all, you can create a folder for VERTEX-CFD and clone from GitHub by using:
```
mkdir VERTEX-CFD
cd VERTEX-CFD
git clone https://github.com/ORNL/VERTEX-CFD.git
```
This will clone the VERTEX-CFD folder. In order to build it, create a new folder as build as follows:
```
mkdir build
cd build
```

In the build directory, depending on the CPU/GPU installation, you will need different environment files. For GPU, you can use `trilinos-cuda-config.sh`. There is no need for a separate configuration script. However, for CPU, you will need `vertex-env-cpu.sh`. Those environment scripts are system specific depending and it may change based on your system. Carefully modify them based on your system. The content of the `vertex-env-cpu.sh` is as follows:
```
#!/bin/bash

install=/<SPACK INSTALL LOCATION>/share/spack/modules/linux-centos7-haswell/
echo $install

module load gcc/13.2.0

module use $install
module load gcc-runtime
module load openmpi
module load cmake
module load ninja
module load boost
module load netcdf-c
module load parmetis
module load intel-oneapi-mkl
module load cgns
module load googletest
module load hdf5
module load zlib-ng
module load trilinos/16.0.0-gcc-13.2.0-qzebvtj
HDF5_DIR=/<SPACK INSTALL LOCATION>/opt/spack/install-centos7-haswell-gcc-13.2.0/hdf5-1.14.5/
ZLIB_DIR=/<SPACK INSTALL LOCATION>/opt/spack/install-centos7-haswell-gcc-13.2.0/zlib-ng-2.2.3/

GCC_ROOT=/<SYSTEM GCC ROOT>/
unset LIBRARY_PATH
```

Please note that, those scripts are system specific. Hence you will need to modify them depending on your system. Please carefully modify them and make sure they are loading the correct modules. The scripts that we shared are succesfully used on CADES (CPU configuration) in ORNL and NERSC Perlmutter (GPU configuration). Once you have the environment script, source the script with one of the lines below based on CPU/GPU installation:

```
source vertex-env-cpu.sh
source trilinos-cuda-env.sh
```

Once the environment script is sourced, VERTEX-CFD needs to be configured. Create a configuration script with `vertex-config.sh` name, which contains the following lines:
```
#!/bin/sh

SOURCE=/<PATH TO CLONED VERTEX-CFD FOLDER>
INSTALL="<PATH TO DESIRED INSTALLATION LOCATION>"
BUILD="RelWithDebInfo"

rm -rf CMake*
rm -rf .ninja*
rm DartConfiguration.tcl
rm CTestTestfile.cmake
rm build.ninja
rm VertexCFDConfig.cmake
rm -rf Testing

# EDIT: Uncomment this line for GPU and update this based on your NVCC WRAPPER location
#NVCC_WRAPPER=<PATH TO NVCC WRAPPER DIRECTORY>

# Unset variable set by spack modules.
# If this is present, any directories present will be treated
# as "implicit" library paths by CMake and it will strip them
# out of the executable RPATHS. Then you have to set
# LD_LIBRARY_PATH appropriately to run jobs.
unset LIBRARY_PATH

cmake \
    -D CMAKE_BUILD_TYPE=${BUILD} \
    -D CMAKE_INSTALL_PREFIX=${INSTALL} \
#    -D CMAKE_CXX_COMPILER=${NVCC_WRAPPER} \
    -D VertexCFD_ENABLE_COVERAGE_BUILD=OFF \
    -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -fdiagnostics-color" \
    -D VertexCFD_ENABLE_TESTING=ON \
    -D Trilinos_ROOT=<PATH TO TRILINOS/16.0.0> \
    \
    ${SOURCE}
```
Please notice that you need to define `NVCC_WRAPPER` and `CMAKE_CXX_COMPILER` for GPU version. Once the configuration script is ready, run it as follows:
```
./vertex-config.sh
```

When configured succesfully, you can use make to install as:
```
make -j install
```
If you wish to limit the number of cores in the installation, append `-j` flag with desired number of cores such as `-j32`. This will build and install VERTEX-CFD. The binary will be located in `<PATH TO DESIRED INSTALLATION LOCATION>/bin/vertexcfd`. Once the installation is completed, unit and regression tests can be run to make sure everything is working as expected. For the instructions for the testing, refer to the following sections.

# Testing
VERTEX-CFD development procedure strictly requires to develop a unit/regression test for each capability added. Hence, it is suggested to run those tests to make sure installation is properly completed and the functionalities are working as expected.

## Unit Tests
VERTEX-CFD has large amount of unit tests. In order to run those, you can use the below job submission scripts written for SLURM scheduler.

```
#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node 32
#SBATCH --time 0:10:00
#SBATCH --output output.log
#SBATCH --error error.log
                                                                                                                                        
source <PATH TO>/trilinos-cuda-config.sh
#source <PATH TO>/vertex-env-cpu.sh # For GPU comment above line and use this line.

export OMP_PROC_BIND=true
export OMP_PLACES=threads

export IOSS_PROPERTIES="COMPOSE_RESULTS=on:MINIMIZE_OPEN_FILES=on:MAXIMUM_NAME_LENGTH=64:DUPLICATE_FIELD_NAME_BEHAVIOR=WARNING"
export EXODUS_NETCDF4=1

export MAP_STRING=slot:pe=${SLURM_CPUS_PER_TASK}

ctest -j ${SLURM_NTASKS_PER_NODE}
```

For other scheduler, such as PBS, you can convert the SLURM submission script. The example output screen for the unit tests is:
```
        Start   1: VertexCFD_KokkosMPI_test_OPENMP_np_1_nt_1
        Start   2: VertexCFD_KokkosMPI_test_OPENMP_np_1_nt_2
        Start   3: VertexCFD_KokkosMPI_test_OPENMP_np_1_nt_4
        Start   5: VertexCFD_KokkosMPI_test_OPENMP_np_2_nt_1
        Start   6: VertexCFD_KokkosMPI_test_OPENMP_np_2_nt_2
        Start   7: VertexCFD_KokkosMPI_test_OPENMP_np_2_nt_4
        Start   8: VertexCFD_KokkosMPI_test_OPENMP_np_4_nt_1
  1/554 Test   #8: VertexCFD_KokkosMPI_test_OPENMP_np_4_nt_1 ..........................................   Passed    2.03 sec
        Start  12: VertexCFD_EvaluatorTestHarness_test_OPENMP_np_1_nt_1
        Start  13: VertexCFD_EvaluatorTestHarness_test_OPENMP_np_1_nt_2
        Start  23: VertexCFD_Version_test_OPENMP_nt_1
  2/554 Test   #2: VertexCFD_KokkosMPI_test_OPENMP_np_1_nt_2 ..........................................   Passed    2.61 sec
        Start  16: VertexCFD_EvaluatorTestHarness_test_OPENMP_np_2_nt_1
  3/554 Test   #1: VertexCFD_KokkosMPI_test_OPENMP_np_1_nt_1 ..........................................   Passed    2.61 sec
        Start  27: VertexCFD_ParameterPack_test_OPENMP_nt_1
  4/554 Test   #7: VertexCFD_KokkosMPI_test_OPENMP_np_2_nt_4 ..........................................   Passed    2.61 sec
        Start   9: VertexCFD_KokkosMPI_test_OPENMP_np_4_nt_2
  5/554 Test   #3: VertexCFD_KokkosMPI_test_OPENMP_np_1_nt_4 ..........................................   Passed    2.62 sec
  .
  .
  .
  553/554 Test #550: VertexCFD_DivergenceAdvectionTest_test_OPENMP_nt_192 ...............................   Passed    0.84 sec
        Start 554: VertexCFD_FullInductionMHDProperties_test_OPENMP_nt_192
  554/554 Test #554: VertexCFD_FullInductionMHDProperties_test_OPENMP_nt_192 ............................   Passed    0.36 sec

  97% tests passed, 14 tests failed out of 554

  Total Test time (real) = 1825.19 sec

  The following tests FAILED:
        11 - VertexCFD_KokkosMPI_test_OPENMP_np_192_nt_1 (Failed)
        22 - VertexCFD_EvaluatorTestHarness_test_OPENMP_np_192_nt_1 (Failed)                                                   
        71 - VertexCFD_MeshManager_test_OPENMP_nt_1 (Failed)
        72 - VertexCFD_MeshManager_test_OPENMP_nt_2 (Failed)
        73 - VertexCFD_MeshManager_test_OPENMP_nt_4 (Failed)
        74 - VertexCFD_MeshManager_test_OPENMP_nt_192 (Failed)
        79 - VertexCFD_InitialConditionManager_test_OPENMP_nt_1 (Failed)
        80 - VertexCFD_InitialConditionManager_test_OPENMP_nt_2 (Failed)
        81 - VertexCFD_InitialConditionManager_test_OPENMP_nt_4 (Failed)
        82 - VertexCFD_InitialConditionManager_test_OPENMP_nt_192 (Failed)
        169 - VertexCFD_ExternalFields_test_OPENMP_np_192_nt_1 (Failed)
        180 - VertexCFD_Restart_test_OPENMP_np_192_nt_1 (Failed)
        191 - VertexCFD_GeometryPrimitives_test_OPENMP_np_192_nt_1 (Failed)
        202 - VertexCFD_WriteMatrix_test_OPENMP_np_192_nt_1 (Failed)       
```
Please note that there are several tests that are used on the continious integration (CI) and they fail on any other system. This is the reason of the failed tests and it is expected to fail on those tests.

## Regression Tests
VERTEX-CFD uses regression tests to make sure the new capabilities are not introducing bugs/errors to the old capabilities. In order to run regression tests, you can use the below submission script written for SLURM scheduler:

```
#!/bin/bash                                                                                                                         
#SBATCH -N 1
#SBATCH --ntasks-per-node 32
#SBATCH --output output.log
#SBATCH --error error.log
#SBATCH --job-name=regression-test

source <PATH TO>/trilinos-cuda-config.sh
#source <PATH TO>/vertex-env-cpu.sh # For GPU comment above line and use this line.

export OMP_PROC_BIND=true
export OMP_PLACES=threads

export IOSS_PROPERTIES="COMPOSE_RESULTS=on:MINIMIZE_OPEN_FILES=on:MAXIMUM_NAME_LENGTH=64:DUPLICATE_FIELD_NAME_BEHAVIOR=WARNING"
export EXODUS_NETCDF4=1

export MAP_STRING=slot:pe=${SLURM_CPUS_PER_TASK}

cd regression_test
pytest -k "test"
```

The example regression test output is:

```
============================= test session starts ==============================
platform linux -- Python 3.9.16, pytest-6.2.5, py-1.11.0, pluggy-1.5.0
rootdir: <PATH TO BUILD DIRECTORY>/regression_test, configfile: pytest.ini
collected 37 items

test_full_induction_mhd/test_current_sheet/test_current_sheet.py::TestCurrentSheet::test_2d_long PASSED [  6%]
test_full_induction_mhd/test_mhd_ldc/test_mhd_ldc.py::TestMHDLDC::test_2d_rotated_mhd_ldc[bx] PASSED [ 13%]
test_full_induction_mhd/test_mhd_ldc/test_mhd_ldc.py::TestMHDLDC::test_2d_rotated_mhd_ldc[by] PASSED [ 20%]
.
.
.
test_incompressible/test_taylor_green_vortex/test_taylor_green_vortex.py::TestTaylorGreenVortex::test_2d_mesh_convergence_laminar PASSED [ 93%]
test_incompressible/test_wale_cavity/test_wale_cavity.py::TestWaleCavity::test_3d_wale_cavity PASSED [100%]
========== 0 failed, 37 passed, 0 deselected in 40525.94s (11:15:25) ==========
```



