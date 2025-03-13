#!/bin/bash
module use <SPACK MODULES LOCATION> # EDIT: Update this with the module location, as an example: module use spack/share/spack/modules/linux-centos7-broadwell/
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
