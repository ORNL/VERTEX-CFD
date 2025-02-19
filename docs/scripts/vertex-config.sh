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
