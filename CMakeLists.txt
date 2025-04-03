#------------------------------------------------------------------------------#
# Project settings
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.12)
include(CheckLanguage)

project(VertexCFD LANGUAGES CXX)
set(PROJECT_VERSION "0.0-dev" )

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)

include(GNUInstallDirs)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Set the exodiff path relative to Trilinos installation path
set(EXODIFF_PATH ${Trilinos_ROOT}/bin/exodiff)

# Set the Trilinos lib path relative to Trilinos installation path
set(TRILINOS_LIB ${Trilinos_ROOT}/lib64)

option(NO_PARMETIS_SUPPORT "No parmetis support" ON)
add_compile_definitions(NO_PARMETIS_SUPPORT)

# Add the installation library dir to the executable RPATH
list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

# Add paths to linked libraries outside the build tree to the RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#------------------------------------------------------------------------------#
# Dependencies
#------------------------------------------------------------------------------#

find_package(CLANG_FORMAT)

find_package(YAPF)

find_package(MPI REQUIRED)

find_package(Trilinos REQUIRED)
if (Trilinos_VERSION VERSION_LESS 16 OR Trilinos_VERSION VERSION_GREATER_EQUAL 17)
    message(FATAL_ERROR "Incompatible Trilinos version ${Trilinos_VERSION}. Only Trilinos version 16 and above are supported.")
endif ()

# FIXME Tacho requires rocblas, rocsparse, and rocsolver with HIP but doesn't
# enforce this dependency explicitly through version 16.0.0
if(Kokkos_ENABLE_HIP AND NOT (ShyLU_NodeTacho_ENABLE_ROCSOLVER AND ShyLU_NodeTacho_ENABLE_ROCSPARSE AND ShyLU_NodeTacho_ENABLE_ROCBLAS))
    message(FATAL_ERROR "ShyLU_NodeTacho wasn't configured with the ROC* TPLs enabled!")
endif()

if (VertexCFD_ENABLE_HYPRE)
  find_package(HYPRE REQUIRED)
  find_package(HYPREDRV REQUIRED)
  add_compile_definitions(VERTEXCFD_HAVE_HYPRE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module)

if (VertexCFD_ENABLE_SUPERLU)
  find_package(SuperLU_dist REQUIRED)
  add_compile_definitions(HAVE_SUPERLUDIST)
endif()

#------------------------------------------------------------------------------#
# Performance Portability
# ------------------------------------------------------------------------------#
# Set the kokkos node type used by panzer/phalanx. We do this because phalanx
# explicitly sets a node type to be used by all of its data structures which
# is subsequently used by panzer. Below we copy the logic from Phalanx for
# choosing the default node type so our choice matches that of Panzer. The
# phalanx CMake variable ${Phalanx_KOKKOS_DEVICE_TYPE} is not exported and
# therefore is not accessible here which would have been perferable.
if (Kokkos_ENABLE_CUDA)
  set(VERTEXCFD_KOKKOS_DEVICE_TYPE "CUDA")
elseif (Kokkos_ENABLE_HIP)
  set(VERTEXCFD_KOKKOS_DEVICE_TYPE "HIP")
elseif (Kokkos_ENABLE_OPENMP)
  set(VERTEXCFD_KOKKOS_DEVICE_TYPE "OPENMP")
elseif (Kokkos_ENABLE_SERIAL)
  set(VERTEXCFD_KOKKOS_DEVICE_TYPE "SERIAL")
else()
  message(FATAL_ERROR "No Kokkos execution space is enabled.")
endif()

# Check that the kokkos device type we want is available.
kokkos_check( DEVICES ${VERTEXCFD_KOKKOS_DEVICE_TYPE} )

# ensure that we can use lambdas if we are using cuda.
if(${VERTEXCFD_KOKKOS_DEVICE_TYPE} STREQUAL "CUDA")
  kokkos_check(OPTIONS CUDA_LAMBDA)
endif()

# output Phalanx/Panzer device type
message( STATUS "Phalanx/Panzer Kokkos device type: ${VERTEXCFD_KOKKOS_DEVICE_TYPE}" )

#------------------------------------------------------------------------------#
# Tests and Documentation
#------------------------------------------------------------------------------#

option(VertexCFD_ENABLE_TESTING "Build tests" OFF)
if(VertexCFD_ENABLE_TESTING)
  include(CTest)
  find_package(GTest REQUIRED)
  enable_testing()
endif()

if (VertexCFD_BUILD_DOC)
  find_package(Doxygen)
  if (DOXYGEN_FOUND)
    # ignore dep packages like gtest/gmock
    set(DOXYGEN_EXCLUDE_PATTERNS "*/_deps/*")
    doxygen_add_docs( vertexcfddocs
      ${PROJECT_SOURCE_DIR}
      COMMENT "Generate doxygen pages"
    )
  else (DOXYGEN_FOUND)
    message(STATUS "Doxygen need to be installed to generate the doxygen documentation")
  endif (DOXYGEN_FOUND)
else (VertexCFD_BUILD_DOC)
  message(STATUS "Doxygen documentation disabled!")
endif (VertexCFD_BUILD_DOC)


##---------------------------------------------------------------------------##
## Code coverage testing
##---------------------------------------------------------------------------##
option(VertexCFD_ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
if(VertexCFD_ENABLE_COVERAGE_BUILD)
  message(STATUS "Enabling coverage build")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -O0")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()

##---------------------------------------------------------------------------##
## Print the revision number to stdout
##---------------------------------------------------------------------------##
find_package(Git)
if(GIT_FOUND AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  execute_process(
    COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE   VertexCFD_GIT_COMMIT_HASH
    )
else()
  set(VertexCFD_GIT_COMMIT_HASH "Not a git repository")
endif()
message(STATUS "VertexCFD Revision = '${VertexCFD_GIT_COMMIT_HASH}'")

##---------------------------------------------------------------------------##
## Library
##---------------------------------------------------------------------------##

add_subdirectory(src)

add_subdirectory(examples)

file(COPY regression_test DESTINATION .)

# Delete the unwanted file from the destination
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/regression_test/vertexcfd_test.py.in)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/regression_test/rename_variables_exodus.py.in)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/regression_test/vertexcfd_test.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/regression_test/vertexcfd_test.py)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/regression_test/rename_variables_exodus.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/regression_test/rename_variables_exodus.py)

##---------------------------------------------------------------------------##
## Package Configuration
##---------------------------------------------------------------------------##
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/VertexCFDConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/VertexCFDConfig.cmake @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/VertexCFDConfig.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VertexCFD)

##---------------------------------------------------------------------------##
## Clang Format
##---------------------------------------------------------------------------##
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.hpp)
  add_custom_target(format-cpp
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
  add_custom_target(format-cpp-dry-run
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file --dry-run -Werror ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()

##---------------------------------------------------------------------------##
## Python Format
##---------------------------------------------------------------------------##
if(YAPF_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES *.py)
  add_custom_target(format-python
    COMMAND ${YAPF_EXECUTABLE} -i -vv ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
  add_custom_target(format-python-dry-run
    COMMAND ${YAPF_EXECUTABLE} -d ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()

##---------------------------------------------------------------------------##
## Python and Clang Format
##---------------------------------------------------------------------------##
if(YAPF_FOUND OR CLANG_FORMAT_FOUND)
  add_custom_target(format)
  if(CLANG_FORMAT_FOUND)
    add_dependencies(format format-cpp)
  endif()
  if(YAPF_FOUND)
    add_dependencies(format format-python)
  endif()
endif()

##---------------------------------------------------------------------------##
## Code coverage report generation
##---------------------------------------------------------------------------##
if(VertexCFD_ENABLE_COVERAGE_BUILD)
  message(CHECK_START "Checking for gcovr")
  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  find_package(GCOVR)
  list(POP_BACK CMAKE_MESSAGE_INDENT)
  if(NOT GCOVR_FOUND)
    message(CHECK_FAIL "not found. Coverage report targets disabled.")
  else()
    message(CHECK_PASS "found. Adding coverage report targets.")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    set(gcovr_filter "'${PROJECT_SOURCE_DIR}/src/.*'")
    message(STATUS "Adding gcovr_xml target")
    add_custom_target(gcovr_xml
        COMMAND ${GCOVR_EXECUTABLE}
        --filter=${gcovr_filter} --xml-pretty --print-summary -o coverage.xml
        -r "${PROJECT_SOURCE_DIR}" .)
    message(STATUS "Adding gcovr_html target")
    add_custom_target(gcovr_html
        COMMAND ${GCOVR_EXECUTABLE}
        --filter=${gcovr_filter} --html-details coverage/
        -r "${PROJECT_SOURCE_DIR}" .)
    list(POP_BACK CMAKE_MESSAGE_INDENT)
  endif()
endif()
