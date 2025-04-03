configure_file( VertexCFD_LinearSolverUnitTestConfig.hpp.cmakein VertexCFD_LinearSolverUnitTestConfig.hpp )

set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/src/test_harness)
include(${TEST_HARNESS_DIR}/TestHarness.cmake)

# Components inside of linear_solvers currently only have
# CUDA implementation
if(${VERTEXCFD_KOKKOS_DEVICE_TYPE} STREQUAL "CUDA")
  VertexCFD_add_tests(
    LIBS VertexCFD
    NAMES
    CusolverGLU
    LocalSolverFactory
    Preconditioner
    PreconditionerFactory
  )
endif()

if(VertexCFD_ENABLE_HYPRE)
  VertexCFD_add_tests(
    LIBS VertexCFD
    NAMES
    Hypre
    HyprePreconditionerFactory
    )
endif()
