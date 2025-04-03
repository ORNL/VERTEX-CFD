set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/src/test_harness)
include(${TEST_HARNESS_DIR}/TestHarness.cmake)

VertexCFD_add_tests(
  LIBS VertexCFD
  NAMES
  IncompressibleLSVOFConvectiveFlux
  IncompressibleLSVOFScalarConvectiveFlux
  IncompressibleLSVOFViscousFlux
  )

