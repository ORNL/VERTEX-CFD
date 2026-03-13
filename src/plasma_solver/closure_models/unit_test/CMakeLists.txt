set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/src/test_harness)
include(${TEST_HARNESS_DIR}/TestHarness.cmake)

set(TEST_NAMES
  PlasmaSpeciesClosureModelFactoryTestHarness
  PlasmaSpeciesConvectiveFlux
  PlasmaSpeciesEOS
  PlasmaSpeciesTimeDerivative
)

VertexCFD_add_tests(
  MPI
  LIBS VertexCFD
  NAMES ${TEST_NAMES}
  )
