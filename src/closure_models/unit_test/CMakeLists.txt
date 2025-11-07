set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/src/test_harness)
include(${TEST_HARNESS_DIR}/TestHarness.cmake)

set(TEST_NAMES
  ConstantScalarField
  ConstantVectorField
  ClosureModelFactoryTestHarness
  ElementLength
  ExternalMagneticField
  MeasureElementLength
  MethodManufacturedSolution
  MethodManufacturedSolutionSource
  MetricTensor
  MetricTensorElementLength
  SingularValueElementLength
  VariableConvectiveFlux
  VariableDiffusionFlux
  VariableSUPGFlux
  VariableTauSUPG
  VectorFieldDivergence
  WallDistance
)

if(VertexCFD_ENABLE_ARBORX)
  list(APPEND TEST_NAMES ExternalInterpolation)
  file(COPY ${CMAKE_SOURCE_DIR}/examples/inputs/rectangle_field.exo DESTINATION .)
  file(COPY ${CMAKE_SOURCE_DIR}/examples/inputs/constant_per_time_step.exo DESTINATION .)
endif()

VertexCFD_add_tests(
  LIBS VertexCFD
  NAMES
  ${TEST_NAMES}
)

VertexCFD_add_tests(
  MPI
  LIBS VertexCFD
  NAMES
  ExternalFields
  )
