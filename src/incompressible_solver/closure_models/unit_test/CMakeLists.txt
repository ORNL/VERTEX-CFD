set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/src/test_harness)
include(${TEST_HARNESS_DIR}/TestHarness.cmake)

set(TEST_NAMES
  IncompressibleTimeDerivative
  IncompressibleConvectiveFlux
  IncompressibleLiftDrag
  IncompressibleViscousFlux
  IncompressibleViscousHeat
  IncompressibleGradDiv
  IncompressibleLocalTimeStepSize
  IncompressibleConstantSource
  IncompressibleBuoyancySource
  IncompressibleRotatingAnnulusExact
  IncompressiblePlanarPoiseuilleExact
  IncompressibleTaylorGreenVortexExactSolution
  IncompressibleNusseltNumber
  IncompressibleErrorNorms
  IncompressibleVariableTimeDerivative
  IncompressibleShearVariables
  IncompressibleEnstrophy
  IncompressibleSUPGExactSolution
  IncompressibleSUPGFlux
  IncompressibleTauSUPG
  )

if(VertexCFD_ENABLE_TENSORFLOW)
  list(APPEND TEST_NAMES IncompressibleConvectiveFluxMachineLearning)
  file(COPY ${CMAKE_SOURCE_DIR}/examples/inputs/incompressible/incompressible_2d_planar_poiseuille_tensorflow.tflite DESTINATION .)
  file(COPY ${CMAKE_SOURCE_DIR}/examples/inputs/incompressible/incompressible_2d_planar_poiseuille_tensorflow_metadata.yml DESTINATION .)
endif()

VertexCFD_add_tests(
  LIBS VertexCFD
  NAMES ${TEST_NAMES}
  )
