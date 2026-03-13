set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/src/test_harness)
include(${TEST_HARNESS_DIR}/TestHarness.cmake)

VertexCFD_add_tests(
  LIBS VertexCFD
  NAMES
  IncompressibleCLSEpsilon
  IncompressibleCLSLambda
  IncompressibleCLSNonReconstructedNormal
  IncompressibleCLSRegularization
  IncompressibleCLSSign
  IncompressibleCLSTimeDerivative
  IncompressibleLSVOFArtificialCompression
  IncompressibleLSVOFBubbleExact
  IncompressibleLSVOFBuoyancySource
  IncompressibleLSVOFConvectiveFlux
  IncompressibleLSVOFEntropyFunction
  IncompressibleLSVOFEntropyFluctuation
  IncompressibleLSVOFEntropyViscosity
  IncompressibleLSVOFErrorNorms
  IncompressibleLSVOFNthPhaseFraction
  IncompressibleLSVOFScalarConvectiveFlux
  IncompressibleLSVOFScalarTimeDerivative
  IncompressibleLSVOFScalarViscousFlux
  IncompressibleLSVOFSurfaceTensionForce
  IncompressibleLSVOFTimeDerivative
  IncompressibleLSVOFVariableProperties
  IncompressibleLSVOFViscousFlux
  IncompressibleLSVOFVortexVelocity
  )

