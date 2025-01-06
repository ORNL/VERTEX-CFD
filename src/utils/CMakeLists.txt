configure_file(VertexCFD_Utils_config.hpp.cmakein VertexCFD_Utils_config.hpp)

set(UTILS_HEADERS
  VertexCFD_EvaluatorBase.hpp
  VertexCFD_Utils_Constants.hpp
  VertexCFD_Utils_ExplicitTemplateInstantiation.hpp
  VertexCFD_Utils_VectorField.hpp
  VertexCFD_Utils_ParameterPack.hpp
  VertexCFD_Utils_NonlinearSolver.hpp
  VertexCFD_Utils_ScalarToVector.hpp
  VertexCFD_Utils_SmoothMath.hpp
  VertexCFD_Utils_VelocityDim.hpp
  VertexCFD_Utils_VelocityLayout.hpp
  VertexCFD_Utils_TypeTraits.hpp
  VertexCFD_Utils_Version.hpp
  VertexCFD_Utils_MatrixMath.hpp
  VertexCFD_Utils_VectorizeOutputFieldNames.hpp
  )

set(UTILS_SOURCES
  VertexCFD_Utils_VelocityDim.cpp
  VertexCFD_Utils_VelocityLayout.cpp
  VertexCFD_Utils_Version.cpp
  )

add_library(Utils ${UTILS_SOURCES})

target_link_libraries(Utils PUBLIC
  MPI::MPI_CXX
  Kokkos::kokkos
  Phalanx::phalanx
  )

target_include_directories(Utils
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS Utils
  EXPORT VertexCFDUtilsTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT VertexCFDUtilsTargets
  FILE VertexCFDUtilsTargets.cmake
  NAMESPACE VertexCFD::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VertexCFD)

install(FILES ${UTILS_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/VertexCFD_Utils_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)

if(VertexCFD_ENABLE_TESTING)
  add_subdirectory(unit_test)
endif()
