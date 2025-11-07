# Locates the tensorFlow library and include directories.

include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

find_path(TensorFlow_flatbuffers_INCLUDE_DIR NAMES flatbuffers
	PATHS
	${TensorFlow_BAZEL_INSTALL_DIR}/external/flatbuffers/_virtual_includes/flatbuffers
        ${TensorFlow_SOURCE_DIR}/bazel-bin/external/flatbuffers/_virtual_includes/flatbuffers)

find_path(TensorFlow_lite_INCLUDE_DIR NAMES tensorflow
	PATHS
	${TensorFlow_SOURCE_DIR})

find_library(TensorFlow_LITE_LIBRARY NAMES tensorflowlite
        HINTS
	${TensorFlow_BAZEL_INSTALL_DIR}/tensorflow/lite
	${TensorFlow_SOURCE_DIR}/bazel-bin/tensorflow/lite)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_lite_INCLUDE_DIR 
	TensorFlow_flatbuffers_INCLUDE_DIR 
	TensorFlow_LITE_LIBRARY)

# set external variables for usage in CMakeLists.txt
if(TENSORFLOW_FOUND AND NOT TARGET TensorFlow)
  set(TensorFlow_LIBRARIES ${TensorFlow_LITE_LIBRARY})
  set(TensorFlow_INCLUDE_DIRS ${TensorFlow_lite_INCLUDE_DIR} ${TensorFlow_flatbuffers_INCLUDE_DIR})
  add_library(TensorFlow UNKNOWN IMPORTED)
  set_target_properties(TensorFlow PROPERTIES
    IMPORTED_LOCATION "${TensorFlow_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorFlow_INCLUDE_DIRS}"
  )
endif()
