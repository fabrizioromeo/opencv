set(sample example-face-detection)

add_android_project(${sample} "${CMAKE_CURRENT_SOURCE_DIR}" LIBRARY_DEPS ${OpenCV_BINARY_DIR} SDK_TARGET 11 ${ANDROID_SDK_TARGET} NATIVE_DEPS opencv_objdetect)
if(TARGET ${sample})
  add_dependencies(opencv_android_examples ${sample})
endif()
