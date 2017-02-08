set(sample example-tutorial-4-opencl)
if(NOT DEFINED ANDROID_OPENCL_SDK)
  message(STATUS "Sample ${sample} is disabled, because ANDROID_OPENCL_SDK is not specified")
  return()
endif()
if(ANDROID_NATIVE_API_LEVEL LESS 14)
  message(STATUS "Sample ${sample} is disabled, because ANDROID_NATIVE_API_LEVEL < 14")
  return()
endif()

include_directories(${ANDROID_OPENCL_SDK}/include)
link_directories(${ANDROID_OPENCL_SDK}/lib/${ANDROID_NDK_ABI_NAME})
add_android_project(${sample} "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_DEPS ${OpenCV_BINARY_DIR}
    SDK_TARGET 21 ${ANDROID_SDK_TARGET}
    NATIVE_DEPS opencv_imgproc -lGLESv2 -lEGL -lOpenCL
    COPY_LIBS YES
)
if(TARGET ${sample})
  add_dependencies(opencv_android_examples ${sample})
endif()
