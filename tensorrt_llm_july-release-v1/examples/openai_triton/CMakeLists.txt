# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Enable C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
set(TARGET_NAME trt_llm_custom_plugins)
project(${TARGET_NAME})

set(CMAKE_VERBOSE_MAKEFILE 1)

# Compile options
set(CMAKE_C_FLAGS "-Wall -pthread ")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -lstdc++")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

set(CMAKE_BUILD_TYPE release)

find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    config: ${CUDA_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

set(TRT_LLM_INCLUDE_DIR "../../cpp")
include_directories(${TRT_LLM_INCLUDE_DIR})
set(TRT_LLM_LIB_DIR "../../tensorrt_llm/libs")
link_directories(${TRT_LLM_LIB_DIR})

# Declare the executable target built from your sources
add_library(
  ${TARGET_NAME} SHARED
  tritonPlugins.cpp
  TritonFlashAttentionPlugin.cpp
  aot/fmha_kernel_fp16.c
  aot/fmha_kernel_fp32.c
  aot/fp16/fmha_kernel_d64_fp16.0eb6b090_0d1d2d3d4d5d67.c
  aot/fp32/fmha_kernel_d64_fp32.7f15949c_0d1d2d3d4d5d67.c)

# Link your application with CUDA libraries
target_link_libraries(${TARGET_NAME} LINK_PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(${TARGET_NAME} LINK_PRIVATE nvinfer)
target_link_libraries(${TARGET_NAME} LINK_PRIVATE nvinfer_plugin_tensorrt_llm)
target_link_libraries(${TARGET_NAME} LINK_PRIVATE cuda)

target_include_directories(${TARGET_NAME} PUBLIC /usr/local/cuda/include)
