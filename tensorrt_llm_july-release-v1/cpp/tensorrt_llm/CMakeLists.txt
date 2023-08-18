set(TARGET_NAME tensorrt_llm)
set(SHARED_TARGET ${TARGET_NAME})
set(SHARED_TARGET
    ${SHARED_TARGET}
    PARENT_SCOPE)
set(STATIC_TARGET ${TARGET_NAME}_static)
set(STATIC_TARGET
    ${STATIC_TARGET}
    PARENT_SCOPE)
set(API_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cutlass_extensions/include
                    ${API_INCLUDE_DIR})

find_package(MPI REQUIRED)
message(STATUS "Using MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
message(STATUS "Using MPI_LIBRARIES: ${MPI_LIBRARIES}")

add_subdirectory(common)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(runtime)

set(TRTLLM_LINK_LIBS
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CUDART_LIB}
    ${CUDNN_LIB}
    ${CMAKE_DL_LIBS}
    ${MPI_LIBRARIES}
    ${TRT_LIB}
    common_src
    kernels_src
    layers_src
    runtime_src)

add_library(tensorrt_llm_batch_manager_static STATIC IMPORTED)
set_property(TARGET tensorrt_llm_batch_manager_static PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/batch_manager/libtensorrt_llm_batch_manager_static.a")

# ################################# SHARED LIBRARY
# ##############################################################################

add_library(${SHARED_TARGET} SHARED)

set_target_properties(
  ${SHARED_TARGET} PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES"
                              CXX_EXTENSIONS "NO")

target_link_libraries(${SHARED_TARGET} PUBLIC ${TRTLLM_LINK_LIBS}
                                              "-Wl,--no-undefined")

# ################################# STATIC LIBRARY
# ##############################################################################

add_library(${STATIC_TARGET} STATIC)

set_target_properties(
  ${STATIC_TARGET} PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES"
                              CXX_EXTENSIONS "NO" POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${STATIC_TARGET} PUBLIC ${TRTLLM_LINK_LIBS})

if(BUILD_PYT)
  add_subdirectory(thop)
endif()

add_subdirectory(plugins)
