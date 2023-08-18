include(FetchContent)

set(SRCS
    iBuffer.cpp
    iTensor.cpp
    bufferManager.cpp
    gptDecoder.cpp
    gptDecoderBatch.cpp
    gptJsonConfig.cpp
    gptSession.cpp
    gptSessionWithDecoderBatch.cpp
    runtimeKernels.cu
    tllmRuntime.cpp
    tllmLogger.cpp
    worldConfig.cpp)

include_directories(${API_INCLUDE_DIR}/tensorrt_llm/runtime)

add_library(runtime_src OBJECT ${SRCS})
set_property(TARGET runtime_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET runtime_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(runtime_src PRIVATE ${MPI_INCLUDE_PATH})

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2)
FetchContent_MakeAvailable(json)

target_link_libraries(runtime_src PUBLIC nlohmann_json::nlohmann_json)
