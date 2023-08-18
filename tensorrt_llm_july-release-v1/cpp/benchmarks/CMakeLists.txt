include_directories(${PROJECT_SOURCE_DIR}/include)

set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")

add_custom_target(benchmarks)

function(add_benchmark test_name test_src)
  add_executable(${test_name} ${test_src})

  target_link_libraries(${test_name} PUBLIC ${SHARED_TARGET} nvinfer_plugin)

  target_compile_features(${test_name} PRIVATE cxx_std_17)
  target_compile_definitions(${test_name}
                             PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")
  add_dependencies(benchmarks ${test_name})
endfunction()

add_benchmark(gptSessionBenchmark gptSessionBenchmark.cpp)
