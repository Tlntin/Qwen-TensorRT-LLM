add_library(th_utils STATIC thUtils.cu torchAllocator.cpp)
set_property(TARGET th_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET th_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(th_utils PUBLIC ${TORCH_LIBRARIES} -lcublas -lcudart
                                      -lcurand)

add_library(th_common SHARED dynamicDecodeOp.cpp weightOnlyQuantOp.cpp
                             gatherTreeOp.cpp fp8Op.cpp)
set_property(TARGET th_common PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(
  th_common PRIVATE ${TORCH_LIBRARIES} th_utils ${Python3_LIBRARIES}
                    ${STATIC_TARGET} "-Wl,--no-undefined")
