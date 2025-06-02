FetchContent_Declare(
  webgpu
  GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
  GIT_TAG        main
)

FetchContent_MakeAvailable(webgpu)

add_library(${LIB_NAME} ${LIB_TYPE}
  wgpu-state.cpp
  msbfs.cpp
  util/wgpu-utils.cpp
  util/shader-locator.cpp
  kernels/frontier-identification.cpp
  kernels/frontier-expansion.cpp
  kernels/bindgroups/bsa.cpp
  kernels/bindgroups/csr.cpp
  kernels/bindgroups/iterative-length.cpp
  kernels/bindgroups/jfq.cpp
)

target_link_libraries(${LIB_NAME} webgpu utils)
target_include_directories(${LIB_NAME} PUBLIC include/)
