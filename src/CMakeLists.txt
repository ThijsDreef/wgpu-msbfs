
FetchContent_Declare(
  webgpu
  GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
  GIT_TAG        wgpu
)
FetchContent_MakeAvailable(webgpu)

add_library(${LIB_NAME} ${LIB_TYPE}
  wgpu-state.cpp
  msbfs.cpp
  core/util/wgpu-utils.cpp
  core/util/shader-locator.cpp
  core/kernels/frontier-identification.cpp
  core/kernels/frontier-expansion.cpp
  core/kernels/bindgroups/bsa.cpp
  core/kernels/bindgroups/csr.cpp
  core/kernels/bindgroups/iterative-length.cpp
  core/kernels/bindgroups/jfq.cpp
)
target_include_directories(${LIB_NAME} PUBLIC include/)
target_link_libraries(${LIB_NAME} webgpu)
