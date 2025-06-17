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
  kernels/set-first-bsak.cpp
  kernels/frontier-expansion.cpp
  kernels/bindgroups/bsa.cpp
  kernels/bindgroups/set-bsak.cpp
  kernels/bindgroups/csr.cpp
  kernels/bindgroups/iterative-length.cpp
  kernels/bindgroups/jfq.cpp
)

target_link_libraries(${LIB_NAME} webgpu utils)
target_include_directories(${LIB_NAME} PUBLIC include/)
target_copy_webgpu_binaries(${LIB_NAME})

if(BACKEND STREQUAL "EMDAWNWEBGPU")
  add_executable(wgpu-msbfs-web main.cpp)
  add_executable(wgpu-msbfs-all main.cpp)

  target_link_options(wgpu-msbfs-web PRIVATE -sMODULARIZE=1 -sEXPORT_ALL -sEXPORT_ES6=1 -sASYNCIFY -sENVIRONMENT='web' -sALLOW_MEMORY_GROWTH -sEXPORTED_RUNTIME_METHODS=["ccall"])
  target_link_options(wgpu-msbfs-all PRIVATE -sASYNCIFY -sALLOW_MEMORY_GROWTH)
  target_link_libraries(wgpu-msbfs-web ${LIB_NAME})
  target_link_libraries(wgpu-msbfs-all ${LIB_NAME})
endif()