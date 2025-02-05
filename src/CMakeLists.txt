FetchContent_Declare(
  webgpu
  GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
  GIT_TAG        main
)
FetchContent_MakeAvailable(webgpu)

add_library(${LIB_NAME} ${LIB_TYPE} msbfs.cpp)
target_include_directories(${LIB_NAME} PUBLIC include/)
target_link_libraries(${LIB_NAME} webgpu)
