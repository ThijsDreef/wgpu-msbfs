add_subdirectory(utils)

if (BACKEND STREQUAL "CUDA")
  add_subdirectory(cuda)
else()
  add_subdirectory(wgpu)
endif()
target_include_directories(${LIB_NAME} PUBLIC include/)
