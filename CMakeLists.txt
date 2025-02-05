cmake_minimum_required(VERSION 3.11)
project(wgpu-msbfs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, defaulting to release")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()

option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCH "Build benchmarks" ON)
option(BUILD_STATIC "Build static library instead of shared" OFF)

if (BUILD_STATIC)
  set(LIB_NAME wgpumsbfs_static)
  set(LIB_TYPE STATIC)
else()
  set(LIB_NAME wgpumsbfs)
  set(LIB_TYPE SHARED)
endif()

include(FetchContent)

# can we lower this without breaking bench / test
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)

if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if (BUILD_BENCH)
  add_subdirectory(benchmarks)
endif()
