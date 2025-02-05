set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           c58e6d0710581e3a08d65c349664128a8d9a2461
)

FetchContent_MakeAvailable(
  googlebenchmark
)

add_executable(bench_wgpumsbfs bench.cpp)

target_link_libraries(bench_wgpumsbfs
  benchmark
  ${LIB_NAME}
)
