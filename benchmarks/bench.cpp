#include "benchmark/benchmark.h"
#include "core/util/file-loader.hpp"
#include "msbfs.hpp"
#include "wgpu-state.hpp"

WGPUState wgpustate = WGPUState();

#define CREATE_BENCHMARK(scale, pairs)                                         \
  static void BM_Scale##scale##Pairs##pairs(benchmark::State &state) {         \
    BinaryLoadedFile files[] = {                                               \
        load_file("data/" #scale "/" #pairs "-src.bin"),                       \
        load_file("data/" #scale "/" #pairs "-dst.bin"),                       \
        load_file("data/" #scale "/v.bin"),                                    \
        load_file("data/" #scale "/e.bin"),                                    \
    };                                                                         \
    TimingInfo info = {0, 0};                                                  \
    for (auto _ : state) {                                                     \
      iterative_length(wgpustate,                                              \
                       {                                                       \
                           .src = (uint32_t *)files[0].data,                   \
                           .dst = (uint32_t *)files[1].data,                   \
                           .length = files[1].length / sizeof(uint32_t),       \
                       },                                                      \
                       {                                                       \
                           .v = (uint32_t *)files[2].data,                     \
                           .e = (uint32_t *)files[3].data,                     \
                           .v_length = files[2].length / sizeof(uint32_t),     \
                           .e_length = files[3].length / sizeof(uint32_t),     \
                       },                                                      \
                       info);                                                  \
    }                                                                          \
    state.counters["Expand"] = benchmark::Counter(                             \
        info.expand_ns / 1000000000.0, benchmark::Counter::kAvgIterations);    \
    state.counters["Identify"] = benchmark::Counter(                           \
        info.identify_ns / 1000000000.0, benchmark::Counter::kAvgIterations);  \
  }                                                                            \
  BENCHMARK(BM_Scale##scale##Pairs##pairs)->Unit(benchmark::kSecond)

CREATE_BENCHMARK(1, 1);
CREATE_BENCHMARK(1, 10);
CREATE_BENCHMARK(1, 100);
CREATE_BENCHMARK(1, 1000);
CREATE_BENCHMARK(1, 2048);
CREATE_BENCHMARK(1, 4096);
CREATE_BENCHMARK(1, 8192);

CREATE_BENCHMARK(3, 1);
CREATE_BENCHMARK(3, 10);
CREATE_BENCHMARK(3, 100);
CREATE_BENCHMARK(3, 1000);
CREATE_BENCHMARK(3, 2048);
CREATE_BENCHMARK(3, 4096);
CREATE_BENCHMARK(3, 8192);

CREATE_BENCHMARK(10, 1);
CREATE_BENCHMARK(10, 10);
CREATE_BENCHMARK(10, 100);
CREATE_BENCHMARK(10, 1000);
CREATE_BENCHMARK(10, 2048);
CREATE_BENCHMARK(10, 4096);
CREATE_BENCHMARK(10, 8192);

CREATE_BENCHMARK(30, 1);
CREATE_BENCHMARK(30, 10);
CREATE_BENCHMARK(30, 100);
CREATE_BENCHMARK(30, 1000);
CREATE_BENCHMARK(30, 2048);
CREATE_BENCHMARK(30, 4096);
CREATE_BENCHMARK(30, 8192);

BENCHMARK_MAIN();
