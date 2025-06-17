#include "benchmark/benchmark.h"
#include "msbfs.hpp"
#include "utils/file-loader.hpp"

#define CREATE_BENCHMARK(scale, pairs)                                         \
  static void BM_Scale##scale##Pairs##pairs(benchmark::State &state) {         \
    BinaryLoadedFile files[] = {                                               \
        load_file("data/" #scale "/" #pairs "-src.bin"),                       \
        load_file("data/" #scale "/" #pairs "-dst.bin"),                       \
        load_file("data/" #scale "/v.bin"),                                    \
        load_file("data/" #scale "/e.bin"),                                    \
        load_file("data/" #scale "/r-v.bin"),                                  \
        load_file("data/" #scale "/r-e.bin"),                                  \
    };                                                                         \
    TimingInfo info = {0, 0};                                                  \
    PathFindingRequest request;                                                \
    request.src = (uint32_t *)files[0].data;                                   \
    request.dst = (uint32_t *)files[1].data;                                   \
    request.length = files[1].length / sizeof(uint32_t);                       \
    CSR csr;                                                                   \
    csr.v = (uint32_t *)files[2].data;                                         \
    csr.e = (uint32_t *)files[3].data;                                         \
    csr.v_length = files[2].length / sizeof(uint32_t);                         \
    csr.e_length = files[3].length / sizeof(uint32_t);                         \
    CSR reverse_csr;                                                           \
    reverse_csr.v = (uint32_t *)files[4].data;                                 \
    reverse_csr.e = (uint32_t *)files[5].data;                                 \
    reverse_csr.v_length = files[4].length / sizeof(uint32_t);                 \
    reverse_csr.e_length = files[5].length / sizeof(uint32_t);                 \
    for (auto _ : state) {                                                     \
      iterative_length(request, csr, reverse_csr, info);                       \
    }                                                                          \
    state.counters["Expand"] = benchmark::Counter(                             \
        info.expand_ns / 1000000000.0, benchmark::Counter::kAvgIterations);    \
    state.counters["Identify"] = benchmark::Counter(                           \
        info.identify_ns / 1000000000.0, benchmark::Counter::kAvgIterations);  \
    free(files[0].data);                                                       \
    free(files[1].data);                                                       \
    free(files[2].data);                                                       \
    free(files[3].data);                                                       \
    free(files[4].data);                                                       \
    free(files[5].data);                                                       \
  }                                                                            \
  BENCHMARK(BM_Scale##scale##Pairs##pairs)->Unit(benchmark::kSecond)

CREATE_BENCHMARK(1, 1);
CREATE_BENCHMARK(1, 10);
CREATE_BENCHMARK(1, 100);
CREATE_BENCHMARK(1, 1000);
CREATE_BENCHMARK(1, 2048);
CREATE_BENCHMARK(1, 4096);
CREATE_BENCHMARK(1, 8192);
CREATE_BENCHMARK(1, 16384);
CREATE_BENCHMARK(1, 32768);
CREATE_BENCHMARK(1, 65536);

CREATE_BENCHMARK(3, 1);
CREATE_BENCHMARK(3, 10);
CREATE_BENCHMARK(3, 100);
CREATE_BENCHMARK(3, 1000);
CREATE_BENCHMARK(3, 2048);
CREATE_BENCHMARK(3, 4096);
CREATE_BENCHMARK(3, 8192);
CREATE_BENCHMARK(3, 16384);
CREATE_BENCHMARK(3, 32768);
CREATE_BENCHMARK(3, 65536);

CREATE_BENCHMARK(10, 1);
CREATE_BENCHMARK(10, 10);
CREATE_BENCHMARK(10, 100);
CREATE_BENCHMARK(10, 1000);
CREATE_BENCHMARK(10, 2048);
CREATE_BENCHMARK(10, 4096);
CREATE_BENCHMARK(10, 8192);
CREATE_BENCHMARK(10, 16384);
CREATE_BENCHMARK(10, 32768);
CREATE_BENCHMARK(10, 65536);

CREATE_BENCHMARK(30, 1);
CREATE_BENCHMARK(30, 10);
CREATE_BENCHMARK(30, 100);
CREATE_BENCHMARK(30, 1000);
CREATE_BENCHMARK(30, 2048);
CREATE_BENCHMARK(30, 4096);
CREATE_BENCHMARK(30, 8192);
CREATE_BENCHMARK(30, 16384);
CREATE_BENCHMARK(30, 32768);
CREATE_BENCHMARK(30, 65536);

CREATE_BENCHMARK(100, 1);
CREATE_BENCHMARK(100, 10);
CREATE_BENCHMARK(100, 100);
CREATE_BENCHMARK(100, 1000);
CREATE_BENCHMARK(100, 2048);
CREATE_BENCHMARK(100, 4096);
CREATE_BENCHMARK(100, 8192);
CREATE_BENCHMARK(100, 16384);
CREATE_BENCHMARK(100, 32768);
CREATE_BENCHMARK(100, 65536);

CREATE_BENCHMARK(300, 1);
CREATE_BENCHMARK(300, 10);
CREATE_BENCHMARK(300, 100);
CREATE_BENCHMARK(300, 1000);
CREATE_BENCHMARK(300, 2048);
CREATE_BENCHMARK(300, 4096);
CREATE_BENCHMARK(300, 8192);
CREATE_BENCHMARK(300, 16384);
CREATE_BENCHMARK(300, 32768);
CREATE_BENCHMARK(300, 65536);

BENCHMARK_MAIN();
