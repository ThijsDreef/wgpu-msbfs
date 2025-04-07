#include "benchmark/benchmark.h"
#include "msbfs.hpp"
#include "wgpu-state.hpp"

#include "sys/mman.h"
#include <cstdint>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#define UNIX

struct mmapped_file {
  void *data;
  int fd;
  size_t length;
};

#ifdef UNIX
mmapped_file file_to_mmap(const char *path) {
  mmapped_file file;
  file.fd = open(path, O_RDONLY);
  struct stat sb;
  fstat(file.fd, &sb);
  file.length = sb.st_size;
  file.data = mmap(NULL, sb.st_size, PROT_WRITE, MAP_PRIVATE, file.fd, 0);
  return file;
}
#endif
WGPUState wgpustate = WGPUState();

#define CREATE_BENCHMARK(scale, pairs)                                         \
  static void BM_Scale##scale##Pairs##pairs(benchmark::State &state) {         \
    mmapped_file files[] = {                                                   \
        file_to_mmap("data/" #scale "/" #pairs "-src.bin"),                    \
        file_to_mmap("data/" #scale "/" #pairs "-dst.bin"),                    \
        file_to_mmap("data/" #scale "/v.bin"),                                 \
        file_to_mmap("data/" #scale "/e.bin"),                                 \
        file_to_mmap("data/" #scale "/r-v.bin"),                               \
        file_to_mmap("data/" #scale "/r-e.bin"),                               \
    };                                                                         \
    TimingInfo info = {0, 0};                                                  \
    for (auto _ : state) {                                                     \
    std::vector<IterativeLengthResult> results =                               \
        iterative_length(wgpustate,                                                \
                         {                                                     \
                             .src = (uint32_t *)files[0].data,                 \
                             .dst = (uint32_t *)files[1].data,                 \
                             .length = files[1].length / sizeof(uint32_t),     \
                         },                                                    \
                         {                                                     \
                             .v = (uint32_t *)files[2].data,                   \
                             .e = (uint32_t *)files[3].data,                   \
                             .v_length = files[2].length / sizeof(uint32_t),   \
                             .e_length = files[3].length / sizeof(uint32_t),   \
                         },                                                    \
                         {                                                     \
                             .v = (uint32_t *)files[4].data,                   \
                             .e = (uint32_t *)files[5].data,                   \
                             .v_length = files[4].length / sizeof(uint32_t),   \
                             .e_length = files[5].length / sizeof(uint32_t),   \
                         });                                                   \
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

BENCHMARK_MAIN();
