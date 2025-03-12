#include "core/util/wgpu-utils.hpp"
#include "msbfs.hpp"
#include "wgpu-state.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <sys/mman.h>

#include "sys/mman.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

struct mmapped_file {
  void *data;
  int fd;
  size_t length;
};

#define UNIX

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
WGPUState state = WGPUState();

bool check_against_csv(const char *path,
                       std::vector<IterativeLengthResult> results) {
  std::ifstream file(path);
  size_t it = 0;
  if (file.is_open()) {
    std::string line;
    std::getline(file, line);
    IterativeLengthResult t;
    while (std::getline(file, line)) {
      while (results[it].length == 99999) {
        it++;
      }
      if (it >= results.size()) {
        std::cout << "More results then in CSV" << std::endl;
        return false;
      }
      sscanf(line.c_str(), "%u,%u,%u", &t.src, &t.dst, &t.length);
      if (t.src != results[it].src || t.dst != results[it].dst ||
          t.length != results[it].length) {
        std::cout << "At index: " << it << std::endl;
        std::cout << "[Error] found: " << results[it].src << " " << results[it].dst << " "
                  << results[it].length << std::endl;
        return false;
      }
      it++;
    }
    file.close();
  }
  return true;
}

#define CREATE_TEST_CASE(scale, pairs)                                         \
  TEST(MSBFSIterativeLength, scale##pairs) {                                   \
    mmapped_file files[] = {                                                   \
        file_to_mmap("data/" #scale "/" #pairs "-src.bin"),                    \
        file_to_mmap("data/" #scale "/" #pairs "-dst.bin"),                    \
        file_to_mmap("data/" #scale "/v.bin"),                                 \
        file_to_mmap("data/" #scale "/e.bin"),                                 \
    };                                                                         \
    std::vector<IterativeLengthResult> results =                               \
        iterative_length_multi_queue(state,                                                \
                         {                                                     \
                             .src = (uint32_t *)files[0].data,                 \
                             .dst = (uint32_t *)files[1].data,                 \
                             .length = files[1].length / sizeof(uint32_t),     \
                         },                                                    \
                         {.v = (uint32_t *)files[2].data,                      \
                          .e = (uint32_t *)files[3].data,                      \
                          .v_length = files[2].length / sizeof(uint32_t),      \
                          .e_length = files[3].length / sizeof(uint32_t)});    \
                                                                               \
    ASSERT_TRUE(check_against_csv(                                             \
"data/" #scale "/" #pairs "-iterativelength-truth.csv", results));             \
  }

CREATE_TEST_CASE(1, 1)
CREATE_TEST_CASE(1, 10)
CREATE_TEST_CASE(1, 100)
CREATE_TEST_CASE(1, 1000)
CREATE_TEST_CASE(1, 2048)
CREATE_TEST_CASE(1, 4096)
CREATE_TEST_CASE(1, 8192)

CREATE_TEST_CASE(3, 1)
CREATE_TEST_CASE(3, 10)
CREATE_TEST_CASE(3, 100)
CREATE_TEST_CASE(3, 1000)
CREATE_TEST_CASE(3, 2048)
CREATE_TEST_CASE(3, 4096)
CREATE_TEST_CASE(3, 8192)

CREATE_TEST_CASE(10, 1)
CREATE_TEST_CASE(10, 10)
CREATE_TEST_CASE(10, 100)
CREATE_TEST_CASE(10, 1000)
CREATE_TEST_CASE(10, 2048)
CREATE_TEST_CASE(10, 4096)
CREATE_TEST_CASE(10, 8192)

CREATE_TEST_CASE(30, 1)
CREATE_TEST_CASE(30, 10)
CREATE_TEST_CASE(30, 100)
CREATE_TEST_CASE(30, 1000)
CREATE_TEST_CASE(30, 2048)
CREATE_TEST_CASE(30, 4096)
CREATE_TEST_CASE(30, 8192)

TEST(MSBFSIterativeLength, GraphBlas) {
  std::vector<uint32_t> src = {0, 0, 0, 0, 0, 0};
  std::vector<uint32_t> dst = {1, 2, 3, 4, 5, 6};
  std::vector<uint32_t> v = {0, 2, 4, 5, 7, 2, 6, 11};
  std::vector<uint32_t> e = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};

  std::vector<IterativeLengthResult> results =
      iterative_length(state,
                       {
                           .src = src.data(),
                           .dst = dst.data(),
                           .length = src.size(),
                       },
                       {
                           .v = v.data(),
                           .e = e.data(),
                           .v_length = v.size(),
                           .e_length = e.size(),
                       });
  std::vector<uint32_t> expected_results = {1, 2, 1, 2, 3, 2};

  EXPECT_EQ(results.size(), expected_results.size());
  for (size_t x = 0; x < expected_results.size(); x++) {
    EXPECT_EQ(results[x].length, expected_results[x]);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
