#include "wgpu-state.hpp"
#include "core/util/wgpu-utils.hpp"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include "msbfs.hpp"

TEST(MSBFSIterativeLength, GraphBlas) {
  WGPUState state = WGPUState();

  std::vector<uint32_t> src = { 0, 0, 0, 0, 0, 0 };
  std::vector<uint32_t> dst = { 1, 2, 3, 4, 5, 6 };
  std::vector<uint32_t> v = {0, 2, 4, 5, 7, 2, 6, 11};
  std::vector<uint32_t> e = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};

  std::vector<uint32_t> results = iterative_length(state, src, dst, v, e);
  std::vector<uint32_t> expected_results = {1, 2, 1, 2, 3, 2};

  EXPECT_EQ(results.size(), expected_results.size());
  for (size_t x = 0; x < expected_results.size(); x++) {
    EXPECT_EQ(results[x], expected_results[x]);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
