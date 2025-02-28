#include "wgpu-state.hpp"
#include <vector>

std::vector<uint32_t> iterative_length(
                      WGPUState& state,
                      std::vector<uint32_t> src,
                      std::vector<uint32_t> dst,
                      std::vector<uint32_t> v,
                      std::vector<uint32_t> e);
