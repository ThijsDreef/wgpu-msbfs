#include "wgpu-state.hpp"
#include <cstdint>
#include <vector>

struct IterativeLengthResult {
  uint32_t src;
  uint32_t dst;
  uint32_t length;
};

struct PathFindingRequest {
  uint32_t *src;
  uint32_t *dst;
  uint64_t length;
};

struct CSR {
  uint32_t *v;
  uint32_t *e;
  uint64_t v_length;
  uint64_t e_length;
};

std::vector<IterativeLengthResult> iterative_length(
                      WGPUState& state,
                      PathFindingRequest request,
                      CSR csr);
