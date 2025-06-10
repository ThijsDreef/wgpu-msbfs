#include "msbfs.hpp"
#include <cstdlib>
#include <cstring>
#include "emscripten/emscripten.h"

extern "C" {
  struct emscripten_output {
    IterativeLengthResult* result;
    uint32_t length;
  };

}

extern "C" {
  EMSCRIPTEN_KEEPALIVE
  emscripten_output e_iterative_length(PathFindingRequest request, CSR csr) {
    std::vector<IterativeLengthResult> result = iterative_length(request, csr);

    emscripten_output output;
    output.result = (IterativeLengthResult*)malloc(sizeof(IterativeLengthResult) * result.size());
    memcpy(output.result, result.data(), sizeof(IterativeLengthResult) * result.size());
    output.length = result.size();
    return output;
  }
  EMSCRIPTEN_KEEPALIVE
  void *allocate_buffer(uint32_t size) {
    return malloc(size);
  }
  EMSCRIPTEN_KEEPALIVE
  void deallocate_buffer(void *data) {
    free(data);
  }
}

int main() {

}
