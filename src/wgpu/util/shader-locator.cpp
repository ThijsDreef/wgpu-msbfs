#include "util/shader-locator.hpp"
#include "utils/file-loader.hpp"
#include <cstring>
#include <stdio.h>

#ifdef USE_FILES
char *getShaderFile(const char *value) { return (char *)load_file(value).data; }
#else
#include "shaders/frontier-expansion.h"
#include "shaders/frontier-identification.h"
#include "shaders/set-first-bsak.h"

char* getShaderPointer(const char* value) {
  if (strcmp(value, "data/shaders/frontier-expansion.wgsl") == 0) {
    return (char*)data_shaders_frontier_expansion_wgsl;
  }
  else if (strcmp(value, "data/shaders/frontier-identification.wgsl") == 0) {
    return (char*)data_shaders_frontier_identification_wgsl;
  }
  else if (strcmp(value, "data/shaders/set-first-bsak.wgsl") == 0) {
    return (char*)data_shaders_set_first_bsak_wgsl;
  }

  return nullptr;
}
#endif
