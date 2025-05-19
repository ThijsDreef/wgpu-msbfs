#include "core/util/shader-locator.hpp"
#include "core/util/file-loader.hpp"
#include <stdio.h>

char* getShaderFile(const char* value) {
  return (char*)load_file(value).data;
}
