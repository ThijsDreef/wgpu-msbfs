#include "core/util/shader-locator.hpp"
#include <stdio.h>

char* getShaderFile(const char* value) {
  FILE *file;
  char *buf;

  file = fopen(value, "rb");
  if (!file) {
    return NULL;
  }

  // Add error checking here?
  fseek(file, 0, SEEK_END);
  long length = ftell(file);
  fseek(file, 0, SEEK_SET);
  buf = new char[length + 1];
  fread(buf, 1, length, file);
  buf[length] = 0;

  return buf;
}
