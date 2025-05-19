#include "core/util/file-loader.hpp"
#include <stdio.h>

BinaryLoadedFile load_file(const char *path) {
  FILE *file;
  char *buf;

  file = fopen(path, "rb");
  if (!file) {
    return {
      .data = nullptr,
      .length = 0,
    };
  }

  // Add error checking here?
  fseek(file, 0, SEEK_END);
  size_t length = ftell(file);
  fseek(file, 0, SEEK_SET);
  buf = new char[length + 1];
  fread(buf, 1, length, file);
  buf[length] = 0;

  return {
    .data = buf,
    .length = length,
  };
}
