#include "utils/file-loader.hpp"
#include <stdio.h>

BinaryLoadedFile load_file(const char *path) {
  FILE *file;
  char *buf;
  BinaryLoadedFile result;

  file = fopen(path, "rb");
  if (!file) {
    result.data = nullptr;
    result.length = 0;
    return result;
  }

  // Add error checking here?
  fseek(file, 0, SEEK_END);
  size_t length = ftell(file);
  fseek(file, 0, SEEK_SET);
  buf = new char[length + 1];
  fread(buf, 1, length, file);
  buf[length] = 0;
  fclose(file);

  result.data = buf;
  result.length = length;
  return result;
}
