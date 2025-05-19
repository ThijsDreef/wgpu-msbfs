#include <cstddef>

struct BinaryLoadedFile {
  void *data;
  size_t length;
};

BinaryLoadedFile load_file(const char* path);
