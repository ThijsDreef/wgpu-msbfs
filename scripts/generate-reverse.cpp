#include <cstdint>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
struct BinaryLoadedFile {
  void *data;
  size_t length;
};

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

  result.data = buf;
  result.length = length;
  return result;
}


struct CSR {
  uint32_t *v;
  uint32_t *e;
  uint64_t v_length;
  uint64_t e_length;
};

CSR reverse_csr(CSR original) {
  CSR reversed_csr;
  reversed_csr.v = new uint32_t[original.v_length];
  reversed_csr.e = new uint32_t[original.e_length];
  reversed_csr.v_length = original.v_length;
  reversed_csr.e_length = original.e_length;

  std::vector<std::vector<uint32_t>> results;
  for (size_t x = 0; x < original.v_length; x++) {
    results.push_back(std::vector<uint32_t>());
  }

  for (uint32_t source = 0; source < original.v_length - 1; source++) {
    uint32_t start = original.v[source];
    uint32_t end = original.v[source + 1];
    for (; start < end; start++) {
      size_t dest = original.e[start];
      results[dest].push_back(source);
    }
  }

  uint32_t offset = 0;
  for (size_t y = 0; y < results.size(); y++) {
    reversed_csr.v[y] = offset;
    for (size_t z = 0; z < results[y].size(); z++) {
      reversed_csr.e[offset] = results[y][z];
      offset++;
    }
  }

  return reversed_csr;
}

int main(int argc, char **argv) {
  struct BinaryLoadedFile vertex = load_file(argv[1]);
  struct BinaryLoadedFile edges = load_file(argv[2]);

  CSR from_csr;

  from_csr.v = (uint32_t *)vertex.data;
  from_csr.e = (uint32_t *)edges.data;
  from_csr.v_length = vertex.length / sizeof(uint32_t);
  from_csr.e_length = edges.length / sizeof(uint32_t);

  CSR reversed_csr = reverse_csr(from_csr);
  FILE *fp = fopen("r-v.bin", "wb");
  fwrite(reversed_csr.v, reversed_csr.v_length, sizeof(uint32_t), fp);
  fclose(fp);
  fp = fopen("r-e.bin", "wb");
  fwrite(reversed_csr.e, reversed_csr.e_length, sizeof(uint32_t), fp);
  fclose(fp);
  return 0;
}
