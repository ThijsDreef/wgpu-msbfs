#include "sys/mman.h"
#include <cstdint>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
struct mmapped_file {
  void *data;
  int fd;
  size_t length;
};

struct mmapped_file file_to_mmap(const char *path) {
  struct mmapped_file file;
  file.fd = open(path, O_RDONLY);
  struct stat sb;
  fstat(file.fd, &sb);
  file.length = sb.st_size;
  file.data = mmap(NULL, sb.st_size, PROT_WRITE, MAP_PRIVATE, file.fd, 0);
  return file;
}
struct CSR {
  uint32_t *v;
  uint32_t *e;
  uint64_t v_length;
  uint64_t e_length;
};

CSR reverse_csr(CSR original) {
  CSR reversed_csr = {
    .v = new uint32_t[original.v_length],
    .e = new uint32_t[original.e_length],
    .v_length = original.v_length,
    .e_length = original.e_length,
  };

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
//  uint32_t v[] = { 0, 1, 2, 3, 4, 5};
//  uint32_t e[] = { 1, 0, 0, 0, 0 };
//  CSR test_csr = {
//    .v = v,
//    .e = e,
//    .v_length = 6,
//    .e_length = 5,
//  };
//
//  CSR test_reverse_csr = reverse_csr(test_csr);
//  for (size_t x = 0; x < test_reverse_csr.v_length; x++) {
//    std::cout << test_reverse_csr.v[x] << ", ";
//  }
//  std::cout << std::endl;
//  for (size_t x = 0; x < test_reverse_csr.e_length; x++) {
//    std::cout << test_reverse_csr.e[x] << ", ";
//  }
//  std::cout << std::endl;
//  return 0;

  struct mmapped_file vertex = file_to_mmap(argv[1]);
  struct mmapped_file edges = file_to_mmap(argv[2]);

  CSR from_csr = {
      .v = (uint32_t *)vertex.data,
      .e = (uint32_t *)edges.data,
      .v_length = vertex.length / sizeof(uint32_t),
      .e_length = edges.length / sizeof(uint32_t),
  };

  CSR reversed_csr = reverse_csr(from_csr);

  FILE *fp = fopen("r-v.bin", "wb");
  fwrite(reversed_csr.v, reversed_csr.v_length, sizeof(uint32_t), fp);
  fclose(fp);
  fp = fopen("r-e.bin", "wb");
  fwrite(reversed_csr.e, reversed_csr.e_length, sizeof(uint32_t), fp);
  fclose(fp);
  return 0;
}
