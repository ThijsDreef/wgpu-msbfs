#include <cstdint>
#include <iostream>
#include <map>
#include "msbfs.hpp"
#include <cassert>
#include <ostream>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

struct SearchInfo {
  uint32_t iteration;
  uint32_t mask;
  uint32_t jfq_length;
};

#define PARTITIONS 8

__global__ void set_first_bsak(uint32_t *bsak, uint32_t* src, uint32_t v_length) {
  for (uint32_t i = threadIdx.x; i < 32; i += blockDim.x) {
    uint32_t offset = (blockIdx.x * v_length);
    atomicOr(bsak + offset + src[i + blockIdx.x * 32], 1u << i);
  }
}

__global__ void identify_step(uint32_t v_length, SearchInfo *info,
                              uint32_t *jfq, uint32_t *dst,
                              uint32_t *path_length, uint32_t *bsa,
                              uint32_t *bsak) {
  uint32_t v_offset = v_length * blockIdx.x;
  uint32_t c_mask = ~(info[blockIdx.x].mask);
  uint32_t iteration = info[blockIdx.x].iteration;
  if (info[blockIdx.x].jfq_length == 0 && info[blockIdx.x].iteration > 0) {
    return;
  }
  info[blockIdx.x].jfq_length = 0;
  __syncthreads();

  for (uint32_t i = threadIdx.x + blockIdx.y * blockDim.x; i < v_length; i += blockDim.x * gridDim.y) {
    uint32_t diff = (bsa[v_offset + i] ^ bsak[v_offset + i]) & c_mask;
    if (diff == 0) continue;

    bsak[v_offset + i] |= bsa[v_offset + i];
    uint32_t id = atomicAdd(&info[blockIdx.x].jfq_length, 1u);
    jfq[v_offset + id] = i;

    uint32_t length = __popc(diff);
    for (uint32_t x = 0; x < length; x++) {
      uint32_t index = 31 - __clz(diff);
      if (dst[index + blockIdx.x * 32] == i) {
        path_length[index + blockIdx.x * 32] = iteration;
        c_mask &= ~(1 << index);
      }
      diff &= ~(1u << index);
    }
  }
  atomicOr(&info[blockIdx.x].mask, ~c_mask);
  info[blockIdx.x].iteration = iteration + 1;
}

struct CudaPartitionedCSR {
  uint32_t *v;
  uint32_t *e;
};

__global__ void expand_step(uint32_t v_length, uint32_t *v, uint32_t *e,
                            uint32_t *e_offsets, SearchInfo *info,
                            uint32_t *jfq, uint32_t *bsa, uint32_t *bsak) {
  const uint32_t csr_v_offset = v_length * blockIdx.z;
  const uint32_t csr_e_offset = e_offsets[blockIdx.z];
  const uint32_t v_offset = v_length * blockIdx.x;
  const uint32_t length = info[blockIdx.x].jfq_length;
  for (uint32_t i = threadIdx.x + blockIdx.y * blockDim.x; i < length; i += blockDim.x * gridDim.y) {
    const uint32_t source = jfq[v_offset + i];
    const uint32_t val = bsa[v_offset + source];

    uint32_t start = v[source + csr_v_offset] + threadIdx.y;
    const uint32_t end = v[source + csr_v_offset + 1];
    for (; start < end; start += blockDim.y) {
      atomicOr(bsak + v_offset + e[start + csr_e_offset], val);
    }
  }
}


std::vector<IterativeLengthResult> iterative_length(PathFindingRequest request,
                                                    CSR csr) {
  TimingInfo timing_info;
  return iterative_length(request, csr, timing_info);
}

uint32_t get_edge_count(CSR original, uint32_t start, uint32_t end) {
  size_t edge_count = 0;
  for (size_t i = 0; i < original.e_length; i++) {
    edge_count += original.e[i] >= start && original.e[i] < end ? 1 : 0;
  }
  return edge_count;
}

CSR fill_partition(CSR original, uint32_t start, uint32_t end) {
  CSR partition;

  partition.v_length = original.v_length;
  partition.e_length = original.e_length;

  partition.v = new uint32_t[partition.v_length];
  partition.e = new uint32_t[partition.e_length];

  uint32_t v_offset = 0;
  for (size_t i = 0; i < original.v_length; i++) {
    partition.v[i] = v_offset;
    for (uint32_t e = original.v[i]; e < original.v[i + 1]; e++) {
      uint32_t dst = original.e[e];
      if (dst >= start && dst < end) {
        partition.e[v_offset] = dst;
        v_offset++;
      }
    }
  }

  partition.e_length = v_offset;

  return partition;
}

uint32_t find_split_point(CSR original, uint32_t start, uint32_t end) {
  uint32_t length = end - start;
  uint32_t midpoint = length / 2 + start;

  uint32_t total_edge_count = get_edge_count(original, start, end);

  uint32_t edge_counts[2];
  float direction;
  do {
    edge_counts[0] = get_edge_count(original, start, midpoint);
    edge_counts[1] = total_edge_count - edge_counts[0];
    direction = edge_counts[0] / (float)edge_counts[1];
    midpoint += length / 64 * (edge_counts[0] > edge_counts[1] ? -1 : 1);
  } while (direction < 0.95 || direction > 1.05);

  return midpoint;
}

void partition(std::vector<CSR> &partitions, CSR original, uint32_t partition_count, uint32_t start, uint32_t end) {
  if (partition_count == 1) {
    partitions.push_back(fill_partition(original, start, end));
    return;
  }
  uint32_t midpoint = find_split_point(original, start, end);
//  uint32_t midpoint = start + (end - start) / 2
  if (midpoint == start) return;

  partition(partitions, original, partition_count / 2, start, midpoint);
  partition(partitions, original, partition_count / 2, midpoint, end);
}

std::vector<IterativeLengthResult> iterative_length(PathFindingRequest request,
                                                    CSR csr, TimingInfo &info) {
  cudaSetDevice(0);
  std::vector<CSR> partitions;
  partition(partitions, csr, PARTITIONS, 0, csr.v_length);
  const size_t WORKGROUPS = 1;
  const size_t SEARCHES_IN_WORKGROUP = 32;
  const size_t PAIRS_IN_PARALLEL = WORKGROUPS * SEARCHES_IN_WORKGROUP;

  uint64_t v_size = csr.v_length * sizeof(uint32_t);
  uint64_t e_size = csr.e_length * sizeof(uint32_t);

  uint32_t *src, *bsa, *bsak, *jfq, *dst, *path_lengths;
  uint32_t * v_buffer;
  uint32_t * e_buffer;

  uint32_t *host_result = new uint32_t[request.length];
  uint32_t debug[6];
  SearchInfo *search_info;

  std::vector<IterativeLengthResult> results;
  results.reserve(request.length);

  cudaMalloc(&v_buffer, v_size * 8);
  cudaMalloc(&e_buffer, e_size);

  uint32_t e_offset[PARTITIONS];
  uint32_t c_offset = 0;
  for (size_t i = 0; i < PARTITIONS; i++) {
    cudaMemcpy(v_buffer + csr.v_length * i, partitions[i].v, v_size, cudaMemcpyHostToDevice);
    cudaMemcpy(e_buffer + c_offset, partitions[i].e, sizeof(uint32_t) * partitions[i].e_length, cudaMemcpyHostToDevice);
    e_offset[i] = c_offset;
    c_offset += partitions[i].e_length;
  }

  uint32_t *c_e_offset;
  cudaMalloc(&c_e_offset, sizeof(uint32_t) * PARTITIONS);
  cudaMemcpy(c_e_offset, e_offset, sizeof(uint32_t) * PARTITIONS, cudaMemcpyHostToDevice);


  cudaMalloc(&bsa, v_size * WORKGROUPS);
  cudaMalloc(&bsak, v_size * WORKGROUPS);
  cudaMalloc(&jfq, v_size * WORKGROUPS);

  cudaMalloc(&dst, sizeof(uint32_t) * request.length);
  cudaMalloc(&path_lengths, sizeof(uint32_t) * request.length);

  cudaMalloc(&search_info, sizeof(SearchInfo) * WORKGROUPS);

  cudaMalloc(&src, request.length * sizeof(uint32_t));


  cudaMemcpy(src, request.src, request.length * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dst, request.dst, request.length * sizeof(uint32_t), cudaMemcpyHostToDevice);


  for (size_t offset = 0; offset < request.length; offset += PAIRS_IN_PARALLEL) {
    // Clear BSA, BSAK, Destinations, Search Info.
    cudaMemset(bsa, 0, v_size * WORKGROUPS);
    cudaMemset(bsak, 0, v_size * WORKGROUPS);
    cudaMemset(search_info, 0, sizeof(SearchInfo) * WORKGROUPS);
    // Setup BSAK
    set_first_bsak<<<WORKGROUPS, 32>>>(bsak, src + offset, csr.v_length);
    dim3 grid(WORKGROUPS, 46 * 2, PARTITIONS);
    dim3 identify_grid(WORKGROUPS, 46 * 2, 1);
    dim3 block(256, 4, 1);
    uint32_t jfq_lengths = 1;
    for (int iteration = 0; jfq_lengths > 0; iteration++) {
      if (iteration % 2 == 1) {
        identify_step<<<identify_grid, 256>>>(csr.v_length, search_info, jfq, dst + offset, path_lengths + offset, bsa, bsak);
        cudaDeviceSynchronize();
        expand_step<<<grid, block>>>(csr.v_length, v_buffer, e_buffer, c_e_offset, search_info, jfq, bsa, bsak);
      } else {
        identify_step<<<identify_grid, 256>>>(csr.v_length, search_info, jfq, dst + offset, path_lengths + offset, bsak, bsa);
        cudaDeviceSynchronize();
        expand_step<<<grid, block>>>(csr.v_length, v_buffer, e_buffer, c_e_offset, search_info, jfq, bsak, bsa);
      }
      cudaDeviceSynchronize();
      if (iteration % 10 == 0) {
        cudaMemcpy(debug, search_info, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        jfq_lengths = debug[2]; // + debug[5];
        // std::cout << jfq_lengths << std::endl;
      }

     }
  }
  cudaMemcpy(host_result, path_lengths, request.length * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  for (size_t j = 0; j < request.length; j++) {
    if (host_result[j] == 0 && request.dst[j] != request.src[j]) {
      continue;
    }
    results.push_back({
      .src = request.src[j],
      .dst = request.dst[j],
      .length = host_result[j],
    });
  }

  for (auto x : partitions) {
    delete[] x.e;
    delete[] x.v;
  }

  cudaFree(v_buffer);
  cudaFree(e_buffer);
  cudaFree(c_e_offset);

  cudaFree(bsa);
  cudaFree(bsak);
  cudaFree(jfq);

  cudaFree(dst);
  cudaFree(path_lengths);

  cudaFree(src);

  cudaFree(search_info);


  return results;
}
