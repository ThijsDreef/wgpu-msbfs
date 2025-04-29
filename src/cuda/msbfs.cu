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
__global__ void expand_step(uint32_t v_length, uint32_t* v, uint32_t* e, SearchInfo *info, uint32_t *jfq,
                            uint32_t *bsa, uint32_t *bsak) {
  const uint32_t v_offset = v_length * blockIdx.x;
  const uint32_t length = info[blockIdx.x].jfq_length;
  for (uint32_t i = threadIdx.x + blockIdx.y * blockDim.x; i < length; i += blockDim.x * gridDim.y) {
    const uint32_t source = jfq[v_offset + i];
    const uint32_t val = bsa[v_offset + source];

    uint32_t start = v[source] + threadIdx.y;
    const uint32_t end = v[source + 1];
    for (; start < end; start += blockDim.y) {
      atomicOr(bsak + v_offset + e[start], val);
    }
  }
}

std::vector<IterativeLengthResult> iterative_length(PathFindingRequest request,
                                                    CSR csr) {
  TimingInfo timing_info;
  return iterative_length(request, csr, timing_info);
}

std::vector<IterativeLengthResult> iterative_length(PathFindingRequest request,
                                                    CSR csr, TimingInfo &info) {
  cudaSetDevice(0);
  const size_t WORKGROUPS = 1;
  const size_t SEARCHES_IN_WORKGROUP = 32;
  const size_t PAIRS_IN_PARALLEL = WORKGROUPS * SEARCHES_IN_WORKGROUP;

  uint64_t v_size = csr.v_length * sizeof(uint32_t);
  uint64_t e_size = csr.e_length * sizeof(uint32_t);

  uint32_t *src, *bsa, *bsak, *jfq, *v_buffer, *e_buffer, *dst, *path_lengths;
  uint32_t *host_result = new uint32_t[request.length];
  uint32_t debug[WORKGROUPS * 3];
  SearchInfo *search_info;

  std::vector<IterativeLengthResult> results;
  results.reserve(request.length);

  cudaMalloc(&v_buffer, v_size);
  cudaMalloc(&e_buffer, e_size);

  cudaMalloc(&bsa, v_size * WORKGROUPS);
  cudaMalloc(&bsak, v_size * WORKGROUPS);
  cudaMalloc(&jfq, v_size * WORKGROUPS);

  cudaMalloc(&dst, sizeof(uint32_t) * request.length);
  cudaMalloc(&path_lengths, sizeof(uint32_t) * request.length);

  cudaMalloc(&search_info, sizeof(SearchInfo) * WORKGROUPS);

  cudaMalloc(&src, request.length * sizeof(uint32_t));

  cudaMemcpy(v_buffer, csr.v, v_size, cudaMemcpyHostToDevice);
  cudaMemcpy(e_buffer, csr.e, e_size, cudaMemcpyHostToDevice);
  cudaMemcpy(src, request.src, request.length * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dst, request.dst, request.length * sizeof(uint32_t), cudaMemcpyHostToDevice);

  CSR cuda_csr = csr;
  cuda_csr.v = v_buffer;
  cuda_csr.e = e_buffer;

  for (size_t offset = 0; offset < request.length; offset += PAIRS_IN_PARALLEL) {
    // Clear BSA, BSAK, Destinations, Search Info.
    cudaMemset(bsa, 0, v_size * WORKGROUPS);
    cudaMemset(bsak, 0, v_size * WORKGROUPS);
    cudaMemset(search_info, 0, sizeof(SearchInfo) * WORKGROUPS);
    // Setup BSAK
    set_first_bsak<<<WORKGROUPS, 32>>>(bsak, src + offset, csr.v_length);
    dim3 grid(WORKGROUPS, 92 / WORKGROUPS, 1);
    dim3 block(128, 8, 1);
    uint32_t jfq_lengths = 1;
    for (int iteration = 0; jfq_lengths > 0; iteration++) {
      if (iteration % 2 == 1) {
        identify_step<<<grid, 256>>>(csr.v_length, search_info, jfq, dst + offset, path_lengths + offset, bsa, bsak);
        cudaDeviceSynchronize();
        expand_step<<<grid, block>>>(cuda_csr.v_length, v_buffer, e_buffer, search_info, jfq, bsa, bsak);
      } else {
        identify_step<<<grid, 256>>>(csr.v_length, search_info, jfq, dst + offset, path_lengths + offset, bsak, bsa);
        cudaDeviceSynchronize();
        expand_step<<<grid, block>>>(cuda_csr.v_length, v_buffer, e_buffer, search_info, jfq, bsak, bsa);
      }
      cudaDeviceSynchronize();
      if (iteration % 10 == 0) {
        cudaMemcpy(debug, search_info, WORKGROUPS * sizeof(SearchInfo), cudaMemcpyDeviceToHost);
        jfq_lengths = 0;
        for (size_t i = 0; i < WORKGROUPS; i++) {
          jfq_lengths += debug[2 + 3 * i];
        }
//        std::cout << jfq_lengths << std::endl;
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

  cudaFree(v_buffer);
  cudaFree(e_buffer);

  cudaFree(bsa);
  cudaFree(bsak);
  cudaFree(jfq);

  cudaFree(dst);
  cudaFree(path_lengths);

  cudaFree(src);

  cudaFree(search_info);


  return results;
}
