#include <cstdint>
#include <iostream>
#include <map>
#include "msbfs.hpp"
#include <cassert>
#include <ostream>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define to_interleaved(var) (var * blockDim.x) + threadIdx.x

struct SearchInfo {
  uint32_t iteration;
  uint32_t mask;
  uint32_t jfq_length;
};

__global__ void set_first_bsak(uint32_t *bsak, uint32_t* src, uint32_t total_length) {
  for (uint32_t i = threadIdx.x; i < 32 && (i + blockIdx.x * 32) < total_length; i += 1) {
    uint32_t v_loc = src[i + (blockIdx.x * 32)] * gridDim.x + blockIdx.x;
    atomicOr(bsak + v_loc, 1u << i);
  }
}

__global__ void identify_step(uint32_t v_length, SearchInfo *info,
                              uint32_t *jfq, uint32_t *dst,
                              uint32_t *path_length, uint32_t *bsa,
                              uint32_t *bsak) {

  const uint32_t id = threadIdx.x;
  uint32_t c_mask = ~(info[id].mask);
  uint32_t iteration = info[id].iteration;
  if (info[id].jfq_length == 0 && info[id].iteration > 0) {
    return;
  }
  info[id].jfq_length = 0;
  __syncthreads();

  for (uint32_t i = blockIdx.x; i < v_length; i += gridDim.x) {
    uint32_t diff = (bsa[to_interleaved(i)] ^ bsak[to_interleaved(i)]) & c_mask;
    if (diff == 0) continue;

    bsak[to_interleaved(i)] |= bsa[to_interleaved(i)];
    jfq[to_interleaved(atomicAdd(&info[id].jfq_length, 1u))] = i;

    uint32_t length = __popc(diff);
    for (uint32_t x = 0; x < length; x++) {
      uint32_t index = 31 - __clz(diff);
      if (dst[index + id * 32] == i) {
        path_length[index + id * 32] = iteration;
        c_mask &= ~(1 << index);
      }
      diff &= ~(1u << index);
    }
  }
  atomicOr(&info[id].mask, ~c_mask);
  info[id].iteration = iteration + 1;
}

__global__ void expand_step(uint32_t v_length, uint32_t* v, uint32_t* e, SearchInfo *info, uint32_t *jfq,
                            uint32_t *bsa, uint32_t *bsak) {
  const uint32_t length = info[threadIdx.x].jfq_length;
  for (uint32_t i = blockIdx.x; i < length; i += gridDim.x) {
    const uint32_t source = jfq[to_interleaved(i)];
    const uint32_t val = bsa[to_interleaved(source)];

    uint32_t start = v[source] + threadIdx.y;
    const uint32_t end = v[source + 1];
    for (; start < end; start += blockDim.y) {
      atomicOr(bsak + (to_interleaved(e[start])), val);
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
  const size_t WORKGROUPS = 8;
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

  uint32_t* bsak_temp = new uint32_t[csr.v_length * WORKGROUPS];

  CSR cuda_csr = csr;
  cuda_csr.v = v_buffer;
  cuda_csr.e = e_buffer;

  for (size_t offset = 0; offset < request.length; offset += PAIRS_IN_PARALLEL) {
    // Clear BSA, BSAK, Destinations, Search Info.
    cudaMemset(bsa, 0, v_size * WORKGROUPS);
    cudaMemset(bsak, 0, v_size * WORKGROUPS);
    cudaMemset(search_info, 0, sizeof(SearchInfo) * WORKGROUPS);
    // Setup BSAK
    set_first_bsak<<<WORKGROUPS, 32>>>(bsak, src + offset, request.length - offset);
//    cudaDeviceSynchronize();
    dim3 grid(92 * 4, 1, 1);
    dim3 block(WORKGROUPS, 4, 1);
    uint32_t jfq_lengths = 1;
    for (int iteration = 0; jfq_lengths > 0; iteration++) {
      if (iteration % 2 == 1) {
        identify_step<<<grid, WORKGROUPS>>>(csr.v_length, search_info, jfq, dst + offset, path_lengths + offset, bsa, bsak);
        cudaDeviceSynchronize();
        expand_step<<<grid, block>>>(cuda_csr.v_length, v_buffer, e_buffer, search_info, jfq, bsa, bsak);
      } else {
        identify_step<<<grid, WORKGROUPS>>>(csr.v_length, search_info, jfq, dst + offset, path_lengths + offset, bsak, bsa);
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
