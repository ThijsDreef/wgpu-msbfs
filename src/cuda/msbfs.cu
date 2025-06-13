#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include "msbfs.hpp"
#include <cassert>
#include <ostream>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#define SEARCHES_PER_ENTRY 32
#define SEARCH_ENTRIES 32
#define SEARCHES_IN_WORKGROUP SEARCHES_PER_ENTRY * SEARCH_ENTRIES

struct SearchInfo {
  uint32_t iteration;
  uint32_t mask[SEARCH_ENTRIES];
  uint32_t jfq_length;
};

__global__ void set_first_bsak(uint32_t *bsak, uint32_t* src, uint32_t request_length) {
  for (uint32_t i = threadIdx.x; i < 32 && i + blockIdx.x * 32 < request_length; i += blockDim.x) {
    atomicOr(bsak + src[i + blockIdx.x * 32] * gridDim.x + blockIdx.x, 1u << i);
  }
}

__global__ void identify_step(uint32_t v_length, SearchInfo *info,
                              uint32_t *jfq, uint32_t *dst,
                              uint32_t *path_length, uint32_t *bsa,
                              uint32_t *bsak) {
  uint32_t c_mask = ~info->mask[threadIdx.x];
  uint32_t iteration = info->iteration;
  if (info[blockIdx.x].jfq_length == 0 && info[blockIdx.x].iteration > 0) {
    return;
  }
  info->jfq_length = 0;
  __syncthreads();

  for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < v_length; i += gridDim.y * blockDim.y) {
    uint32_t diff = (bsa[i * blockDim.x + threadIdx.x] ^ bsak[i * blockDim.x + threadIdx.x]) & c_mask;
    if (__ballot_sync(~0, diff != 0) == 0) {
      continue;
    }

    bsak[i * blockDim.x + threadIdx.x] |= bsa[i * blockDim.x + threadIdx.x];

    uint32_t length = __popc(diff);
    for (uint32_t x = 0; x < length; x++) {
      uint32_t index = 31 - __clz(diff);
      if (dst[index + threadIdx.x * 32] == i) {
        path_length[index + threadIdx.x * 32] = iteration;
        c_mask &= ~(1 << index);
      }
      diff &= ~(1u << index);
    }

    if (threadIdx.x == 0) {
      uint32_t id = atomicAdd(&info->jfq_length, 1u);
      jfq[id] = i;
    }
  }
  atomicOr(&info->mask[threadIdx.x], ~c_mask);
  info->iteration = iteration + 1;
}

__global__ void identify_step_bottom_up(uint32_t v_length, SearchInfo *info,
                              uint32_t *jfq, uint32_t *dst,
                              uint32_t *path_length, uint32_t *bsa,
                              uint32_t *bsak) {
  uint32_t c_mask = ~info->mask[threadIdx.x];
  uint32_t iteration = info->iteration;
  if (info[blockIdx.x].jfq_length == 0 && info[blockIdx.x].iteration > 0) {
    return;
  }
  info->jfq_length = 0;
  __syncthreads();
  for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < v_length; i += gridDim.y * blockDim.y) {
    if (__ballot_sync(~0, ((~(bsak[i * blockDim.x + threadIdx.x] | (~c_mask)))) > 0) == 0) {
      continue;
    }
    uint32_t diff = (bsa[i * blockDim.x + threadIdx.x] ^ bsak[i * blockDim.x + threadIdx.x]) & c_mask;
    bsak[i * blockDim.x + threadIdx.x] |= bsa[i * blockDim.x + threadIdx.x];

    uint32_t length = __popc(diff);
    for (uint32_t x = 0; x < length; x++) {
      uint32_t index = 31 - __clz(diff);
      if (dst[index + threadIdx.x * 32] == i) {
        path_length[index + threadIdx.x * 32] = iteration;
        c_mask &= ~(1 << index);
      }
      diff &= ~(1u << index);
    }

    if (threadIdx.x == 0) {
      uint32_t id = atomicAdd(&info->jfq_length, 1u);
      jfq[id] = i;
    }
  }
  atomicOr(&info->mask[threadIdx.x], ~c_mask);
  info->iteration = iteration + 1;
}
__global__ void expand_step_bottom_up(uint32_t v_length, uint32_t* v, uint32_t* e, SearchInfo *info, uint32_t *jfq,
                                      uint32_t *bsa, uint32_t *bsak) {
  const uint32_t length = info[blockIdx.x].jfq_length;
  for (uint32_t i = blockIdx.y; i < length; i += gridDim.y) {
    const uint32_t source = jfq[i];
    uint32_t val = bsak[source * blockDim.x + threadIdx.x];

    uint32_t start = v[source] + threadIdx.y;
    const uint32_t end = v[source + 1];
    for (; start < end && val != ~0; start += blockDim.y) {
      const uint32_t neighbour = e[start];
      val |= bsa[neighbour * blockDim.x + threadIdx.x];
    }
    atomicOr(&bsak[source * blockDim.x + threadIdx.x], val);
  }
}

__global__ void expand_step_top_down(uint32_t v_length, uint32_t* v, uint32_t* e, SearchInfo *info, uint32_t *jfq,
                            uint32_t *bsa, uint32_t *bsak) {
  const uint32_t length = info->jfq_length;
  for (uint32_t i = blockIdx.y; i < length; i += gridDim.y) {
    const uint32_t source = jfq[i];
    const uint32_t val = bsa[source * blockDim.x + threadIdx.x];

    uint32_t start = v[source] + threadIdx.y;
    const uint32_t end = v[source + 1];
    for (; start < end; start += blockDim.y) {
      atomicOr(bsak + e[start] * blockDim.x + threadIdx.x, val);
    }
  }
}

std::vector<IterativeLengthResult> iterative_length(PathFindingRequest request,
                                                    CSR csr, CSR reverse_csr) {
  TimingInfo timing_info;
  return iterative_length(request, csr, reverse_csr, timing_info);
}

std::vector<IterativeLengthResult> iterative_length(PathFindingRequest request,
                                                    CSR csr, CSR reverse_csr, TimingInfo &info) {
  cudaSetDevice(0);

  uint64_t v_size = csr.v_length * sizeof(uint32_t);
  uint64_t e_size = csr.e_length * sizeof(uint32_t);

  uint32_t *src, *bsa, *bsak, *jfq, *v_buffer, *e_buffer, *r_v_buffer, *r_e_buffer, *dst, *path_lengths;
  uint32_t *host_result = new uint32_t[request.length];
  SearchInfo debug[1];

  uint32_t *d_bsa = new uint32_t[csr.v_length * SEARCH_ENTRIES];

  SearchInfo *search_info;

  std::vector<IterativeLengthResult> results;
  results.reserve(request.length);

  cudaMalloc(&v_buffer, v_size);
  cudaMalloc(&e_buffer, e_size);
  cudaMalloc(&r_v_buffer, v_size);
  cudaMalloc(&r_e_buffer, e_size);

  cudaMalloc(&bsa, v_size * SEARCH_ENTRIES);
  cudaMalloc(&bsak, v_size * SEARCH_ENTRIES);
  cudaMalloc(&jfq, v_size);

  cudaMalloc(&dst, sizeof(uint32_t) * request.length);
  cudaMalloc(&path_lengths, sizeof(uint32_t) * request.length);

  cudaMalloc(&search_info, sizeof(SearchInfo));

  cudaMalloc(&src, request.length * sizeof(uint32_t));

  cudaMemcpy(v_buffer, csr.v, v_size, cudaMemcpyHostToDevice);
  cudaMemcpy(e_buffer, csr.e, e_size, cudaMemcpyHostToDevice);
  cudaMemcpy(r_v_buffer, reverse_csr.v, v_size, cudaMemcpyHostToDevice);
  cudaMemcpy(r_e_buffer, reverse_csr.e, e_size, cudaMemcpyHostToDevice);

  cudaMemcpy(src, request.src, request.length * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dst, request.dst, request.length * sizeof(uint32_t), cudaMemcpyHostToDevice);

  CSR cuda_csr = csr;
  cuda_csr.v = v_buffer;
  cuda_csr.e = e_buffer;

  for (size_t offset = 0; offset < request.length; offset += SEARCHES_IN_WORKGROUP) {
    // Clear BSA, BSAK, Destinations, Search Info.
    cudaMemset(bsa, 0, v_size * SEARCH_ENTRIES);
    cudaMemset(bsak, 0, v_size * SEARCH_ENTRIES);
    cudaMemset(search_info, 0, sizeof(SearchInfo));
    // Setup BSAK
    set_first_bsak<<<SEARCH_ENTRIES, 32>>>(bsak, src + offset, request.length - offset);
    dim3 grid(1, 46 * 6, 1);
    dim3 block(SEARCH_ENTRIES, 4, 1);
    uint32_t jfq_lengths = 1;
    int32_t bottom_up_iterations = 0;
    for (int iteration = 0; jfq_lengths > 0; iteration++) {
      if (iteration % 2 == 1) {
        if (bottom_up_iterations > 0) {
          identify_step_bottom_up<<<grid, dim3(SEARCH_ENTRIES, 4, 1)>>>(csr.v_length, search_info, jfq, dst + offset, path_lengths + offset, bsa, bsak);
          cudaDeviceSynchronize();
          expand_step_bottom_up<<<grid, block>>>(cuda_csr.v_length, r_v_buffer, r_e_buffer, search_info, jfq, bsa, bsak);
        } else {
          identify_step<<<grid, dim3(SEARCH_ENTRIES, 4, 1)>>>(csr.v_length, search_info, jfq, dst + offset, path_lengths + offset, bsa, bsak);
          cudaDeviceSynchronize();
          expand_step_top_down<<<grid, block>>>(cuda_csr.v_length, v_buffer, e_buffer, search_info, jfq, bsa, bsak);
        }

      } else {
        if (bottom_up_iterations > 0) {
          identify_step_bottom_up<<<grid, dim3(SEARCH_ENTRIES, 4, 1)>>>(csr.v_length, search_info, jfq, dst + offset, path_lengths + offset, bsak, bsa);
          cudaDeviceSynchronize();
          expand_step_bottom_up<<<grid, block>>>(cuda_csr.v_length, r_v_buffer, r_e_buffer, search_info, jfq, bsak, bsa);
        } else {
          identify_step<<<grid, dim3(SEARCH_ENTRIES, 4, 1)>>>(csr.v_length, search_info, jfq, dst + offset, path_lengths + offset, bsak, bsa);
          cudaDeviceSynchronize();
          expand_step_top_down<<<grid, block>>>(cuda_csr.v_length, v_buffer, e_buffer, search_info, jfq, bsak, bsa);
        }
      }
      bottom_up_iterations--;
      cudaDeviceSynchronize();
      cudaMemcpy(debug, search_info, sizeof(SearchInfo), cudaMemcpyDeviceToHost);
      jfq_lengths = debug[0].jfq_length;
      if (bottom_up_iterations < 0 && jfq_lengths > csr.v_length / 2 + csr.v_length / 3) {
        bottom_up_iterations = 2;
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
  cudaFree(r_e_buffer);
  cudaFree(r_v_buffer);

  cudaFree(bsa);
  cudaFree(bsak);
  cudaFree(jfq);

  cudaFree(dst);
  cudaFree(path_lengths);

  cudaFree(src);

  cudaFree(search_info);


  return results;
}
