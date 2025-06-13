#include "wgpu/wgpu-state.hpp"
#include "msbfs.hpp"
#include "util/wgpu-utils.hpp"
#include <cstdint>
#include "util/search-info.hpp"
WGPUState state = WGPUState();

std::vector<IterativeLengthResult> iterative_length(PathFindingRequest request, CSR csr, CSR reverse_csr) {
  TimingInfo timing_info;
  return iterative_length(request, csr, reverse_csr, timing_info);
}


std::vector<IterativeLengthResult> iterative_length(PathFindingRequest request, CSR csr, CSR reverse_csr, TimingInfo& timing_info) {
  assert(csr.v_length == reverse_csr.v_length);
  assert(csr.e_length == reverse_csr.e_length);
  uint64_t v_size = csr.v_length * sizeof(uint32_t);
  uint64_t e_size = csr.e_length * sizeof(uint32_t);

  const size_t WORKGROUPS = 1;
  const size_t SEARCHES_PER_WORKGROUP = 32;
  const size_t SEARCHES_PER_THREAD = 32;
  const size_t SEARCHES_IN_WORKGROUP = SEARCHES_PER_THREAD * SEARCHES_PER_WORKGROUP;
  const size_t PAIRS_IN_PARALLEL = WORKGROUPS * SEARCHES_IN_WORKGROUP;

  int flags = wgpu::BufferUsage::Storage |
              wgpu::BufferUsage::CopyDst;
  wgpu::BufferDescriptor desc = getBufferDescriptor(v_size * 2, false, flags);
  wgpu::Buffer v_buffer = state.device.createBuffer(desc);

  desc = getBufferDescriptor(e_size * 2, false, flags);
  wgpu::Buffer e_buffer = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
  desc = getBufferDescriptor(v_size * WORKGROUPS * SEARCHES_PER_THREAD, false, flags);
  wgpu::Buffer bsa = state.device.createBuffer(desc);
  wgpu::Buffer bsak = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage;
  desc = getBufferDescriptor(v_size * WORKGROUPS, false, flags);
  wgpu::Buffer jfq = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
          wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(SearchInfo) * WORKGROUPS, false, flags);
  wgpu::Buffer search_info = state.device.createBuffer(desc);


  flags = wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopyDst |
          wgpu::BufferUsage::CopySrc;
  desc = getBufferDescriptor(sizeof(uint32_t) * request.length, false, flags);
  wgpu::Buffer path_lengths = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::CopyDst |
          wgpu::BufferUsage::MapRead;
  desc = getBufferDescriptor(sizeof(uint32_t) * request.length, false, flags);
  wgpu::Buffer path_lengths_staging = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  wgpu::Buffer jfq_length_staging = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t) * request.length, false, flags);
  wgpu::Buffer all_destinations = state.device.createBuffer(desc);
  wgpu::Buffer all_sources = state.device.createBuffer(desc);

  // Populate buffers
  state.queue.writeBuffer(v_buffer, 0, csr.v, v_size);
  state.queue.writeBuffer(e_buffer, 0, csr.e, e_size);
  state.queue.writeBuffer(v_buffer, v_size, reverse_csr.v, v_size);
  state.queue.writeBuffer(e_buffer, e_size, reverse_csr.e, e_size);
  state.queue.writeBuffer(all_destinations, 0, request.dst, request.length * sizeof(uint32_t));
  state.queue.writeBuffer(all_sources, 0, request.src, request.length * sizeof(uint32_t));

  wgpu::BindGroup expand_groups[] = {
    state.expand->csr_group.getBindGroup(v_buffer, e_buffer, v_size * 2, e_size * 2),
    state.expand->jfq_group.getBindGroup(jfq, search_info, v_size * WORKGROUPS, WORKGROUPS),
    state.expand->bsa_group.getBindGroup(bsak, bsa, v_size * WORKGROUPS * SEARCHES_PER_THREAD),
    state.expand->bsa_group.getBindGroup(bsa, bsak, v_size * WORKGROUPS * SEARCHES_PER_THREAD),
  };

  wgpu::BindGroup identify_groups[] = {
    state.identify->jfq_group.getBindGroup(jfq, search_info, v_size * WORKGROUPS, WORKGROUPS),
    state.identify->length_group.getBindGroup(all_destinations, path_lengths, request.length),
    state.identify->bsa_group.getBindGroup(bsak, bsa, v_size * WORKGROUPS * SEARCHES_PER_THREAD),
    state.identify->bsa_group.getBindGroup(bsa, bsak, v_size * WORKGROUPS * SEARCHES_PER_THREAD),
  };


  wgpu::BindGroup set_bsak_groups[] = {
    state.set_bsak->set_bsak_group.getBindGroup(search_info, all_sources, bsak, csr.v_length, request.length),
  };

  std::vector<IterativeLengthResult> results;
  results.reserve(request.length);

  for (size_t offset = 0; offset < request.length; offset += PAIRS_IN_PARALLEL) {
    state.queue.writeBuffer(search_info, 0, &offset, sizeof(uint32_t));
    size_t pairs_to_solve = request.length - offset > PAIRS_IN_PARALLEL ? PAIRS_IN_PARALLEL : request.length - offset;
    wgpu::CommandEncoder encoder = state.device.createCommandEncoder();
    encoder.clearBuffer(bsa, 0, v_size * WORKGROUPS * SEARCHES_PER_WORKGROUP);
    encoder.clearBuffer(bsak, 0, v_size * WORKGROUPS * SEARCHES_PER_WORKGROUP);
    encoder.clearBuffer(search_info, sizeof(uint32_t), sizeof(SearchInfo) - sizeof(uint32_t));
    wgpu::ComputePassEncoder c_encoder = encoder.beginComputePass();
    c_encoder.setPipeline(state.set_bsak->pipeline);
    c_encoder.setBindGroup(0, set_bsak_groups[0], 0, nullptr);
    c_encoder.dispatchWorkgroups(32, 1, 1);
    c_encoder.end();
    c_encoder.release();
    state.queue.submit(encoder.finish());
    encoder.release();

    uint32_t jfq_length = 1;
#ifdef __EMSCRIPTEN__
    uint32_t target_iterations = 8;
#else
    uint32_t target_iterations = 2;
#endif
    bool bottom_up = false;
    while (jfq_length > 0) {
      encoder = state.device.createCommandEncoder();
      for (size_t iterations = 0; iterations < target_iterations; iterations++) {
        // current hack to syncly set jfq_length back to zero
        for (size_t w = 0; w < WORKGROUPS; w++) {
          encoder.clearBuffer(search_info, (2 + 4 * w) * sizeof(uint32_t), sizeof(uint32_t));
        }
        wgpu::ComputePassEncoder c_encoder = encoder.beginComputePass();
        c_encoder.setPipeline((bottom_up) ? state.identify_bottom_up->pipeline : state.identify->pipeline);
        c_encoder.setBindGroup(0, identify_groups[0], 0, nullptr);
        c_encoder.setBindGroup(1, identify_groups[1], 0, nullptr);
        c_encoder.setBindGroup(2, identify_groups[2 + iterations % 2], 0, nullptr);
        // identify uses 64 warps to find results for 2048 searches
        c_encoder.dispatchWorkgroups(WORKGROUPS, 92 * 8, 1);

        c_encoder.setPipeline((bottom_up) ? state.expand_bottom_up->pipeline : state.expand->pipeline);
        c_encoder.setBindGroup(0, expand_groups[0], 0, nullptr);
        c_encoder.setBindGroup(1, expand_groups[1], 0, nullptr);
        c_encoder.setBindGroup(2, expand_groups[2 + iterations % 2], 0, nullptr);
        // Use 128 * 64 threads to execute the expand step.
        c_encoder.dispatchWorkgroups(WORKGROUPS, 92 * 8, 1);
        c_encoder.end();
        c_encoder.release();
      }
      encoder.copyBufferToBuffer(search_info, sizeof(uint32_t) * 2, jfq_length_staging, 0, sizeof(uint32_t));
      state.queue.submit(encoder.finish());
      encoder.release();
      auto output = getMappedResult(state, jfq_length_staging, sizeof(uint32_t));
      jfq_length = output[0];
      bottom_up = ((double)jfq_length / csr.v_length) > 0.75 && !bottom_up;
    }

  }
  wgpu::CommandEncoder encoder = state.device.createCommandEncoder();
  encoder.copyBufferToBuffer(path_lengths, 0, path_lengths_staging, 0, request.length * sizeof(uint32_t));
  state.queue.submit(encoder.finish());
  encoder.release();

  auto output = getMappedResult(state, path_lengths_staging, request.length * sizeof(uint32_t));
  for (size_t j = 0; j < request.length; j++) {
    if (output[j] == 0 && request.dst[j] != request.src[j]) {
      continue;
    }
    results.push_back({
      .src = request.src[j],
      .dst = request.dst[j],
      .length = output[j],
    });
  }
  bsa.release();
  bsak.release();
  jfq.release();
  v_buffer.release();
  e_buffer.release();
  all_destinations.release();
  jfq_length_staging.release();
  path_lengths.release();
  path_lengths_staging.release();
  search_info.release();

  all_sources.release();

  for (size_t i = 0; i < 4; i++) {
    identify_groups[i].release();
    expand_groups[i].release();
  }
  set_bsak_groups[0].release();

  return results;
}
