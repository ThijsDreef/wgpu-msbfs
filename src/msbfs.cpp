#include "msbfs.hpp"
#include "core/util/wgpu-utils.hpp"
#include <cstdint>
#include <map>

struct SearchInfo {
  uint32_t iteration;
  uint32_t jfq_length;
  uint32_t last_jfq;
  uint32_t mask[32];
};

std::vector<IterativeLengthResult> iterative_length(WGPUState &state, PathFindingRequest request, CSR csr) {
  TimingInfo timing_info;
  return iterative_length(state, request, csr, timing_info);
}


std::vector<IterativeLengthResult> iterative_length(WGPUState &state, PathFindingRequest request, CSR csr, TimingInfo& timing_info) {
  uint64_t v_size = csr.v_length * sizeof(uint32_t);
  uint64_t e_size = csr.e_length * sizeof(uint32_t);

  const size_t WORKGROUPS = 1;
  const size_t SEARCHES_PER_WORKGROUP = 32;
  const size_t SEARCHES_PER_THREAD = 32;
  const size_t SEARCHES_IN_WORKGROUP = SEARCHES_PER_THREAD * SEARCHES_PER_WORKGROUP;
  const size_t PAIRS_IN_PARALLEL = WORKGROUPS * SEARCHES_IN_WORKGROUP;

  int flags = wgpu::BufferUsage::Storage |
              wgpu::BufferUsage::CopyDst;
  wgpu::BufferDescriptor desc = getBufferDescriptor(v_size, false, flags);
  wgpu::Buffer v_buffer = state.device.createBuffer(desc);

  desc = getBufferDescriptor(e_size, false, flags);
  wgpu::Buffer e_buffer = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(v_size * WORKGROUPS * SEARCHES_PER_THREAD, false, flags);
  wgpu::Buffer bsa = state.device.createBuffer(desc);
  wgpu::Buffer bsak = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage;
  desc = getBufferDescriptor(v_size * WORKGROUPS, false, flags);
  wgpu::Buffer jfq = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(SearchInfo) * WORKGROUPS, false, flags);
  wgpu::Buffer search_info = state.device.createBuffer(desc);


  flags = wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t) * PAIRS_IN_PARALLEL, false, flags);
  wgpu::Buffer destinations = state.device.createBuffer(desc);
  flags = wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopyDst |
          wgpu::BufferUsage::CopySrc;
  desc = getBufferDescriptor(sizeof(uint32_t) * PAIRS_IN_PARALLEL, false, flags);
  wgpu::Buffer path_lengths = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::CopyDst |
          wgpu::BufferUsage::MapRead;
  desc = getBufferDescriptor(sizeof(uint32_t) * PAIRS_IN_PARALLEL, false, flags);
  wgpu::Buffer path_lengths_staging = state.device.createBuffer(desc);

  // Populate buffers
  state.queue.writeBuffer(v_buffer, 0, csr.v, v_size);
  state.queue.writeBuffer(e_buffer, 0, csr.e, e_size);

  wgpu::BindGroup expand_groups[] = {
    state.expand->csr_group.getBindGroup(v_buffer, e_buffer, v_size, e_size),
    state.expand->jfq_group.getBindGroup(jfq, search_info, v_size * WORKGROUPS, WORKGROUPS),
    state.expand->bsa_group.getBindGroup(bsak, bsa, v_size * WORKGROUPS * SEARCHES_PER_THREAD),
    state.expand->bsa_group.getBindGroup(bsa, bsak, v_size * WORKGROUPS * SEARCHES_PER_THREAD),
  };

  wgpu::BindGroup identify_groups[] = {
    state.identify->jfq_group.getBindGroup(jfq, search_info, v_size * WORKGROUPS, WORKGROUPS),
    state.identify->length_group.getBindGroup(destinations, path_lengths, PAIRS_IN_PARALLEL),
    state.identify->bsa_group.getBindGroup(bsak, bsa, v_size * WORKGROUPS * SEARCHES_PER_THREAD),
    state.identify->bsa_group.getBindGroup(bsa, bsak, v_size * WORKGROUPS * SEARCHES_PER_THREAD),
  };

  std::vector<IterativeLengthResult> results;
  results.reserve(request.length);

  for (size_t offset = 0; offset < request.length; offset += PAIRS_IN_PARALLEL) {
    size_t pairs_to_solve = request.length - offset > PAIRS_IN_PARALLEL ? PAIRS_IN_PARALLEL : request.length - offset;
    wgpu::CommandEncoder encoder = state.device.createCommandEncoder();
    encoder.clearBuffer(bsa, 0, v_size * WORKGROUPS * SEARCHES_PER_WORKGROUP);
    encoder.clearBuffer(bsak, 0, v_size * WORKGROUPS * SEARCHES_PER_WORKGROUP);
    encoder.clearBuffer(destinations, 0, sizeof(uint32_t) * PAIRS_IN_PARALLEL);
    encoder.clearBuffer(path_lengths, 0, sizeof(uint32_t) * PAIRS_IN_PARALLEL);
    encoder.clearBuffer(search_info, 0, sizeof(SearchInfo) * WORKGROUPS);
    state.queue.submit(encoder.finish());
    encoder.release();
    state.queue.writeBuffer(destinations, 0, request.dst + offset, pairs_to_solve * sizeof(uint32_t));
    size_t pair = offset;
    std::map<uint32_t, uint32_t> to_write;
    for (size_t j = 0; j < pairs_to_solve / 32 + 1; j++) {
      for (size_t x = 0; x < 32 && pair - offset < pairs_to_solve; x++) {
        to_write[request.src[pair] * 32 + j] |= 1 << x;
        pair++;
      }
    }

    for (auto x : to_write) {
      state.queue.writeBuffer(bsak, x.first * sizeof(uint32_t), &x.second, sizeof(uint32_t));
    }

    encoder = state.device.createCommandEncoder();

    for (size_t iterations = 0; iterations < 25; iterations++) {
      // current hack to syncly set jfq_length back to zero
      for (size_t w = 0; w < WORKGROUPS; w++) {
        encoder.clearBuffer(search_info, (1 + 4 * w) * sizeof(uint32_t), sizeof(uint32_t));
      }
      wgpu::ComputePassEncoder c_encoder = encoder.beginComputePass();
      c_encoder.setPipeline(state.identify->pipeline);
      c_encoder.setBindGroup(0, identify_groups[0], 0, nullptr);
      c_encoder.setBindGroup(1, identify_groups[1], 0, nullptr);
      c_encoder.setBindGroup(2, identify_groups[2 + iterations % 2], 0, nullptr);
      // identify uses 64 warps to find results for 2048 searches
      c_encoder.dispatchWorkgroups(WORKGROUPS, 92 * 8, 1);

      c_encoder.setPipeline(state.expand->pipeline);
      c_encoder.setBindGroup(0, expand_groups[0], 0, nullptr);
      c_encoder.setBindGroup(1, expand_groups[1], 0, nullptr);
      c_encoder.setBindGroup(2, expand_groups[2 + iterations % 2], 0, nullptr);
      // Use 128 * 64 threads to execute the expand step.
      c_encoder.dispatchWorkgroups(WORKGROUPS, 92 * 8, 1);
      c_encoder.end();
      c_encoder.release();
    }

    encoder.copyBufferToBuffer(path_lengths, 0, path_lengths_staging, 0, PAIRS_IN_PARALLEL * sizeof(uint32_t));
    state.queue.submit(encoder.finish());
    encoder.release();

    auto output = getMappedResult(state, path_lengths_staging, PAIRS_IN_PARALLEL * sizeof(uint32_t));
    for (size_t j = 0; j < pairs_to_solve; j++) {
      if (output[j] == 0 && request.dst[j + offset] != request.src[j + offset]) {
        continue;
      }
      results.push_back({
          .src = request.src[j + offset],
          .dst = request.dst[j + offset],
          .length = output[j],
      });
    }
  }

  bsa.release();
  bsak.release();
  jfq.release();
  v_buffer.release();
  e_buffer.release();
  destinations.release();
  path_lengths.release();
  path_lengths_staging.release();
  search_info.release();
  for (size_t i = 0; i < 4; i++) {
    identify_groups[i].release();
    expand_groups[i].release();
  }
  /**

  for (size_t offset = 0; offset < request.length; offset += 32) {
    uint32_t pairs_to_solve = request.length - offset > 32 ? 32 : request.length - offset;
    wgpu::CommandEncoder encoder = state.device.createCommandEncoder({});
    encoder.clearBuffer(bsa, 0, v_size);
    encoder.clearBuffer(bsak, 0, v_size);
    encoder.clearBuffer(destinations, 0, sizeof(uint32_t) * 32);
    encoder.clearBuffer(path_length, 0, sizeof(uint32_t) * 32);
    state.queue.submit(encoder.finish());
    encoder.release();

    // Current hack to write some src into start
    std::map<uint32_t, uint32_t> to_write;
    for (size_t j = offset; j < offset + pairs_to_solve; j++) {
      to_write[request.src[j]] |= 1 << (j - offset);
    }

    for (auto x : to_write) {
      state.queue.writeBuffer(bsak, x.first * sizeof(uint32_t), &x.second, sizeof(uint32_t));
    }

    state.queue.writeBuffer(destinations, 0, request.dst + offset, pairs_to_solve * sizeof(uint32_t));
    uint32_t mask_value = 0xffffffff;
    state.queue.writeBuffer(mask, 0, &mask_value, sizeof(uint32_t));

    uint32_t length = 0;
    uint32_t iterations = 0;

    wgpu::ComputePassDescriptor compute_desc;

    do {
      state.queue.writeBuffer(iteration, 0, &iterations, sizeof(uint32_t));
      wgpu::CommandEncoder encoder = state.device.createCommandEncoder({});


      wgpu::ComputePassEncoder compute_pass = encoder.beginComputePass(compute_desc);
      compute_pass.setPipeline(state.expand->pipeline);
      compute_pass.setBindGroup(0, expand_groups[0], 0, nullptr);
      compute_pass.setBindGroup(1, expand_groups[1], 0, nullptr);
      compute_pass.setBindGroup(2, expand_groups[2 + iterations % 2], 0,
                                nullptr);
      uint32_t size = length / 64 + 1;
      compute_pass.dispatchWorkgroups(size > 1024 ? 1024 : size, 1, 1);
      compute_pass.end();
      compute_pass.release();

      compute_pass = encoder.beginComputePass(compute_desc);
      compute_pass.setPipeline(state.identify->pipeline);
      compute_pass.setBindGroup(0, identify_groups[0], 0, nullptr);
      compute_pass.setBindGroup(1, identify_groups[1], 0, nullptr);
      compute_pass.setBindGroup(2, identify_groups[2 + iterations % 2], 0, nullptr);
      compute_pass.dispatchWorkgroups(1, 1, 1);

      compute_pass.end();
      compute_pass.release();
      encoder.copyBufferToBuffer(jfq_length, 0, jfq_length_staging, 0,
                                 sizeof(uint32_t));

      state.queue.submit(encoder.finish());
      encoder.release();

      length = getMappedResult(state, jfq_length_staging, sizeof(uint32_t))[0];
      iterations++;
    } while (length != 0);

    encoder = state.device.createCommandEncoder({});
    encoder.copyBufferToBuffer(path_length, 0, output_staging, 0, sizeof(uint32_t) * 32);
    state.queue.submit(encoder.finish());
    auto output = getMappedResult(state, output_staging, 32 * sizeof(uint32_t));
    for (size_t j = 0; j < pairs_to_solve; j++) {
      if (output[j] == 0 && request.dst[j + offset] != request.src[j + offset]) {
        continue;
      }
      results.push_back({
          .src = request.src[j + offset],
          .dst = request.dst[j + offset],
          .length = output[j],
      });
    }
  }

  // Cleanup
  bsa.release();
  bsak.release();
  jfq.release();
  v_buffer.release();
  e_buffer.release();
  jfq_length.release();
  jfq_length_staging.release();
  destinations.release();
  path_length.release();
  mask.release();
  iteration.release();
  output_staging.release();
  for (size_t i = 0; i < 4; i++) {
    identify_groups[i].release();
    expand_groups[i].release();
  }
  **/
  return results;
}
