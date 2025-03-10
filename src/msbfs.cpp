#include "msbfs.hpp"
#include "core/util/wgpu-utils.hpp"
#include <cstdint>
#include <map>

std::vector<IterativeLengthResult> iterative_length(WGPUState &state, PathFindingRequest request, CSR csr) {
  TimingInfo timing_info;
  return iterative_length(state, request, csr, timing_info);
}


std::vector<IterativeLengthResult> iterative_length(WGPUState &state, PathFindingRequest request, CSR csr, TimingInfo& timing_info) {
  uint64_t v_size = csr.v_length * sizeof(uint32_t);
  uint64_t e_size = csr.e_length * sizeof(uint32_t);

  int flags = wgpu::BufferUsage::CopySrc |
                               wgpu::BufferUsage::Storage |
                               wgpu::BufferUsage::CopyDst;
  wgpu::BufferDescriptor desc = getBufferDescriptor(v_size, false, flags);
  wgpu::Buffer bsa = state.device.createBuffer(desc);
  wgpu::Buffer bsak = state.device.createBuffer(desc);
  wgpu::Buffer jfq = state.device.createBuffer(desc);
  wgpu::Buffer v_buffer = state.device.createBuffer(desc);

  desc = getBufferDescriptor(e_size, false, flags);
  wgpu::Buffer e_buffer = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopySrc;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  wgpu::Buffer jfq_length = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::MapRead |
          wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  wgpu::Buffer jfq_length_staging = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t) * 32, false, flags);
  wgpu::Buffer destinations = state.device.createBuffer(desc);
  wgpu::Buffer path_length = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  wgpu::Buffer mask = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  wgpu::Buffer iteration = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  desc = getBufferDescriptor(sizeof(uint32_t) * 32, false, flags);
  wgpu::Buffer output_staging = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::QueryResolve;
  desc = getBufferDescriptor(sizeof(uint64_t) * 4, false, flags);
  wgpu::Buffer timing_buffer = state.device.createBuffer(desc);
  flags = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  desc = getBufferDescriptor(sizeof(uint64_t) * 4, false, flags);
  wgpu::Buffer timing_staging_buffer = state.device.createBuffer(desc);


  // Timing specifics
  wgpu::QuerySetDescriptor q_desc;
  q_desc.type = wgpu::QueryType::Timestamp;
  q_desc.count = 4;
  wgpu::QuerySet timing_query_set = state.device.createQuerySet(q_desc);

  // Populate buffers
  state.queue.writeBuffer(v_buffer, 0, csr.v, v_size);
  state.queue.writeBuffer(e_buffer, 0, csr.e, e_size);

  wgpu::BindGroup expand_groups[] = {
    state.expand->jfq_group.getBindGroup(jfq, jfq_length, v_size),
    state.expand->csr_group.getBindGroup(v_buffer, e_buffer, v_size, e_size),
    state.expand->bsa_group.getBindGroup(bsa, bsak, v_size),
    state.expand->bsa_group.getBindGroup(bsak, bsa, v_size),
  };

  wgpu::BindGroup identify_groups[] = {
    state.identify->jfq_group.getBindGroup(jfq, jfq_length, v_size),
    state.identify->length_group.getBindGroup(destinations, path_length, iteration, mask),
    state.identify->bsa_group.getBindGroup(bsa, bsak, v_size),
    state.identify->bsa_group.getBindGroup(bsak, bsa, v_size),
  };

  std::vector<IterativeLengthResult> results;
  results.reserve(request.length);

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
    wgpu::ComputePassTimestampWrites cp_timestamp_writes;
    cp_timestamp_writes.beginningOfPassWriteIndex = 0;
    cp_timestamp_writes.endOfPassWriteIndex = 1;
    cp_timestamp_writes.querySet = timing_query_set;
    compute_desc.timestampWrites = &cp_timestamp_writes;

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

      cp_timestamp_writes.beginningOfPassWriteIndex = 2;
      cp_timestamp_writes.endOfPassWriteIndex = 3;
      compute_pass = encoder.beginComputePass(compute_desc);
      compute_pass.setPipeline(state.identify->pipeline);
      compute_pass.setBindGroup(0, identify_groups[0], 0, nullptr);
      compute_pass.setBindGroup(1, identify_groups[1], 0, nullptr);
      compute_pass.setBindGroup(2, identify_groups[2 + iterations % 2], 0, nullptr);
      compute_pass.dispatchWorkgroups(1, 1, 1);

      compute_pass.end();
      compute_pass.release();
      encoder.resolveQuerySet(timing_query_set, 0, 4, timing_buffer, 0);
      encoder.copyBufferToBuffer(timing_buffer, 0, timing_staging_buffer, 0,
                                 sizeof(uint64_t) * 4);
      encoder.copyBufferToBuffer(jfq_length, 0, jfq_length_staging, 0,
                                 sizeof(uint32_t));

      state.queue.submit(encoder.finish());
      encoder.release();

      length = getMappedResult(state, jfq_length_staging, sizeof(uint32_t))[0];
      std::unique_ptr<uint32_t[]> x = getMappedResult(state, timing_staging_buffer, sizeof(uint64_t) * 4);
      auto y = (uint64_t *)x.get();
      timing_info.identify_ns += y[1] - y[0];
      timing_info.expand_ns += y[3] - y[2];
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

  return results;
}
