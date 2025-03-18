#include "msbfs.hpp"
#include "core/util/wgpu-utils.hpp"
#include <cstdint>
#include <map>
#include <vector>

std::vector<IterativeLengthResult> iterative_length(WGPUState &state, PathFindingRequest request, CSR csr) {
  TimingInfo timing_info;
  return iterative_length(state, request, csr, timing_info);
}


std::vector<IterativeLengthResult>
iterative_length_uber(WGPUState &state, PathFindingRequest request, CSR csr) {
  uint64_t v_size = csr.v_length * sizeof(uint32_t);
  uint64_t e_size = csr.e_length * sizeof(uint32_t);

  const size_t MAX_SUBMISSIONS = (134217728 / v_size) * 32;
  // std::cout << "max submissions: " << MAX_SUBMISSIONS << std::endl;
  uint64_t required_submissions = (request.length % MAX_SUBMISSIONS == 0) ? request.length / MAX_SUBMISSIONS : request.length / MAX_SUBMISSIONS + 1;
  uint64_t pairs_per_submission = required_submissions > 1 ? MAX_SUBMISSIONS : request.length;
  if (pairs_per_submission % 32 != 0) {
    pairs_per_submission += 32 - pairs_per_submission % 32;
  }
  // std::cout << "pairs per submission: " << pairs_per_submission << std::endl;
  // std::cout << "required submissions: " << required_submissions << std::endl;




  int flags;
  wgpu::BufferDescriptor desc;

  flags = wgpu::BufferUsage::CopyDst |
          wgpu::BufferUsage::Storage;
  desc = getBufferDescriptor(v_size, false, flags);
  wgpu::Buffer v_buffer = state.device.createBuffer(desc);
  desc = getBufferDescriptor(e_size, false, flags);
  wgpu::Buffer e_buffer = state.device.createBuffer(desc);

  desc = getBufferDescriptor(sizeof(uint32_t) * pairs_per_submission, false, flags);
  wgpu::Buffer destinations = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Storage;
  desc = getBufferDescriptor(sizeof(uint32_t) * pairs_per_submission, false, flags);
  wgpu::Buffer path_lengths = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::Storage;
  desc = getBufferDescriptor(v_size * (pairs_per_submission / 32), false, flags);
  wgpu::Buffer jfq_lengths = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::CopyDst |
          wgpu::BufferUsage::Storage;
  desc = getBufferDescriptor(v_size * (pairs_per_submission / 32), false, flags);
  wgpu::Buffer bsa = state.device.createBuffer(desc);
  wgpu::Buffer bsak = state.device.createBuffer(desc);


  flags = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  desc = getBufferDescriptor(sizeof(uint32_t) * pairs_per_submission, false, flags);
  wgpu::Buffer path_staging = state.device.createBuffer(desc);

  // populate static buffers
  state.queue.writeBuffer(v_buffer, 0, csr.v, v_size);
  state.queue.writeBuffer(e_buffer, 0, csr.e, e_size);

  // prepare results
  std::vector<IterativeLengthResult> results;
  results.reserve(request.length);

  wgpu::BindGroup bindgroup = state.uber->uber_group.getBindGroup(v_buffer, e_buffer, path_lengths, jfq_lengths, destinations, bsa, bsak, v_size, e_size, pairs_per_submission / 32);

  for (size_t x = 0; x < required_submissions; x++) {
    size_t remaining_pairs = (request.length - pairs_per_submission * x);
    size_t to_submit = remaining_pairs > MAX_SUBMISSIONS ? MAX_SUBMISSIONS : remaining_pairs;

    size_t offset = x * MAX_SUBMISSIONS;
    size_t to_dispatch = (to_submit % 32 == 0) ? to_submit / 32 : to_submit / 32 + 1;

    wgpu::CommandEncoder clear_encoder = state.device.createCommandEncoder({});
    clear_encoder.clearBuffer(bsa, 0, v_size * (pairs_per_submission / 32));
    clear_encoder.clearBuffer(bsak, 0, v_size * (pairs_per_submission / 32));
    clear_encoder.clearBuffer(path_lengths, 0, to_dispatch * 32 * sizeof(uint32_t));
    state.queue.submit(clear_encoder.finish());

    std::map<uint32_t, uint32_t> to_write;
    size_t pair = offset;
    for (size_t y = 0; y < to_dispatch; y++) {
      for (size_t z = 0; z < 32 && pair < request.length; z++) {
        to_write[request.src[pair] + csr.v_length * y] |= 1 << z;
        pair++;
      }
    }

    for (auto x : to_write) {
      state.queue.writeBuffer(bsak, x.first * sizeof(uint32_t), &x.second, sizeof(uint32_t));
    }

    state.queue.writeBuffer(destinations, 0, request.dst + x * MAX_SUBMISSIONS, to_dispatch * 32 * sizeof(uint32_t));

    wgpu::CommandEncoder encoder = state.device.createCommandEncoder({});
    wgpu::ComputePassEncoder c_encoder = encoder.beginComputePass({});




    c_encoder.setPipeline(state.uber->pipeline);
    c_encoder.setBindGroup(0, bindgroup, 0, nullptr);
    c_encoder.dispatchWorkgroups(to_dispatch, 1, 1);
    c_encoder.end();
    c_encoder.release();

    encoder.copyBufferToBuffer(path_lengths, 0, path_staging, 0, to_dispatch * 32 * sizeof(uint32_t));

    state.queue.submit(encoder.finish());
    encoder.release();

    auto output = getMappedResult(state, path_staging, to_submit * sizeof(uint32_t));
    for (size_t j = 0; j < to_submit; j++) {
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
  jfq_lengths.release();
  bsak.release();
  destinations.release();
  v_buffer.release();
  e_buffer.release();
  path_lengths.release();
  path_staging.release();

  bindgroup.release();

  return results;
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

  return results;
}
