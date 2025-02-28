#include "msbfs.hpp"
#include "core/util/wgpu-utils.hpp"
#include <cstdint>
#include <map>


std::vector<uint32_t> iterative_length(WGPUState &state, std::vector<uint32_t> src,
                      std::vector<uint32_t> dst, std::vector<uint32_t> v,
                      std::vector<uint32_t> e) {
  assert(src.size() == dst.size());
  uint64_t v_size = v.size() * sizeof(uint32_t);
  uint64_t e_size = e.size() * sizeof(uint32_t);

  WGPUBufferUsageFlags flags = wgpu::BufferUsage::CopySrc |
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

#ifdef DEBUG
  flags = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  desc = getBufferDescriptor(v_size * 3, false, flags);
  wgpu::Buffer debug_staging = state.device.createBuffer(desc);

#endif
  flags = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  desc = getBufferDescriptor(sizeof(uint32_t) * 32, false, flags);
  wgpu::Buffer output_staging = state.device.createBuffer(desc);

  // Populate buffers
  state.queue.writeBuffer(v_buffer, 0, v.data(), v_size);
  state.queue.writeBuffer(e_buffer, 0, e.data(), e_size);

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

  std::vector<uint32_t> results;
  results.reserve(dst.size());

  for (size_t offset = 0; offset < src.size(); offset += 32) {
    uint32_t pairs_to_solve = src.size() - offset > 32 ? 32 : src.size() - offset;
    wgpu::CommandEncoder encoder = state.device.createCommandEncoder({});
    encoder.clearBuffer(bsa, 0, v_size);
    encoder.clearBuffer(bsak, 0, v_size);
    encoder.clearBuffer(destinations, 0, sizeof(uint32_t) * 32);
    encoder.clearBuffer(path_length, 0, sizeof(uint32_t) * 32);
    state.queue.submit(encoder.finish());

    // Current hack to write some src into start
    std::map<uint32_t, uint32_t> to_write;
    for (size_t j = offset; j < 32 && j < src.size(); j++) {
      to_write[src[j]] |= 1 << (j - offset);
    }

    for (auto x : to_write) {
      state.queue.writeBuffer(bsak, x.first * sizeof(uint32_t), &x.second, sizeof(uint32_t));
    }

    state.queue.writeBuffer(destinations, 0, dst.data() + offset, pairs_to_solve * sizeof(uint32_t));
    uint32_t mask_value = 0xffffffff;
    state.queue.writeBuffer(mask, 0, &mask_value, sizeof(uint32_t));

    uint32_t length = 0;
    uint32_t iterations = 0;
    do {
      state.queue.writeBuffer(iteration, 0, &iterations, sizeof(uint32_t));
      wgpu::CommandEncoder encoder = state.device.createCommandEncoder({});
      wgpu::ComputePassEncoder compute_pass = encoder.beginComputePass({});

      compute_pass.setPipeline(state.expand->pipeline);

      compute_pass.setBindGroup(0, expand_groups[0], 0, nullptr);
      compute_pass.setBindGroup(1, expand_groups[1], 0, nullptr);
      compute_pass.setBindGroup(2, expand_groups[2 + iterations % 2], 0, nullptr);
      uint32_t size = length / 64 == 0 ? 1 : length / 64;
      compute_pass.dispatchWorkgroups(size, 1, 1);

      compute_pass.setPipeline(state.identify->pipeline);
      compute_pass.setBindGroup(0, identify_groups[0], 0, nullptr);
      compute_pass.setBindGroup(1, identify_groups[1], 0, nullptr);
      compute_pass.setBindGroup(2, identify_groups[2 + iterations % 2], 0, nullptr);
      compute_pass.dispatchWorkgroups(1, 1, 1);
      compute_pass.end();
      compute_pass.release();

      encoder.copyBufferToBuffer(jfq_length, 0, jfq_length_staging, 0, sizeof(uint32_t));

      state.queue.submit(encoder.finish());
      length = getMappedResult(state, jfq_length_staging, sizeof(uint32_t))[0];

#ifdef DEBUG
      std::cout << "jfq length " << length;

      encoder = state.device.createCommandEncoder({});
      encoder.copyBufferToBuffer(jfq, 0, debug_staging, 0, v_size);
      encoder.copyBufferToBuffer(bsa, 0, debug_staging, v_size, v_size);
      encoder.copyBufferToBuffer(bsak, 0, debug_staging, v_size * 2, v_size);
      encoder.copyBufferToBuffer(path_length, 0, output_staging, 0, sizeof(uint32_t) * 32);
      state.queue.submit(encoder.finish());

      auto x = getMappedResult(state, debug_staging, v_size * 3);
      std::string titles[] = {
        "JFQ",
        "BSA",
        "BSAK",
      };
      size_t loops = 0;
      for (size_t i = 0; i < v.size() * 3; i++) {
        if (i % v.size() == 0) {
          std::cout << std::endl << titles[loops] << ": ";
          loops++;
        }
        std::cout << x[i] << ", ";
      }
      std::cout << std::endl << "Path lengths ";
      auto y = getMappedResult(state, output_staging, 32 * sizeof(uint32_t));
      for (size_t i = 0; i < 32; i++) {
        std:: cout << y[i] << ", ";
      }
      std::cout << std::endl << std::endl;
#endif
      iterations++;
    } while (length != 0);
    encoder = state.device.createCommandEncoder({});
    encoder.copyBufferToBuffer(path_length, 0, output_staging, 0, sizeof(uint32_t) * 32);
    state.queue.submit(encoder.finish());
    auto output = getMappedResult(state, output_staging, 32 * sizeof(uint32_t));
    for (size_t j = offset; j < 32 && j < src.size(); j++) {
      results.push_back(output[j - offset]);
    }
  }


  return results;
}
