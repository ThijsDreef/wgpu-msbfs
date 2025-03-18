#include "core/kernels/bindgroups/uber.hpp"
#include "core/util/wgpu-utils.hpp"
#include <cstdint>


UberGroup::UberGroup(wgpu::Device device) {
  this->device = device;

  wgpu::BindGroupLayoutEntry entries[] = {
    getComputeEntry(0, wgpu::BufferBindingType::ReadOnlyStorage, false, sizeof(uint32_t)),
    getComputeEntry(1, wgpu::BufferBindingType::ReadOnlyStorage, false, sizeof(uint32_t)),
    getComputeEntry(2, wgpu::BufferBindingType::Storage, false, sizeof(uint32_t)),
    getComputeEntry(3, wgpu::BufferBindingType::Storage, false, sizeof(uint32_t)),
    getComputeEntry(4, wgpu::BufferBindingType::ReadOnlyStorage, false, sizeof(uint32_t)),
    getComputeEntry(5, wgpu::BufferBindingType::Storage, false, sizeof(uint32_t)),
    getComputeEntry(6, wgpu::BufferBindingType::Storage, false, sizeof(uint32_t)),
  };

  wgpu::BindGroupLayoutDescriptor desc;
  desc.entries = entries;
  desc.entryCount = 7;
  layout = device.createBindGroupLayout(desc);
}

wgpu::BindGroup UberGroup::getBindGroup(wgpu::Buffer v, wgpu::Buffer e,
                             wgpu::Buffer path_lengths, wgpu::Buffer jfq,
                             wgpu::Buffer destinations, wgpu::Buffer bsa,
                             wgpu::Buffer bsak, size_t v_size, size_t e_size,
                             size_t invocations) {
  wgpu::BindGroupDescriptor desc;

  wgpu::BindGroupEntry entries[] = {
    getBindGroupBufferEntry(v, 0, 0, v_size),
    getBindGroupBufferEntry(e, 1, 0, e_size),
    getBindGroupBufferEntry(path_lengths, 2, 0, sizeof(uint32_t) * 32 * invocations),
    getBindGroupBufferEntry(jfq, 3, 0, v_size * invocations),
    getBindGroupBufferEntry(destinations, 4, 0, sizeof(uint32_t) * 32 * invocations),
    getBindGroupBufferEntry(bsa, 5, 0, v_size * invocations),
    getBindGroupBufferEntry(bsak, 6, 0, v_size * invocations),
  };

  desc.layout = layout;
  desc.entries = entries;
  desc.entryCount = 7;

  return device.createBindGroup(desc);
}
