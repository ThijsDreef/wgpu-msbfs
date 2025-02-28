#include "core/kernels/bindgroups/iterative-length.hpp"
#include "core/util/wgpu-utils.hpp"

IterativeLengthGroup::IterativeLengthGroup(wgpu::Device device) {
  this->device = device;

  wgpu::BindGroupLayoutEntry entries[] = {
    getComputeEntry(0, wgpu::BufferBindingType::ReadOnlyStorage, false, sizeof(uint32_t) * 32),
    getComputeEntry(1, wgpu::BufferBindingType::Storage, false, sizeof(uint32_t) * 32),
    getComputeEntry(2, wgpu::BufferBindingType::Uniform, false, sizeof(uint32_t)),
    getComputeEntry(3, wgpu::BufferBindingType::Storage, false, sizeof(uint32_t)),
  };

  wgpu::BindGroupLayoutDescriptor desc;

  desc.entries = entries;
  desc.entryCount = 4;

  layout = device.createBindGroupLayout(desc);
}

wgpu::BindGroup IterativeLengthGroup::getBindGroup(wgpu::Buffer dst, wgpu::Buffer path_lengths, wgpu::Buffer iteration, wgpu::Buffer mask) {
  wgpu::BindGroupDescriptor desc;

  wgpu::BindGroupEntry entries[] = {
    getBindGroupBufferEntry(dst, 0, 0, sizeof(uint32_t) * 32),
    getBindGroupBufferEntry(path_lengths, 1, 0, sizeof(uint32_t) * 32),
    getBindGroupBufferEntry(iteration, 2, 0, sizeof(uint32_t)),
    getBindGroupBufferEntry(mask, 3, 0, sizeof(uint32_t)),
  };

  desc.layout = layout;
  desc.entryCount = 4;
  desc.entries = entries;

  return device.createBindGroup(desc);
}
