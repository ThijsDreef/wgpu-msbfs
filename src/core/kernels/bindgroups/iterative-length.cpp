#include "core/kernels/bindgroups/iterative-length.hpp"
#include "core/util/wgpu-utils.hpp"

IterativeLengthGroup::IterativeLengthGroup(wgpu::Device device) {
  this->device = device;

  wgpu::BindGroupLayoutEntry entries[] = {
    getComputeEntry(0, wgpu::BufferBindingType::ReadOnlyStorage, false, sizeof(uint32_t) * 64 * 32),
    getComputeEntry(1, wgpu::BufferBindingType::Storage, false, sizeof(uint32_t) * 64 * 32),
  };

  wgpu::BindGroupLayoutDescriptor desc;

  desc.entries = entries;
  desc.entryCount = 2;

  layout = device.createBindGroupLayout(desc);
}

wgpu::BindGroup IterativeLengthGroup::getBindGroup(wgpu::Buffer dst, wgpu::Buffer path_lengths) {
  wgpu::BindGroupDescriptor desc;

  wgpu::BindGroupEntry entries[] = {
    getBindGroupBufferEntry(dst, 0, 0, sizeof(uint32_t) * 64 * 32),
    getBindGroupBufferEntry(path_lengths, 1, 0, sizeof(uint32_t) * 64 * 32),
  };

  desc.layout = layout;
  desc.entryCount = 2;
  desc.entries = entries;

  return device.createBindGroup(desc);
}
