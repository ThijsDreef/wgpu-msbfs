#include "core/kernels/bindgroups/bsa.hpp"
#include "core/util/wgpu-utils.hpp"
#include <cstdint>

BSAGroup::BSAGroup(wgpu::Device device) {
  this->device = device;

  wgpu::BindGroupLayoutEntry entries[] = {
    getComputeEntry(0, wgpu::BufferBindingType::ReadOnlyStorage, false, sizeof(uint32_t)),
    getComputeEntry(1, wgpu::BufferBindingType::Storage, false, sizeof(uint32_t)),
  };
  wgpu::BindGroupLayoutDescriptor desc;
  desc.entries = entries;
  desc.entryCount = 2;
  layout = device.createBindGroupLayout(desc);
}

wgpu::BindGroup BSAGroup::getBindGroup(wgpu::Buffer bsa, wgpu::Buffer bsak, uint64_t length) {
  wgpu::BindGroupDescriptor desc;

  wgpu::BindGroupEntry entries[] = {
    getBindGroupBufferEntry(bsa, 0, 0, length),
    getBindGroupBufferEntry(bsak, 1, 0, length),
  };

  desc.layout = layout;
  desc.entries = entries;
  desc.entryCount = 2;

  return device.createBindGroup(desc);
}
