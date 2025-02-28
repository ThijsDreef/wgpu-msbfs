#include "core/kernels/bindgroups/csr.hpp"
#include "core/util/wgpu-utils.hpp"

CSRGroup::CSRGroup(wgpu::Device device) {
  this->device = device;

  wgpu::BindGroupLayoutEntry entries[] = {
    getComputeEntry(0, wgpu::BufferBindingType::ReadOnlyStorage, false, sizeof(uint32_t)),
    getComputeEntry(1, wgpu::BufferBindingType::ReadOnlyStorage, false, sizeof(uint32_t)),
  };
  wgpu::BindGroupLayoutDescriptor desc;
  desc.entries = entries;
  desc.entryCount = 2;
  layout = device.createBindGroupLayout(desc);
}

wgpu::BindGroup CSRGroup::getBindGroup(wgpu::Buffer v, wgpu::Buffer e, uint64_t v_length, uint64_t e_length) {
  wgpu::BindGroupDescriptor desc;

  wgpu::BindGroupEntry entries[] = {
    getBindGroupBufferEntry(v, 0, 0, v_length),
    getBindGroupBufferEntry(e, 1, 0, e_length),
  };

  desc.layout = layout;
  desc.entries = entries;
  desc.entryCount = 2;

  return device.createBindGroup(desc);
}
