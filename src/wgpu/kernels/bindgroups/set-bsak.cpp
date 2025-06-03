#include "kernels/bindgroups/set-bsak.hpp"
#include "util/wgpu-utils.hpp"

SetBSAKGroup::SetBSAKGroup(wgpu::Device device) {
  this->device = device;

  wgpu::BindGroupLayoutEntry entries[] = {
      getComputeEntry(0, wgpu::BufferBindingType::Storage, false,
                      sizeof(uint32_t)),
      getComputeEntry(1, wgpu::BufferBindingType::ReadOnlyStorage, false,
                      sizeof(uint32_t)),
      getComputeEntry(2, wgpu::BufferBindingType::ReadOnlyStorage, false,
                      sizeof(uint32_t)),
      getComputeEntry(3, wgpu::BufferBindingType::Storage, false,
                      sizeof(uint32_t)),
      getComputeEntry(4, wgpu::BufferBindingType::Storage, false,
                      sizeof(uint32_t)),
  };

  wgpu::BindGroupLayoutDescriptor desc;
  desc.entries = entries;
  desc.entryCount = 5;
  layout = device.createBindGroupLayout(desc);
}

wgpu::BindGroup SetBSAKGroup::getBindGroup(
  wgpu::Buffer offset,
  wgpu::Buffer src,
  wgpu::Buffer dest,
  wgpu::Buffer target_dst,
  wgpu::Buffer bsak,
  uint64_t v_length,
  uint64_t request_length) {

  wgpu::BindGroupEntry entries[] = {
    getBindGroupBufferEntry(offset, 0, 0, sizeof(uint32_t)),
    getBindGroupBufferEntry(src, 1, 0, sizeof(uint32_t) * request_length),
    getBindGroupBufferEntry(dest, 2, 0, sizeof(uint32_t) * request_length),
    getBindGroupBufferEntry(target_dst, 3, 0, sizeof(uint32_t) * 1024),
    getBindGroupBufferEntry(bsak, 4, 0, sizeof(uint32_t) * v_length * 32),
  };

  wgpu::BindGroupDescriptor desc;
  desc.entries = entries;
  desc.entryCount = 5;
  desc.layout = layout;

  return device.createBindGroup(desc);
}
