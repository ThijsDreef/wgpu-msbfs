#include "kernels/bindgroups/set-bsak.hpp"
#include "util/wgpu-utils.hpp"
#include "util/search-info.hpp"

SetBSAKGroup::SetBSAKGroup(wgpu::Device device) {
  this->device = device;

  wgpu::BindGroupLayoutEntry entries[] = {
      getComputeEntry(0, wgpu::BufferBindingType::ReadOnlyStorage, false,
                      sizeof(SearchInfo)),
      getComputeEntry(1, wgpu::BufferBindingType::ReadOnlyStorage, false,
                      sizeof(uint32_t)),
      getComputeEntry(2, wgpu::BufferBindingType::Storage, false,
                      sizeof(uint32_t)),
  };

  wgpu::BindGroupLayoutDescriptor desc;
  desc.entries = entries;
  desc.entryCount = 3;
  layout = device.createBindGroupLayout(desc);
}

wgpu::BindGroup SetBSAKGroup::getBindGroup(
  wgpu::Buffer info,
  wgpu::Buffer src,
  wgpu::Buffer bsak,
  uint64_t v_length,
  uint64_t request_length,
  uint32_t workgroups
) {

  wgpu::BindGroupEntry entries[] = {
    getBindGroupBufferEntry(info, 0, 0, sizeof(SearchInfo) * workgroups),
    getBindGroupBufferEntry(src, 1, 0, sizeof(uint32_t) * request_length),
    getBindGroupBufferEntry(bsak, 2, 0, sizeof(uint32_t) * v_length * workgroups),
  };

  wgpu::BindGroupDescriptor desc;
  desc.entries = entries;
  desc.entryCount = 3;
  desc.layout = layout;

  return device.createBindGroup(desc);
}
