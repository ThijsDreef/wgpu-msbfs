#include "core/kernels/bindgroups/jfq.hpp"
#include "core/util/wgpu-utils.hpp"
#include <cstdint>

JFQGroup::JFQGroup(wgpu::Device device, bool write) {
  this->device = device;
  wgpu::BufferBindingType access =
      write ? wgpu::BufferBindingType::Storage
      : wgpu::BufferBindingType::ReadOnlyStorage;

  wgpu::BindGroupLayoutEntry entries[] = {
    getComputeEntry(0, access, false, sizeof(uint32_t)),
    getComputeEntry(1, access, false, sizeof(uint32_t) * 4),
  };

  wgpu::BindGroupLayoutDescriptor desc;
  desc.entries = entries;
  desc.label = getStringViewFromCString("jfq group");
  desc.entryCount = 2;
  layout = device.createBindGroupLayout(desc);
}


wgpu::BindGroup JFQGroup::getBindGroup(wgpu::Buffer jfq, wgpu::Buffer search_info, uint64_t length, uint32_t workgroups) {
  wgpu::BindGroupDescriptor desc;

  wgpu::BindGroupEntry entries[] = {
    getBindGroupBufferEntry(jfq, 0, 0, length),
    getBindGroupBufferEntry(search_info, 1, 0, sizeof(uint32_t) * 4 * workgroups),
  };

  desc.layout = layout;
  desc.entries = entries;
  desc.entryCount = 2;
  desc.label = getStringViewFromCString("bindgroup jfq");

  return device.createBindGroup(desc);
}
