#pragma once
#include "webgpu/webgpu.hpp"

class BSAGroup {
private:
  wgpu::Device device;
public:
  wgpu::BindGroupLayout layout;
public:
  explicit BSAGroup(wgpu::Device);
  wgpu::BindGroup getBindGroup(wgpu::Buffer bsa, wgpu::Buffer bsak, uint64_t length);
};
