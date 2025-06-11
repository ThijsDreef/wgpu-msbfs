#pragma once
#include "webgpu/webgpu.hpp"

class SetBSAKGroup {
private:
  wgpu::Device device;
public:
  wgpu::BindGroupLayout layout;
public:
  explicit SetBSAKGroup(wgpu::Device);
  wgpu::BindGroup getBindGroup(
    wgpu::Buffer offset,
    wgpu::Buffer src,
    wgpu::Buffer bsak,
    uint64_t v_length,
    uint64_t request_length,
    uint32_t workgroups
  );
};
