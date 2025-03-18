#pragma once
#include "webgpu/webgpu.hpp"

class UberGroup {
private:
  wgpu::Device device;
public:
  wgpu::BindGroupLayout layout;
public:
  explicit UberGroup(wgpu::Device);
  wgpu::BindGroup getBindGroup(wgpu::Buffer v, wgpu::Buffer e, wgpu::Buffer path_lengths, wgpu::Buffer jfq, wgpu::Buffer destinations, wgpu::Buffer bsa, wgpu::Buffer bsak, size_t v_size, size_t e_size, size_t invocations);
};
