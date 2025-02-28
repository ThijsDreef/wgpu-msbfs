#pragma once
#include "webgpu/webgpu.hpp"

class CSRGroup {
private:
  wgpu::Device device;

public:
  wgpu::BindGroupLayout layout;

public:
  explicit CSRGroup(wgpu::Device);
  wgpu::BindGroup getBindGroup(wgpu::Buffer v, wgpu::Buffer e, uint64_t v_length, uint64_t e_length);
};
