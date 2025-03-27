#pragma once
#include "webgpu/webgpu.hpp"

class IterativeLengthGroup {
private:
  wgpu::Device device;
public:
  wgpu::BindGroupLayout layout;
public:
  explicit IterativeLengthGroup(wgpu::Device);
  wgpu::BindGroup getBindGroup(wgpu::Buffer dst, wgpu::Buffer path_lengths, size_t scale_factor);
};
