#pragma once
#include "webgpu/webgpu.hpp"

class IterativeLengthGroup {
private:
  wgpu::Device device;
public:
  wgpu::BindGroupLayout layout;
public:
  explicit IterativeLengthGroup(wgpu::Device);
  wgpu::BindGroup getBindGroup(wgpu::Buffer dst, wgpu::Buffer path_lengths, uint64_t request_length);
};
