#pragma once
#include "webgpu/webgpu.hpp"

class JFQGroup {
private:
  wgpu::Device device;
public:
  wgpu::BindGroupLayout layout;

public:
  explicit JFQGroup(wgpu::Device, bool write);
  wgpu::BindGroup getBindGroup(wgpu::Buffer jfq, wgpu::Buffer jfq_length, uint64_t length);
};
