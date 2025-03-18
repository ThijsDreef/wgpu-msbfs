#pragma once
#include "bindgroups/uber.hpp"
#include "webgpu/webgpu.hpp"

class UberKernel {
private:
  wgpu::Device device;
public:
  wgpu::ComputePipeline pipeline;
  UberKernel(wgpu::Device);
  UberKernel(wgpu::Device device, char *shader_data);
  UberGroup uber_group;
  ~UberKernel();
};
