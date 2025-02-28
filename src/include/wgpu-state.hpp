#pragma once
#include "core/kernels/frontier-expansion.hpp"
#include "core/kernels/frontier-identification.hpp"
#include "webgpu/webgpu.hpp"

class WGPUState {
public:
  explicit WGPUState();
  ~WGPUState();
public:
  wgpu::Instance instance;
  wgpu::Device device;
  wgpu::Queue queue;
  std::unique_ptr<FrontierExpansion> expand;
  std::unique_ptr<FrontierIdentification> identify;
};
