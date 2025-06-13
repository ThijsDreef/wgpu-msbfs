#pragma once
#include "kernels/frontier-expansion.hpp"
#include "kernels/frontier-identification.hpp"
#include "kernels/set-first-bsak.hpp"
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
  std::unique_ptr<FrontierExpansion> expand_bottom_up;
  std::unique_ptr<FrontierIdentification> identify_bottom_up;
  std::unique_ptr<SetFirstBSAK> set_bsak;
};
