#pragma once
#include "webgpu/webgpu.hpp"
#include "wgpu-state.hpp"
#include <vector>

class LengthKernel {
private:
  wgpu::ComputePipeline pipeline;
public:
  explicit LengthKernel(WGPUState wgpu_state);
  ~LengthKernel();
};
