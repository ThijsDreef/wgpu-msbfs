#pragma once

#include "webgpu/webgpu.hpp"
#include "kernels/bindgroups/set-bsak.hpp"

class SetFirstBSAK {
private:
  wgpu::Device device;
  wgpu::BindGroupLayout bind_layouts[1];
public:
  wgpu::ComputePipeline pipeline;
  SetFirstBSAK(wgpu::Device device);
  SetFirstBSAK(wgpu::Device device, char *shader_data);
  SetBSAKGroup set_bsak_group;
  ~SetFirstBSAK();
};
