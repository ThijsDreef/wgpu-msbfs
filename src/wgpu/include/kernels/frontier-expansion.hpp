#pragma once
#include "webgpu/webgpu.hpp"
#include "kernels/bindgroups/jfq.hpp"
#include "kernels/bindgroups/csr.hpp"
#include "kernels/bindgroups/bsa.hpp"

class FrontierExpansion {
private:
  wgpu::Device device;
  wgpu::BindGroupLayout bind_layouts[4];
public:
  wgpu::ComputePipeline pipeline;
  FrontierExpansion(wgpu::Device device);
  FrontierExpansion(wgpu::Device device, char *shader_data);
  JFQGroup jfq_group;
  CSRGroup csr_group;
  BSAGroup bsa_group;
  ~FrontierExpansion();
};
