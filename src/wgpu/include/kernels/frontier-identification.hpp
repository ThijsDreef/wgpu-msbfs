#pragma once
#include "bindgroups/iterative-length.hpp"
#include "webgpu/webgpu.hpp"
#include "kernels/bindgroups/jfq.hpp"
#include "kernels/bindgroups/bsa.hpp"

class FrontierIdentification {
private:
  // This instance is linked to a device
  wgpu::Device device;
public:
  wgpu::ComputePipeline pipeline;
  explicit FrontierIdentification(wgpu::Device device);
  explicit FrontierIdentification(wgpu::Device device, char* shader_data);
  JFQGroup jfq_group;
  IterativeLengthGroup length_group;
  BSAGroup bsa_group;
  ~FrontierIdentification();
};
