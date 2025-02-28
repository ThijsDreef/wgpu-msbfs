#include "core/kernels/frontier-identification.hpp"
#include "core/util/shader-locator.hpp"
#include "core/util/wgpu-utils.hpp"
#include <cstdint>

FrontierIdentification::FrontierIdentification(wgpu::Device device) :
  FrontierIdentification(device, LOCATE_SHADER("data/shaders/frontier-identification.wgsl")) {
}

FrontierIdentification::FrontierIdentification(wgpu::Device device, char* shader_data)
: jfq_group(device, true), bsa_group(device), length_group(device) {
  this->device = device;

  wgpu::ShaderModule shader;
  {
    wgpu::ShaderModuleWGSLDescriptor wgsl_desc;
    wgpu::ShaderModuleDescriptor desc;

    assert(shader_data);

    wgsl_desc.code = shader_data;
    wgsl_desc.chain.sType = wgpu::SType::ShaderModuleWGSLDescriptor;
    desc.nextInChain = &wgsl_desc.chain;

    shader = device.createShaderModule(desc);
#ifdef USE_FILES
    delete[] shader_data;
#endif
  }

  wgpu::BindGroupLayout bind_layouts[] = {
    jfq_group.layout,
    length_group.layout,
    bsa_group.layout,
  };

  wgpu::PipelineLayout pipeline_layout;
  {
    wgpu::PipelineLayoutDescriptor desc;
    desc.bindGroupLayoutCount = 3;
    desc.bindGroupLayouts = reinterpret_cast<WGPUBindGroupLayout*>(bind_layouts);
    pipeline_layout = device.createPipelineLayout(desc);
  }

  {
    wgpu::ComputePipelineDescriptor desc;
    desc.compute.entryPoint = "main";
    desc.compute.module = shader;
    desc.layout = pipeline_layout;
    pipeline = device.createComputePipeline(desc);
  }
  shader.release();
  pipeline_layout.release();
}


FrontierIdentification::~FrontierIdentification() {
  pipeline.release();
}
