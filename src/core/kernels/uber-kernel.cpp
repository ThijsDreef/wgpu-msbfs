#include "core/kernels/uber-kernel.hpp"
#include "core/util/shader-locator.hpp"
#include "core/util/wgpu-utils.hpp"

UberKernel::UberKernel(wgpu::Device device)
: UberKernel(device, LOCATE_SHADER("data/shaders/msbfs-uber.wgsl")) {}

UberKernel::UberKernel(wgpu::Device device, char *shader_data) : uber_group(device) {
  this->device = device;

  wgpu::ShaderModule shader;
  {
    wgpu::ShaderSourceWGSL wgsl_desc;
    wgpu::ShaderModuleDescriptor desc;

    assert(shader_data);

    wgsl_desc.code = getStringViewFromCString(shader_data);
    wgsl_desc.chain.sType = wgpu::SType::ShaderSourceWGSL;
    desc.nextInChain = &wgsl_desc.chain;

    shader = device.createShaderModule(desc);
  }

#ifdef USE_FILES
  delete[] shader_data;
#endif

  wgpu::BindGroupLayout bind_layouts[] = {
      uber_group.layout,
  };

  wgpu::PipelineLayout pipeline_layout;
  {
    wgpu::PipelineLayoutDescriptor desc;
    desc.bindGroupLayoutCount = 1;
    desc.bindGroupLayouts = reinterpret_cast<WGPUBindGroupLayout *>(bind_layouts);
    pipeline_layout = device.createPipelineLayout(desc);
  }

  {
    wgpu::ComputePipelineDescriptor desc;
    desc.compute.entryPoint = getStringViewFromCString("main");
    desc.compute.module = shader;
    desc.layout = pipeline_layout;
    pipeline = device.createComputePipeline(desc);
  }

  pipeline_layout.release();
  shader.release();
}


UberKernel::~UberKernel() {
  pipeline.release();
}
