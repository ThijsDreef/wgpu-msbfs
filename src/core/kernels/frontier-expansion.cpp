#include "core/kernels/frontier-expansion.hpp"
#include "core/util/wgpu-utils.hpp"
#include "core/util/shader-locator.hpp"

FrontierExpansion::FrontierExpansion(wgpu::Device device) :
  FrontierExpansion(device, LOCATE_SHADER("data/shaders/frontier-expansion.wgsl")) {
}

FrontierExpansion::FrontierExpansion(wgpu::Device device, char *shader_data)
: csr_group(device), jfq_group(device, false), bsa_group(device) {
  this->device = device;

  wgpu::ShaderModule shader;
  {
    wgpu::ShaderSourceWGSL wgsl_desc;
    wgpu::ShaderModuleDescriptor desc;
    assert(shader_data);

    wgsl_desc.code = getStringViewFromCString(shader_data);

    wgsl_desc.chain.sType = wgpu::SType::ShaderSourceWGSL;
    desc.nextInChain = &wgsl_desc.chain;
    desc.label = getStringViewFromCString("expand");

    shader = device.createShaderModule(desc);

#ifdef USE_FILES
    delete[] shader_data;
#endif
  }

  wgpu::BindGroupLayout bind_layouts[] = {
    csr_group.layout,
    jfq_group.layout,
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
    desc.compute.entryPoint = getStringViewFromCString("main");
    desc.compute.module = shader;
    desc.layout = pipeline_layout;
    pipeline = device.createComputePipeline(desc);
  }

  shader.release();
  pipeline_layout.release();
}

FrontierExpansion::~FrontierExpansion() {
  pipeline.release();
}
