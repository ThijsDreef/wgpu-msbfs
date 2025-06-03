#include "kernels/set-first-bsak.hpp"
#include "util/shader-locator.hpp"
#include "util/wgpu-utils.hpp"

SetFirstBSAK::SetFirstBSAK(wgpu::Device device)
: SetFirstBSAK(device, LOCATE_SHADER("data/shaders/set-first-bsak.wgsl")) {}

SetFirstBSAK::SetFirstBSAK(wgpu::Device device, char *shader_data)
: set_bsak_group(device) {
  this->device = device;

  wgpu::ShaderModule shader;
  {
        wgpu::ShaderSourceWGSL wgsl_desc;
    wgpu::ShaderModuleDescriptor desc;
    assert(shader_data);

    wgsl_desc.code = getStringViewFromCString(shader_data);

    wgsl_desc.chain.sType = wgpu::SType::ShaderSourceWGSL;
    desc.nextInChain = &wgsl_desc.chain;
    desc.label = getStringViewFromCString("Set first BSAK");

    shader = device.createShaderModule(desc);

#ifdef USE_FILES
    delete[] shader_data;
#endif
  }

  wgpu::BindGroupLayout bind_layouts[] = {
      set_bsak_group.layout,
  };

  wgpu::PipelineLayout pipeline_layout;
  {
    wgpu::PipelineLayoutDescriptor desc;
    desc.bindGroupLayoutCount = 1;
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

SetFirstBSAK::~SetFirstBSAK() {
  pipeline.release();
}
