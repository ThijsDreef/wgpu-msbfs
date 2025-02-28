#define WEBGPU_CPP_IMPLEMENTATION
#include "include/core/kernels/frontier-expansion.hpp"
#include "include/core/kernels/frontier-identification.hpp"
#include <memory>
#include "wgpu-state.hpp"


void error(WGPUErrorType type, const char* msg, void* user) {

}

WGPUState::WGPUState() {
  instance = wgpu::createInstance({});
  wgpu::RequestAdapterOptions options;
  wgpu::Adapter adapter = instance.requestAdapter({});

  device = adapter.requestDevice({});
  queue = device.getQueue();
  expand = std::unique_ptr<FrontierExpansion>(new FrontierExpansion(device));
  identify = std::unique_ptr<FrontierIdentification>(new FrontierIdentification(device));
}

WGPUState::~WGPUState() {
  queue.release();
  device.release();
  instance.release();
}
