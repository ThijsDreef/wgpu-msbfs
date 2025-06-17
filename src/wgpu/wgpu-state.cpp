#define WEBGPU_CPP_IMPLEMENTATION
#include "include/kernels/set-first-bsak.hpp"
#include "kernels/frontier-expansion.hpp"
#include "kernels/frontier-identification.hpp"
#include <memory>
#include "wgpu/wgpu-state.hpp"


void error(WGPUDeviceImpl *const * x, WGPUErrorType type, WGPUStringView msg, void *user, void* s) {
  std::cout << "[WGPU Error] " << type << " ";
  if (msg.data)
    std::cout << msg.data;
  std::cout << std::endl;
}

WGPUState::WGPUState() {

  instance = wgpuCreateInstance({});

  // instance = wgpu::createInstance();
  wgpu::RequestAdapterOptions options;
  options.setDefault();
  options.powerPreference = wgpu::PowerPreference::HighPerformance;

  wgpu::Adapter adapter = instance.requestAdapter(options);

  // wgpu::FeatureName required_features[] = {
  //   wgpu::FeatureName::TimestampQuery,
  // };

  wgpu::DeviceDescriptor device_desc;
  device_desc.setDefault();
  // device_desc.uncapturedErrorCallbackInfo.callback = error;
  // device_desc.requiredFeatures =
  //   reinterpret_cast<WGPUFeatureName *>(required_features);
  // device_desc.requiredFeatureCount = 1;

  wgpu::Limits limits;
  limits.setDefault();
  // TODO: this should probably be dynamically set.
  // This should now reject any GPU not being able to handle SF300
  limits.maxBufferSize = 273255928;
  limits.maxStorageBufferBindingSize = 273255928;
  limits.maxComputeInvocationsPerWorkgroup = 256;
  device_desc.requiredLimits = &limits;


  device = adapter.requestDevice(device_desc);
  queue = device.getQueue();
  expand = std::unique_ptr<FrontierExpansion>(new FrontierExpansion(device));
  identify = std::unique_ptr<FrontierIdentification>(new FrontierIdentification(device));
  set_bsak = std::unique_ptr<SetFirstBSAK>(new SetFirstBSAK(device));
}

WGPUState::~WGPUState() {
  queue.release();
  device.release();
  instance.release();
}
