#define WEBGPU_CPP_IMPLEMENTATION
#include "include/core/kernels/frontier-expansion.hpp"
#include "include/core/kernels/frontier-identification.hpp"
#include <memory>
#include "wgpu-state.hpp"


void error(WGPUDeviceImpl *const * x, WGPUErrorType type, WGPUStringView msg, void *user, void* s) {
  std::cout << "[WGPU Error] " << type << " ";
  if (msg.data)
    std::cout << msg.data;
  std::cout << std::endl;
}

WGPUState::WGPUState() {
  instance = wgpu::createInstance({});
  wgpu::RequestAdapterOptions options;
  options.powerPreference = wgpu::PowerPreference::HighPerformance;
  wgpu::Adapter adapter = instance.requestAdapter(options);

  wgpu::FeatureName required_features[] = {
    wgpu::FeatureName::TimestampQuery,
  };

  wgpu::DeviceDescriptor device_desc;
  device_desc.uncapturedErrorCallbackInfo.callback = error;
  device_desc.requiredFeatures =
    reinterpret_cast<WGPUFeatureName *>(required_features);
  device_desc.requiredFeatureCount = 1;

  wgpu::Limits limits;
  limits.setDefault();
  limits.maxComputeInvocationsPerWorkgroup = 1024;
  limits.maxComputeWorkgroupSizeX = 256;
  limits.maxStorageBuffersPerShaderStage = 7;

  wgpu::AdapterInfo info;
  adapter.getInfo(&info);
  std::cout << info.vendor.data << std::endl;
  std::cout << info.device.data << std::endl;
  device_desc.requiredLimits = &limits;

  device = adapter.requestDevice(device_desc);
  queue = device.getQueue();
  expand = std::unique_ptr<FrontierExpansion>(new FrontierExpansion(device));
  identify = std::unique_ptr<FrontierIdentification>(new FrontierIdentification(device));
}

WGPUState::~WGPUState() {
  queue.release();
  device.release();
  instance.release();
}
