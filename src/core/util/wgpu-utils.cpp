#include "core/util/wgpu-utils.hpp"
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

wgpu::BindGroupLayoutEntry getComputeEntry(uint32_t binding, wgpu::BufferBindingType type, bool dynamic_offset, uint64_t min_bind_size) {
  wgpu::BindGroupLayoutEntry result;

  result.binding = binding;
  result.visibility = wgpu::ShaderStage::Compute;

  result.buffer.minBindingSize = min_bind_size;
  result.buffer.hasDynamicOffset = dynamic_offset;
  result.buffer.type = type;

  return result;
}

wgpu::BindGroupEntry getBindGroupBufferEntry(wgpu::Buffer buffer, uint32_t binding, uint64_t offset, uint64_t size) {
  wgpu::BindGroupEntry result;

  result.buffer = buffer;
  result.binding = binding;
  result.offset = offset;
  result.size = size;

  return result;
}

wgpu::BufferDescriptor getBufferDescriptor(uint64_t size, bool mapped, int usage) {
  wgpu::BufferDescriptor result;

  result.size = size;
  result.mappedAtCreation = mapped;
  result.usage = usage;

  return result;
}

#ifdef WEBGPU_BACKEND_WGPU
std::unique_ptr<uint32_t[]> getMappedResult(WGPUState& state, wgpu::Buffer buffer, uint64_t size) {
  bool done = false;
  std::unique_ptr<uint32_t[]> result =
    std::unique_ptr<uint32_t[]>(new uint32_t[size / sizeof(uint32_t)]);
  auto _ = buffer.mapAsync(wgpu::MapMode::Read, 0, size,
                           [&](wgpu::BufferMapAsyncStatus status) {
                         done = true;
                         memcpy(result.get(), buffer.getConstMappedRange(0, size), size);
                       });

  while (!done) {
    state.device.poll(false);
  }
  buffer.unmap();
  return result;
}
#else

struct CopyData {
  uint64_t size;
  void* dst;
};

static void handleBufferMap(WGPUMapAsyncStatus status, const char* msg, void* user_1, void* user_2) {
  wgpu::Buffer* buffer = (wgpu::Buffer*)user_1;
  CopyData* copy = (CopyData*)user_2;
  const void* data = buffer->getConstMappedRange(0, copy->size);
  memcpy(copy->dst, data, copy->size);
}

std::unique_ptr<uint32_t[]> getMappedResult(WGPUState& state, wgpu::Buffer buffer, uint64_t size) {
  std::unique_ptr<uint32_t[]> result = std::unique_ptr<uint32_t[]>(new uint32_t[size / sizeof(uint32_t)]);
  CopyData copy = { size, result.get() };

  wgpu::BufferMapCallbackInfo2 info;
  info.mode = wgpu::CallbackMode::WaitAnyOnly;
  info.callback = handleBufferMap;
  info.userdata1 = &buffer;
  info.userdata2 = &copy;
  auto x = buffer.mapAsync2(wgpu::MapMode::Read, 0, size, info);
  wgpu::FutureWaitInfo wait_info;
  wait_info.setDefault();
  wait_info.future = x;
  while (!wait_info.completed) {
    state.instance.waitAny(1, &wait_info, 0);
  }
  buffer.unmap();
  return result;
}

#endif
