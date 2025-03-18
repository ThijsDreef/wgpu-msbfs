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



struct CopyData {
  bool done;
  uint64_t size;
  void* dst;
};

static void handleBufferMap(WGPUMapAsyncStatus status, WGPUStringView msg, void* user_1, void* user_2) {
  wgpu::Buffer* buffer = (wgpu::Buffer*)user_1;
  CopyData* copy = (CopyData*)user_2;
  const void *data = buffer->getConstMappedRange(0, copy->size);
  memcpy(copy->dst, data, copy->size);
  copy->done = true;
}

std::unique_ptr<uint32_t[]> getMappedResult(WGPUState& state, wgpu::Buffer buffer, uint64_t size) {
  std::unique_ptr<uint32_t[]> result = std::unique_ptr<uint32_t[]>(new uint32_t[size / sizeof(uint32_t)]);
  CopyData copy = { false, size, result.get() };

  wgpu::BufferMapCallbackInfo info;
  info.mode = wgpu::CallbackMode::WaitAnyOnly;
  info.callback = handleBufferMap;
  info.userdata1 = &buffer;
  info.userdata2 = &copy;
  auto x = buffer.mapAsync(wgpu::MapMode::Read, 0, size, info);
  wgpu::FutureWaitInfo wait_info;
  wait_info.setDefault();
  wait_info.future = x;
  while (!copy.done) {
#ifdef WEBGPU_BACKEND_WGPU
    state.device.poll(true, nullptr);
#else
    state.instance.waitAny(1, &wait_info, 0);
#endif
  }
  buffer.unmap();
  return result;
}

wgpu::StringView getStringViewFromCString(const char *string) {
  wgpu::StringView view;
  view.length = strlen(string);
  view.data = string;
  return view;
}
