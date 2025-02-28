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

wgpu::BufferDescriptor getBufferDescriptor(uint64_t size, bool mapped, WGPUBufferUsageFlags usage) {
  wgpu::BufferDescriptor result;

  result.size = size;
  result.mappedAtCreation = mapped;
  result.usage = usage;

  return result;
}

std::unique_ptr<uint32_t[]> getMappedResult(WGPUState& state, wgpu::Buffer buffer, uint64_t size) {
  bool done = false;
  std::unique_ptr<uint32_t[]> result = std::unique_ptr<uint32_t[]>(new uint32_t[size / sizeof(uint32_t)]);
  auto _ =
      buffer.mapAsync(wgpu::MapMode::Read, 0, size,
                       [&](wgpu::BufferMapAsyncStatus) {
                         done = true;
                         memcpy(result.get(), buffer.getConstMappedRange(0, size), size);
                       });

  while (!done) {
#ifdef WEBGPU_BACKEND_WGPU
    state.queue.submit(0, nullptr);
#else
    state.instance.processEvents();
#endif
    state.device.poll(false);
  }
  buffer.unmap();
  return result;
}
