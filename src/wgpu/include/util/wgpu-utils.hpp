#pragma once
#include <cstdint>
#include <memory>
#include "wgpu/wgpu-state.hpp"


wgpu::BindGroupLayoutEntry getComputeEntry(uint32_t binding, wgpu::BufferBindingType type, bool dynamic_offset, uint64_t min_bind_size);
wgpu::BindGroupEntry getBindGroupBufferEntry(wgpu::Buffer buffer, uint32_t binding, uint64_t offset, uint64_t size);
wgpu::BufferDescriptor getBufferDescriptor(uint64_t size, bool mapped, int usage);
std::unique_ptr<uint32_t[]> getMappedResult(WGPUState &state,
                                            wgpu::Buffer buffer, uint64_t size);
wgpu::StringView getStringViewFromCString(const char* string);
