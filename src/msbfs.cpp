#include "msbfs.hpp"
#include "core/util/wgpu-utils.hpp"
#include <cstdint>
#include <cstring>
#include <map>
#include <thread>

std::vector<IterativeLengthResult> iterative_length(WGPUState &state, PathFindingRequest request, CSR csr) {
  TimingInfo timing_info;
  return iterative_length(state, request, csr, timing_info);
}


struct MSBFSInstance {
  wgpu::Buffer bsa;
  wgpu::Buffer bsak;
  wgpu::Buffer jfq;
  wgpu::Buffer jfq_length;
  wgpu::Buffer iteration;
  wgpu::Buffer destinations;
  wgpu::Buffer path_lengths;
  wgpu::Buffer mask;

  wgpu::Buffer jfq_length_staging;
  wgpu::Buffer path_lengths_staging;

  wgpu::BindGroup identify[4];
  wgpu::BindGroup expand[4];

  uint32_t results[32];

  size_t iterations;
  size_t queue_length;
  size_t async_operations;

  size_t offset;
  size_t length;
};

void advanceMSBFSInstance(WGPUState &state, MSBFSInstance &instance) {
  state.queue.writeBuffer(instance.iteration, 0, &instance.iterations, sizeof(uint32_t));
  wgpu::CommandEncoder encoder = state.device.createCommandEncoder({});


  wgpu::ComputePassEncoder compute_pass = encoder.beginComputePass({});
  compute_pass.setPipeline(state.expand->pipeline);
  compute_pass.setBindGroup(0, instance.expand[0], 0, nullptr);
  compute_pass.setBindGroup(1, instance.expand[1], 0, nullptr);
  compute_pass.setBindGroup(2, instance.expand[2 + instance.iterations % 2], 0,
                            nullptr);
  uint32_t size = instance.queue_length / 64 + 1;
  compute_pass.dispatchWorkgroups(size > 1024 ? 1024 : size, 1, 1);

  compute_pass.setPipeline(state.identify->pipeline);
  compute_pass.setBindGroup(0, instance.identify[0], 0, nullptr);
  compute_pass.setBindGroup(1, instance.identify[1], 0, nullptr);
  compute_pass.setBindGroup(2, instance.identify[2 + instance.iterations % 2], 0, nullptr);
  compute_pass.dispatchWorkgroups(1, 1, 1);

  compute_pass.end();
  compute_pass.release();
  encoder.copyBufferToBuffer(instance.jfq_length, 0, instance.jfq_length_staging, 0,
                             sizeof(uint32_t));

  state.queue.submit(encoder.finish());
  encoder.release();
  instance.async_operations = 1;

  wgpu::BufferMapCallbackInfo callback;
  callback.callback = [](WGPUMapAsyncStatus status, WGPUStringView msg,
                         void *user_1, void *user_2) {
    MSBFSInstance* instance = (MSBFSInstance*)user_1;
    wgpu::Buffer *jfq_buffer = (wgpu::Buffer*)user_2;
    instance->queue_length =
        *(uint32_t *)instance->jfq_length_staging.getConstMappedRange(
                                                                      0,
                                                                      sizeof(uint32_t));
    instance->async_operations--;
    jfq_buffer->unmap();
  };
  callback.mode = wgpu::CallbackMode::AllowProcessEvents;
  callback.userdata1 = &instance;
  callback.userdata2 = &instance.jfq_length_staging;
  instance.jfq_length_staging.mapAsync(wgpu::MapMode::Read, 0,
  sizeof(uint32_t), callback);

  instance.iterations++;
}

MSBFSInstance create_msbfs_instance(WGPUState &state, size_t jfq_max_size) {
  MSBFSInstance result;

  int flags = wgpu::BufferUsage::CopySrc |
                               wgpu::BufferUsage::Storage |
                               wgpu::BufferUsage::CopyDst;
  wgpu::BufferDescriptor desc = getBufferDescriptor(jfq_max_size, false, flags);
  result.bsa = state.device.createBuffer(desc);
  result.bsak = state.device.createBuffer(desc);
  result.jfq = state.device.createBuffer(desc);


  flags = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t) * 32, false, flags);
  result.path_lengths = state.device.createBuffer(desc);
  flags = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t) * 32, false, flags);

  result.destinations = state.device.createBuffer(desc);


  flags = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  result.iteration = state.device.createBuffer(desc);
  flags = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  result.mask = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::Storage;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  result.jfq_length = state.device.createBuffer(desc);
  flags = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  result.jfq_length_staging = state.device.createBuffer(desc);
  desc = getBufferDescriptor(sizeof(uint32_t) * 32, false, flags);
  result.path_lengths_staging = state.device.createBuffer(desc);
  result.async_operations = 0;
  result.iterations = 0;

  return result;
}

#define IN_QUEUE 32

std::vector<IterativeLengthResult> iterative_length_multi_queue(WGPUState &state, PathFindingRequest request, CSR csr) {
  std::vector<IterativeLengthResult> results;
  results.reserve(request.length);
  for (size_t x = 0; x < request.length; x++) {
    results.push_back({.src = request.src[x], .dst = request.dst[x], .length = 99999});
  }
  uint64_t v_size = csr.v_length * sizeof(uint32_t);
  uint64_t e_size = csr.e_length * sizeof(uint32_t);
  int flags = wgpu::BufferUsage::CopySrc |
              wgpu::BufferUsage::Storage |
              wgpu::BufferUsage::CopyDst;
  wgpu::BufferDescriptor desc = getBufferDescriptor(v_size, false, flags);
  wgpu::Buffer v_buffer = state.device.createBuffer(desc);

  desc = getBufferDescriptor(e_size, false, flags);
  wgpu::Buffer e_buffer = state.device.createBuffer(desc);

  // Populate buffers
  state.queue.writeBuffer(v_buffer, 0, csr.v, v_size);
  state.queue.writeBuffer(e_buffer, 0, csr.e, e_size);

  MSBFSInstance instances[IN_QUEUE];
  for (size_t x = 0; x < IN_QUEUE; x++) {
    instances[x] = create_msbfs_instance(state, v_size);
    instances[x].expand[0] = state.expand->jfq_group.getBindGroup(instances[x].jfq, instances[x].jfq_length, v_size);
    instances[x].expand[1] = state.expand->csr_group.getBindGroup(v_buffer, e_buffer, v_size, e_size);
    instances[x].expand[2] = state.expand->bsa_group.getBindGroup(instances[x].bsa, instances[x].bsak, v_size);
    instances[x].expand[3] = state.expand->bsa_group.getBindGroup(instances[x].bsak, instances[x].bsa, v_size);


    instances[x].identify[0] = state.identify->jfq_group.getBindGroup(instances[x].jfq, instances[x].jfq_length, v_size);
    instances[x].identify[1] = state.identify->length_group.getBindGroup(instances[x].destinations, instances[x].path_lengths, instances[x].iteration, instances[x].mask);
    instances[x].identify[2] = state.identify->bsa_group.getBindGroup(instances[x].bsa, instances[x].bsak, v_size);
    instances[x].identify[3] = state.identify->bsa_group.getBindGroup(instances[x].bsak, instances[x].bsa, v_size);
    instances[x].offset = 0;
    instances[x].length = 0;
  }
  size_t offset = 0;
  size_t finished = 0;

  struct Results {
    MSBFSInstance* instance;
    size_t *finished;
    std::vector<IterativeLengthResult>* results;
  };
  Results results_c[IN_QUEUE];
  for (size_t x = 0; x < IN_QUEUE; x++) {
    results_c[x] = { &instances[x], &finished, &results };
  }


  while (finished < request.length) {
    for (size_t x = 0; x < IN_QUEUE; x++) {
      if (instances[x].offset == 0 && instances[x].length == 0) {
        // Fire off new instance
        uint32_t pairs_to_solve = request.length - offset > 32 ? 32 : request.length - offset;
        if (pairs_to_solve == 0) {
          continue;
        }

        instances[x].offset = offset;
        instances[x].length = pairs_to_solve;
        instances[x].iterations = 0;

        wgpu::CommandEncoder encoder = state.device.createCommandEncoder({});
        encoder.clearBuffer(instances[x].bsa, 0, v_size);
        encoder.clearBuffer(instances[x].bsak, 0, v_size);
        encoder.clearBuffer(instances[x].destinations, 0, sizeof(uint32_t) * 32);
        encoder.clearBuffer(instances[x].path_lengths, 0, sizeof(uint32_t) * 32);
        state.queue.submit(encoder.finish());
        encoder.release();

        // Current hack to write some src into start
        std::map<uint32_t, uint32_t> to_write;
        for (size_t j = offset; j < offset + pairs_to_solve; j++) {
          to_write[request.src[j]] |= 1 << (j - offset);
        }

        for (auto y : to_write) {
          state.queue.writeBuffer(instances[x].bsak, y.first * sizeof(uint32_t), &y.second, sizeof(uint32_t));
        }

        state.queue.writeBuffer(instances[x].destinations, 0, request.dst + offset, pairs_to_solve * sizeof(uint32_t));
        uint32_t mask_value = 0xffffffff;
        state.queue.writeBuffer(instances[x].mask, 0, &mask_value, sizeof(uint32_t));

        advanceMSBFSInstance(state, instances[x]);
        offset += pairs_to_solve;
      } else if (instances[x].queue_length == 0 && instances[x].async_operations == 0) {
        // msbfs finished collect results
        instances[x].async_operations = 1;
        wgpu::CommandEncoder encoder = state.device.createCommandEncoder({});
        encoder.copyBufferToBuffer(instances[x].path_lengths, 0,
                                   instances[x].path_lengths_staging, 0,
                                   sizeof(uint32_t) * instances[x].length);
        state.queue.submit(encoder.finish());
        encoder.release();

        wgpu::BufferMapCallbackInfo callback;
        callback.callback = [](WGPUMapAsyncStatus status, WGPUStringView msg,
                               void *user_1, void *user_2) {
          MSBFSInstance* instance = (MSBFSInstance*)user_1;
          Results *r = (Results*)user_2;
          uint32_t* data = (uint32_t*)instance->path_lengths_staging.getConstMappedRange(0, sizeof(uint32_t) * 32);
          size_t offset = instance->offset;
          for (size_t x = 0; x < instance->length; x++) {
            if (data[x] == 0 && r->results->at(x + offset).src != r->results->at(x + offset).dst) continue;
            r->results->at(x + offset).length = data[x];
          }
          *r->finished += instance->length;
          instance->offset = 0;
          instance->length = 0;
          instance->async_operations--;
          instance->path_lengths_staging.unmap();
        };
        callback.mode = wgpu::CallbackMode::AllowProcessEvents;
        callback.userdata1 = &instances[x];
        callback.userdata2 = &results_c[x];
        instances[x].path_lengths_staging.mapAsync(wgpu::MapMode::Read, 0, 32 * sizeof(uint32_t), callback);
      } else if (instances[x].async_operations == 0) {
        // Advance msbfs
        advanceMSBFSInstance(state, instances[x]);
      }
    }
#ifdef WEBGPU_BACKEND_DAWN
    state.instance.processEvents();
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
#endif
#ifdef WEBGPU_BACKEND_WGPU
    state.device.poll(true, nullptr);
#endif
  }

  for (size_t x = 0; x < IN_QUEUE; x++) {
    instances[x].bsa.destroy();
    instances[x].bsak.destroy();
    instances[x].jfq.destroy();
    instances[x].jfq_length.destroy();
    instances[x].jfq_length_staging.destroy();
    instances[x].path_lengths.destroy();
    instances[x].path_lengths_staging.destroy();
    instances[x].destinations.destroy();
    instances[x].iteration.destroy();
    for (size_t y = 0; y < 4; y++) {
      instances[x].expand[y].release();
      instances[x].identify[y].release();
    }
  }

  v_buffer.destroy();
  e_buffer.destroy();

  return results;
}

std::vector<IterativeLengthResult> iterative_length(WGPUState &state, PathFindingRequest request, CSR csr, TimingInfo& timing_info) {
  uint64_t v_size = csr.v_length * sizeof(uint32_t);
  uint64_t e_size = csr.e_length * sizeof(uint32_t);

  int flags = wgpu::BufferUsage::CopySrc |
                               wgpu::BufferUsage::Storage |
                               wgpu::BufferUsage::CopyDst;
  wgpu::BufferDescriptor desc = getBufferDescriptor(v_size, false, flags);
  wgpu::Buffer bsa = state.device.createBuffer(desc);
  wgpu::Buffer bsak = state.device.createBuffer(desc);
  wgpu::Buffer jfq = state.device.createBuffer(desc);
  wgpu::Buffer v_buffer = state.device.createBuffer(desc);

  desc = getBufferDescriptor(e_size, false, flags);
  wgpu::Buffer e_buffer = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopySrc;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  wgpu::Buffer jfq_length = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::MapRead |
          wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  wgpu::Buffer jfq_length_staging = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t) * 32, false, flags);
  wgpu::Buffer destinations = state.device.createBuffer(desc);
  wgpu::Buffer path_length = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  wgpu::Buffer mask = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  desc = getBufferDescriptor(sizeof(uint32_t), false, flags);
  wgpu::Buffer iteration = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  desc = getBufferDescriptor(sizeof(uint32_t) * 32, false, flags);
  wgpu::Buffer output_staging = state.device.createBuffer(desc);

  flags = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::QueryResolve;
  desc = getBufferDescriptor(sizeof(uint64_t) * 4, false, flags);
  wgpu::Buffer timing_buffer = state.device.createBuffer(desc);
  flags = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  desc = getBufferDescriptor(sizeof(uint64_t) * 4, false, flags);
  wgpu::Buffer timing_staging_buffer = state.device.createBuffer(desc);


  // Populate buffers
  state.queue.writeBuffer(v_buffer, 0, csr.v, v_size);
  state.queue.writeBuffer(e_buffer, 0, csr.e, e_size);

  wgpu::BindGroup expand_groups[] = {
    state.expand->jfq_group.getBindGroup(jfq, jfq_length, v_size),
    state.expand->csr_group.getBindGroup(v_buffer, e_buffer, v_size, e_size),
    state.expand->bsa_group.getBindGroup(bsa, bsak, v_size),
    state.expand->bsa_group.getBindGroup(bsak, bsa, v_size),
  };

  wgpu::BindGroup identify_groups[] = {
    state.identify->jfq_group.getBindGroup(jfq, jfq_length, v_size),
    state.identify->length_group.getBindGroup(destinations, path_length, iteration, mask),
    state.identify->bsa_group.getBindGroup(bsa, bsak, v_size),
    state.identify->bsa_group.getBindGroup(bsak, bsa, v_size),
  };

  std::vector<IterativeLengthResult> results;
  results.reserve(request.length);

  for (size_t offset = 0; offset < request.length; offset += 32) {
    uint32_t pairs_to_solve = request.length - offset > 32 ? 32 : request.length - offset;
    wgpu::CommandEncoder encoder = state.device.createCommandEncoder({});
    encoder.clearBuffer(bsa, 0, v_size);
    encoder.clearBuffer(bsak, 0, v_size);
    encoder.clearBuffer(destinations, 0, sizeof(uint32_t) * 32);
    encoder.clearBuffer(path_length, 0, sizeof(uint32_t) * 32);
    state.queue.submit(encoder.finish());
    encoder.release();

    // Current hack to write some src into start
    std::map<uint32_t, uint32_t> to_write;
    for (size_t j = offset; j < offset + pairs_to_solve; j++) {
      to_write[request.src[j]] |= 1 << (j - offset);
    }

    for (auto x : to_write) {
      state.queue.writeBuffer(bsak, x.first * sizeof(uint32_t), &x.second, sizeof(uint32_t));
    }

    state.queue.writeBuffer(destinations, 0, request.dst + offset, pairs_to_solve * sizeof(uint32_t));
    uint32_t mask_value = 0xffffffff;
    state.queue.writeBuffer(mask, 0, &mask_value, sizeof(uint32_t));

    uint32_t length = 0;
    uint32_t iterations = 0;

    wgpu::ComputePassDescriptor compute_desc;

    do {
      state.queue.writeBuffer(iteration, 0, &iterations, sizeof(uint32_t));
      wgpu::CommandEncoder encoder = state.device.createCommandEncoder({});


      wgpu::ComputePassEncoder compute_pass = encoder.beginComputePass(compute_desc);
      compute_pass.setPipeline(state.expand->pipeline);
      compute_pass.setBindGroup(0, expand_groups[0], 0, nullptr);
      compute_pass.setBindGroup(1, expand_groups[1], 0, nullptr);
      compute_pass.setBindGroup(2, expand_groups[2 + iterations % 2], 0,
                                nullptr);
      uint32_t size = length / 64 + 1;
      compute_pass.dispatchWorkgroups(size > 1024 ? 1024 : size, 1, 1);
      compute_pass.end();
      compute_pass.release();

      compute_pass = encoder.beginComputePass(compute_desc);
      compute_pass.setPipeline(state.identify->pipeline);
      compute_pass.setBindGroup(0, identify_groups[0], 0, nullptr);
      compute_pass.setBindGroup(1, identify_groups[1], 0, nullptr);
      compute_pass.setBindGroup(2, identify_groups[2 + iterations % 2], 0, nullptr);
      compute_pass.dispatchWorkgroups(1, 1, 1);

      compute_pass.end();
      compute_pass.release();
      encoder.copyBufferToBuffer(jfq_length, 0, jfq_length_staging, 0,
                                 sizeof(uint32_t));

      state.queue.submit(encoder.finish());
      encoder.release();

      length = getMappedResult(state, jfq_length_staging, sizeof(uint32_t))[0];
      iterations++;
    } while (length != 0);

    encoder = state.device.createCommandEncoder({});
    encoder.copyBufferToBuffer(path_length, 0, output_staging, 0, sizeof(uint32_t) * 32);
    state.queue.submit(encoder.finish());
    auto output = getMappedResult(state, output_staging, 32 * sizeof(uint32_t));
    for (size_t j = 0; j < pairs_to_solve; j++) {
      if (output[j] == 0 && request.dst[j + offset] != request.src[j + offset]) {
        continue;
      }
      results.push_back({
          .src = request.src[j + offset],
          .dst = request.dst[j + offset],
          .length = output[j],
      });
    }
  }

  // Cleanup
  bsa.release();
  bsak.release();
  jfq.release();
  v_buffer.release();
  e_buffer.release();
  jfq_length.release();
  jfq_length_staging.release();
  destinations.release();
  path_length.release();
  mask.release();
  iteration.release();
  output_staging.release();
  for (size_t i = 0; i < 4; i++) {
    identify_groups[i].release();
    expand_groups[i].release();
  }

  return results;
}
