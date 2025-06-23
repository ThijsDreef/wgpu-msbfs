import subprocess
import os

def get_output_name(prefix):
    return f'{prefix}_coalesced-access-bottom-up.json'

cuda = {
    "create_build_directory": ["cmake", "-S", ".", "-B", "build/cuda", "-DBACKEND=CUDA"],
    "build": ["cmake", "--build", "build/cuda"],
    "test": ["./build/cuda/bin/test_wgpumsbfs"],
    "benchmark": ["./build/cuda/bin/bench_wgpumsbfs", f'--benchmark_out={get_output_name('cuda')}']
}

wgpu = {
    "create_build_directory": ["cmake", "-S", ".", "-B", "build/wgpu", "-DBACKEND=WGPU", "-DWEBGPU_BUILD_FROM_SOURCE=OFF"],
    "build": ["cmake", "--build", "build/wgpu"],
    "test": ["./build/wgpu/bin/test_wgpumsbfs"],
    "benchmark": ["./build/wgpu/bin/bench_wgpumsbfs", f'--benchmark_out={get_output_name('wgpu')}']
}

dawn = {
    "create_build_directory": ["cmake", "-S", ".", "-B", "build/dawn", "-DBACKEND=DAWN", "-DWEBGPU_BUILD_FROM_SOURCE=OFF"],
    "build": ["cmake", "--build", "build/dawn"],
    "test": ["./build/dawn/bin/test_wgpumsbfs"],
    "benchmark": ["./build/dawn/bin/bench_wgpumsbfs", f'--benchmark_out={get_output_name('dawn')}']
}


targets = [
    wgpu,
    dawn
]

try:
    subprocess.check_output('nvidia-smi')
    targets.append(cuda)
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    pass

subprocess.run(["python", "scripts/generate-tests.py"])
for target in targets:
    for prop in target:
        subprocess.run(target[prop]).check_returncode()
