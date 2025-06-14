import subprocess
import os

def get_output_name(prefix, workgroups):
    return f'{workgroups}_{prefix}_workgroups.json'

cuda = {
    "name": "cuda",
    "create_build_directory": ["cmake", "-S", ".", "-B", "build/cuda", "-DBACKEND=CUDA"],
    "build": ["cmake", "--build", "build/cuda"],
    "test": ["./build/cuda/tests/test_wgpumsbfs"],
    "benchmark": ["./build/cuda/benchmarks/bench_wgpumsbfs"]
}

wgpu = {
    "name": "wgpu",
    "create_build_directory": ["cmake", "-S", ".", "-B", "build/wgpu", "-DBACKEND=WGPU", "-DWEBGPU_BUILD_FROM_SOURCE=OFF"],
    "build": ["cmake", "--build", "build/wgpu"],
    "test": ["./build/wgpu/tests/test_wgpumsbfs"],
    "benchmark": ["./build/wgpu/benchmarks/bench_wgpumsbfs"]
}

dawn = {
    "name": "dawn",
    "create_build_directory": ["cmake", "-S", ".", "-B", "build/dawn", "-DBACKEND=DAWN", "-DWEBGPU_BUILD_FROM_SOURCE=OFF"],
    "build": ["cmake", "--build", "build/dawn"],
    "test": ["./build/dawn/tests/test_wgpumsbfs"],
    "benchmark": ["./build/dawn/benchmarks/bench_wgpumsbfs"]
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

workgroups = [1, 2, 4]

subprocess.run(["python", "scripts/generate-tests.py"])
for target in targets:
    for prop in target:
        if (prop == "name"):
            continue
        if (prop == "create_build_directory" or prop == "bulid"):
            subprocess.run(target[prop]).check_returncode()
            continue
        for workgroup in workgroups:
            my_env = os.environ.copy()
            my_env['WORKGROUPS'] = str(workgroup)
            if (prop == 'benchmark'):
                args = target[prop]
                args.append(f"--benchmark_out={get_output_name(target["name"], str(workgroup))}")
                subprocess.run(args, env=my_env).check_returncode()
            else:
                subprocess.run(target[prop], env=my_env).check_returncode()
