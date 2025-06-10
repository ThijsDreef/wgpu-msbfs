# WebGPU Multisource Breadth first search
## Reproducing test results
Don't forget to setup a python venv before installing dependencies
```
pip install -r requirements.txt
python scripts/build.py
```
Running this will generate the test data set, ground truth, run correctness
testing and run the benchmark for all three backends CUDA, WGPU and DAWN.
## Development setup
After running the `scripts/build.py` you will have three folder inside of the
build directory cuda, dawn and wgpu. Building after having changed files can be
done using `cmake --build build/${wgpu|cuda|dawn}`.

Tests can be run from the top level folder using `./build/${wgpu|cuda|dawn}/tests/test_wgpumsbfs`.

Benchmarking can be run from the top level folder using `./build/${wgpu|cuda|dawn}/benchmarks/bench_wgpumsbfs`.
## Building for webassembly
When building to webassembly only the main library is built. To generate the
correct build files use the following command.
```
emcmake cmake -S . -B build/emscripten -DBACKEND=EMDAWNWEBGPU
```
To use the generated wasm/js files see the web implementation of this repository.
