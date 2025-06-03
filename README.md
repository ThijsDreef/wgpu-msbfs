# WebGPU Multisource Breadth first search
To build the project first generate the test data using the generate-tests.py in
the scripts folder. Which should be ran from the root folder.

## Reproducing test results
```
pip install -r requirements.txt
python scripts/generate-tests.py
cmake -s . -b build/cuda --DBACKEND=CUDA
cmake -s . -b build/dawn --DBACKEND=DAWN
cmake -s . -b build/wgpu --DBACKEND=WGPU
```
While the WebGPUDistribution dawn source build fails one can use the prebuild
version
```
cmake -s . -b build/dawn-non-source -DBACKEND=DAWN -DWEBGPU_BUILD_FROM_SOURCE=OFF
```


## Building with dawn / wgpu
Setup a cmake build directory using either the -DWEBGPU_BACKEND=DAWN -DWEBGPU_BACKEND=WGPU cmake flags.
