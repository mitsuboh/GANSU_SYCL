## Updates

- Oct 30 2025: gpu_test branch is merged and cofirmed run on Core Ultra2 GPU. Due to JIT compile of device codes, it takes 27 plus 22 minutes only for the first run as shown blelow. After the first run no compile time is needed.

[00:00.000] START: compute_nuclear_repulsion_energy
[00:00.000] END:   compute_nuclear_repulsion_energy after 0.016 ms
[00:00.000] START: compute_core_hamiltonian_matrix
[27:25.462] END:   compute_core_hamiltonian_matrix after 1.6455e+06 ms
[27:25.462] START: precompute_eri_matrix
[49:44.378] END:   precompute_eri_matrix after 1.3389e+06 ms
[49:44.378] START: compute_transform_matrix
[49:44.419] END:   compute_transform_matrix after 41.559 ms

- If you do not like this, you can use AOT compiling by adding CMakelists.txt something like, set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-target_gen -Xs \"-device 0x7d67\"")                                                                                    [Your device id: 0x7d67 ]

# GANSU-SYCL

This repository is an experimental port of the original [GANSU](https://github.com/Yasuaki-Ito/GANSU), which is an open-source quantum chemistry software designed for modern computing architectures. The original implementation was based on CUDA, and this fork adds **SYCL** support to enable cross-platform compatibility with Intel, AMD, and NVIDIA GPUs using oneAPI or other SYCL implementations.

**Note**: This program was ported from CUDA to SYCL-based C++ using the Intel DPC++ Compatibility Tool (DPCT). As a result, it includes DPCT-generated APIs that are specifically designed for Intel GPU architectures and may not be compatible with other platforms.
Please note that this port is still under development and may contain numerous bugs. It has only passed a limited number of example cases from the original code, and no comprehensive or systematic testing has been performed yet.

## What's include

- SYCL ported source codes added (`src_sycl/`, `include_sycl/`)
- SYCL-compatible `CMakeLists_sycl.txt` for building with DPC++/oneAPI
- SYCL-compatible `test_sycl/CMakeLists.txt` for building tests with DPC++/oneAPI
- Original CUDA code preserved under `src/`, `include/`
- Modular design: Choose between CUDA and SYCL at build time

## Repository Structure

```
GANSU_SYCL/
├─  include/ # Original CUDA headers
├─  src/ # Original CUDA source files
├─  include_sycl/ # SYCL ported headers
├─  src_sycl/ # SYCL ported source files
├─  data/ # Sample data and test cases
├─  CMakeLists_cuda.txt # Default CUDA build configuration
├─  CMakeLists_sycl.txt # SYCL-specific CMake configuration
├─  README_org.md # Original README.md
├─  README.md # This file
```

## Building (SYCL Version)

### Requirements

- [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) (includes DPC++ compiler)
- CMake ≥ 3.15
- C++17 or higher
- Other dependencies required for the original code

### Build Instructions (with oneAPI)

```bash
# oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Clone the repository
git clone https://github.com/mitsuboh/GANSU_SYCL.git
cd GANSU_SYCL

# Copy CMakelist.txt
cp CMakelists_sycl.txt CMakelists.txt

# Create build directory
mkdir build_sycl && cd build_sycl

# Configure with SYCL-specific CMake file
cmake -DCMAKE_CXX_COMPILER=icpx ..

# Build the project
make -j
```

## Notes

- The SYCL port aims to replicate the logic of the original CUDA implementation as closely as possible.
- Performance tuning for SYCL devices is still pending, as the port is in trial and has not yet undergone detailed optimization.
- Not all features may be fully supported on non-NVIDIA GPUs yet, particularly for AMD and Intel platforms.

## TODO

- Replace DPCT-generated APIs with standard SYCL APIs to improve compatibility with non-Intel GPUs.
- Expand testing of the SYCL backend using a broader set of examples.
- Perform performance tuning and optimization across various SYCL-supported devices.

## License
This project inherits the license of the original GANSU repository.
See LICENSE for details.

## Acknowledgements
This work has benefited greatly from the GANSU Project, conducted by Professor Yasuaki Ito and his research group at Hiroshima
University in collaboration with Fujitsu Limited. We sincerely thank them for making their results available and for their
valuable contributions.

We would like to thank Intel for granting us access to the Intel® Software Development Platform 1.0.0, which was made available under specific agreement terms. The platform’s capabilities were essential to the success of this work.
