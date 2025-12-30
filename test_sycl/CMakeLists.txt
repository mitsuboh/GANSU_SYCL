cmake_minimum_required(VERSION 3.31)
project(GANSU LANGUAGES CXX)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 17)

# --- Download Google Test ---
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# --- SYCL compile flags for Intel GPU (no CUDA) ---
set(SYCL_COMMON_COMPILE
    -fsycl
    -fsycl-targets=nvptx64-nvidia-cuda
    -fsycl-device-code-split=per_kernel
)

set(SYCL_CUDA_BACKEND_OPTS
    -Xsycl-target-backend=nvptx64-nvidia-cuda "--cuda-gpu-arch=sm_70"
) 

set(ONEMATH_HOME ~/docker_env/oneMath) 

# --- Include directories ---
include_directories(
    ../include_sycl
    ${ONEMATH_HOME}/include
    ${ONEMATH_HOME}/build/bin
    ${ONEMATH_HOME}/examples/include
    /home/mikei/docker_env/oneDPL/include
)

# --- Static libraries ---
add_library(int1e STATIC ../src_sycl/int1e.dp.cpp)
add_library(int2e STATIC ../src_sycl/int2e.dp.cpp)

# Apply SYCL flags to static libraries
target_compile_options(int1e PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_compile_options(int2e PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})

# --- GANSU library ---
set(GANSU_SOURCES
    ../src_sycl/basis_set.cpp
    ../src_sycl/parseXYZ.cpp
    ../src_sycl/hf.dp.cpp
    ../src_sycl/rhf.dp.cpp
    ../src_sycl/uhf.dp.cpp
    ../src_sycl/rohf.dp.cpp
    ../src_sycl/gpu_manager.dp.cpp
    ../src_sycl/gpu_kernels.dp.cpp
    ../src_sycl/profiler.dp.cpp
    ../src_sycl/parameter_manager.cpp
    ../src_sycl/eri.dp.cpp
)
add_library(gansu STATIC ${GANSU_SOURCES})
target_compile_options(gansu PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_link_options(gansu PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)

# --- oneMKL GPU libraries ---
link_directories(${ONEMATH_HOME}/build/lib)
target_link_libraries(gansu PUBLIC
    int1e
    int2e
    onemath
)

# --- Executable ---
set(TEST_SOURCES
    test_gpu_manager.dp.cpp
    test_gpu_kernels.dp.cpp
)
add_executable(test_runner ${TEST_SOURCES})
target_compile_options(test_runner PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_link_options(test_runner PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda ${SYCL_CUDA_BACKEND_OPTS})
target_link_libraries(test_runner PRIVATE
    gansu
    gtest
    gtest_main
    onemath
)

# --- RPATH for runtime libraries ---
set(CMAKE_INSTALL_RPATH ${ONEMATH_HOME}/build/lib)

# --- Enable CTest ---
enable_testing()
add_test(NAME GANSU_Tests COMMAND test_runner)

