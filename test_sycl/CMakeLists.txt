# 
# GANSU: GPU Acclerated Numerical Simulation Utility
# 
# Copyright (c) 2025, Hiroshima University and Fujitsu Limited
# All rights reserved.
# 
# This software is licensed under the BSD 3-Clause License.
# You may obtain a copy of the license in the LICENSE file
# located in the root directory of this source tree or at:
# https://opensource.org/licenses/BSD-3-Clause
#  
# SPDX-License-Identifier: BSD-3-Clause
# 


cmake_minimum_required(VERSION 3.31)
project(GANSU LANGUAGES  CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
include(dpct.cmake)

# Download Google Test using FetchContent
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# Build settings for GTest
FetchContent_MakeAvailable(googletest)


# set compute capability
set(CMAKE_SYCL_ARCHITECTURES 80 86)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set include directories (global)
include_directories(../include_sycl)


# Static libraries
add_library(int1e STATIC ../src_sycl/int1e.dp.cpp)
add_library(int2e STATIC ../src_sycl/int2e.dp.cpp)


set(SOURCES
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

find_package(MKL CONFIG REQUIRED)

add_library(gansu STATIC ${SOURCES})
target_link_libraries(gansu ${MKL_LIB}
        int1e
        int2e
	MKL::MKL
    )


set(TEST_SOURCES
    test_gpu_manager.dp.cpp
    test_gpu_kernels.dp.cpp
)

add_executable(test_runner ${TEST_SOURCES})
target_link_libraries(test_runner gansu
        gtest
        gtest_main
    )


# Enable CTest framework
enable_testing()
add_test(NAME GANSU_Tests COMMAND test_runner)
