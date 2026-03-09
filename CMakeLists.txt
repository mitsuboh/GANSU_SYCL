# 
# GANSU: GPU Accelerated Numerical Simulation Utility
# 
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
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
project(GANSU LANGUAGES CXX)

# SYCL を有効化
add_compile_options(-fsycl -g -O0)
add_link_options(-fsycl)

# --- Include directories ---
include_directories(include_sycl)

# Uncomment the following line if f- or g-orbital basis functions are present.
# add_definitions(-DENABLE_F_INT1E)
# add_definitions(-DENABLE_G_INT1E)
# add_definitions(-DINT1E_MAX_L=4)

# --- 静的ライブラリ ---
add_library(int1e STATIC src_sycl/int1e.dp.cpp)
add_library(int2e STATIC src_sycl/int2e.dp.cpp)
add_library(int2c2e STATIC src_sycl/int2c2e.dp.cpp)
add_library(int3c2e STATIC src_sycl/int3c2e.dp.cpp)
add_library(int2e_direct STATIC src_sycl/int2e_direct.dp.cpp)
add_library(grad_w STATIC src_sycl/grad/grad_w.dp.cpp)
add_library(grad_s STATIC src_sycl/grad/grad_s.dp.cpp)
add_library(grad_k STATIC src_sycl/grad/grad_k.dp.cpp)
add_library(grad_v STATIC src_sycl/grad/grad_v.dp.cpp)
add_library(grad_g STATIC src_sycl/grad/grad_g.dp.cpp)


# --- oneAPI MKL ---
find_package(MKL CONFIG REQUIRED)

# --- GANSU ライブラリ ---
set(GANSU_SOURCES
    src_sycl/basis_set.cpp
    src_sycl/parseXYZ.cpp
    src_sycl/hf.dp.cpp
    src_sycl/rhf.dp.cpp
    src_sycl/uhf.dp.cpp
    src_sycl/rohf.dp.cpp
    src_sycl/gpu_manager.dp.cpp
    src_sycl/gpu_kernels.dp.cpp
    src_sycl/profiler.dp.cpp
    src_sycl/parameter_manager.cpp
    src_sycl/eri.dp.cpp
    src_sycl/eri_ri.dp.cpp
    src_sycl/eri_stored.dp.cpp
    src_sycl/eri_stored_mp4.dp.cpp
    src_sycl/utils.cpp
)
add_library(gansu STATIC ${GANSU_SOURCES})
target_compile_definitions(gansu PRIVATE PROFILE_ENABLED CONSOLE_COLOR_ENABLED)
# target_link_libraries(gansu PRIVATE int1e int2e int2c2e int3c2e MKL::MKL_SYCL)
target_link_libraries(gansu PRIVATE int1e int2e int2c2e int3c2e int2e_direct MKL::MKL_SYCL
        grad_w
        grad_s
        grad_k
        grad_v
        grad_g
        )

# --- 実行ファイル ---
add_executable(HF_main src_sycl/HF_main.dp.cpp)
target_compile_definitions(HF_main PRIVATE PROFILE_ENABLED CONSOLE_COLOR_ENABLED)
target_link_libraries(HF_main PRIVATE gansu)

# --- TBB が必要なら ---
# find_package(TBB REQUIRED)
# target_link_libraries(gansu PRIVATE TBB::tbb)

