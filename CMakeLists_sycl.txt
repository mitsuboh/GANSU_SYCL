cmake_minimum_required(VERSION 3.31)
project(GANSU LANGUAGES CXX)

# SYCL を有効化
add_compile_options(-fsycl)
add_link_options(-fsycl)

# --- Include directories ---
include_directories(include_sycl)

# --- 静的ライブラリ ---
add_library(int1e STATIC src_sycl/int1e.dp.cpp)
add_library(int2e STATIC src_sycl/int2e.dp.cpp)
add_library(int2c2e STATIC src_sycl/int2c2e.dp.cpp)
add_library(int3c2e STATIC src_sycl/int3c2e.dp.cpp)
add_library(int2e_direct STATIC src_sycl/int2e_direct.dp.cpp)

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
target_link_libraries(gansu PRIVATE int1e int2e int2c2e int3c2e int2e_direct MKL::MKL_SYCL)


# --- 実行ファイル ---
add_executable(HF_main src_sycl/HF_main.dp.cpp)
target_compile_definitions(HF_main PRIVATE PROFILE_ENABLED CONSOLE_COLOR_ENABLED)
target_link_libraries(HF_main PRIVATE gansu)

# --- TBB が必要なら ---
# find_package(TBB REQUIRED)
# target_link_libraries(gansu PRIVATE TBB::tbb)

