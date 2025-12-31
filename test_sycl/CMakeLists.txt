cmake_minimum_required(VERSION 3.31)
project(GANSU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_SYCL_ARCHITECTURES 80 86)

# SYCL を有効化
add_compile_options(-fsycl)
add_link_options(-fsycl)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

include_directories(../include_sycl)

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
target_link_libraries(gansu
    int1e
    int2e
    MKL::MKL_SYCL
)

set(TEST_SOURCES
    test_gpu_manager.dp.cpp
    test_gpu_kernels.dp.cpp
)

add_executable(test_runner ${TEST_SOURCES})
target_link_libraries(test_runner
    gansu
    gtest
    gtest_main
)

enable_testing()
add_test(NAME GANSU_Tests COMMAND test_runner)
