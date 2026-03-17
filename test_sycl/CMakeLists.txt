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

link_directories(${CMAKE_SOURCE_DIR}/../build_sycl)

find_package(MKL CONFIG REQUIRED)

set(TEST_SOURCES
    test_gpu_manager.dp.cpp
    test_gpu_kernels.dp.cpp
)

add_executable(test_runner ${TEST_SOURCES})
target_include_directories(test_runner PRIVATE ${CMAKE_SOURCE_DIR}/../include_sycl)

target_link_libraries(test_runner
    gansu
    int1e
    int2e
    int2c2e
    int3c2e
    int2e_direct
    grad_w
    grad_s
    grad_k
    grad_v
    grad_g
    MKL::MKL_SYCL
    gtest
    gtest_main
)

enable_testing()
add_test(NAME GANSU_Tests COMMAND test_runner)
