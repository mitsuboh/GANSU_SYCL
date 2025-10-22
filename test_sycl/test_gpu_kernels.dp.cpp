/*
 * GANSU: GPU Acclerated Numerical Simulation Utility
 *
 * Copyright (c) 2025, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <sycl/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <gtest/gtest.h>
//#include <dpct/blas_utils.hpp>

#include <vector>
#include "../include_sycl/gpu_manager.hpp"
#include "test_util.hpp"

using namespace gansu::gpu;
using real_t = double;

// Test case: invertSqrtElements
TEST(gpu_kernels_test, InvertSqrtElements) {
auto myQ = GPUHandle::syclsolver();
    constexpr size_t size = 4;
    
    // Define input data
    real_t h_input[size] = {1.0, 4.0, 9.0, 16.0};
    real_t expected[size] = {1.0, 0.5, 0.3333, 0.25}; // Expected results (inverse square root)
    
    // Allocate device memory
    real_t* d_vectors;
    d_vectors = sycl::malloc_device<real_t>(size, myQ);

    // Copy input data to device
    myQ.memcpy(d_vectors, h_input, size * sizeof(real_t)).wait();

    // Execute the function
    invertSqrtElements(d_vectors, size);
    
    // Copy the result back to host
    std::vector<real_t> h_output(size);
    copyToHost(h_output, d_vectors, size);
    
    // Validate results
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(h_output[i], expected[i], 1e-4); // Allow small floating-point error
    }
    
    // Free allocated device memory
    sycl::free(d_vectors, myQ);
}


// Test case: Transpose a square matrix in place
TEST(gpu_kernels_test, TransposeMatrixInPlace) {
auto myQ = GPUHandle::syclsolver();
    constexpr int size = 3;
    
    // Define input matrix (row-major order)
    real_t h_input[size * size] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };

    // Expected transposed matrix
    real_t expected[size * size] = {
        1.0, 4.0, 7.0,
        2.0, 5.0, 8.0,
        3.0, 6.0, 9.0
    };

    // Allocate device memory
    real_t* d_matrix;
    d_matrix = sycl::malloc_device<real_t>(size * size, myQ);

    // Copy input matrix to device
    myQ.memcpy(d_matrix, h_input, size * size * sizeof(real_t)).wait();

    // Execute the transpose function
    transposeMatrixInPlace(d_matrix, size);

    // Copy the result back to host
    std::vector<real_t> h_output(size * size);
    copyToHost(h_output, d_matrix, size * size);

    // Validate results
    for (int i = 0; i < size * size; ++i) {
        EXPECT_NEAR(h_output[i], expected[i], 1e-4); // Allow small floating-point error
    }

    // Free allocated device memory
    sycl::free(d_matrix, myQ);
}


// Test case: Create a diagonal matrix from a vector
TEST(gpu_kernels_test, MakeDiagonalMatrix) {
auto myQ = GPUHandle::syclsolver();
    constexpr int size = 4;

    // Input vector
    real_t h_vector[size] = {1.0, 2.0, 3.0, 4.0};

    // Expected diagonal matrix
    real_t expected[size * size] = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0,
        0.0, 0.0, 0.0, 4.0
    };

    // Allocate device memory
    real_t *d_vector, *d_matrix;
    d_vector = sycl::malloc_device<real_t>(size, myQ);
    d_matrix = sycl::malloc_device<real_t>(size * size, myQ);

    // Copy vector to device
    myQ.memcpy(d_vector, h_vector, size * sizeof(real_t)).wait();

    // Execute the function
    makeDiagonalMatrix(d_vector, d_matrix, size);

    // Copy the result back to host
    std::vector<real_t> h_output(size * size);
    copyToHost(h_output, d_matrix, size * size);

    // Validate results
    for (int i = 0; i < size * size; ++i) {
        EXPECT_NEAR(h_output[i], expected[i], 1e-4); // Allow small floating-point error
    }

    // Free allocated device memory
    sycl::free(d_vector, myQ);
    sycl::free(d_matrix, myQ);
}


// Test case: Compute the trace of a square matrix
TEST(gpu_kernels_test, ComputeMatrixTrace) {
auto myQ = GPUHandle::syclsolver();
    constexpr int size = 4;

    // Input matrix (row-major order)
    real_t h_matrix[size * size] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };

    // Expected trace (sum of diagonal elements)
    real_t expected_trace = 1.0 + 6.0 + 11.0 + 16.0; // = 34.0

    // Allocate device memory
    real_t *d_matrix;
    d_matrix = sycl::malloc_device<real_t>(size * size, myQ);

    // Copy matrix to device
    myQ.memcpy(d_matrix, h_matrix, size * size * sizeof(real_t)).wait();

    // Compute trace
    real_t computed_trace = computeMatrixTrace(d_matrix, size);

    // Validate the result
    EXPECT_NEAR(computed_trace, expected_trace, 1e-4); // Allow small floating-point error

    // Free allocated device memory
    sycl::free(d_matrix, myQ);
}


