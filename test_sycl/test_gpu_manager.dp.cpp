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
//#include <dpct/lapack_utils.hpp>

#include <vector>
#include <cmath>
#include "../include_sycl/gpu_manager.hpp"
#include "test_util.hpp"

using namespace gansu::gpu;

using real_t = double;

// Test case for eigenDecomposition
TEST(gpu_manager_test, eigenDecomposition) {
auto myQ = GPUHandle::syclsolver();
    constexpr int size = 2;

    // Input symmetric matrix
    real_t h_matrix[size * size] = {4.0, 1.0, 
                                    1.0, 3.0};
    
    // Expected eigenvalues for [[4, 1], [1, 3]]: λ = {5, 2}
    real_t expected_eigenvalues[size] = {2.381966, 4.618034};

    // Allocate device memory
    real_t *d_matrix, *d_eigenvalues, *d_eigenvectors;
    d_matrix = sycl::malloc_device<real_t>(size * size, myQ);
    d_eigenvalues = sycl::malloc_device<real_t>(size, myQ);
    d_eigenvectors = sycl::malloc_device<real_t>(size * size, myQ);

    // Copy matrix to device
    myQ.memcpy(d_matrix, h_matrix, size * size * sizeof(real_t)).wait();

    // Perform eigen decomposition
    int status = eigenDecomposition(d_matrix, d_eigenvalues, d_eigenvectors, size);

    // Check if decomposition was successful
    EXPECT_EQ(status, 0);

    // Copy results back to host
    std::vector<real_t> h_eigenvalues(size);
    std::vector<real_t> h_eigenvectors(size * size);
    copyToHost(h_eigenvalues, d_eigenvalues, size);
    copyToHost(h_eigenvectors, d_eigenvectors, size * size);

    // Sort eigenvalues (cuSOLVER does not guarantee order)
    std::sort(h_eigenvalues.begin(), h_eigenvalues.end());
    std::sort(expected_eigenvalues, expected_eigenvalues + size);

    // Verify eigenvalues
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(h_eigenvalues[i], expected_eigenvalues[i], 1e-6);
    }

    // Verify eigenvectors are orthonormal (dot product should be close to 0 or 1)
    real_t dot_product = h_eigenvectors[0] * h_eigenvectors[2] + h_eigenvectors[1] * h_eigenvectors[3];
    EXPECT_NEAR(dot_product, 0.0, 1e-6);

    // Cleanup
    sycl::free(d_matrix, myQ);
    sycl::free(d_eigenvalues, myQ);
    sycl::free(d_eigenvectors, myQ);
}



// Matrix comparison helper function
bool isEqualMatrix(const std::vector<double>& mat1, const std::vector<double>& mat2, int N, double tol = 1e-6) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fabs(mat1[i * N + j] - mat2[i * N + j]) > tol) {
                return false;
            }
        }
    }
    return true;
}

// Unit test for invertMatrix
TEST(gpu_manager_test, invertMatrix) {
auto myQ = GPUHandle::syclsolver();
    constexpr int N = 3;
    std::vector<double> h_A = {1, 2, 1, 
                               2, 1, 0, 
                               1, 1, 2};
    
    std::vector<double> h_invA = {-0.4, 0.6, 0.2,
                                  0.8, -0.2, -0.4,
                                  -0.2, -0.2, 0.6};


    double* d_A;
    d_A = sycl::malloc_device<double>(N * N, myQ);
    myQ.memcpy(d_A, h_A.data(), N * N * sizeof(double)).wait();

    // Call the inversion function
    invertMatrix(d_A, N);

    // Copy back result
    std::vector<double> h_result(N * N);
    myQ.memcpy(h_result.data(), d_A, N * N * sizeof(double)).wait();

    // Check if the result is an identity matrix
    EXPECT_TRUE(isEqualMatrix(h_result, h_invA, N));

    sycl::free(d_A, myQ);
}




// Test case for matrixMatrixProduct
TEST(gpu_manager_test, matrixMatrixProduct) {
auto myQ = GPUHandle::syclsolver();
    constexpr int size = 2;

    // Input matrices
    double h_A[size * size] = {1.0, 2.0, 
                               3.0, 4.0};
    double h_B[size * size] = {5.0, 6.0, 
                               7.0, 8.0};
    double expected_C[size * size] = {19.0, 22.0, 
                                      43.0, 50.0}; // A * B

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    d_A = sycl::malloc_device<double>(size * size, myQ);
    d_B = sycl::malloc_device<double>(size * size, myQ);
    d_C = sycl::malloc_device<double>(size * size, myQ);

    // Copy input matrices to device
    myQ.memcpy(d_A, h_A, size * size * sizeof(double));
    myQ.memcpy(d_B, h_B, size * size * sizeof(double)).wait();

    // Perform matrix multiplication (C = A * B)
    matrixMatrixProduct(d_A, d_B, d_C, size, false, false, false);

    // Copy result back to host
    std::vector<double> h_C(size * size);
    copyToHost(h_C, d_C, size * size);

    // Verify result
    for (int i = 0; i < size * size; ++i) {
        EXPECT_NEAR(h_C[i], expected_C[i], 1e-6);
    }

    // Cleanup
    sycl::free(d_A, myQ);
    sycl::free(d_B, myQ);
    sycl::free(d_C, myQ);
}

TEST(gpu_manager_test, matrixMatrixProduct_transposed) {
auto myQ = GPUHandle::syclsolver();
    constexpr int size = 2;

    // Input matrices
    double h_A[size * size] = {1.0, 2.0, 
                               3.0, 4.0};
    double h_B[size * size] = {5.0, 6.0, 
                               7.0, 8.0};
    double expected_C[size * size] = {26.0, 30.0, 
                                      38.0, 44.0}; // A^T * B

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    d_A = sycl::malloc_device<double>(size * size, myQ);
    d_B = sycl::malloc_device<double>(size * size, myQ);
    d_C = sycl::malloc_device<double>(size * size, myQ);

    // Copy input matrices to device
    myQ.memcpy(d_A, h_A, size * size * sizeof(double));
    myQ.memcpy(d_B, h_B, size * size * sizeof(double)).wait();

    // Perform matrix multiplication (C = A^T * B)
    matrixMatrixProduct(d_A, d_B, d_C, size, true, false, false);

    // Copy result back to host
    std::vector<double> h_C(size * size);
    copyToHost(h_C, d_C, size * size);

    // Verify result
    for (int i = 0; i < size * size; ++i) {
        EXPECT_NEAR(h_C[i], expected_C[i], 1e-6);
    }

    // Cleanup
    sycl::free(d_A, myQ);
    sycl::free(d_B, myQ);
    sycl::free(d_C, myQ);
}




// Test case for weightedMatrixSum
TEST(gpu_manager_test, WeightedMatrixSum) {
auto myQ = GPUHandle::syclsolver();
    constexpr int size = 2;

    // Input matrices
    double h_A[size * size] = {1.0, 2.0, 
                               3.0, 4.0};
    double h_B[size * size] = {5.0, 6.0, 
                               7.0, 8.0};
    double expected_C[size * size] = {4.5, 7.0, 
                                      9.5, 12.0}; // 2.0 * A + 0.5 * B

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    d_A = sycl::malloc_device<double>(size * size, myQ);
    d_B = sycl::malloc_device<double>(size * size, myQ);
    d_C = sycl::malloc_device<double>(size * size, myQ);

    // Copy input matrices to device
    myQ.memcpy(d_A, h_A, size * size * sizeof(double));
    myQ.memcpy(d_B, h_B, size * size * sizeof(double)).wait();

    // Perform weighted sum: C = 2.0 * A + 0.5 * B
    weightedMatrixSum(d_A, d_B, d_C, 2.0, 0.5, size);

    // Copy result back to host
    std::vector<double> h_C(size * size);
    copyToHost(h_C, d_C, size * size);

    // Verify result
    for (int i = 0; i < size * size; ++i) {
        EXPECT_NEAR(h_C[i], expected_C[i], 1e-6);
    }

    // Cleanup
    sycl::free(d_A, myQ);
    sycl::free(d_B, myQ);
    sycl::free(d_C, myQ);
}

// Test case for matrixAddition
TEST(gpu_manager_test, matrixAddition) {
auto myQ = GPUHandle::syclsolver();
    constexpr int size = 2;

    // Input matrices
    double h_A[size * size] = {1.0, 2.0, 
                               3.0, 4.0};
    double h_B[size * size] = {5.0, 6.0, 
                               7.0, 8.0};
    double expected_C[size * size] = {6.0, 8.0, 
                                      10.0, 12.0}; // A + B

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    d_A = sycl::malloc_device<double>(size * size, myQ);
    d_B = sycl::malloc_device<double>(size * size, myQ);
    d_C = sycl::malloc_device<double>(size * size, myQ);

    // Copy input matrices to device
    myQ.memcpy(d_A, h_A, size * size * sizeof(double));
    myQ.memcpy(d_B, h_B, size * size * sizeof(double)).wait();

    // Perform addition: C = A + B
    matrixAddition(d_A, d_B, d_C, size);

    // Copy result back to host
    std::vector<double> h_C(size * size);
    copyToHost(h_C, d_C, size * size);

    // Verify result
    for (int i = 0; i < size * size; ++i) {
        EXPECT_NEAR(h_C[i], expected_C[i], 1e-6);
    }

    // Cleanup
    sycl::free(d_A, myQ);
    sycl::free(d_B, myQ);
    sycl::free(d_C, myQ);
}

// Test case for matrixSubtraction
TEST(gpu_manager_test, matrixSubtraction) {
auto myQ = GPUHandle::syclsolver();
    constexpr int size = 2;

    // Input matrices
    double h_A[size * size] = {1.0, 2.0, 
                               3.0, 4.0};
    double h_B[size * size] = {5.0, 6.0, 
                               7.0, 8.0};
    double expected_C[size * size] = {-4.0, -4.0, 
                                      -4.0, -4.0}; // A - B

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    d_A = sycl::malloc_device<double>(size * size, myQ);
    d_B = sycl::malloc_device<double>(size * size, myQ);
    d_C = sycl::malloc_device<double>(size * size, myQ);

    // Copy input matrices to device
    myQ.memcpy(d_A, h_A, size * size * sizeof(double));
    myQ.memcpy(d_B, h_B, size * size * sizeof(double)).wait();

    // Perform subtraction: C = A - B
    matrixSubtraction(d_A, d_B, d_C, size);

    // Copy result back to host
    std::vector<double> h_C(size * size);
    copyToHost(h_C, d_C, size * size);

    // Verify result
    for (int i = 0; i < size * size; ++i) {
        EXPECT_NEAR(h_C[i], expected_C[i], 1e-6);
    }

    // Cleanup
    sycl::free(d_A, myQ);
    sycl::free(d_B, myQ);
    sycl::free(d_C, myQ);
}


// Test case: Perform Cholesky decomposition on a symmetric positive definite matrix
TEST(gpu_manager_test, choleskyDecomposition) {
auto myQ = GPUHandle::syclsolver();
    constexpr int N = 3;

    // Symmetric Positive Definite (SPD) matrix (row-major order)
    double h_A[N * N] = {
        25, 15, -5,
        15, 18,  0,
        -5,  0, 11
    };

    // Expected lower triangular matrix L (result of decomposition)
    double h_expected_L[N * N] = {
        5, 0, 0,
        3, 3, 0,
        -1, 1, 3
    };

    // Allocate device memory
    double *d_A;
    d_A = sycl::malloc_device<double>(N * N, myQ);

    // Copy matrix to device
    myQ.memcpy(d_A, h_A, N * N * sizeof(double)).wait();

    // Perform Cholesky decomposition
    choleskyDecomposition(d_A, N);

    // Copy result back to host
    double h_L[N * N];
    myQ.memcpy(h_L, d_A, N * N * sizeof(double)).wait();

    // Validate results
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <N; ++j) {
            EXPECT_NEAR(h_L[i * N + j], h_expected_L[i * N + j], 1e-6);
        }
    }

    // Free allocated device memory
    sycl::free(d_A, myQ);
}


// Test case: Compute the trace of a matrix
TEST(gpu_manager_test, ComputeMatrixTrace) {
auto myQ = GPUHandle::syclsolver();
    constexpr int N = 4;

    // Define a test matrix (row-major order)
    real_t h_matrix[N * N] = {
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };

    // Expected trace value (sum of diagonal elements: 1 + 6 + 11 + 16)
    real_t expected_trace = 34.0;

    // Allocate device memory
    real_t *d_matrix;
    d_matrix = sycl::malloc_device<real_t>(N * N, myQ);

    // Copy matrix to device
    myQ.memcpy(d_matrix, h_matrix, N * N * sizeof(real_t)).wait();

    // Compute the trace
    real_t computed_trace = computeMatrixTrace(d_matrix, N);

    // Validate the result
    EXPECT_NEAR(computed_trace, expected_trace, 1e-6);

    // Free allocated device memory
    sycl::free(d_matrix, myQ);
}


// Test case: Damping function correctly updates the Fock matrix
TEST(gpu_manager_test, damping) {
auto myQ = GPUHandle::syclsolver();
    constexpr int N = 3;
    constexpr real_t alpha = 0.1;

    // Define test matrices (row-major order)
    real_t h_matrix_old[N * N] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };

    real_t h_matrix_new[N * N] = {
        9.0, 8.0, 7.0,
        6.0, 5.0, 4.0,
        3.0, 2.0, 1.0
    };

    // Expected matrix after damping:
    // F_new = (1-alpha) * F_old + alpha * F_new
    real_t expected_matrix[N * N] = {
        1.8, 2.6, 3.4,
        4.2, 5.0, 5.8,
        6.6, 7.4, 8.2
    };

    // Allocate device memory
    real_t *d_matrix_old, *d_matrix_new;
    d_matrix_old = sycl::malloc_device<real_t>(N * N, myQ);
    d_matrix_new = sycl::malloc_device<real_t>(N * N, myQ);

    // Copy matrices to device
    myQ.memcpy(d_matrix_old, h_matrix_old, N * N * sizeof(real_t));
    myQ.memcpy(d_matrix_new, h_matrix_new, N * N * sizeof(real_t)).wait();

    // Apply damping
    damping(d_matrix_old, d_matrix_new, alpha, N);

    // Copy result back to host
    real_t h_result[N * N];
    real_t h_result_old[N * N];
    myQ.memcpy(h_result, d_matrix_new, N * N * sizeof(real_t));
    myQ.memcpy(h_result_old, d_matrix_old, N * N * sizeof(real_t)).wait();

    // Validate the result
    for (int i = 0; i < N * N; i++) {
        EXPECT_NEAR(h_result[i], expected_matrix[i], 1e-6);
    }
    for (int i = 0; i < N * N; i++) {
        EXPECT_NEAR(h_result_old[i], expected_matrix[i], 1e-6);
    }

    // Free allocated device memory
    sycl::free(d_matrix_old, myQ);
    sycl::free(d_matrix_new, myQ);
}



// Test case: Test the computation of the optimal damping factor for RHF
TEST(gpu_manager_test, computeOptimalDampingFactor_RHF) {
auto myQ = GPUHandle::syclsolver();
    constexpr int N = 3;
    
    // Define matrices (Fock, previous Fock, density, previous density)
    real_t h_fock_matrix[N * N] = {
        2.0, 2.0, 1.0,
        2.0, 3.0, 1.0,
        1.0, 1.0, 2.0
    };
    
    real_t h_prev_fock_matrix[N * N] = {
        1.0, 2.0, 3.0,
        4.0, 6.0, 6.0,
        7.0, 8.0, 10.0
    };
    
    real_t h_density_matrix[N * N] = {
        3.0, 3.0, 3.0,
        3.0, 3.0, 3.0,
        3.0, 3.0, 3.0
    };

    real_t h_prev_density_matrix[N * N] = {
        2.0, 4.0, 6.0,
        4.0, 1.0, 1.0,
        6.0, 1.0, 10.0
    };

    real_t expected_alpha = 0.613207547;

    // Allocate device memory
    real_t *d_fock_matrix, *d_prev_fock_matrix, *d_density_matrix, *d_prev_density_matrix;
    d_fock_matrix = sycl::malloc_device<real_t>(N * N, myQ);
    d_prev_fock_matrix = sycl::malloc_device<real_t>(N * N, myQ);
    d_density_matrix = sycl::malloc_device<real_t>(N * N, myQ);
    d_prev_density_matrix = sycl::malloc_device<real_t>(N * N, myQ);

    // Copy matrices to device
    myQ.memcpy(d_fock_matrix, h_fock_matrix, N * N * sizeof(real_t));
    myQ.memcpy(d_prev_fock_matrix, h_prev_fock_matrix, N * N * sizeof(real_t));
    myQ.memcpy(d_density_matrix, h_density_matrix, N * N * sizeof(real_t));
    myQ
        .memcpy(d_prev_density_matrix, h_prev_density_matrix,
                N * N * sizeof(real_t))
        .wait();

    // Call the function to compute the optimal damping factor
    real_t alpha = computeOptimalDampingFactor_RHF(d_fock_matrix, d_prev_fock_matrix, d_density_matrix, d_prev_density_matrix, N);

    // Check if the damping factor is within an expected range
    EXPECT_NEAR(alpha, expected_alpha, 1e-6);

    // Free device memory
    sycl::free(d_fock_matrix, myQ);
    sycl::free(d_prev_fock_matrix, myQ);
    sycl::free(d_density_matrix, myQ);
    sycl::free(d_prev_density_matrix, myQ);
}
