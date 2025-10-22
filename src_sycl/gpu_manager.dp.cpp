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
#include <dpct/dpct.hpp>
#include "gpu_manager.hpp"
#include "int1e.hpp"
#include "int2e.hpp"
#include "utils.hpp" // THROW_EXCEPTION
#include "int2c2e.hpp"
#include "int3c2e.hpp"

#include <vector>    // std::vector
#include <tuple>     // std::tuple
#include <algorithm> // std::reverse
#include <fstream>
#include <dpct/lib_common_utils.hpp>

//#include <dpct/blas_utils.hpp>

#include <thread>

namespace gansu::gpu{






/**
 * @brief Performs eigenvalue decomposition on a symmetric matrix.
 * 
 * This function computes the eigenvalues and eigenvectors of the matrix, using the cuSOLVER library.
 *
 * @param d_matrix Device pointer to the input symmetric matrix.
 * @param d_eigenvalues Device pointer to store the eigenvalues.
 * @param d_eigenvectors Device pointer to store the eigenvectors.
 * @param size Size of the matrix (size x size).
 * @return Error status (0 if successful).
 * @details Since the eigenvectors are stored in the same memory as the input matrix, the input matrix is copied to a temporary matrix before.
 */
int eigenDecomposition(const real_t *d_matrix, real_t *d_eigenvalues,
                       real_t *d_eigenvectors, const int size) try {
//    dpct::device_ext &dev_ct1 = dpct::get_current_device();
//    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    //cusolverManager cusolver;
    sycl::queue& syclsolverHandle = GPUHandle::syclsolver();
//    int cusolverParams = GPUHandle::cusolverParams();

    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;
    real_t* d_workspace=nullptr;
    real_t* h_workspace=nullptr;

    sycl::event ev;

    // Query the workspace sizes of the device and host memory
try {
    workspaceInBytesOnDevice = oneapi::mkl::lapack::syevd_scratchpad_size<real_t>(
        syclsolverHandle,
        oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, size,
        size
        );
}
catch (oneapi::mkl::lapack::invalid_argument const& e) {
    std::cerr << "Invalid argument in syevd_scratchpad_size: "
              << e.what() << std::endl;
    return -1;
}
catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
    return -2;
}
catch (std::exception const& e) {
    std::cerr << "Other exception: " << e.what() << std::endl;
    return -3;
}

    // workspace allocation
try {
    d_workspace = sycl::malloc_device<real_t>(workspaceInBytesOnDevice, syclsolverHandle);
}
catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
    if (e.code() == sycl::errc::memory_allocation) {
        std::cerr << "Failed to allocate device memory for workspace " << std::endl;
    }
}

    // temporary matrix allocation for d_matrix since the eigenvectors will be stored in the same memory of d_matrix
    real_t * d_temp_matrix;
try {
     d_temp_matrix = sycl::malloc_device<real_t>(size * size, syclsolverHandle);
}
catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
    if (e.code() == sycl::errc::memory_allocation) {
        std::cerr << "Failed to allocate device memory for temporary matrix " << std::endl;
    }
}

    // copy the d_matrix since the eigenvectors will be stored in the same memory
    syclsolverHandle.memcpy(d_temp_matrix, d_matrix, size * size * sizeof(real_t)).wait();

    // Perform eigenvalue decomposition
try {
    ev = oneapi::mkl::lapack::syevd( syclsolverHandle,
        oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, size,
        d_temp_matrix, size,
        d_eigenvalues,
        d_workspace,
        workspaceInBytesOnDevice
    );
 // 実際の実行完了を待つ
    ev.wait_and_throw();
}
catch (oneapi::mkl::lapack::invalid_argument const& e) {
    std::cerr << "Invalid argument in syevd: " << e.what() << std::endl;
}
catch (sycl::exception const& e) {
    std::cerr << "SYCL exception during syevd: " << e.what() << std::endl;
}

    // Copy the eigenvectors to d_eigenvectors
    syclsolverHandle.memcpy(d_eigenvectors, d_temp_matrix, size * size * sizeof(real_t)).wait();

    // transpose the eigenvectors since the eigenvectors are stored by column-major order
//    transposeMatrixInPlace(d_eigenvectors, size);
// 転置処理：A → B（転置＋スケーリング係数 1.0）
try {
    ev = oneapi::mkl::blas::row_major::omatcopy( syclsolverHandle,
        oneapi::mkl::transpose::trans, size, size,
        1.0,                           // スケーリング係数
        d_temp_matrix, size,
        d_eigenvectors,
        size
    );
 // 実際の実行完了を待つ
    ev.wait_and_throw();
}
catch (oneapi::mkl::lapack::invalid_argument const& e) {
    std::cerr << "Invalid argument in omatcopy: " << e.what() << std::endl;
}
catch (sycl::exception const& e) {
    std::cerr << "SYCL exception during omatcopy: " << e.what() << std::endl;
}

    // transpose the eigenvectors since the eigenvectors are stored by column-major order
    // return the error status
    int h_info = 0;

//    if (err == 0) h_info = 0;
    syclsolverHandle.wait_and_throw();
//    q_ct1.memcpy(&h_info, d_info, sizeof(int)).wait();

    // free the temporary memory
    sycl::free(d_temp_matrix, syclsolverHandle);
    sycl::free(d_workspace, syclsolverHandle);
//    sycl::free(h_workspace, syclsolverHandle);
//    sycl::free(d_info, syclsolverHandle);

    return h_info; // 0 if successful
}

catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
//  std::exit(1);
    return(1);
}

/**
 * @brief Computes the product of a matrix and a matrix using cuBLAS.
 * 
 * This function computes the product of a matrix and a matrix using cuBLAS.
 * 
 * @param d_matrix_A Device pointer to the N x N matrix stored by row-major order
 * @param d_matrix_B Device pointer to the N x N matrix stored by row-major order
 * @param d_matrix_C Device pointer to store the result stored by row-major order
 * @param size Size of the matrix (size x size)
 * @param transpose_A Flag to transpose matrix A, default is false
 * @param transpose_B Flag to transpose matrix B, default is false
 * @param initialize_C_to_zero Flag to initialize the matrix C to zero before the computation, default is true. If false, the matrix C is added to the product.
 * @details The matrix product is computed as \f$ C += AB \f$.
 * @details If the flag initialize_C_to_zero is true, the matrix C is initialized to zero before the computation.
 */
 void matrixMatrixProduct(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const int size, const bool transpose_A, const bool transpose_B, const bool accumulate){
    //cublasManager cublas;
    sycl::queue& syclsolverHandle = GPUHandle::syclsolver();
    sycl::event ev;

    const real_t alpha = 1.0;
    const real_t beta = 0.0;

    if (!accumulate){
        syclsolverHandle.memset(d_matrix_C, 0, size * size * sizeof(real_t)).wait();
    }

    const oneapi::mkl::transpose transA =
        (transpose_A) ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
    const oneapi::mkl::transpose transB =
        (transpose_B) ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
/*
    const oneapi::mkl::transpose transA =
        (transpose_A) ? CUBLAS_OP_T : CUBLAS_OP_N;
    const oneapi::mkl::transpose transB =
        (transpose_B) ? CUBLAS_OP_T : CUBLAS_OP_N;
*/
    // Perform matrix matrix multipication
try {
     ev = oneapi::mkl::blas::column_major::gemm(
        syclsolverHandle,
        transB, transA,
        size, size, size,
        alpha,
        d_matrix_B, size,
        d_matrix_A, size,
        beta,
        d_matrix_C, size
    );
 // 実際の実行完了を待つ
    ev.wait_and_throw();
}
catch (oneapi::mkl::lapack::invalid_argument const& e) {
    std::cerr << "Invalid argument in gemm: " << e.what() << std::endl;
}
catch (sycl::exception const& e) {
    std::cerr << "SYCL exception during gemm: " << e.what() << std::endl;
}
}

/**
 * @brief Computes the weighted sum of two matrices using cuBLAS.
 * @param d_matrix_A Device pointer to the size x size matrix
 * @param d_matrix_B Device pointer to the size x size matrix
 * @param d_matrix_C Device pointer to store the result
 * @param weight_A Weight of the matrix A
 * @param weight_B Weight of the matrix B
 * @param size Size of the matrix (size x size)
 * @details The matrix weighted sum is computed as \f$ C = \alpha A + \beta B \f$.
 */
void weightedMatrixSum(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const double weight_A, const double weight_B, const int size) {
    //cublasManager cublas;
   sycl::queue& syclsolverHandle = GPUHandle::syclsolver();
    sycl::event ev;

    const real_t alpha = weight_A;
    const real_t beta = weight_B;

    // Perform weighted matrix sum
try {
     ev = oneapi::mkl::blas::column_major::omatadd(
        syclsolverHandle,
        oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        size, size,
        alpha, d_matrix_A, size,
// May be "dpct::get_value(&alpha, q_ct1)"
        beta, d_matrix_B, size,
        d_matrix_C, size
    );
    ev.wait_and_throw();
}
catch (oneapi::mkl::lapack::invalid_argument const& e) {
    std::cerr << "Invalid argument in omatadd: " << e.what() << std::endl;
}
catch (sycl::exception const& e) {
    std::cerr << "SYCL exception during omatadd: " << e.what() << std::endl;
}
/*
    cublasDgeam(
        cublasHandle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size, size,
        &alpha, d_matrix_A, size,
        &beta, d_matrix_B, size,
        d_matrix_C, size
    );
*/
}


/**
* @brief Computes the addition of two matrices using cuBLAS.
* @param d_matrix_A Device pointer to the size x size matrix
* @param d_matrix_B Device pointer to the size x size matrix
* @param d_matrix_C Device pointer to store the result
* @param size Size of the matrix (size x size)
* @details The matrix subtraction is computed as \f$ C = A + B \f$.
*/
void matrixAddition(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const int size) {
   weightedMatrixSum(d_matrix_A, d_matrix_B, d_matrix_C, 1.0, 1.0, size);
}


/**
 * @brief Computes the subtraction of two matrices using cuBLAS.
 * @param d_matrix_A Device pointer to the size x size matrix
 * @param d_matrix_B Device pointer to the size x size matrix
 * @param d_matrix_C Device pointer to store the result
 * @param size Size of the matrix (size x size)
 * @details The matrix subtraction is computed as \f$ C = A - B \f$.
 */
void matrixSubtraction(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const int size) {
    weightedMatrixSum(d_matrix_A, d_matrix_B, d_matrix_C, 1.0, -1.0, size);
}

/**
 * @brief Computes the inner product of two vectors using cuBLAS.
 * @param d_vector_A Device pointer to the vector A
 * @param d_vector_B Device pointer to the vector B
 * @param size Size of the vector
 * @return The inner product of the two vectors
 * @details The inner product is computed as \f$ result = \sum_{i=1}^{size} A_i B_i \f$.
 */
double innerProduct(const double* d_vector_A, const double* d_vector_B, const int size) {
    //cublasManager cublas;
    sycl::queue& syclsolverHandle = GPUHandle::syclsolver();

    real_t result;
    real_t * d_result = sycl::malloc_device<real_t>(1, syclsolverHandle);
    oneapi::mkl::blas::column_major::dot(
        syclsolverHandle,
        size,
        d_vector_A, 1,
        d_vector_B, 1,
        d_result
    );
    syclsolverHandle.memcpy(&result, d_result, sizeof(real_t)).wait();
    sycl::free(d_result, syclsolverHandle);
/*
    cublasDdot(
        cublasHandle,
        size,
        d_vector_A, 1,
        d_vector_B, 1,
        &result
    );
*/
    return result;
}


/**
 * @brief Computes the inverse of the square root of the vector.
 * 
 * This function computes the inverse of the square root of each value of the vector.
 * 
 * @param d_vectors Device pointer to the vector.
 * @param size Number of the vector.
 */
void invertSqrtElements(real_t* d_vectors, const size_t size, const double threshold) {
    sycl::queue& workq = GPUHandle::syclsolver();
    size_t blockSize = 256;
    size_t numBlocks = (size + blockSize - 1) / blockSize;
    /*
    DPCT1049:56: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        workq.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(numBlocks) * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
            [=](sycl::nd_item<1> item_ct1) {
                inverseSqrt_kernel(d_vectors, size, threshold);
            }).wait();
    }
}

/**
 * @brief Transpose a matrix in place.
 * @param d_matrix Device pointer to the matrix
 * @param size Size of the matrix (size x size)
 * @details This function transposes a matrix in place using shared memory.
 * @details The size of the matrix is size x size.
 */
 void transposeMatrixInPlace(real_t* d_matrix, const int size) {
    sycl::queue& workq = GPUHandle::syclsolver();
    sycl::range<2> blockSize(WARP_SIZE, WARP_SIZE);
    sycl::range<2> gridSize((size + WARP_SIZE - 1) / WARP_SIZE,
                        (size + WARP_SIZE - 1) / WARP_SIZE);
    /*
    DPCT1049:57: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
try {
    auto event = workq.submit([&](sycl::handler& cgh){
    sycl::local_accessor<real_t, 2> s_src(blockSize, cgh);
    sycl::local_accessor<real_t, 2> s_dst(blockSize, cgh);

    cgh.parallel_for(
//    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<2>(gridSize * blockSize, blockSize),
        [=](sycl::nd_item<2> item_ct1) {
            transposeMatrixInPlace_kernel(d_matrix, size, s_src, s_dst);
    });
    });
    event.wait_and_throw();
}
catch (sycl::exception const& e) {
    std::cerr << "transposeMatrixInPlace SYCL exception: " << e.what() << std::endl;
}
catch (std::exception const& e) {
    std::cerr << "transposeMatrixInPlace Other exception: " << e.what() << std::endl;
}
}

/**
 * @brief Make a diagonal matrix from the vector.
 * @param d_vector Device pointer to the vector of size size.
 * @param d_matrix Device pointer to store the diagonal matrix of size size x size.
 * @param size Size of the vector and the matrix.
 * @details This function creates a diagonal matrix, in which the diagonal elements are the elements of the vector.
 */
void makeDiagonalMatrix(const real_t* d_vector, real_t* d_matrix, const int size) {
    //cublasManager cublas;
    sycl::queue& syclsolverHandle = GPUHandle::syclsolver();

    // Set the matrix to zero
    syclsolverHandle.memset(d_matrix, 0, size * size * sizeof(real_t)).wait();

    // Set the diagonal elements to the eigenvalues
    oneapi::mkl::blas::column_major::copy(syclsolverHandle, size, d_vector, 1, d_matrix, size + 1).wait();
}

/**
 * @brief Compute the trace of a matrix (the sum of the diagonal elements)
 * @param d_matrix Device pointer to the matrix
 * @param size Size of the matrix (size x size)
 * @return Trace of the matrix (the sum of the diagonal elements)
 */
 real_t computeMatrixTrace(const real_t *d_matrix, const int size) try {
  sycl::queue& syclsolverHandle = GPUHandle::syclsolver();
    if(size > 1024){ // 1024 is the maximum number of threads per block. If the size is larger than 1024, two or more blocks are required.
        THROW_EXCEPTION("Too many basis functions.");
    }

//    dpct::err0 err;
    real_t zero = 0;
    real_t* d_trace = sycl::malloc_device<real_t>(1, syclsolverHandle);
    syclsolverHandle.memcpy(d_trace, &zero, sizeof(real_t)).wait();

    real_t h_trace = 0.0;

    /*
    DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
//    void getMatrixTrace(queue& q, const float* d_matrix, float* d_trace, int num_basis) {
    syclsolverHandle.submit([&](sycl::handler& cgh) {
        auto red = sycl::reduction(d_trace, sycl::plus<>());
        cgh.parallel_for(sycl::range<1>(size), red, [=](sycl::id<1> i, auto& sum) {
            int idx = i[0];
            sum += d_matrix[idx * size + idx];
        });
    }).wait();
    syclsolverHandle.memcpy(&h_trace, d_trace, sizeof(real_t)).wait();

    sycl::free(d_trace,syclsolverHandle);
    return h_trace;
}
 catch (sycl::exception const &exc) {
   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
             << ", line:" << __LINE__ << std::endl;
   std::exit(1);
 }

/**
 * @brief Compute Core Hamiltonian Matrix (one electron integrals)
 * @param shell_type_infos Information of the shell types
 * @param d_primitive_shells Device pointer to the primitive shells
 * @param d_boys_grid Device pointer to the grid values of the Boys function
 * @param d_cgto_normalization_factors Device pointer to the normalization factors of the contracted Gaussian-type orbitals
 * @param d_overlap_matrix Device pointer to the overlap matrix to store the result
 * @param d_core_hamiltonian_matrix Device pointer to the core Hamiltonian matrix to store the result
 * @param num_atoms Number of atoms
 * @param num_basis Number of basis functions
 * @details This function computes the core Hamiltonian matrix and the overlap matrix.
 */
void computeCoreHamiltonianMatrix(
    const std::vector<ShellTypeInfo> &shell_type_infos, Atom *d_atoms,
    PrimitiveShell *d_primitive_shells, real_t *d_boys_grid,
    real_t *d_cgto_normalization_factors, real_t *d_overlap_matrix,
    real_t *d_core_hamiltonian_matrix, const int num_atoms, const int num_basis,
    const bool verbose) try {
//  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
//    sycl::device wk_dev{sycl::default_selector_v};
//    sycl::context wk_ctx{wk_dev} ;

    sycl::queue& workq = GPUHandle::syclsolver();
    sycl::device work_dev = workq.get_device();
    sycl::context wk_ctx{work_dev} ;
    // compute the core Hamiltonian matrix
    const int threads_per_block = 128; // the number of threads per block

    const int shell_type_count = shell_type_infos.size();

    workq.memset(d_overlap_matrix, 0, sizeof(real_t) * num_basis * num_basis)
        .wait();
    workq
        .memset(d_core_hamiltonian_matrix, 0,
                sizeof(real_t) * num_basis * num_basis)
        .wait();

    // make multi stream
    const int N = (shell_type_count)*(shell_type_count+1) /2;
    std::vector<sycl::queue> streams;
    streams.reserve(N);
    std::vector<sycl::queue> V_streams;
    V_streams.reserve(N);
//    std::vector<dpct::queue_ptr> streams(N);
//    std::vector<dpct::queue_ptr> V_streams(N);

    for (int i = 0; i < N; i++) {
        streams.emplace_back(wk_ctx, work_dev);
        V_streams.emplace_back(wk_ctx, work_dev);
    }

    // Call the kernel functions from (s0|s1),... (e.g. (f|f), (d|f), (d|d), (s|d), (p|d), (d|d) for s, p, d, f shells)
    for (int s0 = shell_type_count-1; s0 >= 0; s0--) {
        for (int s1 = shell_type_count-1; s1 >= s0; s1--) {
            const ShellTypeInfo shell_s0 = shell_type_infos[s0];
            const ShellTypeInfo shell_s1 = shell_type_infos[s1];

            const int num_shell_pairs = (s0==s1) ? (shell_s0.count*(shell_s0.count+1)/2) : (shell_s0.count*shell_s1.count); // the number of pairs of primitive shells = the number of threads
            const int num_blocks = (num_shell_pairs + threads_per_block - 1) / threads_per_block; // the number of blocks

            if(verbose){
                std::cout << "(" << shell_type_to_shell_name(s0) << "|" << shell_type_to_shell_name(s1) << "): ";
                std::cout << "|" << shell_type_to_shell_name(s0) << "|=" << shell_s0.count << ", ";
                std::cout << "|" << shell_type_to_shell_name(s1) << "|=" << shell_s1.count << ", ";
                std::cout << "|[a|b]|=" << num_shell_pairs << ", ";
                std::cout << "num_blocks: " << num_blocks << std::endl;
            }

            int index = (2*(shell_type_count-1)-s0+1)*s0 / 2 + s1;
            // printf("(s0,s1) = (%d, %d), idx = %d\n", s0, s1, index);


            // call the kernel functions
//    dpct::dim3 blocks(num_blocks,1,1);
//    dpct::dim3 threads(threads_per_block,1,1);
    sycl::range<3> blocks(1, 1, num_blocks);
    sycl::range<3> threads(1, 1, threads_per_block);

//launch_overlap_kinetic_kernel( int a, int b, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis)

            streams[index].submit([&](sycl::handler& cgh){
//            sycl::event e1 = streams[index].submit([&](sycl::handler& cgh){
            cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                launch_overlap_kinetic_kernel(
                    s0, s1, d_overlap_matrix,
                    d_core_hamiltonian_matrix, d_primitive_shells,
                    d_cgto_normalization_factors, shell_s0, shell_s1,
                    num_shell_pairs, num_basis);
            });
            });
/*
            dpct::kernel_launcher::launch(
                launch_overlap_kinetic_kernel, num_blocks,
                threads_per_block, 0, streams[index],
                s0, s1, d_overlap_matrix,
                d_core_hamiltonian_matrix, d_primitive_shells,
                d_cgto_normalization_factors, shell_s0, shell_s1,
                num_shell_pairs, num_basis);
*/
            V_streams[index].submit([&](sycl::handler& cgh){
//            cgh.depends_on(e1); //明示的依存
            cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                launch_nuclear_attraction_kernel(
                    s0, s1, d_core_hamiltonian_matrix, d_primitive_shells,
                    d_cgto_normalization_factors, d_atoms, num_atoms, shell_s0,
                    shell_s1, num_shell_pairs, num_basis, d_boys_grid);
            });
            });
/*
            dpct::kernel_launcher::launch(
                launch_nuclear_attraction_kernel, num_blocks,
                threads_per_block, 0, V_streams[index],
                s0, s1, d_core_hamiltonian_matrix, d_primitive_shells,
                d_cgto_normalization_factors, d_atoms, num_atoms, shell_s0,
                shell_s1, num_shell_pairs, num_basis, d_boys_grid);
*/
        }
    }
    // syncronize streams
    for (int i = 0; i < N; i++) {
        streams[i].wait();
        V_streams[i].wait();
    }
//    work_dev.queues_wait_and_throw();

//    dpct::dim3 blocks(int((num_basis + 31) / 32), int((num_basis + 31) / 32));
//    dpct::dim3 threads(32, 32);
    sycl::range<3> blocks(1, int((num_basis + 31) / 32), int((num_basis + 31) / 32));
    sycl::range<3> threads(1, 32, 32);
    /*
    DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
//    dpct::get_default_queue().submit([&](sycl::handler& cgh){
    workq.submit([&](sycl::handler& cgh){
    sycl::local_accessor<real_t, 2> s_mem(sycl::range<2>(32,33), cgh);
    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           Matrix_Symmetrization(d_overlap_matrix, num_basis, s_mem);
                       });
    });
    /*
    DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    workq.submit([&](sycl::handler& cgh){
    sycl::local_accessor<real_t, 2> s_mem(sycl::range<2>(32,33), cgh);
    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           Matrix_Symmetrization(d_core_hamiltonian_matrix,
                                                 num_basis, s_mem);
                       });
    });

    // destory streams
/*
    for (int i = 0; i < N; i++) {
        dev_ct1.destroy_queue(streams[i]);
        dev_ct1.destroy_queue(V_streams[i]);
    }
*/
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int get_index_2to1_horizontal(int i, int j, const int n)
{
    if (i > j) std::swap(i, j);
    return j - static_cast<int>(i * (i - 2 * n + 1) / 2);
}


size_t makeShellPairTypeInfo(const std::vector<ShellTypeInfo>& shell_type_infos, std::vector<ShellPairTypeInfo>& shell_pair_type_infos)
{
    // Make shell-pair type infos: 
    const int shell_type_count = shell_type_infos.size();

    size_t num_primitive_shell_pairs = 0;
    for (int s0 = 0; s0 < shell_type_count; ++s0) {
        for (int s1 = s0; s1 < shell_type_count; ++s1) {
            const ShellTypeInfo shell_s0 = shell_type_infos[s0];
            const ShellTypeInfo shell_s1 = shell_type_infos[s1];
            const size_t num_bra = (s0 == s1) ? shell_s0.count * (shell_s0.count + 1) / 2 : shell_s0.count * shell_s1.count;
            shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)] = {num_bra, num_primitive_shell_pairs};
            num_primitive_shell_pairs += num_bra;
        }
    }

    return num_primitive_shell_pairs;
}

void computeERIMatrix(
    const std::vector<ShellTypeInfo> &shell_type_infos,
    const std::vector<ShellPairTypeInfo> &shell_pair_type_infos,
    const PrimitiveShell *d_primitive_shells, const real_t *d_boys_grid,
    const real_t *d_cgto_normalization_factors, real_t *d_eri_matrix,
    const real_t *d_schwarz_upper_bound_factors,
    const real_t schwarz_screening_threshold, const int num_basis,
    const bool verbose) {
//  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//    sycl::device wk_dev{sycl::default_selector_v};
//    sycl::context wk_ctx{wk_dev} ;
    sycl::queue& workq = GPUHandle::syclsolver();
    sycl::device work_dev = workq.get_device();
    sycl::context wk_ctx{work_dev} ;

    // compute the electron repulsion integrals
    const int threads_per_block = 256; // the number of threads per block
    const int shell_type_count = shell_type_infos.size();

    // Call the kernel functions from (ss|ss),... (e.g. (ss|ss), (ss|sp), (ss|pp), (sp|sp), (sp|pp), (pp|pp) for s and p shells)

    // list shell-quadruples for sorted shell-type (s0, s1, s2, s3)
    std::vector<std::tuple<int, int, int, int>> shell_quadruples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < shell_type_count; ++c) {
                for (int d = c; d < shell_type_count; ++d) {
                    if (a < c || (a == c && b <= d)) {
                        shell_quadruples.emplace_back(a, b, c, d);
                    }
                }
            }
        }
    }
    // reverse the order of the shell_quadruples to make it sorted by (s0, s1, s2, s3)
    std::reverse(shell_quadruples.begin(), shell_quadruples.end());


    // make multi stream
    const int num_kernels = shell_quadruples.size();
//    std::vector<dpct::queue_ptr> streams(num_kernels);
    std::vector<sycl::queue> streams;
    streams.reserve(num_kernels);
    for (int i = 0; i < num_kernels; i++) {
        streams.emplace_back(wk_ctx, work_dev);
//        streams[i] = dev_ct1.create_queue();
    }

    // for-loop for sorted shell-type (s0, s1, s2, s3)
    int stream_id = 0;
    for(const auto& quadruple: shell_quadruples) {
        int s0, s1, s2, s3;
        std::tie(s0, s1, s2, s3) = quadruple;

        const ShellTypeInfo shell_s0 = shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = shell_type_infos[s1];
        const ShellTypeInfo shell_s2 = shell_type_infos[s2];
        const ShellTypeInfo shell_s3 = shell_type_infos[s3];

        const size_t num_bra = (s0==s1) ? shell_s0.count*(shell_s0.count+1)/2 : shell_s0.count*shell_s1.count;
        const size_t num_ket = (s2==s3) ? shell_s2.count*(shell_s2.count+1)/2 : shell_s2.count*shell_s3.count;
        const size_t num_braket = ((s0==s2) && (s1==s3)) ? num_bra*(num_bra+1)/2 : num_bra*num_ket; // equal to the number of threads
        const int num_blocks = (num_braket + threads_per_block - 1) / threads_per_block; // the number of blocks

        const size_t head_bra = shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)].start_index;
        const size_t head_ket = shell_pair_type_infos[get_index_2to1_horizontal(s2, s3, shell_type_count)].start_index;

//    dpct::dim3 blocks(num_blocks,1,1);
//    dpct::dim3 threads(threads_per_block,1,1);
    sycl::range<3> blocks(1, 1, num_blocks);
    sycl::range<3> threads(1, 1, threads_per_block);

            streams[stream_id++].submit([&](sycl::handler& cgh){
            cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                launch_eri_kernel(
                  s0, s1, s2, s3,
                  d_eri_matrix, d_primitive_shells,
                  d_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
                  shell_s3, num_braket, schwarz_screening_threshold,
                  d_schwarz_upper_bound_factors, num_basis, d_boys_grid, head_bra,
                  head_ket);
            });
            });
/*
        dpct::kernel_launcher::launch(
            gpu::launch_eri_kernel, num_blocks, threads_per_block,
            0, streams[stream_id++], s0, s1, s2, s3,
            d_eri_matrix, d_primitive_shells,
            d_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            shell_s3, num_braket, schwarz_screening_threshold,
            d_schwarz_upper_bound_factors, num_basis, d_boys_grid, head_bra,
            head_ket);
*/
        if(verbose){
            std::cout << "(" << shell_type_to_shell_name(s0) << shell_type_to_shell_name(s1) << "|" << shell_type_to_shell_name(s2) << shell_type_to_shell_name(s3) << "): ";
            std::cout << "|" << shell_type_to_shell_name(s0) << "|=" << shell_s0.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s1) << "|=" << shell_s1.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s2) << "|=" << shell_s1.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s3) << "|=" << shell_s1.count << ", ";
            std::cout << "|bra|= " << num_bra << ", " ;
            std::cout << "|ket|= " << num_ket << ", " ;
            std::cout << "|braket|= " << num_braket << ", " ;
            std::cout << "num_blocks: " << num_blocks << std::endl;
        }
    }

    // syncronize streams
    for (int i = 0; i <num_kernels ; i++) {
        streams[i].wait();
    }
//    dev_ct1.queues_wait_and_throw();

    // destory streams
//    for (int i = 0; i < num_kernels; i++) {
//        dev_ct1.destroy_queue(streams[i]);
//    }
}



/**
 * @brief Computes the coefficient matrix from the Fock matrix and the transformation matrix.
 * @param d_fock_matrix Device pointer to the Fock matrix
 * @param d_transform_matrix Device pointer to the transformation matrix
 * @param d_coefficient_matrix Device pointer to store the coefficient matrix
 * @param num_basis Number of basis functions
 * @param d_orbital_energies Device pointer to store the orbital energies, default is nullptr. If nullptr, the orbital energies are stored in the temporary memory allcated inside, otherwise, the orbital energies are stored in the given device memory.
 * @details This function computes the coefficient matrix using the eigenvectors of the Fock matrix by solving the generalized eigenvalue problem \f$FC = SCE \f$.
 * @details To transform the generalized eigenvalue problem to the standard eigenvalue problem \f$FC = CE \f$, the transformation matrix.
 */
void computeCoefficientMatrix(const real_t *d_fock_matrix,
                              const real_t *d_transform_matrix,
                              real_t *d_coefficient_matrix, const int num_basis,
                              real_t *d_orbital_energies) try {
//  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::queue& workq = GPUHandle::syclsolver();
    // allocate temporary memory
    real_t* d_tempMatrix = nullptr;
    real_t* d_tempSymFockMatrix = nullptr;
    real_t* d_tempEigenvectors = nullptr;
    real_t* d_tempEigenvalues = nullptr; // if d_orbital_energies is nullptr, the eigenvalues are stored in d_tempEigenvalues

//    dpct::err0 err;

    try {
        d_tempMatrix = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        if (e.code() == sycl::errc::memory_allocation) {
            std::cerr << "Failed to allocate device memory for temporary matrix " << std::endl;
        }
    }

    try {
        d_tempSymFockMatrix = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        if (e.code() == sycl::errc::memory_allocation) {
            std::cerr << "Failed to allocate device memory for temporary symmetrized Fock matrix " << std::endl;
        }
    }

    try {
        d_tempEigenvectors = sycl::malloc_device<real_t>( num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        if (e.code() == sycl::errc::memory_allocation) {
            std::cerr << "Failed to allocate device memory for temporary temporary eigenvectors " << std::endl;
        }
    }

    if (d_orbital_energies == nullptr){
        try {
            d_tempEigenvalues = sycl::malloc_device<real_t>(num_basis, workq);
        }
        catch (sycl::exception const& e) {
            std::cerr << "SYCL exception: " << e.what() << std::endl;
            if (e.code() == sycl::errc::memory_allocation) {
                std::cerr << "Failed to allocate device memory for temporary temporary eigenvalues " << std::endl;
            }
        }
    } else {
        d_tempEigenvalues = d_orbital_energies;
    }


    // calculate the coefficient matrix using the eigenvectors of the Fock matrix by solving the generalized eigenvalue problem FC = SCE
    // symmetrize the Fock matrix F' = X^T F X
    // temp = X^T F
    matrixMatrixProduct(
        d_transform_matrix, d_fock_matrix, d_tempMatrix, num_basis,
        true, // transpose the transformation matrix X
        false
    );
    // F' = temp X
    matrixMatrixProduct(
        d_tempMatrix, d_transform_matrix, d_tempSymFockMatrix, num_basis,
        false,
        false
    );

    // diagonalize the symmetrized Fock matrix F'C' = C'E
    eigenDecomposition(d_tempSymFockMatrix, d_tempEigenvalues, d_tempEigenvectors, num_basis);

    // obtain the coefficient matrix from the eigenvectors C = X C'
    matrixMatrixProduct(
        d_transform_matrix, d_tempEigenvectors, d_coefficient_matrix, num_basis,
        false,
        false
    );

    // free the temporary memory
    sycl::free(d_tempMatrix, workq);
    sycl::free(d_tempSymFockMatrix, workq);
    sycl::free(d_tempEigenvectors, workq);

    if (d_orbital_energies == nullptr){
            sycl::free(d_tempEigenvalues, workq);
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/**
 * @brief Compute the density matrix for the restricted Hartree-Fock method.
 * @param d_coefficient_matrix Device pointer to the coefficient matrix
 * @param d_density_matrix Device pointer to store the density matrix
 * @param num_electron Number of electrons
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix, each of orbitals has zero or two electrons, using the coefficient matrix.
 * @details Electrons are allocated from the lowest energy orbitals, two by two.
 * @details The density matrix is given by \f$ D_{\mu\nu} = 2 \sum_{i=1}^{N/2} C_{\mu i} C_{\nu i} \f$.
 */
void computeDensityMatrix_RHF(const real_t* d_coefficient_matrix, real_t* d_density_matrix, const int num_electron, const int num_basis) {
    size_t threads_per_block = 256;
    size_t num_blocks = (num_basis * num_basis + threads_per_block - 1) / threads_per_block;
    /*
    DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    sycl::queue& workq = GPUHandle::syclsolver();
    workq.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, threads_per_block),
                          sycl::range<3>(1, 1, threads_per_block)),
        [=](sycl::nd_item<3> item_ct1) {
            computeDensityMatrix_RHF_kernel(d_coefficient_matrix,
                                            d_density_matrix, num_electron,
                                            num_basis);
        });
}


/**
 * @brief Compute the density matrix (alpha or beta spin) for the unrestricted Hartree-Fock method.
 * @param d_coefficient_matrix Device pointer to the coefficient matrix for the alpha spin or beta spin
 * @param d_density_matrix Device pointer to store the density matrix for the alpha spin or beta spin
 * @param num_spin Number of electrons for the alpha spin or beta spin
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix, each of orbitals has zero or one electoron using the coefficient matrix.
 * @details Electrons are allocated from the lowest energy orbitals, one by one.
 * @details The density matrix is given by \f$ D_{\mu\nu} = \sum_{i=1}^{N} C_{\mu i} C_{\nu i} \f$.
 */
void computeDensityMatrix_UHF(const real_t* d_coefficient_matrix, real_t* d_density_matrix, const int num_electron, const int num_basis) {
    size_t threads_per_block = 256;
    size_t num_blocks = (num_basis * num_basis + threads_per_block - 1) / threads_per_block;
    /*
    DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    sycl::queue& workq = GPUHandle::syclsolver();
    workq.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, threads_per_block),
                          sycl::range<3>(1, 1, threads_per_block)),
        [=](sycl::nd_item<3> item_ct1) {
            computeDensityMatrix_UHF_kernel(d_coefficient_matrix,
                                            d_density_matrix, num_electron,
                                            num_basis);
        });
}


/**
 * @brief Compute the density matrix (open- or closed-shell) for the ROHF method.
 * @param d_coefficient_matrix Device pointer to the coefficient matrix for the alpha spin or beta spin
 * @param d_density_matrix_closed Device pointer to store the density matrix for the closed-shell orbitals
 * @param d_density_matrix_open Device pointer to store the density matrix for the open-shell orbitals
 * @param d_density_matrix Device pointer to store the density matrix (sum of the closed- and open-shell orbitals)
 * @param num_closed Number of closed-shell orbitals
 * @param num_open Number of open-shell orbitals
 * @details This function computes the density matrix, each of orbitals has two (closed), one (open), zero (virtual) electoron using the coefficient matrix.
 * @details Electrons are allocated from the lowest energy orbitals.
 */
void computeDensityMatrix_ROHF(const real_t* d_coefficient_matrix, real_t* d_density_matrix_closed, real_t* d_density_matrix_open, real_t* d_density_matrix, const int num_closed, const int num_open, const int num_basis) {
    size_t threads_per_block = 256;
    size_t num_blocks = (num_basis * num_basis + threads_per_block - 1) / threads_per_block;
    /*
    DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    sycl::queue& workq = GPUHandle::syclsolver();
    workq.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, threads_per_block),
                          sycl::range<3>(1, 1, threads_per_block)),
        [=](sycl::nd_item<3> item_ct1) {
            computeDensityMatrix_ROHF_kernel(
                d_coefficient_matrix, d_density_matrix_closed,
                d_density_matrix_open, d_density_matrix, num_closed, num_open,
                num_basis);
        });
}


/**
 * @brief Compute the Fock matrix for the restricted Hartree-Fock method.
 * @param d_density_matrix Device pointer to the density matrix
 * @param core_hamiltonian Device pointer to the core Hamiltonian matrix
 * @param d_eri Device pointer to the electron repulsion integrals
 * @param d_fock_matrix Device pointer to store the Fock matrix
 * @param num_basis Number of basis functions
 * @details This function computes the Fock matrix using the density matrix, core Hamiltonian matrix, and electron repulsion integrals.
 * @details The Fock matrix is given by \f$ F_{\mu\nu} = H_{\mu\nu} + \sum_{\lambda\sigma} D_{\lambda\sigma} ((\mu\nu|\lambda\sigma) - {1 \over 2}(\nu\sigma|\mu\lambda)) \f$.
 */
void computeFockMatrix_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix, const int num_basis) {
    sycl::queue& workq = GPUHandle::syclsolver();
    const int warpsPerBlock = (num_basis + WARP_SIZE - 1) / WARP_SIZE;
    const int threadsPerBlock = WARP_SIZE * warpsPerBlock;
    if (threadsPerBlock > 1024) {
        THROW_EXCEPTION("Too many contracted Gauss-type orbitals.");
    }
    const int num_blocks = num_basis * num_basis;
    //const int num_blocks = num_basis * (num_basis + 1) / 2;
//    dpct::dim3 blocks(num_blocks);
//    dpct::dim3 threads(WARP_SIZE, warpsPerBlock);
    sycl::range<3> blocks(1, 1, num_blocks);
    sycl::range<3> threads(1, warpsPerBlock, WARP_SIZE);
    /*
    DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
// SYCL reduction：初期値 0.0、加算
/*
     auto fock_red = sycl::reduction(d_fock_acc, d_core_acc, sycl::plus<real_t>());
     cgh.parallel_for(
         sycl::nd_range<1>(sycl::range<1>(num_blocks * WARP_SIZE * warpsPerBlock),
                           sycl::range<1>(WARP_SIZE * warpsPerBlock)),
         fock_red,
         [=](sycl::nd_item<1> item, auto &fock_sum) {
*/
    workq.submit([&](sycl::handler& cgh){
    sycl::local_accessor<real_t, 1> s_F_ij(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
            computeFockMatrix_RHF_kernel(d_density_matrix,
                                         d_core_hamiltonian_matrix, d_eri,
                                         d_fock_matrix, num_basis, s_F_ij);
        });
        });
}


/**
 * @brief Compute the Fock matrix for the unrestricted Hartree-Fock method.
 * @param d_density_matrix_a Device pointer to the density matrix for the alpha spin
 * @param d_density_matrix_b Device pointer to the density matrix for the beta spin
 * @param core_hamiltonian Device pointer to the core Hamiltonian matrix
 * @param d_eri Device pointer to the electron repulsion integrals
 * @param d_fock_matrix_a Device pointer to store the Fock matrix for the alpha spin
 * @param d_fock_matrix_b Device pointer to store the Fock matrix for the beta spin
 * @param num_basis Number of basis functions
 * @details This function computes the Fock matrix (alpha and beta spins) using the density matrix, core Hamiltonian matrix, and electron repulsion integrals.
 * @details The Fock matrix is given by 
 *          \f$ F_{\mu\nu}^\alpha = H_{\mu\nu} + \sum_{\lambda\sigma} (D_{\lambda\sigma}^\alpha + D_{\lambda\sigma}^\beta) (\mu\nu|\lambda\sigma) - D_{\lambda\sigma}^\alpha (\nu\sigma|\mu\lambda) \f$.
 *          \f$ F_{\mu\nu}^\beta  = H_{\mu\nu} + \sum_{\lambda\sigma} (D_{\lambda\sigma}^\alpha + D_{\lambda\sigma}^\beta) (\mu\nu|\lambda\sigma) - D_{\lambda\sigma}^\beta  (\nu\sigma|\mu\lambda) \f$.
 */
void computeFockMatrix_UHF(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix_a, real_t* d_fock_matrix_b, const int num_basis) {
    const int warpsPerBlock = (num_basis + WARP_SIZE - 1) / WARP_SIZE;
    const int threadsPerBlock = WARP_SIZE * warpsPerBlock;
    if (threadsPerBlock > 1024) {
        THROW_EXCEPTION("Too many contracted Gauss-type orbitals.");
    }
    const int num_blocks = num_basis * num_basis;
    //const int num_blocks = num_basis * (num_basis + 1) / 2;
//    dpct::dim3 blocks(num_blocks);
//    dpct::dim3 threads(WARP_SIZE, warpsPerBlock);
    sycl::range<3> blocks(1, 1, num_blocks);
    sycl::range<3> threads(1, warpsPerBlock, WARP_SIZE);
    /*
    DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler& cgh){
    sycl::local_accessor<real_t, 1> s_Fa_ij(sycl::range<1>(1), cgh);
    sycl::local_accessor<real_t, 1> s_Fb_ij(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
            computeFockMatrix_UHF_kernel(d_density_matrix_a, d_density_matrix_b,
                                         d_core_hamiltonian_matrix, d_eri,
                                         d_fock_matrix_a, d_fock_matrix_b,
                                         num_basis,
                        s_Fa_ij, s_Fb_ij);
        });
    });
}


/**
 * @brief Compute the Fock matrix for the ROHF method.
 * @param d_density_matrix_closed Device pointer to the density matrix for the closed-shell orbitals
 * @param d_density_matrix_open Device pointer to the density matrix for the open-shell orbitals
 * @param core_hamiltonian Device pointer to the core Hamiltonian matrix
 * @param d_eri Device pointer to the electron repulsion integrals
 * @param d_fock_matrix_closed Device pointer to store the Fock matrix for the closed-shell orbitals
 * @param d_fock_matrix_open Device pointer to store the Fock matrix for the open-shell orbitals
 * @param d_fock_matrix Device pointer to store the unified Fock matrix
 * @details This function computes the Fock matrix using the density matrix, core Hamiltonian matrix, and electron repulsion integrals.
 */
void computeFockMatrix_ROHF(
    const real_t *d_density_matrix_closed, const real_t *d_density_matrix_open,
    const real_t *d_core_hamiltonian_matrix, const real_t *d_coefficient_matrix,
    const real_t *d_overlap_matrix, const real_t *d_eri,
    const ROHF_ParameterSet ROH_parameters, real_t *d_fock_matrix_closed,
    real_t *d_fock_matrix_open, real_t *d_fock_matrix, const int num_closed,
    const int num_open, const int num_basis) try {
//  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::queue& workq = GPUHandle::syclsolver();
    real_t* d_temp_F_MO_closed = nullptr; // Fock matrix for the closed-shell MO
    real_t* d_temp_F_MO_open = nullptr; // Fock matrix for the open-shell MO
    real_t* d_temp_R_MO = nullptr; /// unified Fock matrix R_MO
    real_t* d_temp_matrix1 = nullptr;
    real_t* d_temp_matrix2 = nullptr;

//    dpct::err0 err;

    try {
        d_temp_F_MO_closed = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        if (e.code() == sycl::errc::memory_allocation) {
            std::cerr << "Failed to allocate device memory for temporary Fock matrix for closed-shell orbitals " << std::endl;
        }
    }

    try {
        d_temp_F_MO_open = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        if (e.code() == sycl::errc::memory_allocation) {
            std::cerr << "Failed to allocate device memory for temporary Fock matrix for open-shell orbitals " << std::endl;
        }
    }

    try {
        d_temp_R_MO = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        if (e.code() == sycl::errc::memory_allocation) {
            std::cerr << "Failed to allocate device memory for temporary unified Fock matrix " << std::endl;
        }
    }

    try {
        d_temp_matrix1 = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        if (e.code() == sycl::errc::memory_allocation) {
            std::cerr << "Failed to allocate device memory for temporary matrix 1" << std::endl;
        }
    }

    try {
        d_temp_matrix2 = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        if (e.code() == sycl::errc::memory_allocation) {
            std::cerr << "Failed to allocate device memory for temporary matrix 2" << std::endl;
        }
    }


    { // compute the Fock matrices for the closed- and open-shell orbitals
        const int warpsPerBlock = (num_basis + WARP_SIZE - 1) / WARP_SIZE;
        const int threadsPerBlock = WARP_SIZE * warpsPerBlock;
        if (threadsPerBlock > 1024) {
            THROW_EXCEPTION("Too many contracted Gauss-type orbitals.");
        }
        const int num_blocks = num_basis * num_basis;
        //const int num_blocks = num_basis * (num_basis + 1) / 2;
//        dpct::dim3 blocks(num_blocks);
//        dpct::dim3 threads(WARP_SIZE, warpsPerBlock);
        sycl::range<3> blocks(1, 1, num_blocks);
        sycl::range<3> threads(1, warpsPerBlock, WARP_SIZE);
        /*
        DPCT1049:10: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler& cgh){
        sycl::local_accessor<real_t, 1> s_J_closed_ij(sycl::range<1>(1), cgh);
        sycl::local_accessor<real_t, 1> s_J_open_ij(sycl::range<1>(1), cgh);
        sycl::local_accessor<real_t, 1> s_K_closed_ij(sycl::range<1>(1), cgh);
        sycl::local_accessor<real_t, 1> s_K_open_ij(sycl::range<1>(1), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(blocks * threads, threads),
            [=](sycl::nd_item<3> item_ct1) {
                computeFockMatrix_ROHF_kernel(
                    d_density_matrix_closed, d_density_matrix_open,
                    d_core_hamiltonian_matrix, d_eri, d_fock_matrix_closed,
                    d_fock_matrix_open, num_basis,
                    s_J_closed_ij, s_J_open_ij,
                    s_K_closed_ij, s_K_open_ij);
            });
        });
    }

    { // Transforms the Fock matrices from AO to the MO
        // F_MO_closed = C^T F_AO_closed C
        matrixMatrixProduct(d_coefficient_matrix, d_fock_matrix_closed, d_temp_matrix1, num_basis, true, false);
        matrixMatrixProduct(d_temp_matrix1, d_coefficient_matrix, d_temp_F_MO_closed, num_basis, false, false);

        // F_MO_open = C F_AO_open C
        matrixMatrixProduct(d_coefficient_matrix, d_fock_matrix_open, d_temp_matrix1, num_basis, true, false);
        matrixMatrixProduct(d_temp_matrix1, d_coefficient_matrix, d_temp_F_MO_open, num_basis, false, false);
    }

    { // compute the unified Fock matrix R_MO
        const size_t num_elements = num_basis * (num_basis+1) / 2;
        const size_t threads_per_block = 256;
        const size_t num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
        workq.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, threads_per_block),
                              sycl::range<3>(1, 1, threads_per_block)),
            [=](sycl::nd_item<3> item_ct1) {
                computeUnifiedFockMatrix_ROHF_kernel(
                    d_temp_F_MO_closed, d_temp_F_MO_open, ROH_parameters,
                    d_temp_R_MO, num_closed, num_open, num_basis);
            });
    }

    { // Transform the unified Fock matrix from MO to AO by F_AO = S*C*R_MO*C^T*S
        // temp1 = S*C
        matrixMatrixProduct(d_overlap_matrix, d_coefficient_matrix, d_temp_matrix1, num_basis, false, false);
        // temp2 = temp1 * R_MO
        matrixMatrixProduct(d_temp_matrix1, d_temp_R_MO, d_temp_matrix2, num_basis, false, false);
        // temp1 = temp2 * C^T
        matrixMatrixProduct(d_temp_matrix2, d_coefficient_matrix, d_temp_matrix1, num_basis, false, true);
        // temp2 = temp1 * S
        matrixMatrixProduct(d_temp_matrix1, d_overlap_matrix, d_fock_matrix, num_basis, false, false);
    }

    // free the temporary memory
    sycl::free(d_temp_F_MO_closed, workq);
    sycl::free(d_temp_F_MO_open, workq);
    sycl::free(d_temp_R_MO, workq);
    sycl::free(d_temp_matrix1, workq);
    sycl::free(d_temp_matrix2, workq);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/**
 * @brief Compute the energy for the restricted HF.
 * @param d_density_matrix Device pointer to the density matrix
 * @param d_core_hamiltonian_matrix Device pointer to the core Hamiltonian matrix
 * @param d_fock_matrix Device pointer to the Fock matrix
 * @param num_basis Number of basis functions
 * @return Energy
 */
real_t computeEnergy_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_fock_matrix, const int num_basis) {
   
    real_t energy = 0.0;
    energy += innerProduct(d_density_matrix, d_core_hamiltonian_matrix, num_basis * num_basis);
    energy += innerProduct(d_density_matrix, d_fock_matrix,             num_basis * num_basis);
    energy *= 0.5;

    return energy;
}

/**
 * @brief Compute the energy for the unrestricted HF.
 * @param d_density_matrix_a Device pointer to the density matrix for the alpha spin
 * @param d_density_matrix_b Device pointer to the density matrix for the beta spin
 * @param d_core_hamiltonian_matrix Device pointer to the core Hamiltonian matrix
 * @param d_fock_matrix_a Device pointer to the Fock matrix for the alpha spin
 * @param d_fock_matrix_b Device pointer to the Fock matrix for the beta spin
 * @param num_basis Number of basis functions
 * @return Energy
 */
real_t computeEnergy_UHF(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_core_hamiltonian_matrix, const real_t* d_fock_matrix_a, const real_t* d_fock_matrix_b, const int num_basis) {
    real_t energy = 0.0;

    energy += innerProduct(d_density_matrix_a, d_core_hamiltonian_matrix, num_basis * num_basis);
    energy += innerProduct(d_density_matrix_a, d_fock_matrix_a,           num_basis * num_basis);
    
    energy += innerProduct(d_density_matrix_b, d_core_hamiltonian_matrix, num_basis * num_basis);
    energy += innerProduct(d_density_matrix_b, d_fock_matrix_b,           num_basis * num_basis);
    
    return 0.5 * energy;
}


/**
 * @brief Compute the energy for the ROHF method.
 * @param d_density_matrix_closed Device pointer to the density matrix for the closed orbitals
 * @param d_density_matrix_open Device pointer to the density matrix for the open orbitals
 * @param d_core_hamiltonian_matrix Device pointer to the core Hamiltonian matrix
 * @param d_fock_matrix_closed Device pointer to the Fock matrix for the closed orbitals
 * @param d_fock_matrix_open Device pointer to the Fock matrix for the open orbitals
 * @param num_basis Number of basis functions
 * @return Energy
*/
real_t computeEnergy_ROHF(const real_t* d_density_matrix_closed, const real_t* d_density_matrix_open, const real_t* d_core_hamiltonian_matrix, const real_t* d_fock_matrix_closed, const real_t* d_fock_matrix_open, const int num_basis) {
    real_t energy = 0.0;

    energy +=       innerProduct(d_density_matrix_closed, d_core_hamiltonian_matrix, num_basis * num_basis);
    energy +=       innerProduct(d_density_matrix_closed, d_fock_matrix_closed,      num_basis * num_basis);

    energy +=       innerProduct(d_density_matrix_open, d_core_hamiltonian_matrix,   num_basis * num_basis);
    energy += 2.0 * innerProduct(d_density_matrix_open, d_fock_matrix_open,          num_basis * num_basis); // Note: factor 2.0 only here

    return 0.5 * energy;
}




/**
 * @brief Compute the optimal damping factor for RHF.
 * @param d_fock_matrix Device pointer to the Fock matrix
 * @param d_prev_fock_matrix Device pointer to the previous Fock matrix
 * @param d_density_matrix Device pointer to the density matrix
 * @param d_prev_density_matrix Device pointer to the previous density matrix
 * @param num_basis Number of basis functions
 * @return Optimal damping factor
 * @details This function computes the optimal damping factor for the restricted Hartree-Fock method.
 * @details The damping factor is given as follows:
 * @details \f$ s = \mathrm{Tr}[F_{\mathrm{old}}(D_{\mathrm{new}} - D_{\mathrm{old}})] \f$
 * @details \f$ c = \mathrm{Tr}[(F_{\mathrm{new}} - F_{\mathrm{old}})(D_{\mathrm{new}} - D_{\mathrm{old}})] \f$
 * @details \f$ \alpha = 1 \f$ if \f$ c \le - \frac{s}{2} \f$, otherwise \f$ \alpha = -\frac{s}{2c} \f$
 */
real_t computeOptimalDampingFactor_RHF(const real_t *d_fock_matrix,
                                       const real_t *d_prev_fock_matrix,
                                       const real_t *d_density_matrix,
                                       const real_t *d_prev_density_matrix,
                                       const int num_basis) try {
  sycl::queue& syclsolverHandle = GPUHandle::syclsolver();
    // allocate temporary memory
    real_t* d_tempDiffFockMatrix = nullptr;
    real_t* d_tempDiffDensityMatrix = nullptr;
    real_t* d_tempMatrix = nullptr;

    dpct::err0 err;

    d_tempDiffFockMatrix = sycl::malloc_device<real_t>(num_basis * num_basis, syclsolverHandle);
    d_tempDiffDensityMatrix = sycl::malloc_device<real_t>(num_basis * num_basis, syclsolverHandle);
    d_tempMatrix = sycl::malloc_device<real_t>(num_basis * num_basis, syclsolverHandle);

    // calculate the difference between the Fock matrices
    // \f$ F_{\mathrm{diff}} = F_{\mathrm{new}} - F_{\mathrm{old}}  \f$
    matrixSubtraction(d_fock_matrix, d_prev_fock_matrix, d_tempDiffFockMatrix, num_basis);

    // calculate the difference between the density matrices
    // \f$D_{\mathrm{diff}} = D_{\mathrm{new}} - D_{\mathrm{old}} \f$
    matrixSubtraction(d_density_matrix, d_prev_density_matrix, d_tempDiffDensityMatrix, num_basis);

    // calculate the trace of the product of the difference matrices
    // \f$ s = \mathrm{Tr}[F_{\mathrm{old}}(D_{\mathrm{new}} - D_{\mathrm{old}})] \f$
    real_t s = 0.0;
    matrixMatrixProduct(d_prev_fock_matrix, d_tempDiffDensityMatrix, d_tempMatrix,
        num_basis, false, false);

    s = computeMatrixTrace(d_tempMatrix, num_basis);

    // \f$ c = \mathrm{Tr}[(F_{\mathrm{new}} - F_{\mathrm{old}})(D_{\mathrm{new}} - D_{\mathrm{old}})] \f$
    real_t c = 0.0;
    matrixMatrixProduct(d_tempDiffFockMatrix, d_tempDiffDensityMatrix, d_tempMatrix,
        num_basis, false, false);

    c = computeMatrixTrace(d_tempMatrix, num_basis);

    real_t alpha;
    //std::cout << "s = " << s << ", c = " << c << std::endl;
    if (c <= -s/2.0) {
        alpha = 1.0;
    } else {
        alpha = -0.5 * s / c;
    }

    // free the temporary memory
    sycl::free(d_tempDiffFockMatrix, syclsolverHandle);
    sycl::free(d_tempDiffDensityMatrix, syclsolverHandle);
    sycl::free(d_tempMatrix, syclsolverHandle);

    return alpha;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/**
 * @brief Compute the optimal damping factor for ROHF.
 * @param d_fock_matrix Device pointer to the Fock matrix
 * @param d_prev_fock_matrix Device pointer to the previous Fock matrix
 * @param d_density_matrix Device pointer to the density matrix
 * @param d_prev_density_matrix Device pointer to the previous density matrix
 * @param num_basis Number of basis functions
 * @return Optimal damping factor
 * @details This function just calls the function computeOptimalDampingFactor_RHF.
 */
 real_t computeOptimalDampingFactor_ROHF(const real_t* d_fock_matrix, const real_t* d_prev_fock_matrix, const real_t* d_density_matrix, const real_t* d_prev_density_matrix, const int num_basis) {
    return computeOptimalDampingFactor_RHF(d_fock_matrix, d_prev_fock_matrix, d_density_matrix, d_prev_density_matrix, num_basis);
 }


/**
 * @brief Update the Fock/density matrix using the damping factor.
 * @param d_matrix_old Device pointer to the previous Fock matrix
 * @param d_matrix_new Device pointer to the current Fock matrix
 * @param alpha Damping factor
 * @details This function updates the Fock matrix using the damping factor.
 * @details The updated Fock matrix is given by \f$ F_{\mathrm{new}} = (1-\alpha)F_{\mathrm{old}} + \alpha F_{\mathrm{new}} \f$.
 * @details The current Fock matrix is overwritten with the updated Fock matrix. \f$ F_{\mathrm{old}} = F_{\mathrm{new}} \f$
 */
void damping(real_t *d_matrix_old, real_t *d_matrix_new, const real_t alpha,
             int num_basis) try {
  sycl::queue& syclsolverHandle = GPUHandle::syclsolver();
    real_t* d_tempMatrix;

try {
    d_tempMatrix = sycl::malloc_device<real_t>( num_basis * num_basis, syclsolverHandle);
}
catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
    if (e.code() == sycl::errc::memory_allocation) {
        std::cerr << "Failed to allocate device memory for temporary matrix " << std::endl;
    }
}

    weightedMatrixSum(d_matrix_old, d_matrix_new, d_tempMatrix, 1.0-alpha, alpha, num_basis);

    syclsolverHandle.memcpy(d_matrix_old, d_tempMatrix, num_basis * num_basis * sizeof(real_t));
    syclsolverHandle.memcpy(d_matrix_new, d_tempMatrix, num_basis * num_basis * sizeof(real_t)).wait();

    sycl::free(d_tempMatrix, syclsolverHandle);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/**
 * @brief Compute the DIIS error matrix for RHF, UHF, ROHF.
 * @param d_overlap_matrix Device pointer to the overlap matrix
 * @param d_transform_matrix Device pointer to the transformation matrix
 * @param d_fock_matrix Device pointer to the Fock matrix
 * @param d_density_matrix Device pointer to the density matrix
 * @param d_diis_error_matrix Device pointer to store the DIIS error matrix
 * @param num_basis Number of basis functions
 * @details This function computes the DIIS error matrix.
 * @details The DIIS error matrix is given by \f$ E = FPS - SPF \f$.
 */
void computeDIISErrorMatrix(const real_t *d_overlap_matrix,
                            const real_t *d_transform_matrix,
                            const real_t *d_fock_matrix,
                            const real_t *d_density_matrix,
                            real_t *d_diis_error_matrix, const int num_basis,
                            const bool is_include_transform) try {
//  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::queue& workq = GPUHandle::syclsolver();
    real_t* d_tempFPS;
    real_t* d_tempSPF;
    real_t* d_tempMatrix1;

    dpct::err0 err;

    try {
        d_tempFPS = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string(
                "Failed to allocate device memory for temporary FPS matrix: ") +
            std::string(e.what()));
    }

    try {
        d_tempSPF = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string(
                "Failed to allocate device memory for temporary SPF matrix: ") +
            std::string(e.what()));
    }

    try {
        d_tempMatrix1 = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string(
                "Failed to allocate device memory for temporary matrix 1: ") +
            std::string(e.what()));
    }

    // tempFPS = FPS
    matrixMatrixProduct(d_fock_matrix, d_density_matrix, d_tempMatrix1, num_basis, false, false);
    matrixMatrixProduct(d_tempMatrix1, d_overlap_matrix, d_tempFPS, num_basis, false, false);

    // tempSPF = SPF
    matrixMatrixProduct(d_overlap_matrix, d_density_matrix, d_tempMatrix1, num_basis, false, false);
    matrixMatrixProduct(d_tempMatrix1, d_fock_matrix, d_tempSPF, num_basis, false, false);

    // DIIS error matrix = FPS - SPF
    matrixSubtraction(d_tempFPS, d_tempSPF, d_diis_error_matrix, num_basis);

    if(is_include_transform){
        // tempSPF = X(FPS-SPF)
        matrixMatrixProduct(d_transform_matrix, d_diis_error_matrix, d_tempFPS, num_basis, false, false);

        // DIIS error matrix = X(FPS-SPF)X^T
        matrixMatrixProduct(d_tempFPS, d_transform_matrix, d_diis_error_matrix, num_basis, false, true);
    }

    sycl::free(d_tempMatrix1, workq);
    sycl::free(d_tempFPS, workq);
    sycl::free(d_tempSPF, workq);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/**
 * @brief Compute the Fock matrix by DIIS for RHF, UHF, ROHF.
 * @param d_error_matrices Device pointer to the error matrices
 * @param d_fock_matrices Device pointer to the Fock matrices
 * @param d_new_fock_matrix Device pointer to store the new Fock matrix
 * @param num_prev Number of previous Fock matrices
 * @param num_basis Number of basis functions
 * @details This function computes the Fock matrix by DIIS.
 */
void computeFockMatrixDIIS(real_t *d_error_matrices, real_t *d_fock_matrices,
                           real_t *d_new_fock_matrix, const int num_prev,
                           const int num_basis) try {
    sycl::queue& workq = GPUHandle::syclsolver();
    if (num_prev <= 1){
        THROW_EXCEPTION("DIIS requires at least two previous Fock matrices.");
    }

    const int num_size = num_prev + 1;
    const int lda = num_size;

    // Create the DIIS matrix
    real_t* d_DIIS_matrix;
    real_t* h_DIIS_matrix = new real_t[num_size * num_size];
    if (h_DIIS_matrix == nullptr) {
        THROW_EXCEPTION("Failed to allocate host memory for DIIS matrix.");
    }

    dpct::err0 err;

    try {
        d_DIIS_matrix = sycl::malloc_device<real_t>(num_size * num_size, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for DIIS matrix: ") +
            std::string(e.what()));
    }

    for (int i = 0; i < num_prev; i++){
        for (int j = i; j < num_prev; j++){
            real_t e = innerProduct(&d_error_matrices[i*num_basis*num_basis], &d_error_matrices[j*num_basis*num_basis], num_basis * num_basis);
            h_DIIS_matrix[i * num_size + j] = e;
            h_DIIS_matrix[j * num_size + i] = e;
        }
        h_DIIS_matrix[i * num_size + num_prev] = -1.0;
        h_DIIS_matrix[num_prev * num_size + i] = -1.0;
    }
    h_DIIS_matrix[num_prev * num_size + num_prev] = 0.0;

    workq
        .memcpy(d_DIIS_matrix, h_DIIS_matrix,
                num_size * num_size * sizeof(real_t))
        .wait();

    // Create the right-hand side vector
    real_t* h_DIIS_rhs = new real_t[num_size];
    if (h_DIIS_rhs == nullptr) {
        THROW_EXCEPTION("Failed to allocate host memory for DIIS right-hand side vector.");
    }
    real_t* d_DIIS_rhs;
    try {
        d_DIIS_rhs = sycl::malloc_device<real_t>(num_size, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for DIIS "
                                    "right-hand side vector: ") +
            std::string(e.what()));
    }

    for (int i = 0; i < num_prev; i++){
        h_DIIS_rhs[i] = 0.0;
    }
    h_DIIS_rhs[num_prev] = -1.0;

    workq.memcpy(d_DIIS_rhs, h_DIIS_rhs, num_size * sizeof(real_t)).wait();

    // Solve the linear equation on the device
//    GPUHandle syclsolver;

    // get the workspace size
    std::int64_t work_size;
    work_size = oneapi::mkl::lapack::getrf_scratchpad_size<real_t>(workq, num_size, num_size, lda);
/*    int work_size;
    cusolverDnDgetrf_bufferSize(cusolver.cusolverHandle, num_size, num_size, d_DIIS_matrix, num_size, &work_size);
*/

    // allocate the workspace
    real_t* d_work;
    try {
        d_work = sycl::malloc_device<real_t>(work_size, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for workspace: ") +
            std::string(e.what()));
    }

    // pivot array and info
    int64_t* d_pivot = nullptr;
    int64_t* d_info = nullptr;
    try {
        d_pivot = sycl::malloc_device<int64_t>(num_size, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for pivot array: ") +
            std::string(e.what()));
    }
    try {
        d_info = sycl::malloc_device<int64_t>(1, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for info array: ") +
            std::string(e.what()));
    }

    // LU factorization
    try {
        oneapi::mkl::lapack::getrf(workq, num_size, num_size, d_DIIS_matrix, lda, d_pivot, d_work, work_size).wait();
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception during getrf: " << e.what() << std::endl;
        std::exit(1);
    }
//    cusolverDnDgetrf(cusolver.cusolverHandle, num_size, num_size, d_DIIS_matrix, num_size, d_work, d_pivot, d_info);

    // solve the linear equation
    try {
        oneapi::mkl::lapack::getrs(workq, oneapi::mkl::transpose::nontrans, num_size, 1, d_DIIS_matrix, lda, d_pivot, d_DIIS_rhs, num_size, d_work, work_size).wait();
    } catch (sycl::exception const& e) {                                                                                     std::cerr << "SYCL exception during getrs: " << e.what() << std::endl;
        std::exit(1);
    }
//    cusolverDnDgetrs(cusolver.cusolverHandle, CUBLAS_OP_N, num_size, 1, d_DIIS_matrix, num_size, d_pivot, d_DIIS_rhs, num_size, d_info);

    // copy the result to the host
    workq.memcpy(h_DIIS_rhs, d_DIIS_rhs, num_size * sizeof(real_t)).wait();

    // compute the DIIS Fock matrix (\f$ F_{\mathrm{new}} = \sum_{i=1}^{N} c_i F_i \f$)
    // F = c_1 F_1 + c_2 F_2
    weightedMatrixSum(&d_fock_matrices[0*num_basis*num_basis], &d_fock_matrices[1*num_basis*num_basis], d_new_fock_matrix, h_DIIS_rhs[0], h_DIIS_rhs[1], num_basis);
    for (int i = 2; i < num_prev; i++){
        weightedMatrixSum(d_new_fock_matrix, &d_fock_matrices[i*num_basis*num_basis], d_new_fock_matrix, 1.0, h_DIIS_rhs[i], num_basis);
    }

    // free the memory
    sycl::free(d_DIIS_matrix, workq);
    sycl::free(d_DIIS_rhs, workq);
    sycl::free(d_work, workq);
    sycl::free(d_pivot, workq);
    sycl::free(d_info, workq);

    delete[] h_DIIS_matrix;
    delete[] h_DIIS_rhs;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/**
 * @brief Computes the coefficient matrix from the Fock matrix and the transformation matrix.
 * @param d_fock_matrix Device pointer to the Fock matrix
 * @param d_overlap_matrix Device pointer to the overlap matrix
 * @param d_transform_matrix Device pointer to the transformation matrix
 * @param d_coefficient_matrix Device pointer to store the coefficient matrix
 * @param num_basis Number of basis functions
 * @details This function computes the coefficient matrix using the eigenvectors of the Fock matrix by solving the generalized eigenvalue problem \f$FC = SCE \f$.
 * @details To transform the generalized eigenvalue problem to the standard eigenvalue problem \f$FC = CE \f$, the transformation matrix.
 */
 void computeInitialCoefficientMatrix_GWH(
     const real_t *d_core_hamiltonian_matrix, const real_t *d_overlap_matrix,
     const real_t *d_transform_matrix, real_t *d_coefficient_matrix,
     const int num_basis) try {
//  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::queue& workq = GPUHandle::syclsolver();
    const real_t cx = 1.75;

    dpct::err0 err;

    // allocate temporary memory
    real_t* d_temp_FockMatrix = nullptr;
    real_t* h_temp_FockMatrix = new real_t[num_basis * num_basis];
    try {
        d_temp_FockMatrix = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for temporary Fock matrix: ") +
            std::string(e.what()));
    }

    // Compute the initial Fock matrix
    size_t threads_per_block = 256;
    size_t num_blocks = (num_basis * num_basis + threads_per_block - 1) / threads_per_block;
    /*
    DPCT1049:11: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    workq.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, threads_per_block),
                          sycl::range<3>(1, 1, threads_per_block)),
        [=](sycl::nd_item<3> item_ct1) {
            computeInitialFockMatrix_GWH_kernel(
                d_core_hamiltonian_matrix, d_overlap_matrix, d_temp_FockMatrix,
                num_basis, cx);
        });

    // Diagonalize the Fock matrix
    computeCoefficientMatrix(d_temp_FockMatrix, d_transform_matrix, d_coefficient_matrix, num_basis);

    // free the temporary memory
    sycl::free(d_temp_FockMatrix, workq);
}
 catch (sycl::exception const &exc) {
   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
             << ", line:" << __LINE__ << std::endl;
   std::exit(1);
 }

/**
 * @brief Computes the inverse of an N x N matrix stored in device memory.
 * 
 * This function overwrites the input matrix with its inverse using LU decomposition.
 * The original matrix is destroyed in the process.
 * 
 * @param d_A Pointer to the N x N matrix in device memory (input).
 * @param N The size of the matrix (number of rows/columns).
 */
void invertMatrix(double *d_A, const int N) try {
    sycl::queue& syclsolverHandle = GPUHandle::syclsolver();
    int64_t *d_ipiv;
    real_t *d_work;
            // Leading dimension
    int lda = N;

    // Pivot indices
    d_ipiv = sycl::malloc_device<int64_t>( N, syclsolverHandle);

    // Scratchpad size
    int64_t lwork_rf = oneapi::mkl::lapack::getrf_scratchpad_size<real_t>( syclsolverHandle, N, N, lda);
    real_t *d_work_rf = sycl::malloc_device<real_t>(lwork_rf, syclsolverHandle);

    // LU Decomposition
    oneapi::mkl::lapack::getrf( syclsolverHandle, N, N, d_A, N, d_ipiv, d_work_rf, lwork_rf).wait();

    sycl::free(d_work_rf, syclsolverHandle);

    // Inversion scratchpad size
    int64_t lwork_ri = oneapi::mkl::lapack::getri_scratchpad_size<real_t>( syclsolverHandle, N, lda);
    real_t *d_work_ri = sycl::malloc_device<real_t>(lwork_ri, syclsolverHandle);

    // Inverse computation
    oneapi::mkl::lapack::getri( syclsolverHandle, N, d_A, lda, d_ipiv, d_work_ri, lwork_ri).wait();

    syclsolverHandle.wait_and_throw();

    sycl::free(d_ipiv, syclsolverHandle);
    sycl::free(d_work_ri, syclsolverHandle);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/**
 * @brief Performs Cholesky decomposition on an N x N symmetric positive definite matrix in device memory.
 *
 * The input matrix is overwritten with the result. The decomposition produces a lower triangular
 * matrix L such that A = L * L^T.
 *
 * @param d_A Pointer to the N x N matrix in device memory (input/output).
 * @param N The size of the matrix (number of rows/columns).
 */
void choleskyDecomposition(double *d_A, const int N) try {
    sycl::queue& syclsolverHandle = GPUHandle::syclsolver();

    std::int64_t lwork;
    real_t *d_work = nullptr;

    // Get workspace size
try {
    lwork = oneapi::mkl::lapack::potrf_scratchpad_size<real_t>( syclsolverHandle,
        oneapi::mkl::uplo::upper, N, N);
}
catch (oneapi::mkl::lapack::invalid_argument const& e) {
    std::cerr << "Invalid argument in potrf_scratchpad_size: "
              << e.what() << std::endl;
    return;
}

    // workspace allocation
try {
    d_work = sycl::malloc_device<real_t>(lwork, syclsolverHandle);
}
catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
    if (e.code() == sycl::errc::memory_allocation) {
        std::cerr << "Failed to allocate device memory for workspace " << std::endl;
    }
    return;
}

    // Perform Cholesky decomposition (A -> L, overwriting lower triangular part)
try {
    auto ev = oneapi::mkl::lapack::potrf(syclsolverHandle,
        oneapi::mkl::uplo::upper, N, d_A, N, d_work, lwork
    );
    ev.wait_and_throw();
}
catch (oneapi::mkl::lapack::invalid_argument const& e) {
    std::cerr << "Invalid argument in potrf: " << e.what() << std::endl;
}
catch (sycl::exception const& e) {
    std::cerr << "SYCL exception during potrf: " << e.what() << std::endl;
}

try{
    // Set zero to the upper triangular part
    const int num_threads = 256;
    const int num_blocks = (N * N + num_threads - 1) / num_threads;

    syclsolverHandle.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           setZeroUpperTriangle(d_A, N);
                       }).wait();

    // Cleanup
    sycl::free(d_work, syclsolverHandle);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


/**
 * @brief Solve the equation AX = B for X, where A is a lower triangular matrix.
 * The input matrix B is overwritten with the result X. 
 * @param d_A Pointer to the row x row lower triangular matrix in device memory.
 * @param d_B Pointer to the row x col matrix in device memory (input/output).
 * @param row The number of rows.
 * @param row The number of columns.
 */
void solve_lower_triangular(double *d_A, double *d_B, int row, int col) try {
    sycl::queue& syclsolverHandle = GPUHandle::syclsolver();
    sycl::event ev;

    // 転置
//    transposeMatrixInPlace(d_A, row);
// 転置処理：A → B（転置＋スケーリング係数 1.0）
try {
    ev = oneapi::mkl::blas::row_major::omatcopy( syclsolverHandle,
        oneapi::mkl::transpose::trans, row, row,
        1.0,                           // スケーリング係数
        d_A, row,
        d_A,
        row
    );
 // 実際の実行完了を待つ
    ev.wait_and_throw();
}
catch (oneapi::mkl::lapack::invalid_argument const& e) {
    std::cerr << "Invalid argument in omatcopy: " << e.what() << std::endl;
}
catch (sycl::exception const& e) {
    std::cerr << "SYCL exception during omatcopy: " << e.what() << std::endl;
}

    dpct::err0 err;

    real_t *d_tmp;
    try {
        d_tmp = sycl::malloc_device<real_t>(row * col, syclsolverHandle);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for temporary matrix: ") +
            std::string(e.what()));
    }

    const real_t alpha = 1.0;
    const real_t beta = 0.0; //これ必要

    oneapi::mkl::blas::column_major::omatadd(
        syclsolverHandle,
        oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        row, col,
        alpha, d_B, col,
        beta, nullptr, (row >= col) ? row : col,
        d_tmp, row
    );
/*
    cublasDgeam(
        cublasHandle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        row, col,
        &alpha, d_B, col,
        &beta, nullptr, (row >= col) ? row : col,
        d_tmp, row
    );
*/

    // // Solve A * X = B → X overwrites B
    oneapi::mkl::blas::trsm(
        syclsolverHandle,
        oneapi::mkl::side::left,
        oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit,
        row,
        col,
        alpha,
        d_A, row,
        d_tmp, row
    );
/*
    cublasDtrsm(
        cublasHandle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        row,
        col,
        &alpha,
        d_A, row,
        d_tmp, row
    );
*/

    oneapi::mkl::blas::column_major::omatadd(
        syclsolverHandle,
        oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        col, row,
        alpha,
        d_tmp, row,
        beta,
        nullptr, (row >= col) ? row : col,
        d_B, col
    );
/*
    cublasDgeam(
        cublasHandle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        col, row,
        &alpha,
        d_tmp, row,
        &beta,
        nullptr, (row >= col) ? row : col,
        d_B, col
    );
*/

    sycl::free(d_tmp, syclsolverHandle);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void writeMatrixToFile(std::string filename, double* array, size_t size) {
    std::ofstream outFile(filename);
    if (!outFile) 
        throw std::runtime_error("ファイルを書き込み用に開けませんでした");
    
    for (size_t i = 0; i < size; i++) {
        outFile << std::fixed << std::setprecision(15) << array[i] << "\n";
    }
}

/**
 * @brief Compute the intermediate matrix B for the RI approximation.
 * @param shell_type_infos Information about the basis functions
 * @param d_primitive_shells Pointer to the primitive shells in device memory
 * @param d_cgto_normalization_factors Pointer to the normalization factors of the CGTOs in device memory
 * @param auxiliary_shell_type_infos Information about the auxiliary basis functions
 * @param d_auxiliary_primitive_shells Pointer to the primitive shells of the auxiliary basis functions in device memory
 * @param d_auxiliary_cgto_nomalization_factors Pointer to the normalization factors of the auxiliary CGTOs in device memory
 * @param d_intermediate_matrix_B Pointer to the intermediate matrix B in device memory
 * @param num_basis Number of basis functions
 * @param num_auxiliary_basis Number of auxiliary basis functions
 * @param d_boyst_grid The grid values of the precomputed Boys function
 * @param verbose Whether to print additional information
 * @details This function computes the intermediate matrix B for the RI approximation.
 * @details (1) \f$ A_{pq} = (p|q) \f$. (two-center ERIs)
 * @details (2) \f$ A^{-1} = LL^T \f$. (Cholesky decomposition)
 * @details (3) \f$ B_{\mu\nu}^p = \sum_{q}^{M_\textrm{aux}} (\mu\nu|q)L_{qp}.
 */
void compute_RI_IntermediateMatrixB(
    const std::vector<ShellTypeInfo> &shell_type_infos,
    const std::vector<ShellPairTypeInfo> &shell_pair_type_infos,
    const PrimitiveShell *d_primitive_shells,
    const real_t *d_cgto_normalization_factors,
    const std::vector<ShellTypeInfo> &auxiliary_shell_type_infos,
    const PrimitiveShell *d_auxiliary_primitive_shells,
    const real_t *d_auxiliary_cgto_nomalization_factors,
    real_t *d_intermediate_matrix_B,
    const size_t2 *d_primitive_shell_pair_indices,
    const real_t *d_schwarz_upper_bound_factors,
    const real_t *d_auxiliary_schwarz_upper_bound_factors,
    const real_t schwarz_screening_threshold, const int num_basis,
    const int num_auxiliary_basis, const real_t *d_boys_grid,
    const bool verbose) try {
//  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::queue& workq = GPUHandle::syclsolver();
    dpct::err0 err;

    // Allocate device memory for the two-center ERIs
    real_t* d_two_center_eri;
    try {
        d_two_center_eri = sycl::malloc_device<real_t>(num_auxiliary_basis * num_auxiliary_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for two-center ERIs: ") +
            std::string(e.what()));
    }
    workq
        .memset(d_two_center_eri, 0.0,
                num_auxiliary_basis * num_auxiliary_basis * sizeof(real_t))
        .wait();

    // Compute the two-center ERIs of the auxiliary basis functions
    computeTwoCenterERIs(
        auxiliary_shell_type_infos,
        d_auxiliary_primitive_shells,
        d_auxiliary_cgto_nomalization_factors,
        d_two_center_eri,
        num_auxiliary_basis,
        d_boys_grid,
        d_auxiliary_schwarz_upper_bound_factors,
        schwarz_screening_threshold,
        verbose);


    // // Compute the inverse of the two-center ERI matrix (it is overwritten with its inverse)
    // invertMatrix(d_two_center_eri, num_auxiliary_basis);

    // Cholesky decomposition of the inverse of the two-center ERI matrix (it is overwritten with the result)
    choleskyDecomposition(d_two_center_eri, num_auxiliary_basis);

    // Allocate device memory for the three-center ERIs
    real_t* d_three_center_eri;
    try {
        d_three_center_eri = sycl::malloc_device<real_t>(num_basis * num_basis * num_auxiliary_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for three-center ERIs: ") +
            std::string(e.what()));
    }
    workq
        .memset(d_three_center_eri, 0.0,
                num_basis * num_basis * num_auxiliary_basis * sizeof(real_t))
        .wait();

    // Compute the three-center ERIs of the auxiliary basis functions and the basis functions
    computeThreeCenterERIs(
        shell_type_infos,
        shell_pair_type_infos,
        d_primitive_shells,
        d_cgto_normalization_factors,
        auxiliary_shell_type_infos,
        d_auxiliary_primitive_shells,
        d_auxiliary_cgto_nomalization_factors,
        d_three_center_eri,
        d_primitive_shell_pair_indices,
        num_basis,
        num_auxiliary_basis,
        d_boys_grid,
        d_schwarz_upper_bound_factors,
        d_auxiliary_schwarz_upper_bound_factors,
        schwarz_screening_threshold,
        verbose);


    // Compute the intermediate matrix B
    solve_lower_triangular(d_two_center_eri, d_three_center_eri, num_auxiliary_basis, num_basis*num_basis);
    workq.memcpy(d_intermediate_matrix_B, d_three_center_eri,
                 sizeof(real_t) * num_auxiliary_basis * num_basis * num_basis);

    sycl::free(d_two_center_eri, workq);
    sycl::free(d_three_center_eri, workq);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void computeIntermediateMatrixB(
    const real_t* d_three_center_eri,
    const real_t* d_two_center_eri,
    real_t* d_intermediate_matrix_B,
    const int num_basis,
    const int num_auxiliary_basis)
{
    // B_{\mu\nu}^p = \sum_{q}^{M_\textrm{aux}} (\mu\nu|q)L_{qp}
    // B[p][\mu][\nu] = \sum_{q}^{M_\textrm{aux}} T[q][\mu][\nu] * L[q][p]

    const int num_threads = 256;
    const int num_blocks = (num_auxiliary_basis * num_basis * num_basis + num_threads - 1) / num_threads;
    sycl::queue& workq = GPUHandle::syclsolver();
    workq.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, num_threads),
                          sycl::range<3>(1, 1, num_threads)),
        [=](sycl::nd_item<3> item_ct1) {
            computeRIIntermediateMatrixB_kernel(
                d_three_center_eri, d_two_center_eri, d_intermediate_matrix_B,
                num_basis, num_auxiliary_basis);
        });
}

void computeFockMatrix_RI_RHF(const real_t *d_density_matrix,
                              const real_t *d_core_hamiltonian_matrix,
                              const real_t *d_intermediate_matrix_B,
                              real_t *d_fock_matrix, const int num_basis,
                              const int num_auxiliary_basis) try {
    sycl::queue& workq = GPUHandle::syclsolver();
//    dpct::device_ext &dev_ct1 = dpct::get_current_device();
//    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    //cublasManager cublas;
//    dpct::blas::descriptor_ptr cublasHandle = GPUHandle::cublas();

    dpct::err0 err;

    // the following is used in the two kernels. So, if necessary, it should be changed for each kernel.
    const int num_threads = 256;
    const int num_blocks = (num_basis * num_basis + num_threads - 1) / num_threads;

    ////////////////////////////////// compute J-matrix //////////////////////////////////
    real_t* d_J = nullptr;
    try {
        d_J = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for J matrix: ") +
            std::string(e.what()));
    }

    // W = B D (Matrix(M_aux x M^2 matrix) * Vector (M^2 x 1) )
    real_t* d_W = nullptr;
    try {
        d_W = sycl::malloc_device<real_t>(num_auxiliary_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for W vector: ") +
            std::string(e.what()));
    }

    real_t alpha = 1.0;
    real_t beta = 0.0;

    oneapi::mkl::blas::gemv(workq, oneapi::mkl::transpose::trans, num_basis*num_basis,
                            num_auxiliary_basis, alpha, d_intermediate_matrix_B,
                            num_basis*num_basis, d_density_matrix, 1, beta, d_W, 1);
//    cublasDgemv(cublasHandle, CUBLAS_OP_T, num_basis*num_basis, num_auxiliary_basis, &alpha, d_intermediate_matrix_B, num_basis*num_basis, d_density_matrix, 1, &beta, d_W, 1);

    // J = sum(W[i] * B[i])
    workq.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           weighted_sum_matrices_kernel(
                               d_J, d_intermediate_matrix_B, d_W, num_basis,
                               num_auxiliary_basis, false);
                       });

    // free the memory
    sycl::free(d_W, workq);

    ////////////////////////////////// compute K-matrix //////////////////////////////////
    real_t* d_K = nullptr;
    try {
        d_K = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for K matrix: ") +
            std::string(e.what()));
    }

    // T^p = B^p D^T
    real_t* d_T = nullptr;
    try {
        d_T = sycl::malloc_device<real_t>(num_auxiliary_basis * num_basis * num_basis,workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for T matrix: ") +
            std::string(e.what()));
    }

    // Note: cublasDgemmBatched shoul be used?
    for(int p=0; p<num_auxiliary_basis; p++){
        matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], d_density_matrix, &d_T[p*num_basis*num_basis], num_basis, false, true);
    }


    // V^p = B^p (T^p)^T
    real_t* d_V = nullptr;
    try {
        d_V = sycl::malloc_device<real_t>(num_auxiliary_basis * num_basis * num_basis,workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for V matrix: ") +
            std::string(e.what()));
    }

    // Note: cublasDgemmBatched shoul be used?
    for(int p=0; p<num_auxiliary_basis; p++){
        matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], &d_T[p*num_basis*num_basis], &d_V[p*num_basis*num_basis], num_basis, false, true);
    }

    // K = sum(V^p)
    workq.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           sum_matrices_kernel(d_K, d_V, num_basis,
                                               num_auxiliary_basis, false);
                       });

    // free the memory
    sycl::free(d_T, workq);
    sycl::free(d_V, workq);

    ////////////////////////////////// compute Fock matrix //////////////////////////////////

    // F = H + J - (1/2)*K
    workq.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           computeFockMatrix_RI_RHF_kernel(
                               d_core_hamiltonian_matrix, d_J, d_K,
                               d_fock_matrix, num_basis);
                       });

    // free the memory
    sycl::free(d_J, workq);
    sycl::free(d_K, workq);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void computeFockMatrix_RI_UHF(const real_t *d_density_matrix_a,
                              const real_t *d_density_matrix_b,
                              const real_t *d_core_hamiltonian_matrix,
                              const real_t *d_intermediate_matrix_B,
                              real_t *d_fock_matrix_a, real_t *d_fock_matrix_b,
                              const int num_basis,
                              const int num_auxiliary_basis) try {
//    dpct::device_ext &dev_ct1 = dpct::get_current_device();
//    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    //cublasManager cublas;
//    dpct::blas::descriptor_ptr cublasHandle = GPUHandle::cublas();
     sycl::queue& workq = GPUHandle::syclsolver();

    dpct::err0 err;

    // the following is used in the two kernels. So, if necessary, it should be changed for each kernel.
    const int num_threads = 256;
    const int num_blocks = (num_basis * num_basis + num_threads - 1) / num_threads;

    ////////////////////////////////// compute J-matrix //////////////////////////////////
    real_t* d_J = nullptr;
    real_t* d_density_matrix = nullptr;
    try {
        d_J = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for J matrix: ") +
            std::string(e.what()));
    }
    try {
        d_density_matrix = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for density matrix: ") +
            std::string(e.what()));
    }

    // D = D_a + D_b
    matrixAddition(d_density_matrix_a, d_density_matrix_b, d_density_matrix, num_basis);

    // W = B D (Matrix(M_aux x M^2 matrix) * Vector (M^2 x 1) )
    real_t* d_W = nullptr;
    try {
        d_W = sycl::malloc_device<real_t>(num_auxiliary_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for W vector: ") +
            std::string(e.what()));
    }

    real_t alpha = 1.0;
    real_t beta = 0.0;

    oneapi::mkl::blas::gemv(workq, oneapi::mkl::transpose::trans, num_basis*num_basis,
                            num_auxiliary_basis, alpha, d_intermediate_matrix_B,
                            num_basis*num_basis, d_density_matrix, 1, beta, d_W, 1);
//    cublasDgemv(cublasHandle, CUBLAS_OP_T, num_basis*num_basis, num_auxiliary_basis, &alpha, d_intermediate_matrix_B, num_basis*num_basis, d_density_matrix, 1, &beta, d_W, 1);

    // J = sum(W[i] * B[i])
    workq.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           weighted_sum_matrices_kernel(
                               d_J, d_intermediate_matrix_B, d_W, num_basis,
                               num_auxiliary_basis, false);
                       });

    // free the memory
    sycl::free(d_W, workq);
    sycl::free(d_density_matrix, workq);

    ////////////////////////////////// compute K-matrix //////////////////////////////////
    real_t* d_T = nullptr;
    real_t* d_V = nullptr;
    try {
        d_T = sycl::malloc_device<real_t>(num_auxiliary_basis * num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for T matrix: ") +
            std::string(e.what()));
    }
    try {
        d_V = sycl::malloc_device<real_t>(num_auxiliary_basis * num_basis * num_basis,workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for V matrix: ") +
            std::string(e.what()));
    }

    ////////////// compute Ka-matrix //////////////
    real_t* d_Ka = nullptr;
    try {
        d_Ka = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for Ka matrix: ") +
            std::string(e.what()));
    }

    // T^p = B^p Da^T
    // Note: cublasDgemmBatched should be used?
    for(int p=0; p<num_auxiliary_basis; p++){
        matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], d_density_matrix_a, &d_T[p*num_basis*num_basis], num_basis, false, true);
    }

    // V^p = B^p (T^p)^T
    // Note: cublasDgemmBatched should be used?
    for(int p=0; p<num_auxiliary_basis; p++){
        matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], &d_T[p*num_basis*num_basis], &d_V[p*num_basis*num_basis], num_basis, false, true);
    }

    // Ka = sum(V^p)
    workq.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           sum_matrices_kernel(d_Ka, d_V, num_basis,
                                               num_auxiliary_basis, false);
                       });

    ////////////// compute Kb-matrix //////////////
    real_t* d_Kb = nullptr;
    try {
        d_Kb = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for Kb matrix: ") +
            std::string(e.what()));
    }

    // T^p = B^p Da^T
    // Note: cublasDgemmBatched should be used?
    for(int p=0; p<num_auxiliary_basis; p++){
        matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], d_density_matrix_b, &d_T[p*num_basis*num_basis], num_basis, false, true);
    }

    // V^p = B^p (T^p)^T
    // Note: cublasDgemmBatched should be used?
    for(int p=0; p<num_auxiliary_basis; p++){
        matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], &d_T[p*num_basis*num_basis], &d_V[p*num_basis*num_basis], num_basis, false, true);
    }

    // Kb = sum(V^p)
    workq.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           sum_matrices_kernel(d_Kb, d_V, num_basis,
                                               num_auxiliary_basis, false);
                       });

    // free the memory
    sycl::free(d_T, workq);
    sycl::free(d_V, workq);

    ////////////////////////////////// compute Fock matrix //////////////////////////////////

    // F_a = H + J - K_a
    // F_b = H + J - K_b
    workq.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           computeFockMatrix_RI_UHF_kernel(
                               d_core_hamiltonian_matrix, d_J, d_Ka,
                               d_fock_matrix_a, num_basis);
                       });
    workq.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           computeFockMatrix_RI_UHF_kernel(
                               d_core_hamiltonian_matrix, d_J, d_Kb,
                               d_fock_matrix_b, num_basis);
                       });

    // free the memory
    sycl::free(d_J, workq);
    sycl::free(d_Ka, workq);
    sycl::free(d_Kb, workq);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void computeFockMatrix_RI_ROHF(
    const real_t *d_density_matrix_closed, const real_t *d_density_matrix_open,
    const real_t *d_core_hamiltonian_matrix, const real_t *d_coefficient_matrix,
    const real_t *d_overlap_matrix, const real_t *d_intermediate_matrix_B,
    const ROHF_ParameterSet ROH_parameters, real_t *d_fock_matrix_closed,
    real_t *d_fock_matrix_open, real_t *d_fock_matrix, const int num_closed,
    const int num_open, const int num_basis,
    const int num_auxiliary_basis) try {
//    dpct::device_ext &dev_ct1 = dpct::get_current_device();
//    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::queue& workq = GPUHandle::syclsolver();
    //cublasManager cublas;
//    dpct::blas::descriptor_ptr cublasHandle = GPUHandle::cublas();

    dpct::err0 err;

    real_t* d_temp_F_MO_closed = nullptr; // Fock matrix for the closed-shell MO
    real_t* d_temp_F_MO_open = nullptr; // Fock matrix for the open-shell MO
    real_t* d_temp_R_MO = nullptr; /// unified Fock matrix R_MO
    real_t* d_temp_matrix1 = nullptr;
    real_t* d_temp_matrix2 = nullptr;
    try {
        d_temp_F_MO_closed = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for closed-shell Fock matrix: ") +
            std::string(e.what()));
    }
    try {
        d_temp_F_MO_open = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for open-shell Fock matrix: ") +
            std::string(e.what()));
    }
    try {
        d_temp_R_MO = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for unified Fock matrix: ") +
            std::string(e.what()));
    }
    try {
        d_temp_matrix1 = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for temporary matrix 1: ") +
            std::string(e.what()));
    }
    try {
        d_temp_matrix2 = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for temporary matrix 2: ") +
            std::string(e.what()));
    }

    {// compute the Fock matrices for the closed- and open-shell orbitals using RI approximation

        // the following is used in the two kernels. So, if necessary, it should be changed for each kernel.
        const int num_threads = 256;
        const int num_blocks = (num_basis * num_basis + num_threads - 1) / num_threads;

        ////////////////////////////////// compute J-matrix //////////////////////////////////
        real_t* d_J = nullptr;
        real_t* d_density_matrix = nullptr;
        try {
            d_J = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
        }
        catch (sycl::exception const& e) {
            THROW_EXCEPTION(
                std::string("Failed to allocate device memory for J matrix: ") +
                std::string(e.what()));
        }
        try {
            d_density_matrix = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
        }
        catch (sycl::exception const& e) {
            THROW_EXCEPTION(
                std::string("Failed to allocate device memory for density matrix: ") +
                std::string(e.what()));
        }

        // D = D_closed + D_open
        matrixAddition(d_density_matrix_closed, d_density_matrix_open, d_density_matrix, num_basis);

        // W = B D (Matrix(M_aux x M^2 matrix) * Vector (M^2 x 1) )
        real_t* d_W = nullptr;
        try {
            d_W = sycl::malloc_device<real_t>(num_auxiliary_basis, workq);
        }
        catch (sycl::exception const& e) {
            THROW_EXCEPTION(
                std::string("Failed to allocate device memory for W vector: ") +
                std::string(e.what()));
        }

        real_t alpha = 1.0;
        real_t beta = 0.0;

    oneapi::mkl::blas::gemv(workq, oneapi::mkl::transpose::trans, num_basis*num_basis,
                            num_auxiliary_basis, alpha, d_intermediate_matrix_B,
                            num_basis*num_basis, d_density_matrix, 1, beta, d_W, 1);
//        cublasDgemv(cublasHandle, CUBLAS_OP_T, num_basis*num_basis, num_auxiliary_basis, &alpha, d_intermediate_matrix_B, num_basis*num_basis, d_density_matrix, 1, &beta, d_W, 1);

        // J = sum(W[i] * B[i])
        workq.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, num_threads),
                              sycl::range<3>(1, 1, num_threads)),
            [=](sycl::nd_item<3> item_ct1) {
                weighted_sum_matrices_kernel(d_J, d_intermediate_matrix_B, d_W,
                                             num_basis, num_auxiliary_basis,
                                             false);
            });

        // free the memory
        sycl::free(d_W, workq);

        ////////////////////////////////// compute Kclosed-matrix //////////////////////////////////
        real_t* d_T = nullptr;
        real_t* d_V = nullptr;
        try {
            d_T = sycl::malloc_device<real_t>(num_auxiliary_basis * num_basis * num_basis,workq);
        }
        catch (sycl::exception const& e) {
            THROW_EXCEPTION(
                std::string("Failed to allocate device memory for T matrix: ") +
                std::string(e.what()));
        }
        try {
            d_V = sycl::malloc_device<real_t>(num_auxiliary_basis * num_basis * num_basis,workq);
        }
        catch (sycl::exception const& e) {
            THROW_EXCEPTION(
                std::string("Failed to allocate device memory for V matrix: ") +
                std::string(e.what()));
        }

        ////////////// compute Kclosed-matrix //////////////
        real_t* d_Kclosed = nullptr;
        try {
            d_Kclosed = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
        }
        catch (sycl::exception const& e) {
            THROW_EXCEPTION(
                std::string("Failed to allocate device memory for Kclosed matrix: ") +
                std::string(e.what()));
        }

        // T^p = B^p Da^T
        // Note: cublasDgemmBatched should be used?
        for(int p=0; p<num_auxiliary_basis; p++){
            matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], d_density_matrix, &d_T[p*num_basis*num_basis], num_basis, false, true);
        }

        // V^p = B^p (T^p)^T
        // Note: cublasDgemmBatched should be used?
        for(int p=0; p<num_auxiliary_basis; p++){
            matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], &d_T[p*num_basis*num_basis], &d_V[p*num_basis*num_basis], num_basis, false, true);
        }

        // Kclosed = sum(V^p)
        workq.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, num_threads),
                              sycl::range<3>(1, 1, num_threads)),
            [=](sycl::nd_item<3> item_ct1) {
                sum_matrices_kernel(d_Kclosed, d_V, num_basis,
                                    num_auxiliary_basis, false);
            });

        ////////////// compute Kopen-matrix //////////////
        real_t* d_Kopen = nullptr;
        try {
            d_Kopen = sycl::malloc_device<real_t>(num_basis * num_basis, workq);
        }
        catch (sycl::exception const& e) {
            THROW_EXCEPTION(
                std::string("Failed to allocate device memory for Kopen matrix: ") +
                std::string(e.what()));
        }

        // D = 0.5*D_closed + D_open
        weightedMatrixSum(d_density_matrix_closed, d_density_matrix_open, d_density_matrix, 0.5, 1.0, num_basis);

        // T^p = B^p Da^T
        // Note: cublasDgemmBatched should be used?
        for(int p=0; p<num_auxiliary_basis; p++){
            matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], d_density_matrix, &d_T[p*num_basis*num_basis], num_basis, false, true);
        }

        // V^p = B^p (T^p)^T
        // Note: cublasDgemmBatched should be used?
        for(int p=0; p<num_auxiliary_basis; p++){
            matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], &d_T[p*num_basis*num_basis], &d_V[p*num_basis*num_basis], num_basis, false, true);
        }

        // Kclosed = sum(V^p)
        workq.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, num_threads),
                              sycl::range<3>(1, 1, num_threads)),
            [=](sycl::nd_item<3> item_ct1) {
                sum_matrices_kernel(d_Kopen, d_V, num_basis,
                                    num_auxiliary_basis, false);
            });

        // free the memory
        sycl::free(d_T, workq);
        sycl::free(d_V, workq);

        ////////////////////////////////// compute Fock matrix //////////////////////////////////

        // Fclosed = H + J - 0.5*Kclosed
        // Fopen = 0.5*(H + J - Kopen)
        workq.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, num_threads),
                              sycl::range<3>(1, 1, num_threads)),
            [=](sycl::nd_item<3> item_ct1) {
                computeFockMatrix_RI_ROHF_kernel(
                    d_core_hamiltonian_matrix, d_J, d_Kclosed, d_Kopen,
                    d_fock_matrix_closed, d_fock_matrix_open, num_basis);
            });

        // free the memory
        sycl::free(d_J, workq);
        sycl::free(d_Kclosed, workq);
        sycl::free(d_Kopen, workq);
        sycl::free(d_density_matrix, workq);
    }



    { // Transforms the Fock matrices from AO to the MO
        // F_MO_closed = C^T F_AO_closed C
        matrixMatrixProduct(d_coefficient_matrix, d_fock_matrix_closed, d_temp_matrix1, num_basis, true, false);
        matrixMatrixProduct(d_temp_matrix1, d_coefficient_matrix, d_temp_F_MO_closed, num_basis, false, false);

        // F_MO_open = C F_AO_open C
        matrixMatrixProduct(d_coefficient_matrix, d_fock_matrix_open, d_temp_matrix1, num_basis, true, false);
        matrixMatrixProduct(d_temp_matrix1, d_coefficient_matrix, d_temp_F_MO_open, num_basis, false, false);
    }

    { // compute the unified Fock matrix R_MO
        const size_t num_elements = num_basis * (num_basis+1) / 2;
        const size_t threads_per_block = 256;
        const size_t num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
        workq.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, threads_per_block),
                              sycl::range<3>(1, 1, threads_per_block)),
            [=](sycl::nd_item<3> item_ct1) {
                computeUnifiedFockMatrix_ROHF_kernel(
                    d_temp_F_MO_closed, d_temp_F_MO_open, ROH_parameters,
                    d_temp_R_MO, num_closed, num_open, num_basis);
            });
    }

    { // Transform the unified Fock matrix from MO to AO by F_AO = S*C*R_MO*C^T*S
        // temp1 = S*C
        matrixMatrixProduct(d_overlap_matrix, d_coefficient_matrix, d_temp_matrix1, num_basis, false, false);
        // temp2 = temp1 * R_MO
        matrixMatrixProduct(d_temp_matrix1, d_temp_R_MO, d_temp_matrix2, num_basis, false, false);
        // temp1 = temp2 * C^T
        matrixMatrixProduct(d_temp_matrix2, d_coefficient_matrix, d_temp_matrix1, num_basis, false, true);
        // temp2 = temp1 * S
        matrixMatrixProduct(d_temp_matrix1, d_overlap_matrix, d_fock_matrix, num_basis, false, false);
    }

    // free the temporary memory
    sycl::free(d_temp_F_MO_closed, workq);
    sycl::free(d_temp_F_MO_open, workq);
    sycl::free(d_temp_R_MO, workq);
    sycl::free(d_temp_matrix1, workq);
    sycl::free(d_temp_matrix2, workq);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void computeTwoCenterERIs(
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, 
    const PrimitiveShell* d_auxiliary_primitive_shells, 
    const real_t* d_auxiliary_cgto_nomalization_factors, 
    real_t* d_two_center_eri, 
    const int num_auxiliary_basis,
    const real_t* d_boys_grid,
    const real_t* d_auxiliary_schwarz_upper_bound_factors,
    const real_t schwarz_screening_threshold,
    const bool verbose)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    // ここに２中心積分を計算するコードを書く    
    const int threads_per_block = 128;
    const int auxiliary_shell_type_count = auxiliary_shell_type_infos.size();


    std::vector<std::pair<int, int>> shell_pairs;

    // a <= b の2つ組を作る
    for (int a = 0; a < auxiliary_shell_type_count; ++a) {
        for (int b = a; b < auxiliary_shell_type_count; ++b) {
            shell_pairs.emplace_back(a, b);
        }
    }

    // (a + b) の降順にソート
    std::sort(shell_pairs.begin(), shell_pairs.end(),
        [](const auto& lhs, const auto& rhs) {
            return (lhs.first + lhs.second) > (rhs.first + rhs.second);  // 降順
        });

    // make multi stream
    const int num_kernels = shell_pairs.size();
    std::vector<dpct::queue_ptr> streams(num_kernels);

    // for-loop for sorted shell-type (s0, s1)
    int stream_id = 0;
    for(const auto& pair: shell_pairs) {
        int s0, s1;
        std::tie(s0, s1) = pair;

        const ShellTypeInfo shell_s0 = auxiliary_shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = auxiliary_shell_type_infos[s1];

        const int num_shell_pairs = (s0==s1) ? (shell_s0.count*(shell_s0.count+1)/2) : (shell_s0.count*shell_s1.count); // the number of pairs of primitive shells = the number of threads
        const int num_blocks = (num_shell_pairs + threads_per_block - 1) / threads_per_block; // the number of blocks
        sycl::range<3> blocks(1, 1, num_blocks);
        sycl::range<3> threads(1, 1, threads_per_block);

        // real_t*, PrimitiveShell*, real_t*, ShellTypeInfo, ShellTypeInfo, int, int
            streams[stream_id++]->submit([&](sycl::handler& cgh){
            cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                gpu::launch_2center_kernel(
                    s0, s1, d_two_center_eri,
                    d_auxiliary_primitive_shells, d_auxiliary_cgto_nomalization_factors,
                    shell_s0, shell_s1, num_shell_pairs,
                    d_auxiliary_schwarz_upper_bound_factors, 
                    schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
            });
            });
/*
        dpct::kernel_launcher::launch(
            gpu::get_2center_kernel(s0, s1), num_blocks, threads_per_block, 0,
            streams[stream_id++], d_two_center_eri,
            d_auxiliary_primitive_shells, d_auxiliary_cgto_nomalization_factors,
            shell_s0, shell_s1, num_shell_pairs,
            d_auxiliary_schwarz_upper_bound_factors,
            schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
*/
        if(verbose){
            std::cout << "(" << shell_type_to_shell_name(s0) << "|" << shell_type_to_shell_name(s1) << "): ";
            std::cout << "|" << shell_type_to_shell_name(s0) << "|=" << shell_s0.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s1) << "|=" << shell_s1.count << ", ";
            std::cout << "|[a|b]|=" << num_shell_pairs << ", ";
            std::cout << "num_blocks: " << num_blocks << std::endl;
        }
    }

    // syncronize streams
    dev_ct1.queues_wait_and_throw();

    // destory streams
    for (int i = 0; i < num_kernels; i++) {
        dev_ct1.destroy_queue(streams[i]);
    }
}

inline int calcIdx_triangular_(int a, int b, int N){
    return (int)(a*N - (a*(a-1))/2) + (b-a);
}

void computeThreeCenterERIs(
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells,
    const real_t* d_cgto_nomalization_factors,
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,
    const PrimitiveShell* d_auxiliary_primitive_shells,
    const real_t* d_auxiliary_cgto_nomalization_factors,
    real_t* d_three_center_eri,
    const size_t2* d_primitive_shell_pair_indices,
    const int num_basis,
    const int num_auxiliary_basis,
    const real_t* d_boys_grid,
    const real_t* d_schwarz_upper_bound_factors,
    const real_t* d_auxiliary_schwarz_upper_bound_factors,
    const real_t schwarz_screening_threshold,
    const bool verbose)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
    const int threads_per_block = 128;
    const int shell_type_count = shell_type_infos.size();
    const int auxiliary_shell_type_count = auxiliary_shell_type_infos.size();


    // Call the kernel functions from (ss|s),... (e.g. (ss|s), (ss|p), (sp|s), (sp|p), (pp|s), (pp|p) for s and p shells)

    // list shell-triples for sorted shell-type (s0, s1, s2)
    std::vector<std::tuple<int, int, int>> shell_triples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < auxiliary_shell_type_count; ++c) {
                shell_triples.emplace_back(a, b, c);
            }
        }
    }
    // sort by sum (a + b + c) in descending order
    std::sort(shell_triples.begin(), shell_triples.end(),
        [](const auto& lhs, const auto& rhs) {
            int sum_lhs = std::get<0>(lhs) + std::get<1>(lhs) + std::get<2>(lhs);
            int sum_rhs = std::get<0>(rhs) + std::get<1>(rhs) + std::get<2>(rhs);
            return sum_lhs > sum_rhs;  // 降順
        });


    // make multi stream
    const int num_kernels = shell_triples.size();
    std::vector<dpct::queue_ptr> streams(num_kernels);

    // for-loop for sorted shell-type (s0, s1, s2, s3)
    int stream_id = 0;
    for(const auto& triple: shell_triples) {
        int s0, s1, s2;
        std::tie(s0, s1, s2) = triple;

        const ShellTypeInfo shell_s0 = shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = shell_type_infos[s1];
        const ShellTypeInfo shell_s2 = auxiliary_shell_type_infos[s2];

        const int num_tasks = ( (s0==s1) ? (shell_s0.count*(shell_s0.count+1)/2) : (shell_s0.count*shell_s1.count) ) * shell_s2.count; // the number of pairs of primitive shells = the number of threads
        const int num_blocks = (num_tasks + threads_per_block - 1) / threads_per_block; // the number of blocks
        sycl::range<3> blocks(1, 1, num_blocks);
        sycl::range<3> threads(1, 1, threads_per_block);

        // real_t*, PrimitiveShell*, real_t*, ShellTypeInfo, ShellTypeInfo, int, int
            streams[stream_id++]->submit([&](sycl::handler& cgh){
            const size_t shell_pair_index = shell_pair_type_infos[calcIdx_triangular_(s0, s1, shell_type_count)].start_index;
            const size_t2* dp_ind = &d_primitive_shell_pair_indices[shell_pair_index];
            const double* schwarz_bound = &d_schwarz_upper_bound_factors[shell_pair_index];
            cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                gpu::launch_3center_kernel(
                    s0, s1, s2,
                    d_three_center_eri, d_primitive_shells,
                    d_auxiliary_primitive_shells, d_auxiliary_cgto_nomalization_factors,
                    d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
                    num_tasks, num_basis, 
//                    &d_primitive_shell_pair_indices[shell_pair_type_infos
//                      [calcIdx_triangular_(s0, s1, shell_type_count)] .start_index],
                    dp_ind,
//                    &d_schwarz_upper_bound_factors[shell_pair_type_infos
//                      [calcIdx_triangular_(s0, s1, shell_type_count)] .start_index],
                    schwarz_bound,
                    d_auxiliary_schwarz_upper_bound_factors, 
                    schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
            });
            });
/*
        dpct::kernel_launcher::launch(
            gpu::launch_3center_kernel, num_blocks, threads_per_block,
            0, streams[stream_id++], s0, s1, s2,
            d_three_center_eri, d_primitive_shells,
            d_auxiliary_primitive_shells, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            &d_primitive_shell_pair_indices
                [shell_pair_type_infos[calcIdx_triangular_(s0, s1,
                                                           shell_type_count)]
                     .start_index],
            &d_schwarz_upper_bound_factors
                [shell_pair_type_infos[calcIdx_triangular_(s0, s1,
                                                           shell_type_count)]
                     .start_index],
            d_auxiliary_schwarz_upper_bound_factors,
            schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
*/
        if(verbose){
            std::cout << "(" << shell_type_to_shell_name(s0) << shell_type_to_shell_name(s1) << "|" << shell_type_to_shell_name(s2)<< "): ";
            std::cout << "|" << shell_type_to_shell_name(s0) << "|=" << shell_s0.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s1) << "|=" << shell_s1.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s2) << "|=" << shell_s2.count << ", ";
            std::cout << "|[ab|c]|=" << num_tasks << ", ";
            std::cout << "num_blocks: " << num_blocks << std::endl;
        }

    }

    // syncronize streams
    dev_ct1.queues_wait_and_throw();

    // destory streams
    for (int i = 0; i < num_kernels; i++) {
        dev_ct1.destroy_queue(streams[i]);
    }
}

/**
 * @brief Compute the Schwarz upper bounds for the shell pairs.
 * @param shell_type_infos Information about the basis functions
 * @param shell_pair_type_infos Information about the shell pairs
 * @param d_primitive_shells Pointer to the primitive shells in device memory
 * @param d_boys_grid Pointer to the precomputed grid values of the Boys function in device memory
 * @param d_cgto_normalization_factors Pointer to the normalization factors of the CGTOs in device memory
 * @param d_upper_bound_factors Pointer to store the upper bound factors in device memory to be stored
 * @param verbose Whether to print additional information
 * @details This function computes the Schwarz upper bounds for the shell pairs.
 */
void computeSchwarzUpperBounds(
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells,
    const real_t* d_boys_grid,
    const real_t* d_cgto_normalization_factors,
    real_t* d_upper_bound_factors,
    const bool verbose)
{
    const int threads_per_block = 256; // the number of threads per block
    const int shell_type_count = shell_type_infos.size();

    for (int s0 = 0; s0 < shell_type_count; ++s0) {
        for (int s1 = s0; s1 < shell_type_count; ++s1) {
            const ShellTypeInfo shell_s0 = shell_type_infos[s0];
            const ShellTypeInfo shell_s1 = shell_type_infos[s1];
            const size_t head = shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)].start_index;
            const size_t num_bra = shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)].count;
            const size_t num_blocks = (num_bra + threads_per_block - 1) / threads_per_block; // the number of blocks
//    dpct::dim3 blocks(num_blocks,1,1);
//    dpct::dim3 threads(threads_per_block,1,1);
    sycl::queue& workq = GPUHandle::syclsolver();
    sycl::range<3> blocks(1, 1, num_blocks);
    sycl::range<3> threads(1, 1, threads_per_block);

            workq.submit([&](sycl::handler& cgh){
            cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                launch_schwarz_kernel(
                    s0, s1,
                    d_primitive_shells, d_cgto_normalization_factors,
                    shell_s0, shell_s1, head, num_bra, d_boys_grid,
                    d_upper_bound_factors);
            });
            });
/*
            dpct::kernel_launcher::launch(
                gpu::launch_schwarz_kernel, blocks, threads,
                0, 0, s0, s1,
                d_primitive_shells, d_cgto_normalization_factors,
                shell_s0, shell_s1, head, num_bra, d_boys_grid,
                d_upper_bound_factors);
*/
        }
    }
}

/**
 * @brief Compute the Schwarz upper bounds for the shell pairs.
 * @param shell_aux_type_infos Information about the auxiliary primitive shells
 * @param d_primitive_shells_aux Pointer to the auxiliary primitive shells in device memory
 * @param d_boys_grid Pointer to the precomputed grid values of the Boys function in device memory
 * @param d_cgto_aux_normalization_factors Pointer to the normalization factors of the auxiliary CGTOs in device memory
 * @param d_upper_bound_factors_aux Pointer to store the upper bound factors in device memory to be stored
 * @param verbose Whether to print additional information
 * @details This function computes the Schwarz upper bounds for the shell pairs.
 */
void computeAuxiliarySchwarzUpperBounds(
    const std::vector<ShellTypeInfo>& shell_aux_type_infos, 
    const PrimitiveShell* d_primitive_shells_aux, 
    const real_t* d_boys_grid, 
    const real_t* d_cgto_aux_normalization_factors, 
    real_t* d_upper_bound_factors_aux, 
    const bool verbose)
{
    const int threads_per_block = 256; // the number of threads per block
    const int shell_type_count = shell_aux_type_infos.size();

    for (int s0 = 0; s0 < shell_type_count; ++s0) {
        const ShellTypeInfo shell_s0 = shell_aux_type_infos[s0];
        const size_t head = shell_s0.start_index;
        const size_t num_bra = shell_s0.count;
        const size_t num_blocks = (num_bra + threads_per_block - 1) / threads_per_block; // the number of blocks
        sycl::queue& workq = GPUHandle::syclsolver();
        sycl::range<3> blocks(1, 1, num_blocks);
        sycl::range<3> threads(1, 1, threads_per_block);

            workq.submit([&](sycl::handler& cgh){
            cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                launch_schwarz_aux_kernel(
                    s0, 
                    d_primitive_shells_aux, d_cgto_aux_normalization_factors,
                    shell_s0, head, num_bra, d_boys_grid, d_upper_bound_factors_aux);
            });
            });
/*
        dpct::kernel_launcher::launch(
            gpu::get_schwarz_aux_kernel(s0), num_blocks, threads_per_block, 0,
            0, d_primitive_shells_aux, d_cgto_aux_normalization_factors,
            shell_s0, head, num_bra, d_boys_grid, d_upper_bound_factors_aux);
*/
    }
}



void computeFockMatrix_Direct_RHF(
    const real_t* d_density_matrix,
    const real_t* d_core_hamiltonian_matrix,
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const PrimitiveShell* d_primitive_shells, 
    const real_t* d_cgto_nomalization_factors, 
    const real_t* d_boys_grid, 
    const real_t* d_schwarz_upper_bound_factors,
    const real_t  schwarz_screening_threshold,
    real_t* d_fock_matrix,
    const int num_basis,
    const int verbose)
{
    // ここにDirect-SCF(RHF)のFock行列計算を書く
    THROW_EXCEPTION("Not implemented yet.");
}

void computeMullikenPopulation_RHF(const real_t *d_density_matrix,
                                   const real_t *overlap_matrix,
                                   real_t *mulliken_population_basis,
                                   const int num_basis) try {
//  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::queue& workq = GPUHandle::syclsolver();

    dpct::err0 err;

    real_t* d_mulliken_population = nullptr;
    try {
        d_mulliken_population =sycl::malloc_device<real_t>(num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for Mulliken population: ") +
            std::string(e.what()));
    }

    // Compute the diagonal elements of the product of the density matrix and the overlap matrix
    const size_t threads_per_block = 256;
    const size_t num_blocks = (num_basis + threads_per_block - 1) / threads_per_block;
    workq.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, threads_per_block),
                          sycl::range<3>(1, 1, threads_per_block)),
        [=](sycl::nd_item<3> item_ct1) {
            compute_diagonal_of_product(d_density_matrix, overlap_matrix,
                                        d_mulliken_population, num_basis);
        });

    // Copy the result to the host
    workq
        .memcpy(mulliken_population_basis, d_mulliken_population,
                num_basis * sizeof(real_t))
        .wait();

    // Free the memory for the temporary matrix
    sycl::free(d_mulliken_population, workq);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void computeMullikenPopulation_UHF(const real_t *d_density_matrix_a,
                                   const real_t *d_density_matrix_b,
                                   const real_t *overlap_matrix,
                                   real_t *mulliken_population_basis,
                                   const int num_basis) try {
//  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::queue& workq = GPUHandle::syclsolver();
    dpct::err0 err;

    real_t* d_mulliken_population = nullptr;
    try {
        d_mulliken_population =sycl::malloc_device<real_t>(num_basis, workq);
    }
    catch (sycl::exception const& e) {
        THROW_EXCEPTION(
            std::string("Failed to allocate device memory for Mulliken population: ") +
            std::string(e.what()));
    }

    // Compute the diagonal elements of the product of the density matrix and the overlap matrix
    const size_t threads_per_block = 256;
    const size_t num_blocks = (num_basis + threads_per_block - 1) / threads_per_block;
    workq.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, threads_per_block),
                          sycl::range<3>(1, 1, threads_per_block)),
        [=](sycl::nd_item<3> item_ct1) {
            compute_diagonal_of_product_sum(d_density_matrix_a,
                                            d_density_matrix_b, overlap_matrix,
                                            d_mulliken_population, num_basis);
        });

    // Copy the result to the host
    workq
        .memcpy(mulliken_population_basis, d_mulliken_population,
                num_basis * sizeof(real_t))
        .wait();

    // Free the memory for the temporary matrix
    sycl::free(d_mulliken_population, workq);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void constructERIHash(
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells, 
    const real_t* d_boys_grid, 
    const real_t* d_cgto_normalization_factors, 
    // Hash memoryへのポインタ
    const bool verbose)
{
    // ここにERIを計算してハッシュメモリに追加するコードを書く（GPUカーネルを呼ぶ）
    //const int threads_per_block = 256; // the number of threads per block
    // ...
    // GPUカーネルはgpu_kernels.hppにプロトタイプ宣言、gpu_kernels.cuに実装を記述
    //constructERIHash_kernel<<<1, threads_per_block>>>(shell_type_infos, shell_pair_type_infos, d_primitive_shells, d_cgto_normalization_factors, /* Hash memory へのポインタ, */ verbose);

    THROW_EXCEPTION("Not implemented yet.");

}


void computeFockMatrix_Hash_RHF(
    const real_t* d_density_matrix,
    const real_t* d_core_hamiltonian_matrix,
    // Hash memoryへのポインタ
    real_t* d_fock_matrix,
    const int num_basis,
    const int verbose)
{
    // ここにERIハッシュを使ってFock行列を計算するコードを書く
    // 
    // computeFockMatrix_Hash_RHF_kernel<<<1, 256>>>(d_density_matrix, d_core_hamiltonian_matrix, /* Hash memory へのポインタ */, d_fock_matrix, num_basis);


    THROW_EXCEPTION("Not implemented yet.");
}


} // namespace gansu::gpu
