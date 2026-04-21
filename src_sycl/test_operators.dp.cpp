/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
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
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/lapack.hpp>
#include "test_operators.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"

namespace gansu {

// ========== CUDA Kernels ==========

/**
 * @brief Kernel for diagonal matrix-vector product: y = D * x
 */
void apply_diagonal_kernel(sycl::nd_item<1> item,
    const real_t* d_diagonal,
    const real_t* d_input,
    real_t* d_output,
    int dim
) {
    size_t idx = item.get_global_linear_id();
    if (idx < dim) {
        d_output[idx] = d_diagonal[idx] * d_input[idx];
    }
}

/**
 * @brief Kernel for diagonal preconditioner: y = x / D
 */
void apply_diagonal_preconditioner_kernel(sycl::nd_item<1> item,
    const real_t* d_diagonal,
    const real_t* d_input,
    real_t* d_output,
    int dim
) {
    size_t idx = item.get_global_linear_id();
    if (idx < dim) {
        real_t diag_val = d_diagonal[idx];
        // Avoid division by zero
        d_output[idx] =
            (sycl::fabs(diag_val) > 1e-12) ? d_input[idx] / diag_val : 0.0;
    }
}

// ========== DiagonalOperator Implementation ==========

DiagonalOperator::DiagonalOperator(const std::vector<real_t> &diagonal) try
    : dim_(diagonal.size()), d_diagonal_(nullptr) {
    if (dim_ <= 0) {
        THROW_EXCEPTION("DiagonalOperator: dimension must be positive");
    }

    // Allocate device memory
    sycl::queue& workq = gpu::GPUHandle::syclqueue();
    d_diagonal_ = tracked_syclMalloc<real_t>(dim_, workq);

    // Copy diagonal to device
    try {
        workq.memcpy(d_diagonal_, diagonal.data(), dim_ * sizeof(real_t))
            .wait();
    } catch(const sycl::exception& e) {
        tracked_syclFree(d_diagonal_);
        THROW_EXCEPTION("DiagonalOperator: failed to copy diagonal to device");
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

DiagonalOperator::~DiagonalOperator() {
    if (d_diagonal_) {
        tracked_syclFree(d_diagonal_);
    }
}

void DiagonalOperator::apply(const real_t* d_input, real_t* d_output) const {
    if (!d_input || !d_output) {
        THROW_EXCEPTION("DiagonalOperator::apply: null pointer");
    }
    sycl::queue& workq = gpu::GPUHandle::syclqueue();

    int threads_per_block = 256;
    int maxWG = workq.get_device().get_info<sycl::info::device::max_work_group_size>();
    threads_per_block = std::min(maxWG, threads_per_block);
    int num_blocks = (dim_ + threads_per_block - 1) / threads_per_block;

    sycl::range<1> local(threads_per_block);
    sycl::range<1> global(num_blocks * threads_per_block);

    try{
    workq.submit([&](sycl::handler &cgh) {
        const real_t *d_diagonal__ct0 = d_diagonal_;
        auto dim__ct3 = dim_;

        cgh.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) {
                apply_diagonal_kernel(item, d_diagonal__ct0, d_input, d_output,
                                      dim__ct3);
            });
    }).wait_and_throw();
    } catch(const sycl::exception& e) {
        THROW_EXCEPTION(std::string("DiagonalOperator::apply (SYCL) failed: ") + e.what());
    }
}

void DiagonalOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    if (!d_input || !d_output) {
        THROW_EXCEPTION("DiagonalOperator::apply_preconditioner: null pointer");
    }
    sycl::queue& workq = gpu::GPUHandle::syclqueue();

    int threads_per_block = 256;
    int maxWG = workq.get_device().get_info<sycl::info::device::max_work_group_size>();
    threads_per_block = std::min(maxWG, threads_per_block);
    int num_blocks = (dim_ + threads_per_block - 1) / threads_per_block;

    sycl::range<1> local(threads_per_block);
    sycl::range<1> global(num_blocks * threads_per_block);

    try {
    workq.submit([&](sycl::handler &cgh) {
        const real_t *d_diagonal__ct0 = d_diagonal_;
        auto dim__ct3 = dim_;

        cgh.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) {
                apply_diagonal_preconditioner_kernel(item, d_diagonal__ct0, d_input,
                                                     d_output, dim__ct3);
            });
    });
    } catch(const sycl::exception& e) {
        THROW_EXCEPTION(std::string("DiagonalOperator::apply_preconditioner: "
                                    "kernel launch failed:") + e.what());
    }
}

// ========== SymmetricMatrixOperator Implementation ==========

SymmetricMatrixOperator::SymmetricMatrixOperator(
    const std::vector<real_t> &matrix, int n) try
    : dim_(n), d_matrix_(nullptr), d_diagonal_(nullptr) {
    sycl::queue& workq = gpu::GPUHandle::syclqueue();
    if (dim_ <= 0) {
        THROW_EXCEPTION("SymmetricMatrixOperator: dimension must be positive");
    }

    if (matrix.size() != static_cast<size_t>(dim_ * dim_)) {
        THROW_EXCEPTION("SymmetricMatrixOperator: matrix size mismatch");
    }

    // Allocate device memory for matrix
    try {
        d_matrix_ = tracked_syclMalloc<real_t>(dim_ * dim_, workq);

    // Copy matrix to device
        workq.memcpy(d_matrix_, matrix.data(), dim_ * dim_ * sizeof(real_t))
            .wait();
    } catch(const sycl::exception& e) {
        tracked_syclFree(d_matrix_);
        THROW_EXCEPTION("SymmetricMatrixOperator: failed to copy matrix to device");
    }

    // Extract and store diagonal for preconditioner
    std::vector<real_t> diagonal(dim_);
    for (int i = 0; i < dim_; ++i) {
        diagonal[i] = matrix[i * dim_ + i];
    }

    d_diagonal_ = tracked_syclMalloc<real_t>(dim_, workq);
    try {
        workq.memcpy(d_diagonal_, diagonal.data(), dim_ * sizeof(real_t))
            .wait();
    } catch(const sycl::exception& e) {
        /*
        DPCT1001:19: The statement could not be removed.
        */
        tracked_syclFree(d_matrix_);
        tracked_syclFree(d_diagonal_);
        THROW_EXCEPTION("SymmetricMatrixOperator: failed to copy diagonal to device");
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

SymmetricMatrixOperator::~SymmetricMatrixOperator() {
    if (d_matrix_) {
        tracked_syclFree(d_matrix_);
    }
    if (d_diagonal_) {
        tracked_syclFree(d_diagonal_);
    }
}

void SymmetricMatrixOperator::apply(const real_t *d_input,
                                    real_t *d_output) const try {
    if (!d_input || !d_output) {
        THROW_EXCEPTION("SymmetricMatrixOperator::apply: null pointer");
    }

    sycl::queue& workq = gpu::GPUHandle::syclqueue();

    const real_t alpha = 1.0;
    const real_t beta = 0.0;

    try {
        oneapi::mkl::blas::column_major::gemv(
            workq, oneapi::mkl::transpose::trans, dim_, dim_,
            alpha, d_matrix_, dim_, d_input, 1, beta, d_output, 1);

        workq.wait_and_throw();
    } catch (const sycl::exception& e) {
        THROW_EXCEPTION(std::string("SymmetricMatrixOperator::apply (SYCL) failed: ") + e.what());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void SymmetricMatrixOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    if (!d_input || !d_output) {
        THROW_EXCEPTION("SymmetricMatrixOperator::apply_preconditioner: null pointer");
    }
    sycl::queue& workq = gpu::GPUHandle::syclqueue();

    int threads_per_block = 256;
    int maxWG = workq.get_device().get_info<sycl::info::device::max_work_group_size>();
    threads_per_block = std::min(maxWG, threads_per_block);
    int num_blocks = (dim_ + threads_per_block - 1) / threads_per_block;

    sycl::range<1> local(threads_per_block);
    sycl::range<1> global(num_blocks * threads_per_block);

    try {
    workq.submit([&](sycl::handler &cgh) {
        const real_t *d_diagonal__ct0 = d_diagonal_;
        auto dim__ct3 = dim_;

        cgh.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) {
                apply_diagonal_preconditioner_kernel(item, d_diagonal__ct0, d_input,
                                                     d_output, dim__ct3);
            });
    });
    } catch(const sycl::exception& e) {
        THROW_EXCEPTION(std::string("SymmetricMatrixOperator::apply_"
                                    "preconditioner: kernel launch failed: ") + e.what());
    }
}

} // namespace gansu
