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
#include "davidson_solver.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>

namespace gansu {

// ========== Constructor ==========

DavidsonSolver::DavidsonSolver(const LinearOperator& linear_op,
                               const DavidsonConfig& config)
    : linear_op_(linear_op),
      config_(config),
      dim_(linear_op.dimension()),
      subspace_dim_(0),
      num_iterations_(0),
      d_subspace_vectors_(nullptr),
      d_sigma_vectors_(nullptr),
      d_subspace_matrix_(nullptr),
      d_subspace_eigenvalues_(nullptr),
      d_subspace_eigenvectors_(nullptr),
      d_residuals_(nullptr),
      d_eigenvectors_(nullptr),
      queue_(gpu::GPUHandle::syclqueue())
{
    // Validate configuration
    if (config_.num_eigenvalues <= 0) {
        THROW_EXCEPTION("DavidsonSolver: num_eigenvalues must be positive");
    }
    if (config_.num_eigenvalues > dim_) {
        THROW_EXCEPTION("DavidsonSolver: num_eigenvalues cannot exceed operator dimension");
    }
    // Cap max_subspace_size to dimension (cannot have more orthogonal vectors than dim)
    if (config_.max_subspace_size > dim_) {
        config_.max_subspace_size = dim_;
    }
    if (config_.max_subspace_size < config_.num_eigenvalues) {
        THROW_EXCEPTION("DavidsonSolver: max_subspace_size must be >= num_eigenvalues");
    }
    if (config_.convergence_threshold <= 0.0) {
        THROW_EXCEPTION("DavidsonSolver: convergence_threshold must be positive");
    }

    // Set default initial subspace size if not specified
    if (config_.initial_subspace_size == 0) {
        config_.initial_subspace_size = std::min(2 * config_.num_eigenvalues,
                                                 config_.max_subspace_size);
    }

    // Cap initial_subspace_size to dimension
    if (config_.initial_subspace_size > dim_) {
        config_.initial_subspace_size = dim_;
    }

    if (config_.initial_subspace_size < config_.num_eigenvalues) {
        THROW_EXCEPTION("DavidsonSolver: initial_subspace_size must be >= num_eigenvalues");
    }

    // Allocate host memory
    h_eigenvalues_.resize(config_.num_eigenvalues);
    residual_norms_.resize(config_.num_eigenvalues);

    // Allocate GPU memory
    allocate_memory();

    if (config_.verbose > 0) {
        std::cout << "\n=== Davidson Solver Initialized ===" << std::endl;
        std::cout << "Operator: " << linear_op_.name() << std::endl;
        std::cout << "Dimension: " << dim_ << std::endl;
        std::cout << "Number of eigenvalues: " << config_.num_eigenvalues << std::endl;
        std::cout << "Max subspace size: " << config_.max_subspace_size << std::endl;
        std::cout << "Convergence threshold: " << std::scientific
                  << config_.convergence_threshold << std::endl;
    }
}

// ========== Destructor ==========

DavidsonSolver::~DavidsonSolver() {
    free_memory();
}

// ========== Memory Management ==========

void DavidsonSolver::allocate_memory() {
    using gansu::tracked_syclMalloc;

    // Subspace vectors: dim × max_subspace_size
    d_subspace_vectors_ = tracked_syclMalloc<real_t>(
                       dim_ * config_.max_subspace_size, queue_);

    // Sigma vectors: dim × max_subspace_size
    d_sigma_vectors_ = tracked_syclMalloc<real_t>(
                       dim_ * config_.max_subspace_size, queue_);

    // Subspace matrix: max_subspace_size × max_subspace_size
    d_subspace_matrix_ = tracked_syclMalloc<real_t>(
                       config_.max_subspace_size * config_.max_subspace_size ,queue_);

    // Subspace eigenvalues: max_subspace_size
    d_subspace_eigenvalues_ = tracked_syclMalloc<real_t>(
                       config_.max_subspace_size, queue_);

    // Subspace eigenvectors: max_subspace_size × max_subspace_size
    d_subspace_eigenvectors_ = tracked_syclMalloc<real_t>(
                       config_.max_subspace_size * config_.max_subspace_size ,queue_);

    // Residuals: dim × num_eigenvalues
    d_residuals_ = tracked_syclMalloc<real_t>(
                       dim_ * config_.num_eigenvalues, queue_);

    // Final eigenvectors: dim × num_eigenvalues
    d_eigenvectors_ = tracked_syclMalloc<real_t>(
                       dim_ * config_.num_eigenvalues, queue_);

    if (config_.verbose > 1) {
        size_t total_bytes = (
            static_cast<size_t>(dim_) * config_.max_subspace_size * 2 +  // subspace + sigma
            static_cast<size_t>(config_.max_subspace_size) * config_.max_subspace_size * 2 +  // matrix + eigvecs
            static_cast<size_t>(dim_) * config_.num_eigenvalues * 2 +    // residuals + eigenvectors
            config_.max_subspace_size                                    // eigenvalues
        ) * sizeof(real_t);

        std::cout << "Davidson memory allocated: "
                  << SyclMemoryManager<real_t>::format_bytes(total_bytes) << std::endl;
    }
}

void DavidsonSolver::free_memory() {
    using gansu::tracked_syclFree;

    if (d_subspace_vectors_) tracked_syclFree(d_subspace_vectors_);
    if (d_sigma_vectors_) tracked_syclFree(d_sigma_vectors_);
    if (d_subspace_matrix_) tracked_syclFree(d_subspace_matrix_);
    if (d_subspace_eigenvalues_) tracked_syclFree(d_subspace_eigenvalues_);
    if (d_subspace_eigenvectors_) tracked_syclFree(d_subspace_eigenvectors_);
    if (d_residuals_) tracked_syclFree(d_residuals_);
    if (d_eigenvectors_) tracked_syclFree(d_eigenvectors_);

    d_subspace_vectors_ = nullptr;
    d_sigma_vectors_ = nullptr;
    d_subspace_matrix_ = nullptr;
    d_subspace_eigenvalues_ = nullptr;
    d_subspace_eigenvectors_ = nullptr;
    d_residuals_ = nullptr;
    d_eigenvectors_ = nullptr;
}

// ========== Main Solve Method ==========

bool DavidsonSolver::solve(const real_t* d_initial_guess) {
    sycl::queue& q = queue_;
    // Initialize subspace
    initialize_subspace(d_initial_guess);
    num_iterations_ = 0;

    bool converged = false;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        num_iterations_ = iter + 1;

        if (config_.verbose > 1) {
            std::cout << "\n=== Davidson Iteration " << (iter + 1) << " ===" << std::endl;
            std::cout << "Subspace dimension: " << subspace_dim_ << std::endl;
        }

        // Apply operator to all subspace vectors
        for (int i = 0; i < subspace_dim_; ++i) {
            linear_op_.apply(&d_subspace_vectors_[i * dim_],
                           &d_sigma_vectors_[i * dim_]);
        }
        q.wait_and_throw();

        // Build subspace matrix
        build_subspace_matrix();

        // Solve subspace eigenvalue problem
        solve_subspace_eigenproblem();

        // Compute Ritz vectors and residuals
        compute_ritz_vectors_and_residuals();

        // Check convergence
        converged = check_convergence();

        if (config_.verbose > 1) {
            std::cout << "Eigenvalues: ";
            for (int i = 0; i < std::min(5, config_.num_eigenvalues); ++i) {
                std::cout << std::scientific << std::setprecision(8)
                         << h_eigenvalues_[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "Residual norms: ";
            for (int i = 0; i < config_.num_eigenvalues; ++i) {
                std::cout << std::scientific << std::setprecision(4)
                         << residual_norms_[i] << " ";
            }
            std::cout << std::endl;
        }

        if (converged) {
            if (config_.verbose > 0) {
                std::cout << "\nDavidson converged in " << num_iterations_
                         << " iterations" << std::endl;
            }
            break;
        }

        // Add correction vectors
        add_correction_vectors();

        // Restart if subspace is full (but not if it spans the entire space,
        // since the next iteration will give the exact solution)
        if (subspace_dim_ >= config_.max_subspace_size && subspace_dim_ < dim_) {
            if (config_.verbose > 1) {
                std::cout << "Restarting Davidson subspace" << std::endl;
            }
            restart_subspace();
        }
    }

    if (!converged && config_.verbose > 0) {
        std::cout << "\nWarning: Davidson did not converge in "
                 << config_.max_iterations << " iterations" << std::endl;
        std::cout << "Final residual norms:" << std::endl;
        for (int i = 0; i < config_.num_eigenvalues; ++i) {
            std::cout << "  λ[" << i << "]: " << std::scientific
                     << residual_norms_[i] << std::endl;
        }
    }

    return converged;
}

// ========== Initialization ==========

void DavidsonSolver::initialize_subspace(const real_t *d_initial_guess) {
    sycl::queue& q = queue_;
    if (d_initial_guess != nullptr) {
        // Use provided guess vectors
        q.memcpy(d_subspace_vectors_, d_initial_guess, dim_ * config_.num_eigenvalues * sizeof(real_t));
        subspace_dim_ = config_.num_eigenvalues;
    } else {
        // Generate random initial vectors on host, then copy to device
        int initial_size = config_.initial_subspace_size;
        size_t total_elements = static_cast<size_t>(dim_) * initial_size;

        std::vector<real_t> h_random(total_elements);
        std::mt19937_64 rng(1234ULL);
        std::normal_distribution<real_t> dist(0.0, 1.0);
        for (size_t i = 0; i < total_elements; ++i) {
            h_random[i] = dist(rng);
        }

       q.memcpy(d_subspace_vectors_, h_random.data(), total_elements * sizeof(real_t)) .wait();
        subspace_dim_ = initial_size;
    }

    // Orthogonalize initial vectors
    orthogonalize_vectors(0, subspace_dim_);

    if (config_.verbose > 1) {
        std::cout << "Initialized subspace with " << subspace_dim_
                 << " vectors" << std::endl;
    }
}

// ========== Orthogonalization ==========

void DavidsonSolver::orthogonalize_vectors(int start_index, int num_vectors) {
    sycl::queue& q = queue_;

    using namespace oneapi::mkl::blas::column_major;

    real_t* d_proj = sycl::malloc_shared<real_t>(1, q);
    real_t* d_norm = sycl::malloc_shared<real_t>(1, q);

    for (int i = start_index; i < start_index + num_vectors; ++i) {
        real_t* d_vec_i = &d_subspace_vectors_[i * dim_];

        // Orthogonalize against all previous vectors
        for (int j = 0; j < i; ++j) {
            const real_t* d_vec_j = &d_subspace_vectors_[j * dim_];

            // Compute projection: proj = <v_j | v_i>
            *d_proj = 0.0;
            dot(q, dim_, d_vec_j, 1, d_vec_i, 1, d_proj);
            q.wait_and_throw();
            real_t proj = *d_proj;

            // Subtract projection: v_i -= proj * v_j
            real_t alpha = -proj;
            axpy(q, dim_, alpha, d_vec_j, 1, d_vec_i, 1);
            q.wait_and_throw();
        }

        // Normalize v_i
        *d_norm = 0.0;
        nrm2(q, dim_, d_vec_i, 1, d_norm);
        q.wait_and_throw();
        real_t norm = *d_norm;

        if (norm < 1e-12) {
            if (config_.verbose > 0) {
                std::cout << "Warning: Linear dependence detected in vector "
                         << i << ", replacing with random vector" << std::endl;
            }
            // Replace with random vector (host-side generation to avoid curand even-count issue)
            std::vector<real_t> h_random(dim_);
            std::mt19937_64 rng(1234ULL + i);
            std::normal_distribution<real_t> dist(0.0, 1.0);
            for (int k = 0; k < dim_; ++k) {
                h_random[k] = dist(rng);
            }

            q.memcpy(d_vec_i, h_random.data(), dim_ * sizeof(real_t)).wait();

            // Re-orthogonalize this single vector
            orthogonalize_vectors(i, 1);
            continue;  // Continue to next vector (don't re-normalize)
        }

        real_t inv_norm = 1.0 / norm;
        scal(q, dim_, inv_norm, d_vec_i, 1);
        q.wait_and_throw();
    }
    sycl::free(d_proj, q);
    sycl::free(d_norm, q);
}

// ========== Subspace Matrix Construction ==========

void DavidsonSolver::build_subspace_matrix() try {
    sycl::queue& q = queue_;

    // Compute H = V^T * Σ using DGEMM (single cuBLAS call)
    // V: dim_ × subspace_dim_ stored column-major with lda = dim_
    // Σ: dim_ × subspace_dim_ stored column-major with lda = dim_
    // H: subspace_dim_ × subspace_dim_ stored column-major with lda = subspace_dim_
    // H_ij = <v_i | σ_j> = V[:,i]^T * Σ[:,j]
    const real_t alpha = 1.0;
    const real_t beta = 0.0;

    try {
        oneapi::mkl::blas::column_major::gemm(
            q, oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans, subspace_dim_, subspace_dim_, dim_,
            alpha, d_subspace_vectors_, dim_, d_sigma_vectors_, dim_, beta,
            d_subspace_matrix_, subspace_dim_
        );
        q.wait_and_throw();
    } catch (const sycl::exception& e) {
        THROW_EXCEPTION(std::string("DavidsonSolver::build_subspace_matrix (SYCL) failed: ") + e.what());
    }
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// ========== Subspace Eigenvalue Problem ==========

void DavidsonSolver::solve_subspace_eigenproblem() {
    sycl::queue& q = queue_;

    // Use existing eigenDecomposition from gpu_manager
    int status = gpu::eigenDecomposition(
        d_subspace_matrix_,
        d_subspace_eigenvalues_,
        d_subspace_eigenvectors_,
        subspace_dim_
    );

    if (status != 0) {
        THROW_EXCEPTION("DavidsonSolver: Subspace eigenvalue decomposition failed");
    }

    // Copy eigenvalues to host
    q.memcpy(h_eigenvalues_.data(), d_subspace_eigenvalues_, config_.num_eigenvalues * sizeof(real_t)).wait();

    if (config_.verbose > 2) {
        std::cout << "Subspace eigenvalues:" << std::endl;
        for (int i = 0; i < std::min(5, subspace_dim_); ++i) {
            real_t eval =0;
            q.memcpy(&eval, &d_subspace_eigenvalues_[i], sizeof(real_t)).wait();
            std::cout << "  λ[" << i << "] = " << std::scientific
                     << std::setprecision(10) << eval << std::endl;
        }
    }
}

// ========== Ritz Vectors and Residuals ==========

void DavidsonSolver::compute_ritz_vectors_and_residuals() {
    sycl::queue& q = queue_;
    using namespace oneapi::mkl::blas::column_major;

    for (int i = 0; i < config_.num_eigenvalues; ++i) {
        // Compute Ritz vector: ψ_i = V * c_i (linear combination of basis vectors)
        // eigenDecomposition transposes eigenvectors to row-major, so eigenvector i
        // is in COLUMN i (stride = subspace_dim_, starting at index i)
        const real_t alpha = 1.0;
        const real_t beta = 0.0;

        // Use Dgemv: y = alpha * A * x + beta * y
        gemv(
            q, oneapi::mkl::transpose::nontrans, dim_,
            subspace_dim_, alpha, d_subspace_vectors_, dim_,
            &d_subspace_eigenvectors_[i], subspace_dim_, beta,
            &d_eigenvectors_[i * dim_], 1);
        q.wait_and_throw();

        // Compute Hψ_i = Σ * c_i (linear combination of sigma vectors)
        gemv(
            q, oneapi::mkl::transpose::nontrans, dim_,
            subspace_dim_, alpha, d_sigma_vectors_, dim_,
            &d_subspace_eigenvectors_[i], subspace_dim_, beta,
            &d_residuals_[i * dim_], 1);
        q.wait_and_throw();

        // Compute residual: r_i = Hψ_i - λ_i * ψ_i
        real_t minus_lambda = -h_eigenvalues_[i];
        axpy(
            q, dim_, minus_lambda, &d_eigenvectors_[i * dim_],
            1, &d_residuals_[i * dim_], 1);
        q.wait_and_throw();

        // Compute residual norm
        real_t res_norm = 0.0;;
        nrm2(q, dim_,&d_residuals_[i * dim_], 1, &res_norm);
        q.wait_and_throw();
        residual_norms_[i] = res_norm;
    }
}

// ========== Convergence Check ==========

bool DavidsonSolver::check_convergence() {
    for (int i = 0; i < config_.num_eigenvalues; ++i) {
        if (residual_norms_[i] > config_.convergence_threshold) {
            return false;
        }
    }
    return true;
}

// ========== Correction Vectors ==========

void DavidsonSolver::add_correction_vectors() {
    sycl::queue& q = queue_;
    using namespace oneapi::mkl::blas::column_major;
    int num_new_vectors = 0;

    real_t* d_norm = sycl::malloc_shared<real_t>(1, q);

    for (int i = 0; i < config_.num_eigenvalues; ++i) {
        if (residual_norms_[i] <= config_.convergence_threshold) {
            continue;
        }
        // Check if we have space for more vectors
        if (subspace_dim_ + num_new_vectors >= config_.max_subspace_size) {
            break;
        }
        real_t* d_correction = &d_subspace_vectors_[(subspace_dim_ + num_new_vectors) * dim_];

        // Apply preconditioner to residual
        if (config_.use_preconditioner) {
            linear_op_.apply_preconditioner(&d_residuals_[i * dim_], d_correction);
        } else {
            q.memcpy(d_correction, &d_residuals_[i * dim_], static_cast<size_t>(dim_) * sizeof(real_t)).wait();
        }

        // Normalize correction vector
        * d_norm = 0.0;
        nrm2(q, dim_, d_correction, 1, d_norm);
        q.wait_and_throw();
        real_t norm = *d_norm;

        if (norm > 1e-12) {
            real_t inv_norm = -1.0 / norm;  // Negative for correction direction
            scal(q, dim_, inv_norm, d_correction, 1);
            q.wait_and_throw();
            ++num_new_vectors;
        }
    }

    sycl::free(d_norm, q);

    if (num_new_vectors > 0) {
        // Orthogonalize new vectors against existing subspace
        orthogonalize_vectors(subspace_dim_, num_new_vectors);
        subspace_dim_ += num_new_vectors;

        if (config_.verbose > 2) {
            std::cout << "Added " << num_new_vectors
                     << " correction vectors (subspace dim: "
                     << subspace_dim_ << ")" << std::endl;
        }
    }
}

// ========== Subspace Restart ==========

void DavidsonSolver::restart_subspace() {
    sycl::queue& q = queue_;
    // Keep only the num_eigenvalues Ritz vectors with lowest eigenvalues
    // Copy current Ritz vectors to beginning of subspace
    for (int i = 0; i < config_.num_eigenvalues; ++i) {
        q.memcpy(&d_subspace_vectors_[i * dim_], &d_eigenvectors_[i * dim_], dim_ * sizeof(real_t));
    }

    subspace_dim_ = config_.num_eigenvalues;
}

// ========== Public Getters ==========

void DavidsonSolver::copy_eigenvectors_to_host(real_t* h_output) {
    sycl::queue& q = queue_;
    if (!h_output) {
        THROW_EXCEPTION("DavidsonSolver::copy_eigenvectors_to_host: null pointer");
    }

    q.memcpy(h_output, d_eigenvectors_, dim_ * config_.num_eigenvalues * sizeof(real_t)) .wait();
}

} // namespace gansu
