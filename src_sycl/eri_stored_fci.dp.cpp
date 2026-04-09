/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Mitsuru Ikei
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
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <assert.h>

#include "rhf.hpp"
#include "eri_stored.hpp"
#include "fci_hamiltonian.hpp"
#include "davidson_solver.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"
#include "fci.hpp"

namespace gansu {

// ========================================================================
//  Forward declarations for functions defined in eri_stored.cu
// ========================================================================
void transform_ao_eri_to_mo_eri_full(
    const double* d_eri_ao, const double* d_C, int nao, double* d_eri_mo);

// ========================================================================
//  Host utility: generate all bit strings with exactly k bits set in n positions
// ========================================================================
/**
 * @brief Count trailing zeros in a 64-bit integer (portable)
 */
static int ctzll_portable(uint64_t v) {
    if (v == 0) return 64;
    int n = 0;
    while ((v & 1ULL) == 0) { v >>= 1; ++n; }
    return n;
}

static void generate_combinations(int n, int k, std::vector<uint64_t>& result) {
    result.clear();
    if (k == 0) {
        result.push_back(0ULL);
        return;
    }
    if (k > n || n > 64) return;

    // Generate lexicographically ordered bit strings using Gosper's hack
    uint64_t v = (1ULL << k) - 1;  // first combination: lowest k bits set
    uint64_t limit = 1ULL << n;

    while (v < limit) {
        result.push_back(v);
        // Gosper's hack: next combination with same number of bits
        uint64_t t = v | (v - 1);
        uint64_t w = (t + 1) | (((~t & -(~t)) - 1) >> (ctzll_portable(v) + 1));
        v = w;
    }
}

// ========================================================================
//  Device utility functions
// ========================================================================

/**
 * @brief Count excitations between two bit strings
 * @return Number of orbitals that differ (excitation level = return/2 for each spin)
 */
inline int count_excitations(uint64_t str1, uint64_t str2) {
    return sycl::popcount(str1 ^ str2) / 2;
}

/**
 * @brief Compute phase factor for single excitation p -> q in a string
 *
 * Phase = (-1)^(number of occupied orbitals between positions p and q)
 */
inline real_t compute_phase(uint64_t str, int p, int q) {
    int lo = sycl::min(p, q);
    int hi = sycl::max(p, q);
    // Mask for bits strictly between lo and hi
    uint64_t mask = ((1ULL << hi) - 1) & ~((1ULL << (lo + 1)) - 1);
    int n_between = sycl::popcount(str & mask);
    return (n_between % 2 == 0) ? 1.0 : -1.0;
}

/**
 * @brief Find the single differing orbital between two strings
 *        that has one excitation (p in str1, q in str2)
 *
 * @param str1 Source string (has orbital p occupied)
 * @param str2 Target string (has orbital q occupied)
 * @param[out] p Orbital removed (in str1 but not str2)
 * @param[out] q Orbital added   (in str2 but not str1)
 */
inline void find_single_excitation(uint64_t str1, uint64_t str2,
                                                   int &p, int &q)
{
    uint64_t diff = str1 ^ str2;
    uint64_t removed = diff & str1;  // bits in str1 not in str2
    uint64_t added   = diff & str2;  // bits in str2 not in str1
    p = sycl::ctz<long long int>(removed) -
        1; // 0-indexed position of removed orbital
    q = sycl::ctz<long long int>(added) -
        1; // 0-indexed position of added orbital
}

/**
 * @brief Find the two differing orbitals for a double excitation
 */
inline void find_double_excitation(uint64_t str1, uint64_t str2,
                                                   int &p1, int &p2, int &q1,
                                                   int &q2)
{
    uint64_t diff = str1 ^ str2;
    uint64_t removed = diff & str1;
    uint64_t added   = diff & str2;

    p1 = sycl::ctz<long long int>(removed) - 1;
    removed &= removed - 1;  // clear lowest bit
    p2 = sycl::ctz<long long int>(removed) - 1;

    q1 = sycl::ctz<long long int>(added) - 1;
    added &= added - 1;
    q2 = sycl::ctz<long long int>(added) - 1;
}

/**
 * @brief Compute phase for double excitation (p1,p2 -> q1,q2) in a string
 *
 * The phase is the product of individual annihilation/creation phases.
 */
inline real_t compute_double_phase(uint64_t str, int p1, int p2,
                                                   int q1, int q2)
{
    // Phase for annihilating p1 from str
    real_t phase = compute_phase(str, p1, 0);
    // Count occupied orbitals below p1
    uint64_t mask_p1 = (1ULL << p1) - 1;
    int n_below_p1 = sycl::popcount(str & mask_p1);
    phase = (n_below_p1 % 2 == 0) ? 1.0 : -1.0;

    // After removing p1
    uint64_t str2 = str & ~(1ULL << p1);
    uint64_t mask_p2 = (1ULL << p2) - 1;
    int n_below_p2 = sycl::popcount(str2 & mask_p2);
    phase *= (n_below_p2 % 2 == 0) ? 1.0 : -1.0;

    // After removing p2
    uint64_t str3 = str2 & ~(1ULL << p2);
    uint64_t mask_q1 = (1ULL << q1) - 1;
    int n_below_q1 = sycl::popcount(str3 & mask_q1);
    phase *= (n_below_q1 % 2 == 0) ? 1.0 : -1.0;

    // After adding q1
    uint64_t str4 = str3 | (1ULL << q1);
    uint64_t mask_q2 = (1ULL << q2) - 1;
    int n_below_q2 = sycl::popcount(str4 & mask_q2);
    phase *= (n_below_q2 % 2 == 0) ? 1.0 : -1.0;

    return phase;
}

/**
 * @brief Compute phase for single excitation p -> q
 *
 * Phase = (-1)^(number of occupied orbitals that must be anticommuted past)
 */
inline real_t compute_single_phase(uint64_t str, int p, int q) {
    // Annihilate p, then create q
    uint64_t mask_p = (1ULL << p) - 1;
    int n_below_p = sycl::popcount(str & mask_p);
    real_t phase = (n_below_p % 2 == 0) ? 1.0 : -1.0;

    uint64_t str2 = str & ~(1ULL << p);
    uint64_t mask_q = (1ULL << q) - 1;
    int n_below_q = sycl::popcount(str2 & mask_q);
    phase *= (n_below_q % 2 == 0) ? 1.0 : -1.0;

    return phase;
}


// ========================================================================
//  MO integral access helper (device)
// ========================================================================

/**
 * @brief Access 1-electron MO integral h_pq
 */
inline real_t h1_mo(const real_t *h1, int M, int p, int q) {
    return h1[p * M + q];
}

/**
 * @brief Access 2-electron MO integral (pq|rs) in chemist's notation
 */
inline real_t eri_mo(const real_t *eri, int M, int p, int q,
                                     int r, int s) {
    return eri[((size_t(p) * M + q) * M + r) * M + s];
}


// ========================================================================
//  GPU Kernel: Compute diagonal elements H_II
// ========================================================================
void fci_diagonal_kernel(sycl::nd_item<1> item,
    const real_t* __restrict__ d_h1,
    const real_t* __restrict__ d_eri,
    const uint64_t* __restrict__ d_alpha_strings,
    const uint64_t* __restrict__ d_beta_strings,
    real_t* __restrict__ d_diagonal,
    int M, int num_alpha_det, int num_beta_det)
{
//    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item.get_global_linear_id();
    int num_det = num_alpha_det * num_beta_det;
    if (idx >= num_det) return;

    int ia = idx / num_beta_det;
    int ib = idx % num_beta_det;
    uint64_t alpha = d_alpha_strings[ia];
    uint64_t beta  = d_beta_strings[ib];

    real_t diag = 0.0;

    // 1-electron contributions: Σ_i h_ii for occupied orbitals
    for (int p = 0; p < M; ++p) {
        if (alpha & (1ULL << p)) diag += h1_mo(d_h1, M, p, p);
        if (beta  & (1ULL << p)) diag += h1_mo(d_h1, M, p, p);
    }

    // 2-electron contributions
    // alpha-alpha: Σ_{p<q} [(pp|qq) - (pq|qp)]
    for (int p = 0; p < M; ++p) {
        if (!(alpha & (1ULL << p))) continue;
        for (int q = p + 1; q < M; ++q) {
            if (!(alpha & (1ULL << q))) continue;
            diag += eri_mo(d_eri, M, p, p, q, q) - eri_mo(d_eri, M, p, q, q, p);
        }
    }

    // beta-beta: Σ_{p<q} [(pp|qq) - (pq|qp)]
    for (int p = 0; p < M; ++p) {
        if (!(beta & (1ULL << p))) continue;
        for (int q = p + 1; q < M; ++q) {
            if (!(beta & (1ULL << q))) continue;
            diag += eri_mo(d_eri, M, p, p, q, q) - eri_mo(d_eri, M, p, q, q, p);
        }
    }

    // alpha-beta: Σ_{p∈α, q∈β} (pp|qq)
    for (int p = 0; p < M; ++p) {
        if (!(alpha & (1ULL << p))) continue;
        for (int q = 0; q < M; ++q) {
            if (!(beta & (1ULL << q))) continue;
            diag += eri_mo(d_eri, M, p, p, q, q);
        }
    }

    d_diagonal[idx] = diag;
}


// ========================================================================
//  GPU Kernel: Sigma vector  σ = H * C
// ========================================================================
void fci_sigma_kernel(sycl::nd_item<1> item,
                      const real_t *__restrict__ d_h1,
                      const real_t *__restrict__ d_eri,
                      const uint64_t *__restrict__ d_alpha_strings,
                      const uint64_t *__restrict__ d_beta_strings,
                      const real_t *__restrict__ d_C,
                      real_t *__restrict__ d_sigma, int M, int num_alpha_det,
                      int num_beta_det)
{
    int idx_I = item.get_global_linear_id();
    int num_det = num_alpha_det * num_beta_det;
    if (idx_I >= num_det) return;

    int ia_I = idx_I / num_beta_det;
    int ib_I = idx_I % num_beta_det;
    uint64_t alpha_I = d_alpha_strings[ia_I];
    uint64_t beta_I  = d_beta_strings[ib_I];

    real_t sigma_I = 0.0;

    // Loop over all determinants J
    for (int ia_J = 0; ia_J < num_alpha_det; ++ia_J) {
        uint64_t alpha_J = d_alpha_strings[ia_J];
        int exc_alpha = sycl::popcount(alpha_I ^ alpha_J) / 2;
        if (exc_alpha > 2) continue;

        for (int ib_J = 0; ib_J < num_beta_det; ++ib_J) {
            uint64_t beta_J = d_beta_strings[ib_J];
            int exc_beta = sycl::popcount(beta_I ^ beta_J) / 2;

            int total_exc = exc_alpha + exc_beta;
            if (total_exc > 2) continue;

            int idx_J = ia_J * num_beta_det + ib_J;
            real_t C_J = d_C[idx_J];
            real_t H_IJ = 0.0;

            if (total_exc == 0) {
                // ---- Diagonal: same determinant ----
                // 1-electron
                for (int p = 0; p < M; ++p) {
                    if (alpha_I & (1ULL << p)) H_IJ += h1_mo(d_h1, M, p, p);
                    if (beta_I  & (1ULL << p)) H_IJ += h1_mo(d_h1, M, p, p);
                }
                // alpha-alpha 2e
                for (int p = 0; p < M; ++p) {
                    if (!(alpha_I & (1ULL << p))) continue;
                    for (int q = p + 1; q < M; ++q) {
                        if (!(alpha_I & (1ULL << q))) continue;
                        H_IJ += eri_mo(d_eri, M, p, p, q, q) - eri_mo(d_eri, M, p, q, q, p);
                    }
                }
                // beta-beta 2e
                for (int p = 0; p < M; ++p) {
                    if (!(beta_I & (1ULL << p))) continue;
                    for (int q = p + 1; q < M; ++q) {
                        if (!(beta_I & (1ULL << q))) continue;
                        H_IJ += eri_mo(d_eri, M, p, p, q, q) - eri_mo(d_eri, M, p, q, q, p);
                    }
                }
                // alpha-beta 2e
                for (int p = 0; p < M; ++p) {
                    if (!(alpha_I & (1ULL << p))) continue;
                    for (int q = 0; q < M; ++q) {
                        if (!(beta_I & (1ULL << q))) continue;
                        H_IJ += eri_mo(d_eri, M, p, p, q, q);
                    }
                }

            } else if (exc_alpha == 1 && exc_beta == 0) {
                // ---- Single alpha excitation ----
                int p, q;
                find_single_excitation(alpha_I, alpha_J, p, q);
                real_t phase = compute_single_phase(alpha_I, p, q);

                H_IJ = h1_mo(d_h1, M, p, q);
                // alpha-alpha: Σ_{k∈α, k≠p} [(pq|kk) - (pk|kq)]
                uint64_t common_alpha = alpha_I & alpha_J;
                for (int k = 0; k < M; ++k) {
                    if (common_alpha & (1ULL << k)) {
                        H_IJ += eri_mo(d_eri, M, p, q, k, k) - eri_mo(d_eri, M, p, k, k, q);
                    }
                }
                // alpha-beta: Σ_{k∈β} (pq|kk)
                for (int k = 0; k < M; ++k) {
                    if (beta_I & (1ULL << k)) {
                        H_IJ += eri_mo(d_eri, M, p, q, k, k);
                    }
                }
                H_IJ *= phase;

            } else if (exc_alpha == 0 && exc_beta == 1) {
                // ---- Single beta excitation ----
                int p, q;
                find_single_excitation(beta_I, beta_J, p, q);
                real_t phase = compute_single_phase(beta_I, p, q);

                H_IJ = h1_mo(d_h1, M, p, q);
                // beta-beta: Σ_{k∈β, k≠p} [(pq|kk) - (pk|kq)]
                uint64_t common_beta = beta_I & beta_J;
                for (int k = 0; k < M; ++k) {
                    if (common_beta & (1ULL << k)) {
                        H_IJ += eri_mo(d_eri, M, p, q, k, k) - eri_mo(d_eri, M, p, k, k, q);
                    }
                }
                // beta-alpha: Σ_{k∈α} (pq|kk)
                for (int k = 0; k < M; ++k) {
                    if (alpha_I & (1ULL << k)) {
                        H_IJ += eri_mo(d_eri, M, p, q, k, k);
                    }
                }
                H_IJ *= phase;

            } else if (exc_alpha == 2 && exc_beta == 0) {
                // ---- Double alpha-alpha excitation ----
                int p1, p2, q1, q2;
                find_double_excitation(alpha_I, alpha_J, p1, p2, q1, q2);
                real_t phase = compute_double_phase(alpha_I, p1, p2, q1, q2);
                // <p1 p2 || q1 q2> = (p1 q1 | p2 q2) - (p1 q2 | p2 q1)
                H_IJ = phase * (eri_mo(d_eri, M, p1, q1, p2, q2) - eri_mo(d_eri, M, p1, q2, p2, q1));

            } else if (exc_alpha == 0 && exc_beta == 2) {
                // ---- Double beta-beta excitation ----
                int p1, p2, q1, q2;
                find_double_excitation(beta_I, beta_J, p1, p2, q1, q2);
                real_t phase = compute_double_phase(beta_I, p1, p2, q1, q2);
                H_IJ = phase * (eri_mo(d_eri, M, p1, q1, p2, q2) - eri_mo(d_eri, M, p1, q2, p2, q1));

            } else if (exc_alpha == 1 && exc_beta == 1) {
                // ---- Mixed alpha-beta excitation ----
                int pa, qa, pb, qb;
                find_single_excitation(alpha_I, alpha_J, pa, qa);
                find_single_excitation(beta_I, beta_J, pb, qb);
                real_t phase_a = compute_single_phase(alpha_I, pa, qa);
                real_t phase_b = compute_single_phase(beta_I, pb, qb);
                // (pa qa | pb qb)  — Coulomb only, no exchange for different spins
                H_IJ = phase_a * phase_b * eri_mo(d_eri, M, pa, qa, pb, qb);
            }

            sigma_I += H_IJ * C_J;
        }
    }

    d_sigma[idx_I] = sigma_I;
}


// ========================================================================
//  GPU Kernel: Diagonal preconditioner  δ_i = r_i / H_ii
// ========================================================================
void fci_preconditioner_kernel(sycl::nd_item<1> item,
    const real_t* __restrict__ d_diagonal,
    const real_t* __restrict__ d_input,
    real_t* __restrict__ d_output,
    int n)
{
    int idx = item.get_global_linear_id();
    if (idx >= n) return;
    real_t diag = d_diagonal[idx];
    d_output[idx] = (sycl::fabs(diag) > 1e-12) ? d_input[idx] / diag : 0.0;
}


// ========================================================================
//  FCIHamiltonianOperator Implementation
// ========================================================================

FCIHamiltonianOperator::FCIHamiltonianOperator(
    const real_t* d_h1_mo,
    const real_t* d_eri_mo,
    int num_orbitals,
    int num_alpha,
    int num_beta)
    : d_h1_mo_(d_h1_mo),
      d_eri_mo_(d_eri_mo),
      num_orbitals_(num_orbitals),
      num_alpha_(num_alpha),
      num_beta_(num_beta),
      num_alpha_det_(0),
      num_beta_det_(0),
      num_det_(0),
      d_alpha_strings_(nullptr),
      d_beta_strings_(nullptr),
      d_diagonal_(nullptr)
{
    if (num_orbitals_ > 64) {
        THROW_EXCEPTION("FCIHamiltonianOperator: max 64 spatial orbitals supported");
    }
    if (num_alpha_ > num_orbitals_ || num_beta_ > num_orbitals_) {
        THROW_EXCEPTION("FCIHamiltonianOperator: more electrons than orbitals");
    }

    generate_determinants();
    compute_diagonal();
}

FCIHamiltonianOperator::~FCIHamiltonianOperator() {
    if (d_alpha_strings_) tracked_syclFree(d_alpha_strings_);
    if (d_beta_strings_)  tracked_syclFree(d_beta_strings_);
    if (d_diagonal_)      tracked_syclFree(d_diagonal_);
}

void FCIHamiltonianOperator::generate_determinants() {
    sycl::queue& q_ct1 = gpu::GPUHandle::syclqueue();
    // Generate all alpha strings
    std::vector<uint64_t> alpha_strings, beta_strings;
    generate_combinations(num_orbitals_, num_alpha_, alpha_strings);
    generate_combinations(num_orbitals_, num_beta_,  beta_strings);

    num_alpha_det_ = static_cast<int>(alpha_strings.size());
    num_beta_det_  = static_cast<int>(beta_strings.size());
    num_det_ = num_alpha_det_ * num_beta_det_;

    // Copy to device
    d_alpha_strings_ = tracked_syclMalloc<uint64_t>(num_alpha_det_, q_ct1);
    d_beta_strings_ = tracked_syclMalloc<uint64_t>(num_beta_det_ , q_ct1);

    q_ct1.memcpy(d_alpha_strings_, alpha_strings.data(),
                 num_alpha_det_ * sizeof(uint64_t));
    q_ct1
        .memcpy(d_beta_strings_, beta_strings.data(),
                num_beta_det_ * sizeof(uint64_t))
        .wait();
}

void FCIHamiltonianOperator::compute_diagonal() {
    sycl::queue& q_ct1 = gpu::GPUHandle::syclqueue();
    d_diagonal_ = tracked_syclMalloc<real_t>(num_det_, q_ct1);

    size_t maxWG = q_ct1.get_device().get_info<sycl::info::device::max_work_group_size>();
    size_t threads = 256;
    threads = std::min(maxWG, threads);
    size_t blocks = (num_det_ + threads - 1) / threads;
    sycl::range<1> local(threads);
    sycl::range<1> global(blocks * threads);
 

    q_ct1.submit([&](sycl::handler &cgh) {
        auto d_h1_mo__ct0 = d_h1_mo_;
        auto d_eri_mo__ct1 = d_eri_mo_;
        const uint64_t *d_alpha_strings__ct2 = d_alpha_strings_;
        const uint64_t *d_beta_strings__ct3 = d_beta_strings_;
        auto d_diagonal__ct4 = d_diagonal_;
        auto num_orbitals__ct5 = num_orbitals_;
        auto num_alpha_det__ct6 = num_alpha_det_;
        auto num_beta_det__ct7 = num_beta_det_;

        cgh.parallel_for(sycl::nd_range<1>(global, local),
                         [=](sycl::nd_item<1> item) {
                             fci_diagonal_kernel(item,
                                 d_h1_mo__ct0, d_eri_mo__ct1,
                                 d_alpha_strings__ct2, d_beta_strings__ct3,
                                 d_diagonal__ct4, num_orbitals__ct5,
                                 num_alpha_det__ct6, num_beta_det__ct7);
                         });
    });
    q_ct1.wait_and_throw();
}

void FCIHamiltonianOperator::apply(const real_t* d_input, real_t* d_output) const {
    sycl::queue& q_ct1 = gpu::GPUHandle::syclqueue();
    size_t maxWG = q_ct1.get_device().get_info<sycl::info::device::max_work_group_size>();
    size_t threads = 128;
    threads = std::min(maxWG, threads);
    size_t blocks = (num_det_ + threads - 1) / threads;
    sycl::range<1> local(threads);
    sycl::range<1> global(blocks * threads);

    q_ct1.submit([&](sycl::handler &cgh) {
        auto d_h1_mo__ct0 = d_h1_mo_;
        auto d_eri_mo__ct1 = d_eri_mo_;
        const uint64_t *d_alpha_strings__ct2 = d_alpha_strings_;
        const uint64_t *d_beta_strings__ct3 = d_beta_strings_;
        auto num_orbitals__ct6 = num_orbitals_;
        auto num_alpha_det__ct7 = num_alpha_det_;
        auto num_beta_det__ct8 = num_beta_det_;

        cgh.parallel_for(sycl::nd_range<1>(global, local),
                         [=](sycl::nd_item<1> item) {
                             fci_sigma_kernel(item,
                                 d_h1_mo__ct0, d_eri_mo__ct1,
                                 d_alpha_strings__ct2, d_beta_strings__ct3,
                                 d_input, d_output, num_orbitals__ct6,
                                 num_alpha_det__ct7, num_beta_det__ct8);
                         });
    });
}

void FCIHamiltonianOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    sycl::queue& q_ct1 = gpu::GPUHandle::syclqueue();
    size_t maxWG = q_ct1.get_device().get_info<sycl::info::device::max_work_group_size>();
    size_t threads = 256;
    threads = std::min(maxWG, threads);
    size_t blocks = (num_det_ + threads - 1) / threads;
    sycl::range<1> local(threads);
    sycl::range<1> global(blocks * threads);

    q_ct1.submit([&](sycl::handler &cgh) {
        const real_t *d_diagonal__ct0 = d_diagonal_;
        auto num_det__ct3 = num_det_;

        cgh.parallel_for(sycl::nd_range<1>(global, local),
                         [=](sycl::nd_item<1> item) {
                             fci_preconditioner_kernel(item, d_diagonal__ct0, d_input,
                                                       d_output, num_det__ct3);
                         });
    });
}


// ========================================================================
//  compute_fci_energy()
// ========================================================================

real_t ERI_Stored_RHF::compute_fci_energy() {
    PROFILE_FUNCTION();

    const int num_occ = rhf_.get_num_electrons() / 2;
    const int num_basis = rhf_.get_num_basis();
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
    const real_t* d_C = coefficient_matrix.device_ptr();
    const real_t* d_eri_ao = eri_matrix_.device_ptr();


    std::cout << "\n=== Full-CI Calculation ===" << std::endl;
    std::cout << "Number of basis functions (spatial orbitals): " << num_basis << std::endl;
    std::cout << "Number of electrons: " << rhf_.get_num_electrons() << std::endl;
    std::cout << "Number of occupied orbitals: " << num_occ << std::endl;

    sycl::queue& q_ct1 = gpu::GPUHandle::syclqueue();
    // ------------------------------------------------------------------
    // Step 1: Transform AO ERIs to MO ERIs
    // ------------------------------------------------------------------
    real_t* d_eri_mo = nullptr;
    d_eri_mo = tracked_syclMalloc<real_t>((size_t)num_basis * num_basis * num_basis * num_basis, q_ct1);
    transform_ao_eri_to_mo_eri_full(d_eri_ao, d_C, num_basis, d_eri_mo);

    // ------------------------------------------------------------------
    // Step 2: Compute 1-electron MO integrals  h_MO = C^T * h_AO * C
    // ------------------------------------------------------------------
    DeviceHostMatrix<real_t>& core_H = rhf_.get_core_hamiltonian_matrix();
    const real_t* d_h_ao = core_H.device_ptr();

    // Temporary for h_AO * C
    real_t* d_temp = nullptr;
    real_t* d_h1_mo = nullptr;
    d_temp = tracked_syclMalloc<real_t>((size_t)num_basis * num_basis, q_ct1);
    d_h1_mo = tracked_syclMalloc<real_t>((size_t)num_basis * num_basis, q_ct1);

    // temp = h_AO * C
    gpu::matrixMatrixProduct(d_h_ao, d_C, d_temp, num_basis, false, false, false);
    // h1_MO = C^T * temp
    gpu::matrixMatrixProduct(d_C, d_temp, d_h1_mo, num_basis, true, false, false);
    tracked_syclFree(d_temp);
    

    // ------------------------------------------------------------------
    // Step 3: Count determinants
    // ------------------------------------------------------------------
    // C(M, N_alpha) * C(M, N_beta)
    std::vector<uint64_t> alpha_test;
    generate_combinations(num_basis, num_occ, alpha_test);
    int num_alpha_det = static_cast<int>(alpha_test.size());
    long long num_det = (long long)num_alpha_det * num_alpha_det;

    std::cout << "Number of alpha determinants: " << num_alpha_det << std::endl;
    std::cout << "Number of beta determinants:  " << num_alpha_det << std::endl;
    std::cout << "Total FCI dimension:          " << num_det << std::endl;

    if (num_det > 500000) {
        std::cout << "Warning: FCI dimension is very large. This may be slow." << std::endl;
    }

    // ------------------------------------------------------------------
    // Step 4: Solve FCI problem    
    // ------------------------------------------------------------------
    
    double E_fci_electronic = 0.0;
    real_t nuclearE = rhf_.get_nuclear_repulsion_energy();
    E_fci_electronic = fci(d_h1_mo, d_eri_mo, num_basis, num_occ*2, num_alpha_det, num_det, nuclearE);

    real_t E_hf_electronic  = rhf_.get_energy();
    real_t E_corr = E_fci_electronic - E_hf_electronic;

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "\n=== FCI Results ===" << std::endl;
    std::cout << "HF electronic energy:  " << E_hf_electronic << " Hartree" << std::endl;
    std::cout << "FCI electronic energy: " << E_fci_electronic << " Hartree" << std::endl;
    std::cout << "FCI correlation energy: " << E_corr << " Hartree" << std::endl;
    std::cout << "FCI total energy:      "
              << E_fci_electronic + rhf_.get_nuclear_repulsion_energy()
              << " Hartree" << std::endl;

    // Cleanup
    tracked_syclFree(d_eri_mo);
    tracked_syclFree(d_h1_mo);

    return E_corr;
}


} // namespace gansu
