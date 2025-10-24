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
#include "gpu_kernels.hpp"
#include "utils.hpp"

namespace gansu::gpu{


/**
 * @brief CUDA kernel for inverse of square root for individual values of input vectors
 * @param d_eigenvalues Device pointer storing the eigenvalues as a vector
 * @param size Size of the input vector
 * @details This function computes the inverse of the square root of each element of the input vector.
 *         The input vector is modified in place.
 */
/*
 SYCL_EXTERNAL void inverseSqrt_kernel(real_t *d_eigenvalues, const size_t size,
                                       const double threshold) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    size_t idx = item_ct1.get_local_id(0);
    if (idx < size) {
        double value = d_eigenvalues[idx];
        if (value < threshold) {
            d_eigenvalues[idx] = 0.0; // Avoid division by zero
        }else{
            d_eigenvalues[idx] = 1.0 / sycl::sqrt(value);
        }
    }
}
*/



/**
 * @brief CUDA kernel for computing the density matrix for restricted Hartree-Fock
 * @param d_coefficient_matrix Device pointer to the coefficient matrix
 * @param d_density_matrix Device pointer to the density matrix, each of orbital elements has exactly 2 electrons
 * @param num_electron Number of electrons, must be even
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix using the coefficient matrix.
 * @details The density matrix is given by \f$ D_{ij} = 2 \sum_{k=1}^{N/2} C_{ik} C_{jk} \f$.
 */
/*
 SYCL_EXTERNAL void computeDensityMatrix_RHF_kernel(
     const real_t *d_coefficient_matrix, real_t *d_density_matrix,
     const int num_electron, const size_t num_basis) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (id >= num_basis * num_basis) return;

    size_t i = id / num_basis;
    size_t j = id % num_basis;

    real_t sum = 0.0;
    for (size_t k = 0; k < num_electron / 2; k++) {
        sum += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k];
    }
    d_density_matrix[id] = 2.0 * sum;
}
*/
/**
 * @brief CUDA kernel for computing the density matrix for unrestricted Hartree-Fock
 * @param d_coefficient_matrix Device pointer to the coefficient matrix (alpha or beta)
 * @param d_density_matrix Device pointer to the density matrix (alpha or beta)
 * @param num_spin Number of electrons, must be number of electrons for the alpha or beta spin
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix using the coefficient matrix.
 * @details The density matrix is given by \f$ D_{ij} = \sum_{k=1}^{N} C_{ik} C_{jk} \f$.
 */
/*
 SYCL_EXTERNAL void
 computeDensityMatrix_UHF_kernel(const double *d_coefficient_matrix,
                                 double *d_density_matrix, const int num_spin,
                                 const size_t num_basis) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (id >= num_basis * num_basis) return;

    size_t i = id / num_basis;
    size_t j = id % num_basis;

    real_t sum = 0.0;
    for (size_t k = 0; k < num_spin; k++) {
        sum += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k];
    }
    d_density_matrix[id] = sum;
}
*/


/**
 * @brief CUDA kernel for computing the density matrix for ROHF
 * @param d_coefficient_matrix Device pointer to the coefficient matrix
 * @param d_density_matrix_closed Device pointer to the density matrix (closed-shell)
 * @param d_density_matrix_oepn Device pointer to the density matrix (open-shell)
 * @param d_density_matrix Device pointer to the density matrix (sum of closed-shell and open-shell)
 * @param num_closed Number of closed-shell orbitals
 * @param num_open Number of open-shell orbitals
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix using the coefficient matrix.
 */
/*
 SYCL_EXTERNAL void computeDensityMatrix_ROHF_kernel(
     const double *d_coefficient_matrix, double *d_density_matrix_closed,
     double *d_density_matrix_open, double *d_density_matrix,
     const int num_closed, const int num_open, const size_t num_basis) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (id >= num_basis * num_basis) return;

    size_t i = id / num_basis;
    size_t j = id % num_basis;

    real_t sum_closed = 0.0;
    for (size_t k = 0; k < num_closed; k++) {
        sum_closed += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k];
    }
    sum_closed *= 2.0; // closedd shell (2 electrons per orbital)
    d_density_matrix_closed[id] = sum_closed; 

    real_t sum_open = 0.0;
    for (size_t k = num_closed; k < num_closed+num_open; k++) {
        sum_open += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k];
    }
    sum_open *= 1.0; // open shell (1 electron per orbital)
    d_density_matrix_open[id] = sum_open;

    d_density_matrix[id] = sum_closed + sum_open;
}
*/


/**
 * @brief transposeMatrixInPlace_kernel CUDA kernel for transposing a matrix in place
 * @param d_matrix Device pointer to the matrix
 * @param size Size of the matrix
 */
/*
SYCL_EXTERNAL void transposeMatrixInPlace_kernel(real_t* d_matrix, int size,
                                sycl::local_accessor<real_t, 2> s_src,
                                sycl::local_accessor<real_t, 2> s_dst)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<2>();

    const int xid = item_ct1.get_global_id(0);
    const int yid = item_ct1.get_global_id(1);
    const int l_xid = item_ct1.get_local_id(0);
    const int l_yid = item_ct1.get_local_id(1);
    const int b_xid = item_ct1.get_group(0);
    const int b_yid = item_ct1.get_group(1);

    bool in_bounds = (xid < size) && (yid < size);
    bool do_work = in_bounds && !(b_xid < b_yid || xid < yid);

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (do_work) {
    //__shared__ real_t s_src[WARP_SIZE][WARP_SIZE];
    //__shared__ real_t s_dst[WARP_SIZE][WARP_SIZE];

    s_src[l_yid][l_xid] = d_matrix[yid * size + xid];
    s_dst[l_yid][l_xid] = d_matrix[xid * size + yid];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (do_work) {
    d_matrix[yid * size + xid] = s_dst[l_yid][l_xid];
    d_matrix[xid * size + yid] = s_src[l_yid][l_xid];
    }
}
*/

/**
 * @brief CUDA kernel for computing weight sum matices sum(W[i] * B[i]).
 *
 * @param d_J Output result matrix (MxM) in device memory.
 * @param d_B Input matrices (N matrices of size MxM).
 * @param d_W Scalars (size N).
 * @param M Dimension of matrices (M x M).
 * @param N Number of matrices.
 * @param accumulated If true, the result is accumulated to the output matrix.
 */
/*
SYCL_EXTERNAL void weighted_sum_matrices_kernel(double *d_J, const double *d_B,
                                                const double *d_W, const int M,
                                                const int N,
                                                const bool accumulated) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t id = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (id >= M * M) return;

    double sum = 0.0;
    for (int j = 0; j < N; ++j) {
        sum += d_W[j] * d_B[j * M * M + id];  // Apply scalar multiplication and accumulate
    }

    if(accumulated){
        d_J[id] += sum;
    }else{
        d_J[id] = sum;
    }
}
*/

/**
 * @brief CUDA kernel for computing sum matices sum(B[i]).
 *
 * @param d_J Output result matrix (MxM) in device memory.
 * @param d_B Input matrices (N matrices of size MxM).
 * @param M Dimension of matrices (M x M).
 * @param N Number of matrices.
 * @param accumulated If true, the result is accumulated to the output matrix.
 */
/*
SYCL_EXTERNAL void sum_matrices_kernel(double *d_K, const double *d_B,
                                       const int M, const int N,
                                       const bool accumulated) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t id = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (id >= M * M) return;

    double sum = 0.0;
    for (int p = 0; p < N; p++) {
        sum += d_B[p * M * M + id];  // Apply scalar multiplication and accumulate
    }

    if(accumulated){
        d_K[id] += sum;
    }else{
        d_K[id] = sum;
    }
}
*/

/*
SYCL_EXTERNAL
void computeFockMatrix_RHF_kernel(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix, int num_basis,
      sycl::local_accessor<real_t, 1> s_F_ij)
{
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int bra = item.get_group_linear_id();
    const int i = bra / num_basis;
    const int j = bra % num_basis;
    const size_t l = item.get_local_linear_id();

    if (item.get_local_id(2) == 0 && item.get_local_id(1) == 0) {
        s_F_ij[0] = 0.0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    real_t sum = 0.0;
    if (l < num_basis) {
        for (int k = 0; k < num_basis; ++k) {
            size_t eid1 = get_1d_indexM4(i, j, k, l, num_basis);
            size_t eid2 = get_1d_indexM4(i, k, j, l, num_basis);
            sum += (d_eri[eid1] - 0.5 * d_eri[eid2]) * d_density_matrix[k * num_basis + l];
        }
    }
//    item.barrier(sycl::access::fence_space::local_space);
//    if (item.get_local_id(2) == 0 && item.get_local_id(1) == 0) {
//       sycl::ext::oneapi::experimental::printf("<%d-%f> ",l,sum);
//    }
//    item.barrier(sycl::access::fence_space::local_space);

//    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    sum = reduce_over_group(sg, sum, std::plus<>());
//    item.barrier(sycl::access::fence_space::local_space);
//    if (item.get_local_id(2) == 0 && item.get_local_id(1) == 0) {
//       if(bra==0)sycl::ext::oneapi::experimental::printf("<<%d %f>> ",l,sum);
//    }
    item.barrier(sycl::access::fence_space::local_space);

    if (sg.leader()) {
        sycl::atomic_ref<real_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
        sycl::access::address_space::local_space> atomic_F(s_F_ij[0]);
        atomic_F.fetch_add(sum);
    }

    item.barrier(sycl::access::fence_space::global_and_local);
//    if (item.get_local_id(2) == 0 && item.get_local_id(1) == 0) {
//       if(bra==0)sycl::ext::oneapi::experimental::printf("<<<%d %f>>> ",l,s_F_ij[0]);
//    }

    if (item.get_local_id(2) == 0 && item.get_local_id(1) == 0) {
        d_fock_matrix[bra] = s_F_ij[0] + d_core_hamiltonian_matrix[bra];
    }
//    item.barrier(sycl::access::fence_space::local_space);
}
*/
/*
SYCL_EXTERNAL
void computeFockMatrix_UHF_kernel(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix_a, real_t* d_fock_matrix_b, int num_basis,
      sycl::local_accessor<real_t, 1> s_Fa_ij, sycl::local_accessor<real_t, 1> s_Fb_ij)
//                                  real_t *s_Fa_ij, real_t *s_Fb_ij)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int bra = item_ct1.get_group(2);
    const int i = bra / num_basis;
    const int j = bra % num_basis;


//    const size_t l = item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
//                     item_ct1.get_local_id(2);
    const size_t l = item_ct1.get_local_linear_id();

    if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0) {
        s_Fa_ij[0] = 0.0;
        s_Fb_ij[0] = 0.0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    real_t sum_a = 0.0;
    real_t sum_b = 0.0;
    size_t eid1, eid2;
    if (l < num_basis) {
        for (int k = 0; k < num_basis; ++k) {
            eid1 = get_1d_indexM4(i, j, k, l, num_basis);
            //eid2 = get_1d_indexM4(i, l, k, j, num_basis);
            eid2 = get_1d_indexM4(i, k, j, l, num_basis);
            sum_a += (d_density_matrix_a[num_basis * k + l]+d_density_matrix_b[num_basis * k + l]) * d_eri[eid1] - d_density_matrix_a[num_basis * k + l] * d_eri[eid2];
            sum_b += (d_density_matrix_a[num_basis * k + l]+d_density_matrix_b[num_basis * k + l]) * d_eri[eid1] - d_density_matrix_b[num_basis * k + l] * d_eri[eid2];
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum_a += dpct::shift_sub_group_left(
            sycl::ext::oneapi::this_work_item::get_sub_group(), sum_a, offset);
        sum_b += dpct::shift_sub_group_left(
            sycl::ext::oneapi::this_work_item::get_sub_group(), sum_b, offset);
    }

    if (item_ct1.get_local_id(2) == 0) {

        sycl::atomic_ref<real_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> atomic_Fa(s_Fa_ij[0]);
        sycl::atomic_ref<real_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> atomic_Fb(s_Fb_ij[0]);

        // 加算操作
        atomic_Fa.fetch_add(sum_a);
        atomic_Fb.fetch_add(sum_b);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0) {
        d_fock_matrix_a[bra] = s_Fa_ij[0] + d_core_hamiltonian_matrix[bra];
        d_fock_matrix_b[bra] = s_Fb_ij[0] + d_core_hamiltonian_matrix[bra];
        //g_fock[uid] = g_fock[lid] = s_F_ij[0] + d_core_hamiltonian_matrix[uid];   // 2-fold symmetry
        //g_fock[bra] = s_F_ij[0];  // use cuBLAS
    }
}
*/
/*
SYCL_EXTERNAL
void computeFockMatrix_ROHF_kernel(const real_t* d_density_matrix_closed, const real_t* d_density_matrix_open, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix_closed, real_t* d_fock_matrix_open, int num_basis,
      sycl::local_accessor<real_t, 1> s_J_closed_ij, sycl::local_accessor<real_t, 1> s_J_open_ij,
      sycl::local_accessor<real_t, 1> s_K_closed_ij, sycl::local_accessor<real_t, 1> s_K_open_ij)
{
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int bra = item.get_group_linear_id();
    const int i = bra / num_basis;
    const int j = bra % num_basis;


    const size_t l = item.get_local_linear_id();

    if (item.get_local_id(2) == 0 && item.get_local_id(1) == 0) {
        s_J_closed_ij[0] = 0.0;
        s_J_open_ij[0] = 0.0;
        s_K_closed_ij[0] = 0.0;
        s_K_open_ij[0] = 0.0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    real_t J_closed = 0.0;
    real_t J_open = 0.0;
    real_t K_closed = 0.0;
    real_t K_open = 0.0;
    if (l < num_basis) {
        for (int k = 0; k < num_basis; ++k) {
            const real_t eri_ijkl = d_eri[get_1d_indexM4(i, j, k, l, num_basis)];
            const real_t eri_ikjl = d_eri[get_1d_indexM4(i, k, j, l, num_basis)];
            J_closed += d_density_matrix_closed[num_basis * k + l] * eri_ijkl;
            J_open   += d_density_matrix_open  [num_basis * k + l] * eri_ijkl;
            K_closed += d_density_matrix_closed[num_basis * k + l] * eri_ikjl;
            K_open   += d_density_matrix_open  [num_basis * k + l] * eri_ikjl;
        }
    }

    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    J_closed = reduce_over_group(sg, J_closed, std::plus<>());
    J_open   = reduce_over_group(sg, J_open, std::plus<>());
    K_closed = reduce_over_group(sg, K_closed, std::plus<>());
    K_open   = reduce_over_group(sg, K_open, std::plus<>());
    item.barrier(sycl::access::fence_space::local_space);

//    if (item_ct1.get_local_id(2) == 0) {
    if (sg.leader()) {

        sycl::atomic_ref<real_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> atomic_J_closed(s_J_closed_ij[0]);
        sycl::atomic_ref<real_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> atomic_J_open(s_J_open_ij[0]);
        sycl::atomic_ref<real_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> atomic_K_closed(s_K_closed_ij[0]);
        sycl::atomic_ref<real_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> atomic_K_open(s_K_open_ij[0]);

        // 加算操作
        atomic_J_closed.fetch_add(J_closed);
        atomic_J_open.fetch_add(J_open);
        atomic_K_closed.fetch_add(K_closed);
        atomic_K_open.fetch_add(K_open);
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (item.get_local_id(2) == 0 && item.get_local_id(1) == 0) {
        d_fock_matrix_closed[bra] = d_core_hamiltonian_matrix[bra] + s_J_closed_ij[0] - 0.5 * s_K_closed_ij[0] + s_J_open_ij[0] - 0.5 * s_K_open_ij[0];
        d_fock_matrix_open[bra]  = 0.5 * (d_core_hamiltonian_matrix[bra] + s_J_closed_ij[0] - 0.5 * s_K_closed_ij[0] + s_J_open_ij[0] - s_K_open_ij[0]);
    }
}
*/

/*
SYCL_EXTERNAL void computeUnifiedFockMatrix_ROHF_kernel(
    const real_t *d_fock_mo_closed_matrix, const real_t *d_fock_mo_open_matrix,
    const ROHF_ParameterSet rohf_params, real_t *d_unified_fock_matrix,
    const int num_closed, const int num_open, const size_t num_basis) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (id >= num_basis * (num_basis+1) / 2) return;

    const size_t2 ij = index1to2(id, true);
    size_t i,j;
    if(ij.x < ij.y){
        i = ij.x;
        j = ij.y;
    }else{
        i = ij.y;
        j = ij.x;
    }

    enum SHELL_TYPE {CLOSED, OPEN, VIRTUAL};
    SHELL_TYPE shell_i, shell_j;
    if(i < num_closed) shell_i = CLOSED;
    else if(i < num_closed+num_open) shell_i = OPEN;
    else shell_i = VIRTUAL;
    if(j < num_closed) shell_j = CLOSED;
    else if(j < num_closed+num_open) shell_j = OPEN;
    else shell_j = VIRTUAL;

    const auto Acc = rohf_params.Acc;
    const auto Bcc = rohf_params.Bcc;
    const auto Aoo = rohf_params.Aoo;
    const auto Boo = rohf_params.Boo;
    const auto Avv = rohf_params.Avv;
    const auto Bvv = rohf_params.Bvv;

    real_t d = 0.0;

    if(shell_i == CLOSED && shell_j == CLOSED){ // closed-closed
        d = 2.0 * (Acc*d_fock_mo_open_matrix[i*num_basis+j] + Bcc*(d_fock_mo_closed_matrix[i*num_basis+j] - d_fock_mo_open_matrix[i*num_basis+j]));
    }else if(shell_i == CLOSED && shell_j == OPEN){ // closed-open
        d = 2.0 * (d_fock_mo_closed_matrix[i*num_basis+j] - d_fock_mo_open_matrix[i*num_basis+j]);
    }else if(shell_i == CLOSED && shell_j == VIRTUAL){ // closed-virtual
        d = d_fock_mo_closed_matrix[i*num_basis+j];
    }else if(shell_i == OPEN && shell_j == OPEN){ // open-open
        d = 2.0 * (Aoo*d_fock_mo_open_matrix[i*num_basis+j] + Boo*(d_fock_mo_closed_matrix[i*num_basis+j] - d_fock_mo_open_matrix[i*num_basis+j]));
    }else if(shell_i == OPEN && shell_j == VIRTUAL){ // open-virtual
        d = 2.0 * d_fock_mo_open_matrix[i*num_basis+j];
    }else if(shell_i == VIRTUAL && shell_j == VIRTUAL){ // virtual-virtual
        d = 2.0 * (Avv*d_fock_mo_open_matrix[i*num_basis+j] + Bvv*(d_fock_mo_closed_matrix[i*num_basis+j] - d_fock_mo_open_matrix[i*num_basis+j]));
    }

    // 2-fold symmetry
    d_unified_fock_matrix[i*num_basis+j] = d;
    if(i != j) d_unified_fock_matrix[j*num_basis+i] = d;
}
*/


/**
 * @brief CUDA kernel for computing the trace of a matrix
 * @param d_matrix Device pointer to the matrix
 * @param d_trace Device pointer to the trace
 * @param num_basis Number of basis functions
 * @details This function computes the trace of a matrix.
 */
/*
SYCL_EXTERNAL void getMatrixTrace(const double *d_matrix, double *d_trace,
                                  const int num_basis, double &s_trace)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (item_ct1.get_local_id(2) >= num_basis) return;

    if (item_ct1.get_local_id(2) == 0) {
        s_trace = 0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &s_trace, d_matrix[num_basis * item_ct1.get_local_id(2) +
                           item_ct1.get_local_id(2)]);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0) {
        d_trace[0] = s_trace;
    }
}
*/

/**
 * @brief CUDA kernel for computing the initial Fock matrix in GWH method
 * @param d_core_hamiltonian_matrix Device pointer to the core Hamiltonian matrix
 * @param d_overlap_matrix Device pointer to the overlap matrix
 * @param d_fock_matrix Device pointer to the initial Fock matrix
 * @param num_basis Number of basis functions
 * @param c_x Constant c_x
 */
/*
SYCL_EXTERNAL void computeInitialFockMatrix_GWH_kernel(
    const double *d_core_hamiltonian_matrix, const double *d_overlap_matrix,
    double *d_fock_matrix, const int num_basis, const double c_x) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (id >= num_basis * num_basis) return;

    size_t p = id / num_basis;
    size_t q = id % num_basis;

    d_fock_matrix[id] = c_x * d_overlap_matrix[id] * (d_core_hamiltonian_matrix[p*num_basis+p] + d_core_hamiltonian_matrix[q*num_basis+q]) / 2.0;
}
*/
/*
SYCL_EXTERNAL void computeRIIntermediateMatrixB_kernel(
    const double *d_three_center_eri, const double *d_matrix_L,
    double *d_matrix_B, const int num_basis, const int num_auxiliary_basis) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                      item_ct1.get_local_id(2);
    if (id >= num_auxiliary_basis * num_basis * num_basis) return;

    const size_t p = id / (num_basis*num_basis);
    const size_t id2 = (id % (num_basis*num_basis)) ;
    const size_t mu = id2 / num_basis;
    const size_t nu = id2 % num_basis;

    real_t sum = 0.0;
    for (int q = 0; q < num_auxiliary_basis; q++) {
        sum += d_three_center_eri[q*num_basis*num_basis + mu*num_basis + nu] * d_matrix_L[q*num_auxiliary_basis + p];
    }
    d_matrix_B[id] = sum;
}
*/
/*
SYCL_EXTERNAL void computeFockMatrix_RI_RHF_kernel(
    const double *d_core_hamiltonian_matrix, const double *d_J_matrix,
    const double *d_K_matrix, double *d_Fock_matrix, const int num_basis) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t id = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (id >= num_basis * num_basis) return;

    d_Fock_matrix[id] = d_core_hamiltonian_matrix[id] + d_J_matrix[id] - 0.5*d_K_matrix[id];
}
*/
/*
SYCL_EXTERNAL void computeFockMatrix_RI_UHF_kernel(
    const double *d_core_hamiltonian_matrix, const double *d_J_matrix,
    const double *d_K_matrix, double *d_Fock_matrix, const int num_basis) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t id = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (id >= num_basis * num_basis) return;

    d_Fock_matrix[id] = d_core_hamiltonian_matrix[id] + d_J_matrix[id] - d_K_matrix[id];
}
*/
/*
SYCL_EXTERNAL void computeFockMatrix_RI_ROHF_kernel(
    const double *d_core_hamiltonian_matrix, const double *d_J_matrix,
    const double *d_K_matrix_closed, const double *d_K_matrix_open,
    double *d_Fock_matrix_closed, double *d_Fock_matrix_open,
    const int num_basis) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t id = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (id >= num_basis * num_basis) return;

    d_Fock_matrix_closed[id] = d_core_hamiltonian_matrix[id] + d_J_matrix[id] - 0.5*d_K_matrix_closed[id];
    d_Fock_matrix_open[id] = 0.5 * (d_core_hamiltonian_matrix[id] + d_J_matrix[id] - d_K_matrix_open[id]);
}
*/
/*
 * @brief Sets zeros to the upper triangular part of the matrix
 *
 * @param d_A Pointer to the N x N matrix in device memory (input/output).
 * @param N The size of the matrix (number of rows/columns).
 */
/*
 SYCL_EXTERNAL void setZeroUpperTriangle(double *d_A, const int N) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                      item_ct1.get_local_id(2);
    const size_t row = id / N;
    const size_t col = id % N;
    if (row < N && col < N && col > row) {
        d_A[row * N + col] = 0.0;
    }
}
*/


/**
 * @brief CUDA kernel for computing the diagonal of the product of two matrices A and B
 * @param A Device pointer to the first matrix (row-major)
 * @param B Device pointer to the second matrix (row-major)
 * @param diag Device pointer to the output diagonal vector
 * @param N Size of the matrices (N x N)
 */
/*
SYCL_EXTERNAL void compute_diagonal_of_product(const double *A, const double *B,
                                               double *diag, const int N) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[i * N + k] * B[k * N + i];  // Diagonal element of the product matrix stored in row-major order
        }
        diag[i] = sum;
    }
}
*/

/**
 * @brief CUDA kernel for computing the diagonal of the sum of two matrices A and B, multiplied by a third matrix C
 * @param A Device pointer to the first matrix (row-major)
 * @param B Device pointer to the second matrix (row-major)
 * @param C Device pointer to the third matrix (row-major)
 * @param diag Device pointer to the output diagonal vector
 * @param N Size of the matrices (N x N)
 */
/*
SYCL_EXTERNAL void compute_diagonal_of_product_sum(const double *A,
                                                   const double *B,
                                                   const double *C,
                                                   double *diag, const int N)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i >= N) return;

    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        double a_plus_b = A[i * N + k] + B[i * N + k]; // (A + B)[i][k]
        double c = C[k * N + i];                       // C[k][i] (row-major)
        sum += a_plus_b * c;
    }
    diag[i] = sum;
}
*/


void constructERIHash_kernel(const std::vector<ShellTypeInfo> shell_type_infos, const std::vector<ShellPairTypeInfo> shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_cgto_normalization_factors, /* Hash memoryへのポインタ, */ const bool verbose)
{
    // ここにERIを計算して、ハッシュテーブルに格納する処理を実装する
}

void computeFockMatrix_Hash_RHF_kernel(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, /* Hash memoryへのポインタ, */ real_t* d_fock_matrix, const int num_basis, const int verbose)
{
    // ハッシュテーブルを使用してFock行列を計算する処理を実装する
}



} // namespace gansu::gpu
