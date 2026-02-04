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

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <algorithm>

#include <cstdlib>  // std::getenv
#include <string>   // std::string
#include <fstream>

//#include <omp.h>

#include "rhf.hpp"
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/lapack.hpp>
//#include <dpct/blas_utils.hpp>


#include <cmath>
#include "eri_stored.hpp"

namespace gansu{

// // #threads = M * Mvir * Maux

void nu2a_(sycl::nd_item<1> item, short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B_p_mu_nu, double* d_B_p_mu_a)
{
    long long seq = item.get_global_id(0);
    if (seq >= (long long)norbs * nvir * naux) {
        return;
    }

    const short p = seq / (norbs * nvir);
    seq %= (norbs * nvir);

    const int a = seq % nvir;
    const int mu = seq / nvir;

    double tmp = 0.0;
    for (short nu = 0; nu < norbs; ++nu) {
        tmp += d_C[norbs * nu + (a + nocc)] * d_B_p_mu_nu[p*(norbs*norbs) + mu*norbs + nu];
    }
    d_B_p_mu_a[p*(norbs*nvir) + mu*nvir + a] = tmp;
}


// #threads = Mocc * Mvir * Maux

void mu2i_(sycl::nd_item<1> item, short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B_p_mu_a, double* d_B_p_i_a)
{
    long long seq = item.get_global_id(0);
    if (seq >= (long long)nocc * nvir * naux) {
        return;
    }

    const short p = seq / (nocc * nvir);
    seq %= (nocc * nvir);

    const int a = seq % nvir;
    const int i = seq / nvir;

    double tmp = 0.0;
    for (short mu = 0; mu < norbs; ++mu) {
        tmp += d_C[norbs * mu + i] * d_B_p_mu_a[p*(norbs*nvir) + mu*nvir + a];
    }
    d_B_p_i_a[p*(nocc*nvir) + i*nvir + a] = tmp;
}

// void nu2a_dgemm(short norbs, short nocc, short nvir, short naux, double *d_C,
//                 double *d_B_p_mu_nu, double *d_B_p_mu_a,
//                 dpct::blas::descriptor_ptr &handle) {
void nu2a_dgemm(sycl::queue& workq, short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B_p_mu_nu, double* d_B_p_mu_a) {
//    sycl::queue& workq = gpu::GPUHandle::syclqueue();

    // if(col_A != row_B) throw exception("行数と列数が不一致\n");

    const double alpha = 1.0;
    const double beta = 0.0;

//    workq.memset(d_B_p_mu_a, 0, norbs * norbs * naux * sizeof(double)).wait();
    workq.memset(d_B_p_mu_a, 0, norbs * nvir * naux * sizeof(double)).wait();

    oneapi::mkl::blas::column_major::gemm(
        workq,
        oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        nvir, naux * norbs, norbs,
        alpha,
        d_C + nocc, norbs,
        d_B_p_mu_nu, norbs,
        beta,
        d_B_p_mu_a, nvir  
    ).wait();

/*
    cublasDgemm(
        handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        nvir, naux * norbs, norbs, 
        &alpha, 
        &d_C[nocc], norbs, 
        d_B_p_mu_nu, norbs, 
        &beta, 
        d_B_p_mu_a, nvir
    );
*/    
    // cublasDestroy(handle);
}

void mu2i_dgemm(sycl::queue& workq, short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B_p_mu_a, double* d_B_p_i_a) {
    const double alpha = 1.0;
    const double beta  = 0.0;

    int row = naux * norbs;
    int col = nvir;

    // (pμ,a) → (p,a,μ)
    oneapi::mkl::blas::column_major::omatcopy(
        workq,
        oneapi::mkl::transpose::trans,
        row, col,
        alpha,
        d_B_p_mu_a, row,
        d_B_p_i_a, col
    ).wait();

    // GEMM
//    workq.memset(d_B_p_mu_a, 0, norbs * norbs * naux * sizeof(double)).wait();
    workq.memset(d_B_p_mu_a, 0, norbs * nvir * naux * sizeof(double)).wait();

    oneapi::mkl::blas::column_major::gemm(
        workq,
        oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        nocc, naux * nvir, norbs,
        alpha,
        d_C, norbs,
        d_B_p_i_a, norbs,
        beta,
        d_B_p_mu_a, nocc
    ).wait();

    // (p,i,a) → (p,μ,a)
    oneapi::mkl::blas::column_major::omatcopy(
        workq,
        oneapi::mkl::transpose::trans,
        col, naux * nocc,
        alpha,
        d_B_p_mu_a, col,
        d_B_p_i_a, naux * nocc
    ).wait();
}

    
void transform_intermediate_matrix(
                                   sycl::queue& workq,
                                   short norbs, short nocc, short nvir,
                                   short naux, double *d_C, double *d_B,
                                   double *d_tmp
                                   ) {
    nu2a_dgemm(workq, norbs, nocc, nvir, naux, d_C, d_B, d_tmp);
    mu2i_dgemm(workq, norbs, nocc, nvir, naux, d_C, d_tmp, d_B);
}





void nu2a_dgemm( short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B_p_mu_nu, double* d_B_p_mu_a){

    sycl::queue& workq = gpu::GPUHandle::syclqueue();

    const double alpha = 1.0;
    const double beta  = 0.0;

    workq.memset( d_B_p_mu_a, 0,
        norbs * nvir * naux * sizeof(double)
//        norbs * norbs * naux * sizeof(double) // bug?
    ).wait();

    // 仮想 MO ブロックの先頭
    double* d_C_vir = d_C + nocc * norbs;
//    double* d_C_vir = d_C + nocc // bug?

    oneapi::mkl::blas::column_major::gemm(
        workq,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        nvir, naux * norbs, norbs,
        alpha,
        d_C_vir, norbs,
        d_B_p_mu_nu, norbs,
        beta,
        d_B_p_mu_a, nvir
    ).wait();
}

void mu2i_dgemm(short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B_p_mu_a, double* d_B_p_i_a){

    sycl::queue& workq = gpu::GPUHandle::syclqueue();

    const double alpha = 1.0;
    const double beta  = 0.0;

    double* d_B_mu_pa = sycl::malloc_device<double>(norbs * naux * nvir, workq);
    workq.memset( d_B_mu_pa, 0, norbs * nvir * naux * sizeof(double)).wait();

    oneapi::mkl::blas::column_major::omatcopy(
        workq,
        oneapi::mkl::transpose::trans,
        naux * norbs, nvir,
        alpha,
        d_B_p_mu_a, naux * norbs,
        d_B_mu_pa, norbs
    ).wait();

    double* d_C_occ = d_C;

    oneapi::mkl::blas::column_major::gemm(
        workq,
        oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        nocc, naux * nvir, norbs,
        alpha,
        d_C_occ, norbs,
        d_B_mu_pa, norbs,
        beta,
        d_B_p_i_a, nocc
    ).wait();

    sycl::free(d_B_mu_pa, workq);
}





void transform_intermediate_matrix(short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B, double* d_tmp){
    nu2a_dgemm(norbs, nocc, nvir, naux, d_C, d_B, d_tmp);
    mu2i_dgemm(norbs, nocc, nvir, naux, d_C, d_tmp, d_B);
}








inline size_t2 index1to2_upper_wo_trace(const uint64_t index, const int n){
    size_t r2 = (2.0 * n - 1.0 -
                 sycl::sqrt((2.0 * n - 1.0) * (2.0 * n - 1.0) - 8.0 * index)) /
                2.0;
    size_t r1 = (r2*r2 - (2.0*n - 3.0)*r2 + 2.0*index) / 2.0 + 1.0;

    return {r2,r1};
}




inline uint64_t calc_i(uint64_t id, int s, int k) {   // s:nocc_stride, k: nocc_block
    return ((uint64_t)1.0 + sycl::sqrt(1.0 + 4.0 * (2.0 * id + s * (s - 1)))) /
           2.0;
}



inline uint64_t calc_exclusive_prefix_num_j(int i, int s){
    return (i*(i-1) - s*(s-1)) / 2.0;
}


inline uint64_t calc_j(uint64_t id, int i, int s){
    return id - calc_exclusive_prefix_num_j(i,s);
}


struct energy_kernel1 {};
struct energy_kernel2 {};
struct energy_kernel3 {};
struct energy_kernel4 {};


template <typename Term>
void ri_rmp2_kernel_body(
    sycl::nd_item<1> item,
    int nocc,
    int nocc_block,
    int nvir,
    int i,
    int naux,
    double* d_iajb,
    double* d_eps,
    double* d_energy,
    sycl::local_accessor<double, 1> sh_tmp
 );


// i>j, a>b

//void calc_RI_RMP2_energy_kernel1(int nocc, int nocc_block, int nvir, int nocc_stride, int naux, double* d_iajb, double* d_eps, double* energy,
//                                 double *sh_tmp){
template <>
void ri_rmp2_kernel_body<energy_kernel1>(sycl::nd_item<1> item, int nocc, int nocc_block, int nvir, int nocc_stride, int naux,
                                        double* d_iajb, double* d_eps, double* d_energy, sycl::local_accessor<double, 1> sh_tmp)
{
    size_t tid = item.get_global_linear_id();
    size_t lid = item.get_local_id(0);
    size_t total_threads = ((size_t)nocc_block * nocc_stride + nocc_block*(nocc_block-1)/2) * nvir*(nvir-1)/2;

//    if (tid >= total_threads) return; ==> shm_tmp[0] =0
    sh_tmp[lid] = 0.0;
    item.barrier(sycl::access::fence_space::local_space);

    double val = 0.0;

    if (tid  < total_threads){
        size_t id = tid;

        size_t2 ab = index1to2_upper_wo_trace(id % (nvir * (nvir-1) / 2), nvir);
        id /= (nvir * (nvir-1) / 2);

        const size_t i = calc_i(id, nocc_stride, nocc_block); // full-rangeの値が出てくる(0~kでなく、stride~stride+k)
        const size_t j = calc_j(id, i, nocc_stride);

        if(i < nocc){
            double iajb = d_iajb[(i-nocc_stride)*nvir*nocc*nvir + ab.x*nocc*nvir + j*nvir + ab.y];
            double ibja = d_iajb[(i-nocc_stride)*nvir*nocc*nvir + ab.y*nocc*nvir + j*nvir + ab.x];        
            val = 4.0 * ((iajb-ibja)*(iajb-ibja) + iajb*ibja) / (d_eps[i] + d_eps[j] - d_eps[ab.x+nocc] - d_eps[ab.y+nocc]);
        }
    }

    size_t group_size = item.get_local_range().size();

    sh_tmp[lid] = val;
    item.barrier(sycl::access::fence_space::local_space);

    if (lid == 0) {
        double block_sum = 0.0;
        for (size_t i = 0; i < group_size; ++i) {
            block_sum += sh_tmp[i];
        }
    // device-wide reduction
        atomic_add(d_energy, block_sum);
    }

}

// i>j, a

//void calc_RI_RMP2_energy_kernel2(int nocc, int nocc_block, int nvir, int nocc_stride, int naux, double* d_iajb, double* d_eps, double* energy,
//                                 double *sh_tmp){
template <>
void ri_rmp2_kernel_body<energy_kernel2>(sycl::nd_item<1> item, int nocc, int nocc_block, int nvir, int nocc_stride, int naux,
                                        double* d_iajb, double* d_eps, double* d_energy, sycl::local_accessor<double, 1> sh_tmp)
{
    size_t tid = item.get_global_linear_id();
    size_t lid = item.get_local_id(0);
    size_t total_threads = ((size_t)nocc_block * nocc_stride + nocc_block*(nocc_block-1)/2) * nvir;

//    if (tid >= total_threads) return;
    sh_tmp[0] = 0.0;
    item.barrier(sycl::access::fence_space::local_space);

    double val = 0.0;

    if (tid  < total_threads){
        size_t id = tid;

        const size_t a = id % nvir;
        id /= nvir;

        const size_t i = calc_i(id, nocc_stride, nocc_block); // full-rangeの値が出てくる(0~kでなく、stride~stride+k)
        const size_t j = calc_j(id, i, nocc_stride);

        if(i < nocc){
            double iaja = d_iajb[(i-nocc_stride)*nvir*nocc*nvir + a*nocc*nvir + j*nvir + a];     
            val = 2.0*iaja*iaja / (d_eps[i] + d_eps[j] - 2.0*d_eps[a+nocc]);
        }
    }

    size_t group_size = item.get_local_range().size();

    sh_tmp[lid] = val;
    item.barrier(sycl::access::fence_space::local_space);

    if (lid == 0) {
        double block_sum = 0.0;
        for (size_t i = 0; i < group_size; ++i) {
            block_sum += sh_tmp[i];
        }
    // device-wide reduction
        atomic_add(d_energy, block_sum);
    }

}


// i, a>b

//void calc_RI_RMP2_energy_kernel3(int nocc, int nocc_block, int nvir, int nocc_stride, int naux, double* d_iajb, double* d_eps, double* energy,
//                                 double *sh_tmp){
template <>
void ri_rmp2_kernel_body<energy_kernel3>(sycl::nd_item<1> item, int nocc, int nocc_block, int nvir, int nocc_stride, int naux,
                                        double* d_iajb, double* d_eps, double* d_energy, sycl::local_accessor<double, 1> sh_tmp)
{
    size_t tid = item.get_global_linear_id();
    size_t lid = item.get_local_id(0);
    size_t total_threads = (uint64_t)nocc_block * nvir * (nvir - 1.0) / 2;
//    if (tid >= total_threads) return;

    sh_tmp[0] = 0.0;
    item.barrier(sycl::access::fence_space::local_space);

    double val = 0.0;

    if (tid  < total_threads){
        size_t id = tid;

        size_t2 ab = index1to2_upper_wo_trace(id % (nvir * (nvir-1) / 2), nvir);
        const size_t a = ab.x, b = ab.y;

        const size_t i = id / (nvir * (nvir-1) / 2);  // このiは0~k-1

        if((i + nocc_stride) < nocc){
            double iaib = d_iajb[i*nvir*nocc*nvir + a*nocc*nvir + (i+nocc_stride)*nvir + b];     
            val = 2.0*iaib*iaib / (2.0*d_eps[i+nocc_stride] - d_eps[a+nocc] - d_eps[b+nocc]);
        }
    }

    size_t group_size = item.get_local_range().size();

    sh_tmp[lid] = val;
    item.barrier(sycl::access::fence_space::local_space);

    if (lid == 0) {
        double block_sum = 0.0;
        for (size_t i = 0; i < group_size; ++i) {
            block_sum += sh_tmp[i];
        }
    // device-wide reduction
        atomic_add(d_energy, block_sum);
    }

}

// i, a

//void calc_RI_RMP2_energy_kernel4(int nocc, int nocc_block, int nvir, int nocc_stride, int naux, double* d_iajb, double* d_eps, double* energy,
//                                 double *sh_tmp){
template <>
void ri_rmp2_kernel_body<energy_kernel4>(sycl::nd_item<1> item, int nocc, int nocc_block, int nvir, int nocc_stride, int naux,
                                        double* d_iajb, double* d_eps, double* d_energy, sycl::local_accessor<double, 1> sh_tmp)
{
    size_t tid = item.get_global_linear_id();
    size_t lid = item.get_local_id(0);
    size_t total_threads = (uint64_t)nocc_block * nvir;
//    if (tid >= total_threads) return;

    sh_tmp[0] = 0.0;
    item.barrier(sycl::access::fence_space::local_space);

    double val = 0.0;

    if (tid  < total_threads){
        size_t id = tid;
        const size_t a = id % nvir;
        const size_t i = id / nvir;  // このiは0~k-1

        if((i + nocc_stride) < nocc){
            double iaia = d_iajb[i*nvir*nocc*nvir + a*nocc*nvir + (i+nocc_stride)*nvir + a];     
            val = 0.5*iaia*iaia / (d_eps[i+nocc_stride] - d_eps[a+nocc]);
        }
    }

    size_t group_size = item.get_local_range().size();

    sh_tmp[lid] = val;
    item.barrier(sycl::access::fence_space::local_space);

    if (lid == 0) {
        double block_sum = 0.0;
        for (size_t i = 0; i < group_size; ++i) {
            block_sum += sh_tmp[i];
        }
    // device-wide reduction
        atomic_add(d_energy, block_sum);
    }
}


/*
int search_maximum_k(int mocc, int mvir) {
    size_t free_mem_bytes, total_mem_bytes;
*/
    /*
    DPCT1106:65: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
/*
    dpct::get_current_device().get_memory_info(free_mem_bytes, total_mem_bytes);

    return std::min(free_mem_bytes/(mocc * mvir * mvir * sizeof(double)), (size_t)mocc);    
}
*/
int search_maximum_k(sycl::queue& workq, int mocc, int mvir) {
    if (mocc <= 0 || mvir <= 0) return 0;

    size_t total_mem = workq.get_device().get_info<sycl::info::device::global_mem_size>();
    size_t usable_mem = total_mem * 7 / 10;

    const size_t bytes_per_k = static_cast<size_t>(mocc) * static_cast<size_t>(mvir)
                             * static_cast<size_t>(mvir) * sizeof(double);

    size_t max_k = std::min( usable_mem / bytes_per_k, static_cast<size_t>(mocc));

    return static_cast<int>(max_k);
}


void search_k_and_syclmalloc_4cERI(sycl::queue& workq, int mocc, int mvir, int &k, double **d_iajb) {
    k = search_maximum_k(workq, mocc, mvir);

    while (true) {
        try {
            *d_iajb = sycl::malloc_device<double>(k * mvir * mocc * mvir, workq);
            break; // 成功したらループ脱出
        } catch (sycl::exception const &exc) {
            std::cerr << "malloc_device failed for k=" << k << ": " << exc.what() << "\n";
            k = int(k * 0.9); // 減らして再試行
            if (k == 0) {
                std::cerr << "Allocation failed completely.\n";
                std::exit(1);
            }
        }
    }
}

/*
void search_k_and_cudamalloc_4cERI(int mocc, int mvir, int &k, double **d_iajb,
                                   dpct::queue_ptr &stream) try {
    k = search_maximime_k(mocc, mvir);
    // k = (int)(k*mvir / 32) * 32;
    // k = 10;

    while (cudaMallocAsync((void **)d_iajb,
                           sizeof(double) * k * mvir * mocc * mvir,
                           stream) != 0) {
        k *= 0.9;
    }

    // printf("k = %d\n",k);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
*/

template <typename Term>
struct ri_rmp2;

template <typename Term>
void launch_ri_rmp2_kernel(sycl::queue &workq, size_t num_blocks, int num_threads, int nocc, int nocc_block, int nvir,
                              int i, int naux, double* d_iajb, double* d_eps, double* d_energy)
{
    size_t global_size = num_blocks * num_threads;

    workq.submit([&](sycl::handler &h){
        sycl::local_accessor<double, 1> sh_tmp(num_threads, h);

        h.parallel_for<ri_rmp2<Term>>(sycl::nd_range<1>(global_size, num_threads),
            [=](sycl::nd_item<1> item){
               ri_rmp2_kernel_body<Term>(item, nocc, nocc_block, nvir, i,
               naux, d_iajb, d_eps, d_energy, sh_tmp);
            });
    }).wait();
}


real_t ERI_RI_RHF::compute_mp2_energy() {
    PROFILE_FUNCTION();

    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = rhf_.get_num_basis() - nocc;
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies = rhf_.get_orbital_energies();

    sycl::queue& workq = gpu::GPUHandle::syclqueue();

    real_t *d_C = coefficient_matrix.device_ptr();
    real_t *d_eps = orbital_energies.device_ptr();
    real_t *d_intermediate_matrix_B = intermediate_matrix_B_.device_ptr();
    const int num_auxiliary_basis = num_auxiliary_basis_;

    // 中間バッファ
//    real_t* d_tmp = sycl::malloc_device<real_t>(num_auxiliary_basis * num_basis_ * num_basis_, workq);
    real_t* d_tmp = sycl::malloc_device<real_t>(num_basis_ * nvir * num_auxiliary_basis, workq);

    // エネルギー用スカラー
    real_t* d_energy = sycl::malloc_device<real_t>(1, workq);
    real_t zero = 0.0;
    workq.memcpy(d_energy, &zero, sizeof(real_t)).wait();

    double *d_iajb = nullptr;
    int nocc_block;

    // CUDA版 search_k_and_cudamalloc_4cERI に相当
    search_k_and_syclmalloc_4cERI(workq, nocc, nvir, nocc_block, &d_iajb);

    // intermediate matrix 変換
    transform_intermediate_matrix(workq, num_basis_, nocc, nvir, num_auxiliary_basis, d_C, d_intermediate_matrix_B, d_tmp);
    sycl::free(d_tmp, workq);

    const int num_threads = 1024;
    size_t num_blocks_1 = 0;
    size_t num_blocks_2 = 0;
    size_t num_blocks_3 = ((size_t)(nocc_block * nvir * (nvir - 1.0) / 2) + num_threads - 1) / num_threads;
    size_t num_blocks_4 = ((size_t)(nocc_block * nvir) + num_threads - 1) / num_threads;

    // niter: ブロックごとのループ回数
    int niter = (nocc + nocc_block - 1) / nocc_block;

    for(int i = 0; i < nocc; i += nocc_block) {
        int curr_block = std::min(nocc_block, nocc - i);

        // GEMM: d_iajb = B * C^T 相当
        oneapi::mkl::blas::column_major::gemm(
            workq,
            oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
            nocc * nvir, curr_block * nvir, num_auxiliary_basis,
            1.0,                      // alpha
            d_intermediate_matrix_B, nocc * nvir,
            &d_intermediate_matrix_B[i * nvir], nocc * nvir,
            0.0,                     // beta
            d_iajb, nocc * nvir
        ).wait();

        // RI-RMP2 カーネル呼び出し相当
        launch_ri_rmp2_kernel<energy_kernel1>(
            workq, num_blocks_1, num_threads,
            nocc, nocc_block, nvir, i, num_auxiliary_basis,
            d_iajb, d_eps, d_energy
        );

        launch_ri_rmp2_kernel<energy_kernel2>(
            workq, num_blocks_2, num_threads,
            nocc, nocc_block, nvir, i, num_auxiliary_basis,
            d_iajb, d_eps, d_energy
        );

        launch_ri_rmp2_kernel<energy_kernel3>(
            workq, num_blocks_3, num_threads,
            nocc, nocc_block, nvir, i, num_auxiliary_basis,
            d_iajb, d_eps, d_energy
        );

        launch_ri_rmp2_kernel<energy_kernel4>(
            workq, num_blocks_4, num_threads,
            nocc, nocc_block, nvir, i, num_auxiliary_basis,
            d_iajb, d_eps, d_energy
        );

        }

    // ホストにコピーして取得
    real_t energy;
    workq.memcpy(&energy, d_energy, sizeof(real_t)).wait();
    sycl::free(d_iajb, workq);
    sycl::free(d_energy, workq);

    printf("RMP2_energy: %.10f\n", energy);
    printf("RMP2_total_energy: %.10f\n", rhf_.get_total_energy() + energy);
    printf("(nocc, nvir, naux) = (%d, %d, %d)\n", nocc, nvir, num_auxiliary_basis);

    return energy;
}


} // namespace gansu
