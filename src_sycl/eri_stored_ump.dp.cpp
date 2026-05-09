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
#include <iomanip>
#include <iostream>
#include <assert.h>

#include "uhf.hpp"
#include "eri_stored.hpp"
#include "device_host_memory.hpp"

#include "ao2mo.syh"
#include <cmath>

#define FULLMASK 0xffffffff

namespace gansu {



inline sycl::event compute_ump2_energy_contrib_ss( sycl::queue& q,
    double* d_second_energy,
    const double* eri, const double* eps,
    int nocc, int nvir)
{
    const size_t total = (size_t)nocc * nvir * nocc * nvir;

    return q.submit([&](sycl::handler& cgh) {
        auto red = sycl::reduction(d_second_energy, sycl::plus<double>());

        cgh.parallel_for( sycl::range<1>(total), red, [=](sycl::id<1> idx, auto& sum) {
                size_t t = idx[0];

                int i = t / (nvir * nocc * nvir);
                t %= (nvir * nocc * nvir);

                int a = t / (nocc * nvir);
                t %= (nocc * nvir);

                int j = t / nvir;
                int b = t % nvir;

                double iajb = eri[idx];
                double ibja = eri[(size_t)i*nvir*nocc*nvir + b*nocc*nvir + j*nvir + a];

                double denom = eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b];

                sum += iajb * (2.0 * iajb - ibja) / denom;
            }
        );
    });
}




inline sycl::event compute_ump2_energy_contrib_os(sycl::queue& q,
    double* d_second_energy, const double* eri,
    const double* eps_a, const double* eps_b,
    int nocc_a, int nvir_a,
    int nocc_b, int nvir_b)
{
    const std::size_t total = (std::size_t)nocc_a * nvir_a * nocc_b * nvir_b;

    return q.submit([&](sycl::handler& cgh) {
        auto red = sycl::reduction(d_second_energy, sycl::plus<double>());

        cgh.parallel_for( sycl::range<1>(total), red, [=](sycl::id<1> id, auto& sum) {
                std::size_t t = id[0];

                int b = t % nvir_b; t /= nvir_b;
                int j = t % nocc_b; t /= nocc_b;
                int a = t % nvir_a; t /= nvir_a;
                int i = (int)t;

                std::size_t idx_iajb =
                    (((std::size_t)i * nvir_a + a) * nocc_b + j) * nvir_b + b;

                double iajb = eri[idx_iajb];

                double denom = eps_a[i] + eps_b[j] - eps_a[nocc_a + a] - eps_b[nocc_b + b];

                sum += 2.0 * iajb * iajb / denom;
            }
        );
    });
}





double ump2_from_aoeri_via_required_moeri(
    double* d_eri_ao,
    const double* d_coefficient_matrix_al,
    const double* d_coefficient_matrix_be,
    const double* d_orbital_energies_al,
    const double* d_orbital_energies_be,
    const int num_basis, 
    const int num_occupied_orbitals_al,
    const int num_occupied_orbitals_be)
{
    sycl::queue& q_ct1 = gpu::GPUHandle::syclqueue();
    double* d_eri_tmp1 = nullptr;
    double* d_eri_tmp2 = nullptr;
    const size_t num_basis_2 = num_basis * num_basis;
    const int max_num_occ = std::max(num_occupied_orbitals_al, num_occupied_orbitals_be);
    d_eri_tmp1 = tracked_syclMalloc<double>(num_basis_2 * num_basis_2, q_ct1);
    d_eri_tmp2 = tracked_syclMalloc<double>(max_num_occ * num_basis_2 * num_basis, q_ct1);
    if (!d_eri_tmp1) { THROW_EXCEPTION("syclMalloc failed for d_eri_tmp_1."); }
    if (!d_eri_tmp2) { THROW_EXCEPTION("syclMalloc failed for d_eri_tmp_2."); }

    const int num_virtual_orbitals_al = num_basis - num_occupied_orbitals_al;
    const int num_virtual_orbitals_be = num_basis - num_occupied_orbitals_be;

    double* d_second_energy = nullptr;
    d_second_energy = tracked_syclMalloc<double>(1, q_ct1);
    q_ct1.memset(d_second_energy, 0, sizeof(double)).wait();

    const int num_threads_per_warp = 32;
    const int num_warps_per_block = 32;
    const int num_threads_per_block = num_threads_per_warp * num_warps_per_block;

    float time_aa, time_bb, time_ab;
//    dpct::event_ptr begin, end;
//    begin = new sycl::event();
//    end = new sycl::event();

//    dpct::sync_barrier(begin);
    // Compute alpha-alpha energy contribution
    {
        std::string str = "Computing 1st term... ";
        PROFILE_ELAPSED_TIME(str);

        q_ct1.memcpy(d_eri_tmp1, d_eri_ao, sizeof(double) * num_basis_2 * num_basis_2);
        q_ct1.memset(d_eri_tmp2, 0,
                    sizeof(double) * max_num_occ * num_basis_2 * num_basis)
        .wait();

        // AO ERIs (d_eri_tmp1) will be overwritten with (ia|jb) MO ERIs (d_eri_mo_ovov)
        transform_eri_ao2mo_dgemm_ovov(q_ct1, d_eri_tmp1, d_eri_tmp2, d_coefficient_matrix_al, num_occupied_orbitals_al, num_virtual_orbitals_al);
        q_ct1.wait_and_throw();
        double* d_eri_mo_ovov_aa = d_eri_tmp1;

        // aaaa
        require_fp64(q_ct1);

        sycl::event e = compute_ump2_energy_contrib_ss(q_ct1,
            d_second_energy, d_eri_mo_ovov_aa, d_orbital_energies_al, num_occupied_orbitals_al, num_virtual_orbitals_al);

        q_ct1.wait_and_throw();
        uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = e.get_profiling_info<sycl::info::event_profiling::command_end>();

        time_aa = (end - start) * 1e-6; // ns → ms
        printf("alpha-alpha: %.2f [ms]\n", time_aa);
    }
//    dpct::sync_barrier(end);
//    end->wait_and_throw();
//    time_aa =
//        (end->get_profiling_info<sycl::info::event_profiling::command_end>() -
//         begin->get_profiling_info<
//             sycl::info::event_profiling::command_start>()) /
//        1000000.0f;
//    printf("alpha-alpha: %.2f [ms]\n", time_aa);

//    dpct::sync_barrier(begin);
    // Compute beta-beta energy contribution
    {
        std::string str = "Computing 2nd term... ";
        PROFILE_ELAPSED_TIME(str);

        q_ct1.memcpy(d_eri_tmp1, d_eri_ao, sizeof(double) * num_basis_2 * num_basis_2);
        q_ct1
            .memset(d_eri_tmp2, 0, sizeof(double) * max_num_occ * num_basis_2 * num_basis)
            .wait();

        // AO ERIs (d_eri_tmp1) will be overwritten with (ia|jb) MO ERIs (d_eri_mo_ovov)
        transform_eri_ao2mo_dgemm_ovov(q_ct1, d_eri_tmp1, d_eri_tmp2, d_coefficient_matrix_be, num_occupied_orbitals_be, num_virtual_orbitals_be);
        q_ct1.wait_and_throw();
        double* d_eri_mo_ovov_bb = d_eri_tmp1;

        require_fp64(q_ct1);
        sycl::event e = compute_ump2_energy_contrib_ss(q_ct1,
            d_second_energy, d_eri_mo_ovov_bb, d_orbital_energies_be, num_occupied_orbitals_be, num_virtual_orbitals_be);
        q_ct1.wait_and_throw();
        uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = e.get_profiling_info<sycl::info::event_profiling::command_end>();

        time_bb = (end - start) * 1e-6; // ns → ms
        printf("beta-beta: %.2f [ms]\n", time_bb);
    }
//    dpct::sync_barrier(end);
//    end->wait_and_throw();
//    time_bb =
//        (end->get_profiling_info<sycl::info::event_profiling::command_end>() -
//         begin->get_profiling_info<
//             sycl::info::event_profiling::command_start>()) /
//        1000000.0f;
//    printf("beta-beta: %.2f [ms]\n", time_bb);

//    dpct::sync_barrier(begin);
    // Compute alpha-beta energy contribution
    {
        std::string str = "Computing 3rd term... ";
        PROFILE_ELAPSED_TIME(str);

        q_ct1.memcpy(d_eri_tmp1, d_eri_ao,
                     sizeof(double) * num_basis_2 * num_basis_2);
        q_ct1
            .memset(d_eri_tmp2, 0,
                    sizeof(double) * max_num_occ * num_basis_2 * num_basis)
            .wait();

        // AO ERIs (d_eri_tmp1) will be overwritten with (ia|jb) MO ERIs (d_eri_mo_ovov)
        transform_eri_ao2mo_dgemm_ovov_os(q_ct1, d_eri_tmp1, d_eri_tmp2, d_coefficient_matrix_al, d_coefficient_matrix_be, num_occupied_orbitals_al, num_virtual_orbitals_al, num_occupied_orbitals_be, num_virtual_orbitals_be);
        q_ct1.wait_and_throw();
        double* d_eri_mo_ovov_ab = d_eri_tmp1;
       
        require_fp64(q_ct1);

        sycl::event e = compute_ump2_energy_contrib_os(q_ct1,
            d_second_energy, d_eri_mo_ovov_ab, d_orbital_energies_al, d_orbital_energies_be,
            num_occupied_orbitals_al, num_virtual_orbitals_al, num_occupied_orbitals_be,
            num_virtual_orbitals_be);

        q_ct1.wait_and_throw();
        uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = e.get_profiling_info<sycl::info::event_profiling::command_end>();

        time_ab = (end - start) * 1e-6; // ns → ms
        printf("alpha-beta: %.2f [ms]\n", time_ab);
    }
//    dpct::sync_barrier(end);
//    end->wait_and_throw();
//    time_ab =
//        (end->get_profiling_info<sycl::info::event_profiling::command_end>() -
//         begin->get_profiling_info<
//             sycl::info::event_profiling::command_start>()) /
//        1000000.0f;
//    printf("alpha-beta: %.2f [ms]\n", time_ab);


    double h_second_energy = 0.0;
    q_ct1.memcpy(&h_second_energy, d_second_energy, sizeof(double)).wait();
    std::cout << "UMP2 correlation energy: " << std::setprecision(12) << h_second_energy << std::endl;

    tracked_syclFree(d_eri_tmp1);
    tracked_syclFree(d_eri_tmp2);
    tracked_syclFree(d_second_energy);

    return h_second_energy;
}










real_t ERI_Stored_UHF::compute_mp2_energy() 
{
    PROFILE_FUNCTION();

    const int num_basis = uhf_.get_num_basis();
    const int num_occ_al = uhf_.get_num_alpha_spins();
    const int num_occ_be = uhf_.get_num_beta_spins();

    DeviceHostMatrix<real_t>& coefficient_matrix_al = uhf_.get_coefficient_matrix_a();
    DeviceHostMatrix<real_t>& coefficient_matrix_be = uhf_.get_coefficient_matrix_b();
    DeviceHostMemory<real_t>& orbital_energies_al = uhf_.get_orbital_energies_a();
    DeviceHostMemory<real_t>& orbital_energies_be = uhf_.get_orbital_energies_b();

    //const real_t* d_C = coefficient_matrix.device_ptr();
    //const real_t* d_eps = orbital_energies.device_ptr();
    //real_t* d_eri = eri_matrix_.device_ptr();

    //const real_t E_UMP2 = 1.0;
    const real_t E_UMP2 = ump2_from_aoeri_via_required_moeri(
        eri_matrix_.device_ptr(), 
        coefficient_matrix_al.device_ptr(), 
        coefficient_matrix_be.device_ptr(), 
        orbital_energies_al.device_ptr(),
        orbital_energies_be.device_ptr(),
        num_basis, 
        num_occ_al, 
        num_occ_be
    );

    std::cout << "UMP2 energy test" << std::endl;

    return E_UMP2;
}














}   // namespace gansu
