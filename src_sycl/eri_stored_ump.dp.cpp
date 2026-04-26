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
#include <dpct/dpct.hpp>
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





//*
void compute_ump2_energy_contrib_ss(
    double* g_energy_second, 
    const double* g_eri_mo, const double* g_eps, 
    const int num_occupied, const int num_virtual, double &s_tmp)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0) {
        s_tmp = 0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    double tmp = 0.0;
    const size_t seq =
        (((size_t)item_ct1.get_local_range(2) * item_ct1.get_local_range(1)) *
         item_ct1.get_group(2)) +
        item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
        item_ct1.get_local_id(2);
    if (seq < (size_t)num_occupied * num_virtual * (size_t)num_occupied * num_virtual) {
        const int ia = seq / (num_occupied * num_virtual);
        const int jb = seq % (num_occupied * num_virtual);
        const int i = ia / num_virtual;
        const int a = ia % num_virtual;
        const int j = jb / num_virtual;
        const int b = jb % num_virtual;

        const double iajb = g_eri_mo[ovov2seq(i, a, j, b, num_occupied, num_virtual)];
        const double jaib = g_eri_mo[ovov2seq(j, a, i, b, num_occupied, num_virtual)];
        //tmp = iajb * (2 * iajb - jaib) / (g_eps[i] + g_eps[j] - g_eps[num_occupied + a] - g_eps[num_occupied + b]);
        tmp = iajb * (iajb - jaib) / (g_eps[i] + g_eps[j] - g_eps[num_occupied + a] - g_eps[num_occupied + b]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        tmp += dpct::shift_sub_group_left(
            sycl::ext::oneapi::this_work_item::get_sub_group(), tmp, offset);
    }
    if (item_ct1.get_local_id(2) == 0) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &s_tmp, tmp);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            g_energy_second, s_tmp * 0.5);
    }
}
/**/


//*
void compute_ump2_energy_contrib_os(
    double* g_energy_second, const double* g_eri_mo, 
    const double* g_eps_al, const double* g_eps_be, 
    const int num_occupied_al, const int num_virtual_al, 
    const int num_occupied_be, const int num_virtual_be, double &s_tmp)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0) {
        s_tmp = 0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    double tmp = 0.0;
    const size_t seq =
        (((size_t)item_ct1.get_local_range(2) * item_ct1.get_local_range(1)) *
         item_ct1.get_group(2)) +
        item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
        item_ct1.get_local_id(2);
    if (seq < (size_t)num_occupied_al * num_virtual_al * (size_t)num_occupied_be * num_virtual_be) {
        const int ia = seq / (num_occupied_be * num_virtual_be);
        const int jb = seq % (num_occupied_be * num_virtual_be);
        const int i = ia / num_virtual_al;
        const int a = ia % num_virtual_al;
        const int j = jb / num_virtual_be;
        const int b = jb % num_virtual_be;

        const double iajb = g_eri_mo[ovov2seq_aabb(i, a, j, b, num_occupied_al, num_virtual_al, num_occupied_be, num_virtual_be)];
        tmp = (iajb * iajb) / (g_eps_al[i] + g_eps_be[j] - g_eps_al[num_occupied_al + a] - g_eps_be[num_occupied_be + b]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        tmp += dpct::shift_sub_group_left(
            sycl::ext::oneapi::this_work_item::get_sub_group(), tmp, offset);
    }
    if (item_ct1.get_local_id(2) == 0) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &s_tmp, tmp);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            g_energy_second, s_tmp);
    }
}
/**/





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
//  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
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
    dpct::event_ptr begin, end;
    begin = new sycl::event();
    end = new sycl::event();

    dpct::sync_barrier(begin);
    // Compute alpha-alpha energy contribution
    {
        std::string str = "Computing 1st term... ";
        PROFILE_ELAPSED_TIME(str);

        q_ct1.memcpy(d_eri_tmp1, d_eri_ao,
                     sizeof(double) * num_basis_2 * num_basis_2);
        q_ct1
            .memset(d_eri_tmp2, 0,
                    sizeof(double) * max_num_occ * num_basis_2 * num_basis)
            .wait();

        // AO ERIs (d_eri_tmp1) will be overwritten with (ia|jb) MO ERIs (d_eri_mo_ovov)
        transform_eri_ao2mo_dgemm_ovov(q_ct1, d_eri_tmp1, d_eri_tmp2, d_coefficient_matrix_al, num_occupied_orbitals_al, num_virtual_orbitals_al);
        q_ct1.wait_and_throw();
        double* d_eri_mo_ovov_aa = d_eri_tmp1;

        const size_t total = (size_t)num_occupied_orbitals_al * num_virtual_orbitals_al * num_occupied_orbitals_al * num_virtual_orbitals_al;
        const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
        const dpct::dim3 blocks(num_blocks);
        const dpct::dim3 threads(num_threads_per_warp, num_warps_per_block);

        // aaaa
        /*
        DPCT1049:58: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(q_ct1.get_device(),
                                         {sycl::aspect::fp64});

            q_ct1.submit([&](sycl::handler &cgh) {
                sycl::local_accessor<double, 0> s_tmp_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            compute_ump2_energy_contrib_ss(
                                d_second_energy, d_eri_mo_ovov_aa,
                                d_orbital_energies_al, num_occupied_orbitals_al,
                                num_virtual_orbitals_al, s_tmp_acc_ct1);
                        });
            });
        }
        q_ct1.wait_and_throw();
    }
    dpct::sync_barrier(end);
    end->wait_and_throw();
    time_aa =
        (end->get_profiling_info<sycl::info::event_profiling::command_end>() -
         begin->get_profiling_info<
             sycl::info::event_profiling::command_start>()) /
        1000000.0f;
    printf("alpha-alpha: %.2f [ms]\n", time_aa);

    dpct::sync_barrier(begin);
    // Compute beta-beta energy contribution
    {
        std::string str = "Computing 2nd term... ";
        PROFILE_ELAPSED_TIME(str);

        q_ct1.memcpy(d_eri_tmp1, d_eri_ao,
                     sizeof(double) * num_basis_2 * num_basis_2);
        q_ct1
            .memset(d_eri_tmp2, 0,
                    sizeof(double) * max_num_occ * num_basis_2 * num_basis)
            .wait();

        // AO ERIs (d_eri_tmp1) will be overwritten with (ia|jb) MO ERIs (d_eri_mo_ovov)
        transform_eri_ao2mo_dgemm_ovov(q_ct1, d_eri_tmp1, d_eri_tmp2, d_coefficient_matrix_be, num_occupied_orbitals_be, num_virtual_orbitals_be);
        q_ct1.wait_and_throw();
        double* d_eri_mo_ovov_bb = d_eri_tmp1;

        const size_t total = (size_t)num_occupied_orbitals_be * num_virtual_orbitals_be * num_occupied_orbitals_be * num_virtual_orbitals_be;
        const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
        const dpct::dim3 blocks(num_blocks);
        const dpct::dim3 threads(num_threads_per_warp, num_warps_per_block);

        // bbbb
        /*
        DPCT1049:59: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(q_ct1.get_device(),
                                         {sycl::aspect::fp64});

            q_ct1.submit([&](sycl::handler &cgh) {
                sycl::local_accessor<double, 0> s_tmp_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            compute_ump2_energy_contrib_ss(
                                d_second_energy, d_eri_mo_ovov_bb,
                                d_orbital_energies_be, num_occupied_orbitals_be,
                                num_virtual_orbitals_be, s_tmp_acc_ct1);
                        });
            });
        }
        q_ct1.wait_and_throw();
    }
    dpct::sync_barrier(end);
    end->wait_and_throw();
    time_bb =
        (end->get_profiling_info<sycl::info::event_profiling::command_end>() -
         begin->get_profiling_info<
             sycl::info::event_profiling::command_start>()) /
        1000000.0f;
    printf("beta-beta: %.2f [ms]\n", time_bb);

    dpct::sync_barrier(begin);
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

        const size_t total = (size_t)num_occupied_orbitals_al * num_virtual_orbitals_al * num_occupied_orbitals_be * num_virtual_orbitals_be;
        const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
        const dpct::dim3 blocks(num_blocks);
        const dpct::dim3 threads(num_threads_per_warp, num_warps_per_block);

        // aabb
        /*
        DPCT1049:60: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(q_ct1.get_device(),
                                         {sycl::aspect::fp64});

            q_ct1.submit([&](sycl::handler &cgh) {
                sycl::local_accessor<double, 0> s_tmp_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            compute_ump2_energy_contrib_os(
                                d_second_energy, d_eri_mo_ovov_ab,
                                d_orbital_energies_al, d_orbital_energies_be,
                                num_occupied_orbitals_al,
                                num_virtual_orbitals_al,
                                num_occupied_orbitals_be,
                                num_virtual_orbitals_be, s_tmp_acc_ct1);
                        });
            });
        }
        q_ct1.wait_and_throw();
    }
    dpct::sync_barrier(end);
    end->wait_and_throw();
    time_ab =
        (end->get_profiling_info<sycl::info::event_profiling::command_end>() -
         begin->get_profiling_info<
             sycl::info::event_profiling::command_start>()) /
        1000000.0f;
    printf("alpha-beta: %.2f [ms]\n", time_ab);


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
