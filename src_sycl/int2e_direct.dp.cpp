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
//#include <dpct/dpct.hpp>
#include <cstdio>
#include <cstdint>
#include <cmath>

#include "boys.hpp"
#include "int2e.hpp"
#include "types.hpp"
#include "utils_cuda.hpp"
#include "Et_functions.hpp"
#include "int2fock.syh"

namespace gansu::gpu{


#define TASK_GROUP_SIZE 16

SYCL_EXTERNAL
void composeFockMatrix(sycl::nd_item<1> item,
    real_t* g_fock_matrix, real_t* g_fock_matrix_replicas, const real_t* g_int1e, const int num_basis, const int num_fock_replicas)
{
    const int num_utm_elements = num_basis * (num_basis + 1) / 2;
    const int idx_linear = item.get_global_linear_id();
    if (idx_linear >= num_utm_elements)  return; 

//    const int nu = sycl::vec<double, 1>{
//            ((sycl::sqrt(sycl::vec<int, 1>{(8 * idx_linear + 1)} .convert<double, sycl::rounding_mode::rte>()[0]) - 1) / 2)}
//            .convert<int, sycl::rounding_mode::rtn>()[0];
    const int nu = static_cast<int>(
        sycl::floor((sycl::sqrt(8.0 * idx_linear + 1.0) - 1.0) * 0.5));

    const int mu = idx_linear - nu * (nu + 1) / 2;

    real_t fock_value = 0.0;
    const int stride = num_basis * num_basis;
    for (int i = 0; i < num_fock_replicas; ++i) {
        fock_value += g_fock_matrix_replicas[stride * i + (num_basis * mu + nu)];
    }
    fock_value += g_int1e[num_basis * mu + nu];
    g_fock_matrix[num_basis * mu + nu] = fock_value;
    if (mu != nu) {
        g_fock_matrix[num_basis * nu + mu] = fock_value;
    }
}

SYCL_EXTERNAL
void ssss2e_dynamic(const sycl::nd_item<1>& item_ct1,
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    const sycl::local_accessor<int, 1>& s_ket_group_idx,
    const sycl::local_accessor<bool, 1>& s_significant_flag) 

{
    int ab, cd;
    int bra_group_idx = 0;
    int ket_group_idx;

    int primitive_shell_index_a, primitive_shell_index_b;
    int primitive_shell_index_c, primitive_shell_index_d;
    bool is_bra_asymmetric, is_ket_asymmetric, is_braket_asymmetric;
    const int num_bra_groups = (num_bra + TASK_GROUP_SIZE - 1) / TASK_GROUP_SIZE;

    while (bra_group_idx < num_bra_groups && bra_group_idx < g_min_skipped_column[bra_group_idx]) {
        ab = (TASK_GROUP_SIZE * bra_group_idx) +
             (item_ct1.get_local_id(0) / TASK_GROUP_SIZE);
        while (true) {
            if (item_ct1.get_local_id(0) == 0) {
                s_significant_flag[0] = false;
                ket_group_idx = bra_group_idx + atomic_add(g_counter, bra_group_idx, 1);
                if (ket_group_idx < g_min_skipped_column[bra_group_idx]) {
                    if (g_schwarz_upper_bound_factors[head_bra + TASK_GROUP_SIZE * bra_group_idx] * g_schwarz_upper_bound_factors[head_ket + TASK_GROUP_SIZE * ket_group_idx] < schwarz_screening_threshold) {
                        atomic_min(g_min_skipped_column, bra_group_idx, ket_group_idx);
                        //s_significant_flag = false;
                    }
                    else {
                        s_significant_flag[0] = true;
                        s_ket_group_idx[0] = ket_group_idx;
                    }
                }
                else {
                    //s_significant_flag = false;
                }
            }
            /*
            DPCT1118:0: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);

            if (s_significant_flag[0]) {
                cd = (TASK_GROUP_SIZE * s_ket_group_idx[0]) +
                     (item_ct1.get_local_id(0) % TASK_GROUP_SIZE);
                if (ab <= cd && cd < num_ket) {
                    primitive_shell_index_a =
                        g_primitive_shell_pair_indices[head_bra + ab].x() +
                        shell_s0.start_index;
                    primitive_shell_index_b =
                        g_primitive_shell_pair_indices[head_bra + ab].y() +
                        shell_s1.start_index;
                    primitive_shell_index_c =
                        g_primitive_shell_pair_indices[head_ket + cd].x() +
                        shell_s2.start_index;
                    primitive_shell_index_d =
                        g_primitive_shell_pair_indices[head_ket + cd].y() +
                        shell_s3.start_index;
                    is_bra_asymmetric = (primitive_shell_index_a != primitive_shell_index_b);
                    is_ket_asymmetric = (primitive_shell_index_c != primitive_shell_index_d);
                    is_braket_asymmetric = ((primitive_shell_index_a != primitive_shell_index_c) || (primitive_shell_index_b != primitive_shell_index_d));
                    //is_braket_asymmetric = (ab != cd);
                    ssss2fock(
                        g_primitive_shells[primitive_shell_index_a],
                        g_primitive_shells[primitive_shell_index_b],
                        g_primitive_shells[primitive_shell_index_c],
                        g_primitive_shells[primitive_shell_index_d],
                        is_bra_asymmetric, is_ket_asymmetric,
                        is_braket_asymmetric,
                        g_fock_matrix_replicas +
                            num_basis * num_basis *
                                (item_ct1.get_local_id(0) % num_fock_replicas),
                        num_basis, g_boys_grid, g_density_matrix,
                        g_cgto_normalization_factors);
                }
            }
            else {
                break;
            }
        }
        bra_group_idx++;
    }
}

SYCL_EXTERNAL
void sssp2e_dynamic(const sycl::nd_item<1>& item_ct1,
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    const sycl::local_accessor<int, 1>& s_ket_group_idx,
    const sycl::local_accessor<bool, 1>& s_significant_flag) 
{
    //*
    int ab, cd;
    int bra_group_idx = 0;
    int ket_group_idx;

    int primitive_shell_index_a, primitive_shell_index_b;
    int primitive_shell_index_c, primitive_shell_index_d;
    bool is_bra_asymmetric; //is_ket_asymmetric, is_braket_asymmetric;
    const int num_bra_groups = (num_bra + TASK_GROUP_SIZE - 1) / TASK_GROUP_SIZE;

    while (bra_group_idx < num_bra_groups && 0 < g_min_skipped_column[bra_group_idx]) {
        ab = (TASK_GROUP_SIZE * bra_group_idx) +
             (item_ct1.get_local_id(0) / TASK_GROUP_SIZE);
        while (true) {
            if (item_ct1.get_local_id(0) == 0) {
                s_significant_flag[0] = false;
                ket_group_idx = atomic_add(g_counter, bra_group_idx, 1);
                if (ket_group_idx < g_min_skipped_column[bra_group_idx]) {
                    if (g_schwarz_upper_bound_factors[head_bra + TASK_GROUP_SIZE * bra_group_idx] * g_schwarz_upper_bound_factors[head_ket + TASK_GROUP_SIZE * ket_group_idx] < schwarz_screening_threshold) {
                        atomic_min(g_min_skipped_column, bra_group_idx, ket_group_idx);
                        //s_significant_flag = false;
                    }
                    else {
                        s_significant_flag[0] = true;
                        s_ket_group_idx[0] = ket_group_idx;
                    }
                }
                else {
                    //s_significant_flag = false;
                }
            }
            /*
            DPCT1118:1: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);

            if (s_significant_flag[0]) {
                cd = (TASK_GROUP_SIZE * s_ket_group_idx[0]) +
                     (item_ct1.get_local_id(0) % TASK_GROUP_SIZE);
                if (ab < num_bra && cd < num_ket) {
                    primitive_shell_index_a =
                        g_primitive_shell_pair_indices[head_bra + ab].x() +
                        shell_s0.start_index;
                    primitive_shell_index_b =
                        g_primitive_shell_pair_indices[head_bra + ab].y() +
                        shell_s1.start_index;
                    primitive_shell_index_c =
                        g_primitive_shell_pair_indices[head_ket + cd].x() +
                        shell_s2.start_index;
                    primitive_shell_index_d =
                        g_primitive_shell_pair_indices[head_ket + cd].y() +
                        shell_s3.start_index;
                    is_bra_asymmetric = (primitive_shell_index_a != primitive_shell_index_b);
                    //is_ket_asymmetric = (primitive_shell_index_c != primitive_shell_index_d);
                    //is_braket_asymmetric = ((primitive_shell_index_a != primitive_shell_index_c) || (primitive_shell_index_b != primitive_shell_index_d));
                    sssp2fock(
                        g_primitive_shells[primitive_shell_index_a],
                        g_primitive_shells[primitive_shell_index_b],
                        g_primitive_shells[primitive_shell_index_c],
                        g_primitive_shells[primitive_shell_index_d],
                        is_bra_asymmetric, // is_ket_asymmetric,
                                           // is_braket_asymmetric,
                        g_fock_matrix_replicas +
                            num_basis * num_basis *
                                (item_ct1.get_local_id(0) % num_fock_replicas),
                        num_basis, g_boys_grid, g_density_matrix,
                        g_cgto_normalization_factors);
                }
            }
            else {
                break;
            }
        }
        ++bra_group_idx;
    }
    /**/
}

SYCL_EXTERNAL
void sspp2e_dynamic(const sycl::nd_item<1>& item_ct1,
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    const sycl::local_accessor<int, 1>& s_ket_group_idx,
    const sycl::local_accessor<bool, 1>& s_significant_flag) 
{
    //*
    int ab, cd;
    int bra_group_idx = 0;
    int ket_group_idx;

    int primitive_shell_index_a, primitive_shell_index_b;
    int primitive_shell_index_c, primitive_shell_index_d;
    bool is_bra_asymmetric, is_ket_asymmetric; //is_braket_asymmetric;
    const int num_bra_groups = (num_bra + TASK_GROUP_SIZE - 1) / TASK_GROUP_SIZE;

    while (bra_group_idx < num_bra_groups && 0 < g_min_skipped_column[bra_group_idx]) {
        ab = (TASK_GROUP_SIZE * bra_group_idx) +
             (item_ct1.get_local_id(0) / TASK_GROUP_SIZE);
        while (true) {
            if (item_ct1.get_local_id(0) == 0) {
                s_significant_flag[0] = false;
                ket_group_idx = atomic_add(g_counter, bra_group_idx, 1);
                if (ket_group_idx < g_min_skipped_column[bra_group_idx]) {
                    if (g_schwarz_upper_bound_factors[head_bra + TASK_GROUP_SIZE * bra_group_idx] * g_schwarz_upper_bound_factors[head_ket + TASK_GROUP_SIZE * ket_group_idx] < schwarz_screening_threshold) {
                        atomic_min(g_min_skipped_column, bra_group_idx, ket_group_idx);
                        //s_significant_flag = false;
                    }
                    else {
                        s_significant_flag[0] = true;
                        s_ket_group_idx[0] = ket_group_idx;
                    }
                }
                else {
                    //s_significant_flag = false;
                }
            }
            /*
            DPCT1118:2: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);

            if (s_significant_flag[0]) {
                cd = (TASK_GROUP_SIZE * s_ket_group_idx[0]) +
                     (item_ct1.get_local_id(0) % TASK_GROUP_SIZE);
                if (ab < num_bra && cd < num_ket) {
                    primitive_shell_index_a =
                        g_primitive_shell_pair_indices[head_bra + ab].x() +
                        shell_s0.start_index;
                    primitive_shell_index_b =
                        g_primitive_shell_pair_indices[head_bra + ab].y() +
                        shell_s1.start_index;
                    primitive_shell_index_c =
                        g_primitive_shell_pair_indices[head_ket + cd].x() +
                        shell_s2.start_index;
                    primitive_shell_index_d =
                        g_primitive_shell_pair_indices[head_ket + cd].y() +
                        shell_s3.start_index;
                    is_bra_asymmetric = (primitive_shell_index_a != primitive_shell_index_b);
                    is_ket_asymmetric = (primitive_shell_index_c != primitive_shell_index_d);
                    //is_braket_asymmetric = ((primitive_shell_index_a != primitive_shell_index_c) || (primitive_shell_index_b != primitive_shell_index_d));
                    sspp2fock(
                        g_primitive_shells[primitive_shell_index_a],
                        g_primitive_shells[primitive_shell_index_b],
                        g_primitive_shells[primitive_shell_index_c],
                        g_primitive_shells[primitive_shell_index_d],
                        is_bra_asymmetric,
                        is_ket_asymmetric, // is_braket_asymmetric,
                        g_fock_matrix_replicas +
                            num_basis * num_basis *
                                (item_ct1.get_local_id(0) % num_fock_replicas),
                        num_basis, g_boys_grid, g_density_matrix,
                        g_cgto_normalization_factors);
                }
            }
            else {
                break;
            }
        }
        ++bra_group_idx;
    }
    /**/
}

SYCL_EXTERNAL
void spsp2e_dynamic(const sycl::nd_item<1>& item_ct1,
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    const sycl::local_accessor<int, 1>& s_ket_group_idx,
    const sycl::local_accessor<bool, 1>& s_significant_flag) 
{
    //*
    int ab, cd;
    int bra_group_idx = 0;
    int ket_group_idx;

    int primitive_shell_index_a, primitive_shell_index_b;
    int primitive_shell_index_c, primitive_shell_index_d;
    bool /*is_bra_asymmetric, is_ket_asymmetric, */is_braket_asymmetric;
    const int num_bra_groups = (num_bra + TASK_GROUP_SIZE - 1) / TASK_GROUP_SIZE;

    while (bra_group_idx < num_bra_groups && bra_group_idx < g_min_skipped_column[bra_group_idx]) {
        ab = (TASK_GROUP_SIZE * bra_group_idx) +
             (item_ct1.get_local_id(0) / TASK_GROUP_SIZE);
        while (true) {
            if (item_ct1.get_local_id(0) == 0) {
                s_significant_flag[0] = false;
                ket_group_idx = bra_group_idx + atomic_add(g_counter, bra_group_idx, 1);
                if (ket_group_idx < g_min_skipped_column[bra_group_idx]) {
                    if (g_schwarz_upper_bound_factors[head_bra + TASK_GROUP_SIZE * bra_group_idx] * g_schwarz_upper_bound_factors[head_ket + TASK_GROUP_SIZE * ket_group_idx] < schwarz_screening_threshold) {
                        atomic_min(g_min_skipped_column, bra_group_idx, ket_group_idx);
                        //s_significant_flag = false;
                    }
                    else {
                        s_significant_flag[0] = true;
                        s_ket_group_idx[0] = ket_group_idx;
                    }
                }
                else {
                    //s_significant_flag = false;
                }
            }
            /*
            DPCT1118:3: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);

            if (s_significant_flag[0]) {
                cd = (TASK_GROUP_SIZE * s_ket_group_idx[0]) +
                     (item_ct1.get_local_id(0) % TASK_GROUP_SIZE);
                if (ab <= cd && cd < num_ket) {
                    primitive_shell_index_a =
                        g_primitive_shell_pair_indices[head_bra + ab].x() +
                        shell_s0.start_index;
                    primitive_shell_index_b =
                        g_primitive_shell_pair_indices[head_bra + ab].y() +
                        shell_s1.start_index;
                    primitive_shell_index_c =
                        g_primitive_shell_pair_indices[head_ket + cd].x() +
                        shell_s2.start_index;
                    primitive_shell_index_d =
                        g_primitive_shell_pair_indices[head_ket + cd].y() +
                        shell_s3.start_index;
                    //is_bra_asymmetric = (primitive_shell_index_a != primitive_shell_index_b);
                    //is_ket_asymmetric = (primitive_shell_index_c != primitive_shell_index_d);
                    is_braket_asymmetric = ((primitive_shell_index_a != primitive_shell_index_c) || (primitive_shell_index_b != primitive_shell_index_d));
                    //is_braket_asymmetric = (ab != cd);
                    spsp2fock(
                        g_primitive_shells[primitive_shell_index_a],
                        g_primitive_shells[primitive_shell_index_b],
                        g_primitive_shells[primitive_shell_index_c],
                        g_primitive_shells[primitive_shell_index_d],
                        /*is_bra_asymmetric, is_ket_asymmetric, */
                        is_braket_asymmetric,
                        g_fock_matrix_replicas +
                            num_basis * num_basis *
                                (item_ct1.get_local_id(0) % num_fock_replicas),
                        num_basis, g_boys_grid, g_density_matrix,
                        g_cgto_normalization_factors);
                }
            }
            else {
                break;
            }
        }
        ++bra_group_idx;
    }
    /**/
}

SYCL_EXTERNAL
void sppp2e_dynamic(const sycl::nd_item<1>& item_ct1,
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    const sycl::local_accessor<int, 1>& s_ket_group_idx,
    const sycl::local_accessor<bool, 1>& s_significant_flag) 
{
    int ab, cd;
    int bra_group_idx = 0;
    int ket_group_idx;

    int primitive_shell_index_a, primitive_shell_index_b;
    int primitive_shell_index_c, primitive_shell_index_d;
    //bool is_bra_asymmetric, is_ket_asymmetric, is_braket_asymmetric;
    bool is_ket_asymmetric;
    const int num_bra_groups = (num_bra + TASK_GROUP_SIZE - 1) / TASK_GROUP_SIZE;

    while (bra_group_idx < num_bra_groups && 0 < g_min_skipped_column[bra_group_idx]) {
        ab = (TASK_GROUP_SIZE * bra_group_idx) +
             (item_ct1.get_local_id(0) / TASK_GROUP_SIZE);
        while (true) {
            if (item_ct1.get_local_id(0) == 0) {
                s_significant_flag[0] = false;
                ket_group_idx = atomic_add(g_counter, bra_group_idx, 1);
                if (ket_group_idx < g_min_skipped_column[bra_group_idx]) {
                    if (g_schwarz_upper_bound_factors[head_bra + TASK_GROUP_SIZE * bra_group_idx] * g_schwarz_upper_bound_factors[head_ket + TASK_GROUP_SIZE * ket_group_idx] < schwarz_screening_threshold) {
                        atomic_min(g_min_skipped_column, bra_group_idx, ket_group_idx);
                        //s_significant_flag = false;
                    }
                    else {
                        s_significant_flag[0] = true;
                        s_ket_group_idx[0] = ket_group_idx;
                    }
                }
                else {
                    //s_significant_flag = false;
                }
            }
            /*
            DPCT1118:4: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);

            if (s_significant_flag[0]) {
                cd = (TASK_GROUP_SIZE * s_ket_group_idx[0]) +
                     (item_ct1.get_local_id(0) % TASK_GROUP_SIZE);
                if (ab < num_bra && cd < num_ket) {
                    primitive_shell_index_a =
                        g_primitive_shell_pair_indices[head_bra + ab].x() +
                        shell_s0.start_index;
                    primitive_shell_index_b =
                        g_primitive_shell_pair_indices[head_bra + ab].y() +
                        shell_s1.start_index;
                    primitive_shell_index_c =
                        g_primitive_shell_pair_indices[head_ket + cd].x() +
                        shell_s2.start_index;
                    primitive_shell_index_d =
                        g_primitive_shell_pair_indices[head_ket + cd].y() +
                        shell_s3.start_index;
                    //is_bra_asymmetric = (primitive_shell_index_a != primitive_shell_index_b);
                    is_ket_asymmetric = (primitive_shell_index_c != primitive_shell_index_d);
                    //is_braket_asymmetric = ((primitive_shell_index_a != primitive_shell_index_c) || (primitive_shell_index_b != primitive_shell_index_d));
                    sppp2fock(
                        g_primitive_shells[primitive_shell_index_a],
                        g_primitive_shells[primitive_shell_index_b],
                        g_primitive_shells[primitive_shell_index_c],
                        g_primitive_shells[primitive_shell_index_d],
                        // is_bra_asymmetric, is_ket_asymmetric,
                        // is_braket_asymmetric,
                        is_ket_asymmetric,
                        g_fock_matrix_replicas +
                            num_basis * num_basis *
                                (item_ct1.get_local_id(0) % num_fock_replicas),
                        num_basis, g_boys_grid, g_density_matrix,
                        g_cgto_normalization_factors);
                }
            }
            else {
                break;
            }
        }
        ++bra_group_idx;
    }
    /**/
}

SYCL_EXTERNAL
void pppp2e_dynamic(const sycl::nd_item<1>& item_ct1,
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    const sycl::local_accessor<int, 1>& s_ket_group_idx,
    const sycl::local_accessor<bool, 1>& s_significant_flag) 
{
    int ab, cd;               // bra and ket index of shell pairs
    int bra_group_idx = 0;  // row index of task groups
    int ket_group_idx;      // column index of task groups

    int primitive_shell_index_a, primitive_shell_index_b;
    int primitive_shell_index_c, primitive_shell_index_d;
    bool is_bra_asymmetric, is_ket_asymmetric, is_braket_asymmetric;
    const int num_bra_groups = (num_bra + TASK_GROUP_SIZE - 1) / TASK_GROUP_SIZE;

    while (bra_group_idx < num_bra_groups && bra_group_idx < g_min_skipped_column[bra_group_idx]) {
        ab = (TASK_GROUP_SIZE * bra_group_idx) +
             (item_ct1.get_local_id(0) / TASK_GROUP_SIZE);
        while (true) {
            if (item_ct1.get_local_id(0) == 0) {
                s_significant_flag[0] = false;
                ket_group_idx = bra_group_idx + atomic_add(g_counter, bra_group_idx, 1);
                if (ket_group_idx < g_min_skipped_column[bra_group_idx]) {
                    if (g_schwarz_upper_bound_factors[head_bra + TASK_GROUP_SIZE * bra_group_idx] * g_schwarz_upper_bound_factors[head_ket + TASK_GROUP_SIZE * ket_group_idx] < schwarz_screening_threshold) {
                        atomic_min(g_min_skipped_column, bra_group_idx, ket_group_idx);
                        //s_significant_flag = false;
                    }
                    else {
                        s_significant_flag[0] = true;
                        s_ket_group_idx[0] = ket_group_idx;
                    }
                }
                else {
                    //s_significant_flag = false;
                }
            }
            /*
            DPCT1118:5: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);

            if (s_significant_flag[0]) {
                cd = (TASK_GROUP_SIZE * s_ket_group_idx[0]) +
                     (item_ct1.get_local_id(0) % TASK_GROUP_SIZE);
                if (ab <= cd && cd < num_ket) {
                    primitive_shell_index_a =
                        g_primitive_shell_pair_indices[head_bra + ab].x() +
                        shell_s0.start_index;
                    primitive_shell_index_b =
                        g_primitive_shell_pair_indices[head_bra + ab].y() +
                        shell_s1.start_index;
                    primitive_shell_index_c =
                        g_primitive_shell_pair_indices[head_ket + cd].x() +
                        shell_s2.start_index;
                    primitive_shell_index_d =
                        g_primitive_shell_pair_indices[head_ket + cd].y() +
                        shell_s3.start_index;
                    is_bra_asymmetric = (primitive_shell_index_a != primitive_shell_index_b);
                    is_ket_asymmetric = (primitive_shell_index_c != primitive_shell_index_d);
                    is_braket_asymmetric = (ab != cd);
                    pppp2fock(
                        g_primitive_shells[primitive_shell_index_a],
                        g_primitive_shells[primitive_shell_index_b],
                        g_primitive_shells[primitive_shell_index_c],
                        g_primitive_shells[primitive_shell_index_d],
                        is_bra_asymmetric, is_ket_asymmetric,
                        is_braket_asymmetric,
                        g_fock_matrix_replicas +
                            num_basis * num_basis *
                                (item_ct1.get_local_id(0) % num_fock_replicas),
                        num_basis, g_boys_grid, g_density_matrix,
                        g_cgto_normalization_factors);
                }
            }
            else {
                break;
            }
        }
        ++bra_group_idx;
    }
}

/*
DPCT1110:6: The total declared local variable size in device function
ssss2e_direct exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void ssss2e_direct(real_t *g_fock_matrix,
                   const PrimitiveShell *g_primitive_shells,
                   const sycl::int2 *g_primitive_shell_pair_indices,
                   const real_t *g_cgto_normalization_factors,
                   const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                   const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
                   const size_t num_braket,
                   const real_t schwarz_screening_threshold,
                   const real_t *g_schwarz_upper_bound_factors,
                   const int num_basis, const real_t *g_boys_grid,
                   const real_t *g_density_matrix, const size_t head_bra,
                   const size_t head_ket, const int num_fock_replicas)
{
    // 2D grid and 1D block to linear thread index
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx_linear = item_ct1.get_global_linear_id();
    if (idx_linear >= num_braket) {
        return;
    }

    // Calculate row and column indices from linear index (vertical thread assignment)
    /*
    DPCT1013:14: The rounding mode could not be specified and the generated code
    may have different accuracy than the original code. Verify the correctness.
    SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
    standard.
    */
    int cd =
        sycl::vec<double, 1>{
            ((sycl::sqrt(sycl::vec<long long, 1>{static_cast<long long>(8 * idx_linear + 1) }
                             .convert<double, sycl::rounding_mode::rte>()[0]) -
              1) /
             2)}
            .convert<int, sycl::rounding_mode::rtn>()[0];
    const int ab = head_bra + idx_linear - static_cast<size_t>(cd) * (cd + 1) / 2;
    cd += head_ket;

    // Task-wise Schwarz screening
    if (g_schwarz_upper_bound_factors[ab] * g_schwarz_upper_bound_factors[cd] < schwarz_screening_threshold) {
        return;
    }

    const int primitive_shell_index_a =
        g_primitive_shell_pair_indices[ab].x() + shell_s0.start_index;
    const int primitive_shell_index_b =
        g_primitive_shell_pair_indices[ab].y() + shell_s1.start_index;
    const int primitive_shell_index_c =
        g_primitive_shell_pair_indices[cd].x() + shell_s2.start_index;
    const int primitive_shell_index_d =
        g_primitive_shell_pair_indices[cd].y() + shell_s3.start_index;

    const PrimitiveShell a = g_primitive_shells[primitive_shell_index_a];
    const PrimitiveShell b = g_primitive_shells[primitive_shell_index_b];
    const PrimitiveShell c = g_primitive_shells[primitive_shell_index_c];
    const PrimitiveShell d = g_primitive_shells[primitive_shell_index_d];

    const bool is_bra_asymmetric = (primitive_shell_index_a != primitive_shell_index_b);
    const bool is_ket_asymmetric = (primitive_shell_index_c != primitive_shell_index_d);
    const bool is_braket_asymmetric = (ab != cd);
    //const bool is_braket_asymmetric = (primitive_shell_index_a != primitive_shell_index_c) || (primitive_shell_index_b != primitive_shell_index_d);

    ssss2fock(
        a, b, c, d, is_bra_asymmetric, is_ket_asymmetric, is_braket_asymmetric,
        g_fock_matrix + num_basis * num_basis *
                            (item_ct1.get_local_id(2) % num_fock_replicas),
        num_basis, g_boys_grid, g_density_matrix, g_cgto_normalization_factors);
}


//*
/*
DPCT1110:7: The total declared local variable size in device function
sssp2e_direct exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void sssp2e_direct(real_t *g_fock_matrix,
                   const PrimitiveShell *g_primitive_shells,
                   const sycl::int2 *g_primitive_shell_pair_indices,
                   const real_t *g_cgto_normalization_factors,
                   const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                   const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
                   const size_t num_braket,
                   const real_t schwarz_screening_threshold,
                   const real_t *g_schwarz_upper_bound_factors,
                   const int num_basis, const real_t *g_boys_grid,
                   const real_t *g_density_matrix, const size_t head_bra,
                   const size_t head_ket, const int num_fock_replicas)
{
    // 2D grid and 1D block to linear thread index
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx_linear = item_ct1.get_global_linear_id();
    if (idx_linear >= num_braket) {
        return;
    }

    //const int cd = (idx_linear % num_sp) + num_ss;
    const int num_ket = shell_s2.count * shell_s3.count;
    const int cd = head_ket + (idx_linear % num_ket);
    const int ab = head_bra + (idx_linear / num_ket);

    if (g_schwarz_upper_bound_factors[ab] * g_schwarz_upper_bound_factors[cd] < schwarz_screening_threshold) {
        return;
    }

    const int primitive_shell_index_a =
        g_primitive_shell_pair_indices[ab].x() + shell_s0.start_index;
    const int primitive_shell_index_b =
        g_primitive_shell_pair_indices[ab].y() + shell_s1.start_index;
    const int primitive_shell_index_c =
        g_primitive_shell_pair_indices[cd].x() + shell_s2.start_index;
    const int primitive_shell_index_d =
        g_primitive_shell_pair_indices[cd].y() + shell_s3.start_index;

    PrimitiveShell a = g_primitive_shells[primitive_shell_index_a];
    PrimitiveShell b = g_primitive_shells[primitive_shell_index_b];
    PrimitiveShell c = g_primitive_shells[primitive_shell_index_c];
    PrimitiveShell d = g_primitive_shells[primitive_shell_index_d];

    const bool is_bra_asymmetric = (primitive_shell_index_a != primitive_shell_index_b);
    //const bool is_ket_asymmetric = (primitive_shell_index_c != primitive_shell_index_d);
    //const bool is_braket_asymmetric = (ab != cd);

    sssp2fock(
        a, b, c, d, is_bra_asymmetric,
        g_fock_matrix + num_basis * num_basis *
                            (item_ct1.get_local_id(2) % num_fock_replicas),
        num_basis, g_boys_grid, g_density_matrix, g_cgto_normalization_factors);
}
/**/

/*
DPCT1110:8: The total declared local variable size in device function
sspp2e_direct exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void sspp2e_direct(real_t *g_fock_matrix,
                   const PrimitiveShell *g_primitive_shells,
                   const sycl::int2 *g_primitive_shell_pair_indices,
                   const real_t *g_cgto_normalization_factors,
                   const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                   const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
                   const size_t num_braket,
                   const real_t schwarz_screening_threshold,
                   const real_t *g_schwarz_upper_bound_factors,
                   const int num_basis, const real_t *g_boys_grid,
                   const real_t *g_density_matrix, const size_t head_bra,
                   const size_t head_ket, const int num_fock_replicas)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx_linear = item_ct1.get_global_linear_id();
    if (idx_linear >= num_braket) {
        return;
    }

    const int num_ket = shell_s2.count * (shell_s2.count + 1) / 2;
    const int cd = head_ket + (idx_linear % num_ket);
    const int ab = head_bra + (idx_linear / num_ket);

    if (g_schwarz_upper_bound_factors[ab] * g_schwarz_upper_bound_factors[cd] < schwarz_screening_threshold) {
        return;
    }

    const int primitive_shell_index_a =
        g_primitive_shell_pair_indices[ab].x() + shell_s0.start_index;
    const int primitive_shell_index_b =
        g_primitive_shell_pair_indices[ab].y() + shell_s1.start_index;
    const int primitive_shell_index_c =
        g_primitive_shell_pair_indices[cd].x() + shell_s2.start_index;
    const int primitive_shell_index_d =
        g_primitive_shell_pair_indices[cd].y() + shell_s3.start_index;

    PrimitiveShell a = g_primitive_shells[primitive_shell_index_a];
    PrimitiveShell b = g_primitive_shells[primitive_shell_index_b];
    PrimitiveShell c = g_primitive_shells[primitive_shell_index_c];
    PrimitiveShell d = g_primitive_shells[primitive_shell_index_d];

    const bool is_bra_asymmetric = (primitive_shell_index_a != primitive_shell_index_b);
    const bool is_ket_asymmetric = (primitive_shell_index_c != primitive_shell_index_d);
    //const bool is_braket_asymmetric = (ab != cd);

    sspp2fock(
        a, b, c, d, is_bra_asymmetric, is_ket_asymmetric,
        g_fock_matrix + num_basis * num_basis *
                            (item_ct1.get_local_id(2) % num_fock_replicas),
        num_basis, g_boys_grid, g_density_matrix, g_cgto_normalization_factors);
}

/*
DPCT1110:9: The total declared local variable size in device function
spsp2e_direct exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void spsp2e_direct(real_t *g_fock_matrix,
                   const PrimitiveShell *g_primitive_shells,
                   const sycl::int2 *g_primitive_shell_pair_indices,
                   const real_t *g_cgto_normalization_factors,
                   const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                   const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
                   const size_t num_braket,
                   const real_t schwarz_screening_threshold,
                   const real_t *g_schwarz_upper_bound_factors,
                   const int num_basis, const real_t *g_boys_grid,
                   const real_t *g_density_matrix, const size_t head_bra,
                   const size_t head_ket, const int num_fock_replicas)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx_linear = item_ct1.get_global_linear_id();
    if (idx_linear >= num_braket) {
        return;
    }

    /*
    DPCT1013:15: The rounding mode could not be specified and the generated code
    may have different accuracy than the original code. Verify the correctness.
    SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
    standard.
    */
    int cd =
        sycl::vec<double, 1>{
            ((sycl::sqrt(sycl::vec<long long, 1>{static_cast<long long>(8 * idx_linear + 1)}
                             .convert<double, sycl::rounding_mode::rte>()[0]) -
              1) /
             2)}
            .convert<int, sycl::rounding_mode::rtn>()[0];
    const int ab = head_bra + idx_linear - static_cast<size_t>(cd) * (cd + 1) / 2;
    cd += head_ket;

    if (g_schwarz_upper_bound_factors[ab] * g_schwarz_upper_bound_factors[cd] < schwarz_screening_threshold) {
        return;
    }

    const int primitive_shell_index_a =
        g_primitive_shell_pair_indices[ab].x() + shell_s0.start_index;
    const int primitive_shell_index_b =
        g_primitive_shell_pair_indices[ab].y() + shell_s1.start_index;
    const int primitive_shell_index_c =
        g_primitive_shell_pair_indices[cd].x() + shell_s2.start_index;
    const int primitive_shell_index_d =
        g_primitive_shell_pair_indices[cd].y() + shell_s3.start_index;

    PrimitiveShell a = g_primitive_shells[primitive_shell_index_a];
    PrimitiveShell b = g_primitive_shells[primitive_shell_index_b];
    PrimitiveShell c = g_primitive_shells[primitive_shell_index_c];
    PrimitiveShell d = g_primitive_shells[primitive_shell_index_d];

    //const bool is_bra_asymmetric = (primitive_shell_index_a != primitive_shell_index_b);
    //const bool is_ket_asymmetric = (primitive_shell_index_c != primitive_shell_index_d);
    const bool is_braket_asymmetric = (ab != cd);

    spsp2fock(
        a, b, c, d, is_braket_asymmetric,
        g_fock_matrix + num_basis * num_basis *
                            (item_ct1.get_local_id(2) % num_fock_replicas),
        num_basis, g_boys_grid, g_density_matrix, g_cgto_normalization_factors);
}

/*
DPCT1110:10: The total declared local variable size in device function
sppp2e_direct exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void sppp2e_direct(real_t *g_fock_matrix,
                   const PrimitiveShell *g_primitive_shells,
                   const sycl::int2 *g_primitive_shell_pair_indices,
                   const real_t *g_cgto_normalization_factors,
                   const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                   const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
                   const size_t num_braket,
                   const real_t schwarz_screening_threshold,
                   const real_t *g_schwarz_upper_bound_factors,
                   const int num_basis, const real_t *g_boys_grid,
                   const real_t *g_density_matrix, const size_t head_bra,
                   const size_t head_ket, const int num_fock_replicas)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx_linear = item_ct1.get_global_linear_id();
    if (idx_linear >= num_braket) {
        return;
    }

    const int num_ket = shell_s2.count * (shell_s2.count + 1) / 2;
    const int cd = head_ket + (idx_linear % num_ket);
    const int ab = head_bra + (idx_linear / num_ket);

    if (g_schwarz_upper_bound_factors[ab] * g_schwarz_upper_bound_factors[cd] < schwarz_screening_threshold) {
        return;
    }

    const int primitive_shell_index_a =
        g_primitive_shell_pair_indices[ab].x() + shell_s0.start_index;
    const int primitive_shell_index_b =
        g_primitive_shell_pair_indices[ab].y() + shell_s1.start_index;
    const int primitive_shell_index_c =
        g_primitive_shell_pair_indices[cd].x() + shell_s2.start_index;
    const int primitive_shell_index_d =
        g_primitive_shell_pair_indices[cd].y() + shell_s3.start_index;

    PrimitiveShell a = g_primitive_shells[primitive_shell_index_a];
    PrimitiveShell b = g_primitive_shells[primitive_shell_index_b];
    PrimitiveShell c = g_primitive_shells[primitive_shell_index_c];
    PrimitiveShell d = g_primitive_shells[primitive_shell_index_d];

    //const bool is_bra_asymmetric = (primitive_shell_index_a != primitive_shell_index_b);
    const bool is_ket_asymmetric = (primitive_shell_index_c != primitive_shell_index_d);
    //const bool is_braket_asymmetric = (ab != cd);

    sppp2fock(
        a, b, c, d, is_ket_asymmetric,
        g_fock_matrix + num_basis * num_basis *
                            (item_ct1.get_local_id(2) % num_fock_replicas),
        num_basis, g_boys_grid, g_density_matrix, g_cgto_normalization_factors);
}


//*
/*
DPCT1110:11: The total declared local variable size in device function
pppp2e_direct exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void pppp2e_direct(real_t *g_fock_matrix,
                   const PrimitiveShell *g_primitive_shells,
                   const sycl::int2 *g_primitive_shell_pair_indices,
                   const real_t *g_cgto_normalization_factors,
                   const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                   const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
                   const size_t num_braket,
                   const real_t schwarz_screening_threshold,
                   const real_t *g_schwarz_upper_bound_factors,
                   const int num_basis, const real_t *g_boys_grid,
                   const real_t *g_density_matrix, const size_t head_bra,
                   const size_t head_ket, const int num_fock_replicas)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx_linear = item_ct1.get_global_linear_id();
    if (idx_linear >= num_braket) {
        return;
    }

    /*
    DPCT1013:16: The rounding mode could not be specified and the generated code
    may have different accuracy than the original code. Verify the correctness.
    SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
    standard.
    */
    int cd =
        sycl::vec<double, 1>{
            ((sycl::sqrt(sycl::vec<long long, 1>{static_cast<long long>(8 * idx_linear + 1)}
                             .convert<double, sycl::rounding_mode::rte>()[0]) -
              1) /
             2)}
            .convert<int, sycl::rounding_mode::rtn>()[0];
    const int ab = head_bra + idx_linear - static_cast<size_t>(cd) * (cd + 1) / 2;
    cd += head_ket;

    if (g_schwarz_upper_bound_factors[ab] * g_schwarz_upper_bound_factors[cd] < schwarz_screening_threshold) {
        return;
    }

    const int primitive_shell_index_a =
        g_primitive_shell_pair_indices[ab].x() + shell_s0.start_index;
    const int primitive_shell_index_b =
        g_primitive_shell_pair_indices[ab].y() + shell_s1.start_index;
    const int primitive_shell_index_c =
        g_primitive_shell_pair_indices[cd].x() + shell_s2.start_index;
    const int primitive_shell_index_d =
        g_primitive_shell_pair_indices[cd].y() + shell_s3.start_index;

    PrimitiveShell a = g_primitive_shells[primitive_shell_index_a];
    PrimitiveShell b = g_primitive_shells[primitive_shell_index_b];
    PrimitiveShell c = g_primitive_shells[primitive_shell_index_c];
    PrimitiveShell d = g_primitive_shells[primitive_shell_index_d];

    const bool is_bra_asymmetric = (primitive_shell_index_a != primitive_shell_index_b);
    const bool is_ket_asymmetric = (primitive_shell_index_c != primitive_shell_index_d);
    const bool is_braket_asymmetric = (ab != cd);

    pppp2fock(
        a, b, c, d, is_bra_asymmetric, is_ket_asymmetric, is_braket_asymmetric,
        g_fock_matrix + num_basis * num_basis *
                            (item_ct1.get_local_id(2) % num_fock_replicas),
        num_basis, g_boys_grid, g_density_matrix, g_cgto_normalization_factors);
}
/**/









inline 
void add2fock_general(double val, double* g_fock, 
                      int mu, int nu, int la, int si, int num_basis, const double* g_dens) 
{
    if (mu > nu) {
        swap_indices(mu, nu);
    }
    if (la > si) {
        swap_indices(la, si);
    }
    if (mu > la || (mu == la && nu > si)) {
        swap_indices(mu, la);
        swap_indices(nu, si);
    }

    bool is_sym_bra = false;
    bool is_sym_ket = false;
    bool is_sym_braket = false;
    if (mu == nu) is_sym_bra = true;
    if (la == si) is_sym_ket = true;
    if (mu == la && nu == si) is_sym_braket = true;



   if (is_sym_bra && is_sym_ket && is_sym_braket) {
        // printf("1.\n");
        atomic_add(g_fock, num_basis * mu + nu, 0.5 * g_dens[num_basis * la + si] * val);
    }
    else if (is_sym_bra && is_sym_ket) {
        // printf("2\n");
        atomic_add(g_fock, num_basis * mu + nu, 1.0 * g_dens[num_basis * la + si] * val);
        atomic_add(g_fock, num_basis * la + si, 1.0 * g_dens[num_basis * mu + nu] * val);
        atomic_add(g_fock, num_basis * mu + la, (-0.5) * g_dens[num_basis * nu + si] * val);
    }
    else if (is_sym_bra) {
        // printf("3.\n");
        atomic_add(g_fock, num_basis * mu + nu, 2.0 * g_dens[num_basis * la + si] * val);
        atomic_add(g_fock, num_basis * la + si, 1.0 * g_dens[num_basis * mu + nu] * val);
        atomic_add(g_fock, num_basis * mu + la, ((mu == la) ? -1.0 : -0.5) * g_dens[num_basis * nu + si] * val);
        atomic_add(g_fock, ((nu <= si) ? num_basis * nu + si : num_basis * si + nu), (-0.5) * g_dens[num_basis * mu + la] * val);
    }
    else if (is_sym_ket) {
        // printf("4.\n");
        atomic_add(g_fock, num_basis * mu + nu, 1.0 * g_dens[num_basis * la + si] * val);
        atomic_add(g_fock, num_basis * la + si, 2.0 * g_dens[num_basis * mu + nu] * val);
        atomic_add(g_fock, num_basis * mu + la, (-0.5) * g_dens[num_basis * nu + si] * val);
        atomic_add(g_fock, ((nu <= si) ? num_basis * nu + si : num_basis * si + nu), ((nu == si) ? -1.0 : -0.5) * g_dens[num_basis * mu + la] * val);
    }
    else if (is_sym_braket) {
        // printf("5.\n");
        atomic_add(g_fock, num_basis * mu + nu, 2.0 * g_dens[num_basis * la + si] * val);
        atomic_add(g_fock, ((mu <= la) ? num_basis * mu + la : num_basis * la + mu), ((mu == la) ? -0.5 : -0.25) * g_dens[num_basis * nu + si] * val);
        atomic_add(g_fock, ((nu <= si) ? num_basis * nu + si : num_basis * si + nu), ((nu == si) ? -0.5 : -0.25) * g_dens[num_basis * mu + la] * val);
        atomic_add(g_fock, ((mu <= si) ? num_basis * mu + si : num_basis * si + mu), ((mu == si) ? -0.5 : -0.25) * g_dens[num_basis * nu + la] * val);
        atomic_add(g_fock, ((nu <= la) ? num_basis * nu + la : num_basis * la + nu), ((nu == la) ? -0.5 : -0.25) * g_dens[num_basis * mu + si] * val);
    }
    else {
        // printf("6.\n");
        atomic_add(g_fock, num_basis * mu + nu, 2.0 * g_dens[num_basis * la + si] * val);
        atomic_add(g_fock, num_basis * la + si, 2.0 * g_dens[num_basis * mu + nu] * val);
        atomic_add(g_fock, ((mu <= la) ? num_basis * mu + la : num_basis * la + mu), ((mu == la) ? -1.0 : -0.5) * g_dens[num_basis * nu + si] * val);
        atomic_add(g_fock, ((nu <= si) ? num_basis * nu + si : num_basis * si + nu), ((nu == si) ? -1.0 : -0.5) * g_dens[num_basis * mu + la] * val);
        atomic_add(g_fock, ((mu <= si) ? num_basis * mu + si : num_basis * si + mu), ((mu == si) ? -1.0 : -0.5) * g_dens[num_basis * nu + la] * val);
        atomic_add(g_fock, ((nu <= la) ? num_basis * nu + la : num_basis * la + nu), ((nu == la) ? -1.0 : -0.5) * g_dens[num_basis * mu + si] * val);
    }
}

/*
DPCT1110:12: The total declared local variable size in device function
MD_direct_SCF_1T1SP exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
SYCL_EXTERNAL
void launch_MD_direct_SCF_1T1SP(const sycl::nd_item<1>& item,  
    real_t *g_fock, const real_t *g_dens, const PrimitiveShell *g_shell,
    const int num_fock_replicas, const real_t *g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads, const real_t swartz_screening_threshold,
    const real_t *g_upper_bound_factors,
    const sycl::int2 *d_primitive_shell_pair_indices, const int num_basis,
    const real_t *g_boys_grid, const size_t head_bra, const size_t head_ket)
//    dpct::accessor<int, dpct::constant, 3> loop_to_ang,
//    dpct::accessor<int, dpct::constant, 2> tuv_list)
{
    // 通し番号indexの計算
//    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t id = item.get_global_linear_id();

//    const size_t id =
//        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
//        item_ct1.get_local_id(2);

    if (id >= num_threads) return;

    const double size_Rmid=1377;

    //使い捨ての中間体R_mid
    double R_mid[3*1377];

    //解を格納する配列R
    double R[2925];

    //thread内で結果を保持するメモリ
    double thread_val=0.0;
    
    // Compute 4D index from thread id
    int ket_size;
    if(shell_s2.start_index == shell_s3.start_index){
        ket_size = (shell_s2.count * (shell_s2.count+1)) / 2;
    }else{
        ket_size = shell_s2.count*shell_s3.count;
    }

    // const size_t2 abcd = index1to2(id, false, ket_size);
    const size_t2 abcd = index1to2(id, (shell_s0.start_index == shell_s2.start_index && shell_s1.start_index == shell_s3.start_index), ket_size);
    const sycl::int2 ab = d_primitive_shell_pair_indices[head_bra + abcd.x];
    const sycl::int2 cd = d_primitive_shell_pair_indices[head_ket + abcd.y];

    // Task-wise Schwarz screening
    if (g_upper_bound_factors[head_bra + abcd.x] * g_upper_bound_factors[head_ket + abcd.y] < swartz_screening_threshold) {
        return;
    }

    // Obtain primitive shells [ab|cd]
    const size_t primitive_index_a = ab.x() + shell_s0.start_index;
    const size_t primitive_index_b = ab.y() + shell_s1.start_index;
    const size_t primitive_index_c = cd.x() + shell_s2.start_index;
    const size_t primitive_index_d = cd.y() + shell_s3.start_index;

    const PrimitiveShell a = g_shell[primitive_index_a];
    const PrimitiveShell b = g_shell[primitive_index_b];
    const PrimitiveShell c = g_shell[primitive_index_c];
    const PrimitiveShell d = g_shell[primitive_index_d];
        
    // Obtain basis index (ij|kl)
    const size_t size_a = a.basis_index;
    const size_t size_b = b.basis_index;
    const size_t size_c = c.basis_index;
    const size_t size_d = d.basis_index;

    bool is_bra_symmetric = (primitive_index_a == primitive_index_b);
    bool is_ket_symmetric = (primitive_index_c == primitive_index_d);
    bool is_braket_symmetric = utm_id(primitive_index_a,primitive_index_b) == utm_id(primitive_index_c,primitive_index_d);
    
    //使用データを取得，レジスタに書き込み

    //指数部
    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double gamma = c.exponent;
    const double delta = d.exponent;
    const double p = alpha+beta;
    const double q = gamma+delta;
    const double xi = p*q / (p+q);

    //係数部
    const double coef_a = a.coefficient;
    const double coef_b = b.coefficient;
    const double coef_c = c.coefficient;
    const double coef_d = d.coefficient;

    //座標
    const double pos_A[3] = {a.coordinate.x, a.coordinate.y, a.coordinate.z};
    const double pos_B[3] = {b.coordinate.x, b.coordinate.y, b.coordinate.z};
    const double pos_C[3] = {c.coordinate.x, c.coordinate.y, c.coordinate.z};
    const double pos_D[3] = {d.coordinate.x, d.coordinate.y, d.coordinate.z};

    const double pos_P[3] = {(alpha*pos_A[0]+beta*pos_B[0])/(alpha+beta), (alpha*pos_A[1]+beta*pos_B[1])/(alpha+beta), (alpha*pos_A[2]+beta*pos_B[2])/(alpha+beta)};
    const double pos_Q[3] = {(gamma*pos_C[0]+delta*pos_D[0])/(gamma+delta), (gamma*pos_C[1]+delta*pos_D[1])/(gamma+delta), (gamma*pos_C[2]+delta*pos_D[2])/(gamma+delta)};

    //角運動量の総和
    const int orbital_A = a.shell_type;
    const int orbital_B = b.shell_type;
    const int orbital_C = c.shell_type;
    const int orbital_D = d.shell_type;

    //軌道間距離の二乗
    const double dist = ((pos_P[0]-pos_Q[0])*(pos_P[0]-pos_Q[0]) + (pos_P[1]-pos_Q[1])*(pos_P[1]-pos_Q[1]) + (pos_P[2]-pos_Q[2])*(pos_P[2]-pos_Q[2]));


    const int K=orbital_A + orbital_B + orbital_C + orbital_D;
    
    double Boys[25];
    getIncrementalBoys(K, xi*dist, g_boys_grid, Boys);

    //Boys関数の値を計算(Single)
    for(int i=0; i <= K; i++){
        Boys[i] *= (right2left_binary_woif((-2*xi), i));
    }

    //各ERIを計算
    //事前計算⇒実際のERI計算の順に実行
    //p軌道の場合lmn_aが0:px, 1:py, 2:pz軌道のように対応付け
    //d以上はconstant配列のloop_to_angを参照

    
    
    double Norm_A, Norm_B, Norm_C, Norm_D;
    double Norm;

    int t,u,v,tau,nu,phi;
    int t_max;
    int u_max;
    int v_max;
    int tau_max;
    int nu_max;
    int phi_max;

    int tid=0;
    
    int iter_max;


    // 方位量子数l,m,nの値をループ変数から導出
    for(int lmn_a=0; lmn_a<comb_max(orbital_A); lmn_a++){
        int l1=loop_to_ang[orbital_A][lmn_a][0]; int m1=loop_to_ang[orbital_A][lmn_a][1]; int n1=loop_to_ang[orbital_A][lmn_a][2];
        Norm_A = calcNorm(alpha, l1, m1, n1);

        for(int lmn_b=0; lmn_b<comb_max(orbital_B); lmn_b++){                  
            int l2=loop_to_ang[orbital_B][lmn_b][0]; int m2=loop_to_ang[orbital_B][lmn_b][1]; int n2=loop_to_ang[orbital_B][lmn_b][2];
            Norm_B = calcNorm(beta, l2, m2, n2);

            for(int lmn_c=0; lmn_c<comb_max(orbital_C); lmn_c++){
                int l3=loop_to_ang[orbital_C][lmn_c][0]; int m3=loop_to_ang[orbital_C][lmn_c][1]; int n3=loop_to_ang[orbital_C][lmn_c][2];
                Norm_C = calcNorm(gamma, l3, m3, n3);

                for(int lmn_d=0; lmn_d<comb_max(orbital_D); lmn_d++){
                    int l4=loop_to_ang[orbital_D][lmn_d][0]; int m4=loop_to_ang[orbital_D][lmn_d][1]; int n4=loop_to_ang[orbital_D][lmn_d][2];
                    Norm_D = calcNorm(delta, l4, m4, n4);



                    if(size_c==size_d && lmn_c > lmn_d) continue; // sspp, sppp,...
                    if(size_a==size_b && lmn_a > lmn_b) continue; // pppd,...


                    Norm = Norm_A * Norm_B * Norm_C * Norm_D;
                    // 前回のループの計算結果をクリア
                    thread_val=0.0;
                    // 事前計算部
                    //初期値：Boysとして計算済
                    //Step 0: Boys関数評価
                    R[0]=Boys[0];
                    for(int i=0; i <= K; i++){
                        R_mid[i]=Boys[i];
                    }
                    
                    // ループ変数の設定
                    t_max = l1+l2+1;
                    u_max = m1+m2+1;
                    v_max = n1+n2+1;
                    tau_max = l3+l4+1;
                    nu_max = m3+m4+1;
                    phi_max = n3+n4+1;

                    for(int k=1; k <= K; k++){//Step 1~Kの計算
                        // t+u+v=kとなる全ペアに対して適切な計算
                        // 0~K-kまでそれぞれ必要⇒ループでやる
        
        
                        for(int z=0; z<=(K+1)*comb_max(k); z++){
                        
                            int i = z/comb_max(k);
        
                            if(i <= K-k){
                                t=tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][0];
                                u=tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][1];
                                v=tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][2];
        
                                if((t <= (t_max+tau_max-2)) && (u <= (u_max+nu_max-2)) && (v <= (v_max+phi_max-2))){
                                    if(t >= 1){
                                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[0] - pos_Q[0])*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_Rmid)];
                                    }
                                    else if(u >= 1){
                                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[1] - pos_Q[1])*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_Rmid)];
                                    }
                                    else{
                                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[2] - pos_Q[2])*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_Rmid)];
                                    }
                                }
                            }
                        }//step kの全計算が終了
        

                        //必要な結果を配列Rに書き込み
                        for(int i=0; i<=comb_max(k); i++){
                            R[static_cast<int>(k*(k+1)*(k+2)/6) + i] = R_mid[(k%3)*static_cast<int>(size_Rmid) + i];
                        }

                    }
                    //事前計算完了
                    

                    // ERI計算部
                    iter_max=t_max*u_max*v_max*tau_max*nu_max*phi_max + 1;
                    for(int i=0; i<iter_max; i++){
                        // MD法6重ループを管理する6変数を各Threadに割り当て
                        tid=i;
                        phi = tid % phi_max;
                        tid /= phi_max;
                        nu = tid % nu_max;
                        tid /= nu_max;
                        tau = tid % tau_max;
                        tid /= tau_max;
                        v = tid % v_max;
                        tid /= v_max;
                        u = tid % u_max;
                        tid /= u_max;
                        t=tid;


                        double my_val = 0.0;

                        ////特定の(t,u,v,tau,nu,phi)に対応する結果をmy_valとして持つ
                        if(t <= t_max-1 && u<=u_max-1 && v<=v_max-1 && tau<=tau_max-1 && nu<=nu_max-1 && phi<=phi_max-1){
                            int k=t+u+v+tau+nu+phi;
                            my_val = MD_Et_NonRecursion(l1, l2, t, alpha, beta, (pos_A[0]-pos_B[0])) * MD_Et_NonRecursion(m1, m2, u, alpha, beta, (pos_A[1]-pos_B[1])) * MD_Et_NonRecursion(n1, n2, v, alpha, beta, (pos_A[2]-pos_B[2])) * MD_Et_NonRecursion(l3, l4, tau, gamma, delta, (pos_C[0]-pos_D[0])) * MD_Et_NonRecursion(m3, m4, nu, gamma, delta, (pos_C[1]-pos_D[1])) * MD_Et_NonRecursion(n3, n4, phi, gamma, delta, (pos_C[2]-pos_D[2])) * (1 - 2*((tau+nu+phi)&1)) * R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k,u+nu,v+phi,0,0,0)];
                            // thread_valに足しこんでMD法の結果を得る
                            thread_val += my_val * 2 * M_PI_2_5 /
                                          (p * q * sycl::sqrt((p + q))) *
                                          coef_a * coef_b * coef_c * coef_d;
                        } 
                    }

                    thread_val *= Norm  * g_cgto_normalization_factors[size_a + lmn_a]
                                        * g_cgto_normalization_factors[size_b + lmn_b]
                                        * g_cgto_normalization_factors[size_c + lmn_c]
                                        * g_cgto_normalization_factors[size_d + lmn_d];


                    if(!is_bra_symmetric && size_a == size_b) thread_val *= 2.0;
                    if(!is_ket_symmetric && size_c == size_d) thread_val *= 2.0;

                    if(utm_id(size_a,size_b) == utm_id(size_c,size_d)) {
                        if(!is_braket_symmetric) thread_val *= 2.0;
                        if(twoDim2oneDim(size_a+lmn_a,size_b+lmn_b,num_basis) != twoDim2oneDim(size_c+lmn_c,size_d+lmn_d,num_basis)) thread_val *= 0.5;
                    }



                    // Global Memoryへ書き込み
                    // 汎用カーネルでは全要素判定(case1)
                    add2fock_general(thread_val,
                                     // g_fock,
                                     g_fock + num_basis * num_basis * (item.get_local_id() % num_fock_replicas),
                                     size_a + lmn_a, size_b + lmn_b,
                                     size_c + lmn_c, size_d + lmn_d, num_basis,
                                     g_dens);
                }
            }
        }
    }
    return;
}

/*
// This is a new trial --- maybe later again

template <typename KernelFunc>
void launch_kernel(
    sycl::queue& workq,
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    int &s_ket_group_idx, bool &s_significant_flag)
{
    workq.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) {
                KernelFunc(
                    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
                    const sycl::int2 *g_primitive_shell_pair_indices,
                    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
                    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
                    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
                    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
                    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
                    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
                    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
                    int &s_ket_group_idx, bool &s_significant_flag)
                    item
                );
            });
    });
}

void launch_eri_kernel_dynamic(
     sycl::queue& workq,
     int s0,
     int s1,
     int s2,
     int s3,
     real_t* g_fock_matrix_replicas, 
     const PrimitiveShell* g_primitive_shells, 
     const sycl::int2* g_primitive_shell_pair_indices, 
     const real_t* g_cgto_normalization_factors, 
     const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
     const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, 
     const real_t schwarz_screening_threshold, 
     const real_t* g_schwarz_upper_bound_factors, 
     const int num_basis, 
     const real_t* g_boys_grid, 
     const real_t* g_density_matrix, 
     int* g_counter, int* g_min_skipped_column,
     const size_t head_bra, const size_t head_ket, 
     const size_t num_bra, const size_t num_ket, 
     const int num_fock_replicas,
     const int num_cuda_blocks,
     const int num_threads_per_block)
{
auto kind = select_eri_kernel(s0, s1, s2, s3);

switch (kind) {
case eri_kernel_kind::ssss:
    launch_kernel<ssss2e_dynamic>(workq, 
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
    break;
case eri_kernel_kind::sssp:
    launch_kernel<sssp2e_dynamic>(workq, 
                (g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
    break;
case eri_kernel_kind::sspp:
    launch_kernel<sspp2e_dynamic>(workq,
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
    break;
case eri_kernel_kind::spsp:
    launch_kernel<spsp2e_dynamic>(workq,
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
    break;
case eri_kernel_kind::sppp:
    launch_kernel<sppp2e_dynamic>(workq,
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
    break;
case eri_kernel_kind::pppp:
    launch_kernel<pppp2e_dynamic>(workq,
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
    break;
}

}
*/

SYCL_EXTERNAL
void launch_eri_kernel_dynamic(
     const sycl::nd_item<1>& item,
     int a,
     int b,
     int c,
     int d,
     real_t* g_fock_matrix_replicas, 
     const PrimitiveShell* g_primitive_shells, 
     const sycl::int2* g_primitive_shell_pair_indices, 
     const real_t* g_cgto_normalization_factors, 
     const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
     const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, 
     const real_t schwarz_screening_threshold, 
     const real_t* g_schwarz_upper_bound_factors, 
     const int num_basis, 
     const real_t* g_boys_grid, 
     const real_t* g_density_matrix, 
     int* g_counter, int* g_min_skipped_column,
     const size_t head_bra, const size_t head_ket, 
     const size_t num_bra, const size_t num_ket, 
     const int num_fock_replicas,
     const sycl::local_accessor<int, 1>& s_ket_group_idx,
     const sycl::local_accessor<bool, 1>& s_significant_flag)
{
    if (a > b) std::swap(a, b);
    if (c > d) std::swap(c, d);
    if (a > c || (a == c && b > d)) {
        std::swap(a, c);
        std::swap(b, d);
    }

    if (a == 0 && b == 0 && c == 0 && d == 0) ssss2e_dynamic(item,
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas,
                 s_ket_group_idx, s_significant_flag);
    else if (a == 0 && b == 0 && c == 0 && d == 1) sssp2e_dynamic(item,
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas,
                 s_ket_group_idx, s_significant_flag);
    else if (a == 0 && b == 0 && c == 1 && d == 1) sspp2e_dynamic(item, 
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas,
                 s_ket_group_idx, s_significant_flag);
    else if (a == 0 && b == 1 && c == 0 && d == 1) spsp2e_dynamic(item, 
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas,
                 s_ket_group_idx, s_significant_flag);
    else if (a == 0 && b == 1 && c == 1 && d == 1) sppp2e_dynamic(item, 
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas,
                 s_ket_group_idx, s_significant_flag);
    else if (a == 1 && b == 1 && c == 1 && d == 1) pppp2e_dynamic(item, 
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas,
                 s_ket_group_idx, s_significant_flag);
}



} // namespace gansu::gpu
