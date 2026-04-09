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

#ifndef INT2E_DIRECT_CUH
#define INT2E_DIRECT_CUH

#include <sycl/sycl.hpp>
//#include <dpct/dpct.hpp>
#include "boys.hpp"
#include "types.hpp"
#include "utils_cuda.hpp"

namespace gansu::gpu{


//SYCL_EXTERNAL void composeFockMatrix(sycl::nd_item<1> item,
//    real_t* g_fock_matrix, real_t* g_fock_matrix_replicas, const real_t* g_int1e, const int num_basis, const int num_fock_replicas);
SYCL_EXTERNAL void composeFockMatrix(sycl::nd_item<1> item,
    real_t* g_fock_matrix, real_t* g_fock_matrix_replicas, const real_t* g_int1e, const int num_basis, const int num_fock_replicas, bool is_first_call);

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
                   const size_t head_ket, const int num_fock_replicas);


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
                   const size_t head_ket, const int num_fock_replicas);


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
                   const size_t head_ket, const int num_fock_replicas);


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
                   const size_t head_ket, const int num_fock_replicas);


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
                   const size_t head_ket, const int num_fock_replicas);


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
                   const size_t head_ket, const int num_fock_replicas);


SYCL_EXTERNAL void ssss2e_dynamic(const sycl::nd_item<1>& item, 
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
//    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    const int num_primitive_shells, const real_t* g_boys_grid, const real_t* g_density_matrix,
    const real_t* g_density_matrix_diff_shell, int* g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    const sycl::local_accessor<int, 1>& s_ket_group_idx,
    const sycl::local_accessor<bool, 1>& s_significant_flag,
    const sycl::local_accessor<real_t, 1>& s_schwarz_upper_bound);


SYCL_EXTERNAL void sssp2e_dynamic(const sycl::nd_item<1>& item, 
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
//    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    const int num_primitive_shells, const real_t* g_boys_grid, const real_t* g_density_matrix,
    const real_t* g_density_matrix_diff_shell, int* g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    const sycl::local_accessor<int, 1>& s_ket_group_idx,
    const sycl::local_accessor<bool, 1>& s_significant_flag,
    const sycl::local_accessor<real_t, 1>& s_schwarz_upper_bound);


SYCL_EXTERNAL void sspp2e_dynamic(const sycl::nd_item<1>& item, 
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
//    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    const int num_primitive_shells, const real_t* g_boys_grid, const real_t* g_density_matrix,
    const real_t* g_density_matrix_diff_shell, int* g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    const sycl::local_accessor<int, 1>& s_ket_group_idx,
    const sycl::local_accessor<bool, 1>& s_significant_flag,
    const sycl::local_accessor<real_t, 1>& s_schwarz_upper_bound);


SYCL_EXTERNAL void spsp2e_dynamic(const sycl::nd_item<1>& item, 
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
//    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    const int num_primitive_shells, const real_t* g_boys_grid, const real_t* g_density_matrix,
    const real_t* g_density_matrix_diff_shell, int* g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    const sycl::local_accessor<int, 1>& s_ket_group_idx,
    const sycl::local_accessor<bool, 1>& s_significant_flag,
    const sycl::local_accessor<real_t, 1>& s_schwarz_upper_bound);


SYCL_EXTERNAL void sppp2e_dynamic(const sycl::nd_item<1>& item, 
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
//    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    const int num_primitive_shells, const real_t* g_boys_grid, const real_t* g_density_matrix,
    const real_t* g_density_matrix_diff_shell, int* g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    const sycl::local_accessor<int, 1>& s_ket_group_idx,
    const sycl::local_accessor<bool, 1>& s_significant_flag,
    const sycl::local_accessor<real_t, 1>& s_schwarz_upper_bound);


SYCL_EXTERNAL void pppp2e_dynamic(const sycl::nd_item<1>& item, 
    real_t *g_fock_matrix_replicas, const PrimitiveShell *g_primitive_shells,
    const sycl::int2 *g_primitive_shell_pair_indices,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2,
    const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold,
    const real_t *g_schwarz_upper_bound_factors, const int num_basis,
//    const real_t *g_boys_grid, const real_t *g_density_matrix, int *g_counter,
    const int num_primitive_shells, const real_t* g_boys_grid, const real_t* g_density_matrix,
    const real_t* g_density_matrix_diff_shell, int* g_counter,
    int *g_min_skipped_column, const size_t head_bra, const size_t head_ket,
    const size_t num_bra, const size_t num_ket, const int num_fock_replicas,
    const sycl::local_accessor<int, 1>& s_ket_group_idx,
    const sycl::local_accessor<bool, 1>& s_significant_flag,
    const sycl::local_accessor<real_t, 1>& s_schwarz_upper_bound);


SYCL_EXTERNAL void launch_MD_direct_SCF_1T1SP(
    const sycl::nd_item<1>& item,
    real_t *g_fock, const real_t *g_dens, const PrimitiveShell *g_shell,
    const int num_fock_replicas, const real_t *g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads, const real_t swartz_screening_threshold,
    const real_t *g_upper_bound_factors,
    const sycl::int2 *d_primitive_shell_pair_indices, const int num_basis,
    const real_t *g_boys_grid, const size_t head_bra, const size_t head_ket);

using eri_kernel_direct_t = void (*)(real_t *, const PrimitiveShell *,
                                     const sycl::int2 *, const real_t *,
                                     const ShellTypeInfo, const ShellTypeInfo,
                                     const ShellTypeInfo, const ShellTypeInfo,
                                     const size_t, const real_t, const real_t *,
                                     const int, const real_t *, const real_t *,
                                     const size_t, const size_t, const int);
using eri_kernel_dynamic_t = void (*)(real_t *, const PrimitiveShell *,
                                      const sycl::int2 *, const real_t *,
                                      const ShellTypeInfo, const ShellTypeInfo,
                                      const ShellTypeInfo, const ShellTypeInfo,
                                      const real_t, const real_t *, const int,
//                                      const real_t *, const real_t *, int *,
const int, const real_t*, const real_t*,
const real_t*, int*,
                                      int *, const size_t, const size_t,
                                      const size_t, const size_t, const int);


enum class eri_kernel_kind {
    ssss,
    sssp,
    sspp,
    spsp,
    sppp,
    pppp
};

inline eri_kernel_kind select_eri_kernel(int a, int b, int c, int d) {
    if (a > b) std::swap(a, b);
    if (c > d) std::swap(c, d);
    if (a > c || (a == c && b > d)) {
        std::swap(a, c);
        std::swap(b, d);
    }

    if      (a == 0 && b == 0 && c == 0 && d == 0) return eri_kernel_kind::ssss;
    else if (a == 0 && b == 0 && c == 0 && d == 1) return eri_kernel_kind::sssp;
    else if (a == 0 && b == 0 && c == 1 && d == 1) return eri_kernel_kind::sspp;
    else if (a == 0 && b == 1 && c == 0 && d == 1) return eri_kernel_kind::spsp;
    else if (a == 0 && b == 1 && c == 1 && d == 1) return eri_kernel_kind::sppp;
    else if (a == 1 && b == 1 && c == 1 && d == 1) return eri_kernel_kind::pppp;
    else throw std::runtime_error("Invalid shell type");
}

/*
void launch_get_eri_kernel_dynamic_sycl(
     sycl::queue& workq,
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
     const int num_cuda_blocks,
     const int num_threads_per_block)
{
    if (a > b) std::swap(a, b);
    if (c > d) std::swap(c, d);
    if (a > c || (a == c && b > d)) {
        std::swap(a, c);
        std::swap(b, d);
    }

    if (!((a==0 && b==0 && c==0 && d==0) || (a==0 && b==0 && c==0 && d==1) || (a==0 && b==0 && c==1 && d==1) ||
       (a==0 && b==1 && c==0 && d==1) || (a==0 && b==1 && c==1 && d==1) || (a==1 && b==1 && c==1 && d==1))) {
        std::cerr << "Invalid shell type: " << a << "," << b << "," << c << "," << d << "\n";
        std::exit(1);
    }

    workq.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(num_cuda_blocks * num_threads_per_block),
                sycl::range<1>(num_threads_per_block)
            ),
            [=](sycl::nd_item<1> item) {

                const size_t tid = item.get_global_linear_id();

    if (a == 0 && b == 0 && c == 0 && d == 0) ssss2e_dynamic 
                (g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
    else if (a == 0 && b == 0 && c == 0 && d == 1) sssp2e_dynamic
                (g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
    else if (a == 0 && b == 0 && c == 1 && d == 1) sspp2e_dynamic
                (g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
    else if (a == 0 && b == 1 && c == 0 && d == 1) spsp2e_dynamic
                (g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
    else if (a == 0 && b == 1 && c == 1 && d == 1) sppp2e_dynamic
                (g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
    else if (a == 1 && b == 1 && c == 1 && d == 1) pppp2e_dynamic
                (g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 g_boys_grid, g_density_matrix, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
            }
        );
    });

}
*/


inline void launch_eri_kernel_dynamic(
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
     const int num_primitive_shells,
     const real_t* g_boys_grid, 
     const real_t* g_density_matrix, 
     const real_t* g_density_matrix_diff_shell,
     int* g_counter, int* g_min_skipped_column,
     const size_t head_bra, const size_t head_ket, 
     const size_t num_bra, const size_t num_ket, 
     const int num_fock_replicas,
     const sycl::local_accessor<int, 1>& s_ket_group_idx,
     const sycl::local_accessor<bool, 1>& s_significant_flag,
     const sycl::local_accessor<real_t, 1>& s_schwarz_upper_bound)
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
                 num_primitive_shells, g_boys_grid, g_density_matrix,
                 g_density_matrix_diff_shell, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas,
                 s_ket_group_idx, s_significant_flag,s_schwarz_upper_bound);
    else if (a == 0 && b == 0 && c == 0 && d == 1) sssp2e_dynamic(item,
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 num_primitive_shells, g_boys_grid, g_density_matrix,
                 g_density_matrix_diff_shell, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas,
                 s_ket_group_idx, s_significant_flag,s_schwarz_upper_bound);
    else if (a == 0 && b == 0 && c == 1 && d == 1) sspp2e_dynamic(item, 
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 num_primitive_shells, g_boys_grid, g_density_matrix,
                 g_density_matrix_diff_shell, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas,
                 s_ket_group_idx, s_significant_flag,s_schwarz_upper_bound);
    else if (a == 0 && b == 1 && c == 0 && d == 1) spsp2e_dynamic(item, 
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 num_primitive_shells, g_boys_grid, g_density_matrix,
                 g_density_matrix_diff_shell, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas,
                 s_ket_group_idx, s_significant_flag,s_schwarz_upper_bound);
    else if (a == 0 && b == 1 && c == 1 && d == 1) sppp2e_dynamic(item, 
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 num_primitive_shells, g_boys_grid, g_density_matrix,
                 g_density_matrix_diff_shell, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas,
                 s_ket_group_idx, s_significant_flag,s_schwarz_upper_bound);
    else if (a == 1 && b == 1 && c == 1 && d == 1) pppp2e_dynamic(item, 
                 g_fock_matrix_replicas, g_primitive_shells, g_primitive_shell_pair_indices, 
                 g_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, g_schwarz_upper_bound_factors, num_basis, 
                 num_primitive_shells, g_boys_grid, g_density_matrix,
                 g_density_matrix_diff_shell, g_counter, g_min_skipped_column,
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas,
                 s_ket_group_idx, s_significant_flag,s_schwarz_upper_bound);

}




} // namespace gansu::gpu

#endif
