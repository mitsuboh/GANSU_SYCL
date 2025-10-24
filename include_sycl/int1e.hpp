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



#ifndef INT1E_CUH
#define INT1E_CUH

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "types.hpp"
#include "utils_cuda.hpp"
#include "boys.hpp"

namespace gansu::gpu{

// 2点間の距離を求める関数（2乗済み）
inline double calc_dist_GPU(const Coordinate &coord1,
                                   const Coordinate &coord2) {
    return (coord1.x-coord2.x)*(coord1.x-coord2.x) + (coord1.y-coord2.y)*(coord1.y-coord2.y) + (coord1.z-coord2.z)*(coord1.z-coord2.z);
}
inline double calc_dist_GPU(const sycl::double3 &coord1,
                                   const Coordinate &coord2) {
    return (coord1.x() - coord2.x) * (coord1.x() - coord2.x) +
           (coord1.y() - coord2.y) * (coord1.y() - coord2.y) +
           (coord1.z() - coord2.z) * (coord1.z() - coord2.z);
}

inline double calc_Norms(double alpha, double beta, int ijk, int lmn) {
    return dpct::pow(2.0, ijk + lmn) * dpct::pow(2.0 / M_PI, 1.5) *
           dpct::pow(alpha, (2.0 * ijk + 3.0) / 4.0) *
           dpct::pow(beta, (2.0 * lmn + 3.0) / 4.0);
}

inline int calc_result_index(int y, int x, int sumCGTO) {
    return (y<=x) ? y*sumCGTO + x : x*sumCGTO + y;
}

inline void Matrix_Symmetrization(double *matrix, int n,
                                         sycl::local_accessor<real_t, 2> sh_mem) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    if (item_ct1.get_group(1) > item_ct1.get_group(2)) return;

    int src_block = item_ct1.get_group(1) * 32 * n + item_ct1.get_group(2) * 32;
    int dst_block = item_ct1.get_group(2) * 32 * n + item_ct1.get_group(1) * 32;

    if (item_ct1.get_group(2) * 32 + item_ct1.get_local_id(2) < n ||
        item_ct1.get_group(1) * 32 + item_ct1.get_local_id(1) < n) {
        sh_mem[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] =
            matrix[src_block + item_ct1.get_local_id(1) * n +
                   item_ct1.get_local_id(2)];
    }
    /*
    DPCT1065:129: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (item_ct1.get_group(1) == item_ct1.get_group(2) &&
            item_ct1.get_local_id(1) <= item_ct1.get_local_id(2) ||
        (dst_block + item_ct1.get_local_id(1) * n + item_ct1.get_local_id(2) >=
         n * n)) return;

    matrix[dst_block + item_ct1.get_local_id(1) * n +
           item_ct1.get_local_id(2)] =
        sh_mem[item_ct1.get_local_id(2)][item_ct1.get_local_id(1)];
}


// MD method(overlap and kinetic integral)
#include "md_kernel.txt"

// OS method(overlap and kinetic integral)
#include "os_kernel.txt"

// 2025-05-26 define a function to target matrices

SYCL_EXTERNAL
void compute_kinetic_energy_integral(
    real_t *g_overlap, real_t *g_kinetic, const PrimitiveShell *g_shell,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const size_t num_threads,
    const int num_basis);

SYCL_EXTERNAL
void compute_nuclear_attraction_integral(
    real_t *g_nucattr, const PrimitiveShell *g_shell,
    const real_t *g_cgto_normalization_factors, const Atom *g_atom,
    const int num_atoms, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis,
    const real_t *g_boys_grid);


inline void launch_overlap_kinetic_kernel( int a, int b, real_t* g_overlap, real_t* g_kinetic,
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                        const size_t num_threads,
                        const int num_basis)
{
    int flag=0;

    if(flag){
        if (a == 0 && b == 0) overlap_kinetic_MDss(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 0 && b == 1) overlap_kinetic_MDsp(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 1 && b == 1) overlap_kinetic_MDpp(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 0 && b == 2) overlap_kinetic_MDsd(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 1 && b == 2) overlap_kinetic_MDpd(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 2 && b == 2) overlap_kinetic_MDdd(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 0 && b == 3) overlap_kinetic_MDsf(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 1 && b == 3) overlap_kinetic_MDpf(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 2 && b == 3) overlap_kinetic_MDdf(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 3 && b == 3) overlap_kinetic_MDff(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else compute_kinetic_energy_integral(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        // else THROW_EXCEPTION("Only up to f-orbitals are supported in MD method for overlap and kinetic integrals");
    }
    else{
        if (a == 0 && b == 0) overlap_kinetic_OSss(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 0 && b == 1) overlap_kinetic_OSsp(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 1 && b == 1) overlap_kinetic_OSpp(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 0 && b == 2) overlap_kinetic_OSsd(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 1 && b == 2) overlap_kinetic_OSpd(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 2 && b == 2) overlap_kinetic_OSdd(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 0 && b == 3) overlap_kinetic_OSsf(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 1 && b == 3) overlap_kinetic_OSpf(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 2 && b == 3) overlap_kinetic_OSdf(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 3 && b == 3) overlap_kinetic_OSff(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else compute_kinetic_energy_integral(g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        // else THROW_EXCEPTION("Only up to f-orbitals are supported in OS method for overlap and kinetic integrals");
    }
}


inline void launch_nuclear_attraction_kernel(int a, int b, real_t* g_nucattr,
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,
                        const Atom* g_atom, const int num_atoms,
                        const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                        const size_t num_threads,
                        const int num_basis, const real_t* g_boys_grid)
{
    int flag=0;

    if(flag){
        if (a == 0 && b == 0) nuclear_attraction_MDss(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 0 && b == 1) nuclear_attraction_MDsp(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 1 && b == 1) nuclear_attraction_MDpp(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 0 && b == 2) nuclear_attraction_MDsd(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 1 && b == 2) nuclear_attraction_MDpd(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 2 && b == 2) nuclear_attraction_MDdd(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 0 && b == 3) nuclear_attraction_MDsf(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 1 && b == 3) nuclear_attraction_MDpf(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 2 && b == 3) nuclear_attraction_MDdf(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 3 && b == 3) nuclear_attraction_MDff(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else compute_nuclear_attraction_integral(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        // else THROW_EXCEPTION("Only up to f-orbitals are supported in MD method for nuclear attraction integrals");
    }
    else{
        if (a == 0 && b == 0) nuclear_attraction_OSss(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 0 && b == 1) nuclear_attraction_OSsp(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 1 && b == 1) nuclear_attraction_OSpp(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 0 && b == 2) nuclear_attraction_OSsd(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 1 && b == 2) nuclear_attraction_OSpd(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 2 && b == 2) nuclear_attraction_OSdd(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 0 && b == 3) nuclear_attraction_OSsf(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 1 && b == 3) nuclear_attraction_OSpf(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 2 && b == 3) nuclear_attraction_OSdf(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 3 && b == 3) nuclear_attraction_OSff(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else compute_nuclear_attraction_integral(g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        // else THROW_EXCEPTION("Only up to f-orbitals are supported in OS method for nuclear attraction integrals");
    }

}




} // namespace gansu::gpu

#endif
