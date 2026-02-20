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



#ifndef INT1E_CUH
#define INT1E_CUH

#include <sycl/sycl.hpp>
#include "types.hpp"
#include "utils_cuda.hpp"
#include "boys.hpp"

namespace gansu::gpu{

struct double3 {
    double x, y, z;
};

inline double3 make_double3(double x, double y, double z) {
    return double3{x, y, z};
}

// something wrong how to define below so temporary macro ND_DIST_GPU
// 2点間の距離を求める関数（2乗済み）
#ifndef ND_DIST_GPU
inline double calc_dist_GPU(const Coordinate &coord1,
                                   const Coordinate &coord2) {
    return (coord1.x-coord2.x)*(coord1.x-coord2.x) + (coord1.y-coord2.y)*(coord1.y-coord2.y) + (coord1.z-coord2.z)*(coord1.z-coord2.z);
}
inline double calc_dist_GPU(const double3 &coord1,
                                   const Coordinate &coord2) {
    return (coord1.x - coord2.x) * (coord1.x - coord2.x) +
           (coord1.y - coord2.y) * (coord1.y - coord2.y) +
           (coord1.z - coord2.z) * (coord1.z - coord2.z);
}
#define ND_DIST_GPU
#endif

inline double calc_Norms(double alpha, double beta, int ijk, int lmn) {
    return sycl::pow(2.0, ijk + lmn) * sycl::pow(2.0 / M_PI, 1.5) *
           sycl::pow(alpha, (2.0 * ijk + 3.0) / 4.0) *
           sycl::pow(beta, (2.0 * lmn + 3.0) / 4.0);
}

inline int calc_result_index(int y, int x, int sumCGTO) {
    return (y<=x) ? y*sumCGTO + x : x*sumCGTO + y;
}

/*
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
*/
    /*
    DPCT1065:129: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
/*
    item_ct1.barrier();

    if (item_ct1.get_group(1) == item_ct1.get_group(2) &&
            item_ct1.get_local_id(1) <= item_ct1.get_local_id(2) ||
        (dst_block + item_ct1.get_local_id(1) * n + item_ct1.get_local_id(2) >=
         n * n)) return;

    matrix[dst_block + item_ct1.get_local_id(1) * n +
           item_ct1.get_local_id(2)] =
        sh_mem[item_ct1.get_local_id(2)][item_ct1.get_local_id(1)];
}
*/

inline void matrixSymmetrization(const sycl::id<1>& item, double* g_matrix, const int num_basis)
{
    size_t idx = item[0];
    size_t mu = idx / num_basis;
    size_t nu = idx % num_basis;

    if (mu < nu) {
        g_matrix[num_basis * nu + mu] = g_matrix[num_basis * mu + nu];
    }

}


// int1e_method for device
enum Int1eMethod {
    int1e_md = 0,
    int1e_os = 1,
    int1e_hybrid = 2,
};


// MD method(overlap and kinetic integral)
//#include "md_kernel.txt"

// OS method(overlap and kinetic integral)
//#include "os_kernel.txt"

// Definition of kernels for calculating one-electron integra
#include "../src_sycl/one_integral/d_s_int1e_kernel.txt"


// If the F or G orbit swas enabled when compiling, include the following.
#if INT1E_MAX_L >= 3
    // f kernels
    #include "../src_sycl/one_integral/f_s_int1e_kernel.txt"
#else
inline void overlap_kinetic_MDsf(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_MDpf(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_MDdf(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_MDff(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void nuclear_attraction_MDsf(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_MDpf(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_MDdf(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_MDff(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
    
inline void overlap_kinetic_OSsf(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_OSpf(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_OSdf(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_OSff(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void nuclear_attraction_OSsf(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_OSpf(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_OSdf(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_OSff(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
    
#endif

// If the G orbit was enabled when compiling, include the following.
#if INT1E_MAX_L >= 4
    // g kernels
    #include "../src_sycl/one_integral/g_s_int1e_kernel.txt"
#else
inline void overlap_kinetic_MDsg(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_MDpg(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_MDdg(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_MDfg(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_MDgg(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void nuclear_attraction_MDsg(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_MDpg(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_MDdg(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_MDfg(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_MDgg(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}

inline void overlap_kinetic_OSsg(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_OSpg(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_OSdg(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_OSfg(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void overlap_kinetic_OSgg(const sycl::nd_item<1>& item_ct1, real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis) {}
inline void nuclear_attraction_OSsg(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_OSpg(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_OSdg(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_OSfg(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}
inline void nuclear_attraction_OSgg(const sycl::nd_item<1>& item_ct1, real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid) {}

#endif




// 2025-05-26 define a function to target matrices

SYCL_EXTERNAL
void compute_kinetic_energy_integral(
    const sycl::nd_item<1>& item_ct1,
    real_t *g_overlap, real_t *g_kinetic, const PrimitiveShell *g_shell,
    const real_t *g_cgto_normalization_factors, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const size_t num_threads,
    const int num_basis);

SYCL_EXTERNAL
void compute_nuclear_attraction_integral(
    const sycl::nd_item<1>& item_ct1,
    real_t *g_nucattr, const PrimitiveShell *g_shell,
    const real_t *g_cgto_normalization_factors, const Atom *g_atom,
    const int num_atoms, const ShellTypeInfo shell_s0,
    const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis,
    const real_t *g_boys_grid);


inline void launch_overlap_kinetic_kernel(const sycl::nd_item<1>& item_ct1, int a, int b, const Int1eMethod int1e_method,
    real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                        const size_t num_threads,
                        const int num_basis)
{
    const int idx = b*(b+1)/2 + a;

    switch(idx) {
        case 0: // ss
            if(int1e_method == int1e_md) overlap_kinetic_MDss(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSss(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_OSss(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

        case 1: // sp
            if(int1e_method == int1e_md) overlap_kinetic_MDsp(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSsp(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_OSsp(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

        case 2: // pp
            if(int1e_method == int1e_md) overlap_kinetic_MDpp(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSpp(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_OSpp(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

        case 3: // sd
            if(int1e_method == int1e_md) overlap_kinetic_MDsd(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSsd(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_MDsd(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

        case 4: // pd
            if(int1e_method == int1e_md) overlap_kinetic_MDpd(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSpd(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_OSpd(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

        case 5: // dd
            if(int1e_method == int1e_md) overlap_kinetic_MDdd(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSdd(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_OSdd(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

#if INT1E_MAX_L >= 3
        case 6: // sf
            if(int1e_method == int1e_md) overlap_kinetic_MDsf(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSsf(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_MDsf(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

        case 7: // pf
            if(int1e_method == int1e_md) overlap_kinetic_MDpf(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSpf(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_OSpf(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

        case 8: // df
            if(int1e_method == int1e_md) overlap_kinetic_MDdf(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSdf(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_OSdf(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

        case 9: // ff
            if(int1e_method == int1e_md) overlap_kinetic_MDff(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSff(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_OSff(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;
#endif

#if INT1E_MAX_L >= 4
        case 10: // sg
            if(int1e_method == int1e_md) overlap_kinetic_MDsg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSsg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_MDsg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

        case 11: // pg
            if(int1e_method == int1e_md) overlap_kinetic_MDpg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSpg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_OSpg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

        case 12: // dg
            if(int1e_method == int1e_md) overlap_kinetic_MDdg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSdg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_OSdg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

        case 13: // fg
            if(int1e_method == int1e_md) overlap_kinetic_MDfg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSfg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_OSfg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;

        case 14: // gg
            if(int1e_method == int1e_md) overlap_kinetic_MDgg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else if(int1e_method == int1e_os) overlap_kinetic_OSgg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            else overlap_kinetic_OSgg(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;
#endif

        default:
            compute_kinetic_energy_integral(item_ct1, g_overlap, g_kinetic, g_shell, g_cgto_normalization_factors, shell_s0, shell_s1, num_threads, num_basis);
            break;
    }

/*
    int flag=0;

    if(flag){
        if (a == 0 && b == 0) overlap_kinetic_MDss(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 0 && b == 1) overlap_kinetic_MDsp(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 1 && b == 1) overlap_kinetic_MDpp(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 0 && b == 2) overlap_kinetic_MDsd(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 1 && b == 2) overlap_kinetic_MDpd(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 2 && b == 2) overlap_kinetic_MDdd(item_ct1,  g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 0 && b == 3) overlap_kinetic_MDsf(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 1 && b == 3) overlap_kinetic_MDpf(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 2 && b == 3) overlap_kinetic_MDdf(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 3 && b == 3) overlap_kinetic_MDff(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else compute_kinetic_energy_integral(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        // else THROW_EXCEPTION("Only up to f-orbitals are supported in MD method for overlap and kinetic integrals");
    }
    else{
        if (a == 0 && b == 0) overlap_kinetic_OSss(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 0 && b == 1) overlap_kinetic_OSsp(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 1 && b == 1) overlap_kinetic_OSpp(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 0 && b == 2) overlap_kinetic_OSsd(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 1 && b == 2) overlap_kinetic_OSpd(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 2 && b == 2) overlap_kinetic_OSdd(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 0 && b == 3) overlap_kinetic_OSsf(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 1 && b == 3) overlap_kinetic_OSpf(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 2 && b == 3) overlap_kinetic_OSdf(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else if (a == 3 && b == 3) overlap_kinetic_OSff(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        else compute_kinetic_energy_integral(item_ct1, g_overlap, g_kinetic,
                          g_shell,
                          g_cgto_normalization_factors,
                          shell_s0,
                          shell_s1,
                          num_threads, num_basis);
        // else THROW_EXCEPTION("Only up to f-orbitals are supported in OS method for overlap and kinetic integrals");
    }
*/
}

inline void launch_nuclear_attraction_kernel(const sycl::nd_item<1>& item_ct1, int a, int b, const Int1eMethod method, real_t* g_nucattr,
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,
                        const Atom* g_atom, const int num_atoms,
                        const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                        const size_t num_threads,
                        const int num_basis, const real_t* g_boys_grid)
{

    const int idx = b*(b+1)/2 + a;

    switch(idx)
    {
        // 0–5 : Hybrid = OS
        case 0:
            if(method == int1e_md)
                nuclear_attraction_MDss(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSss(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

        case 1:
            if(method == int1e_md)
                nuclear_attraction_MDsp(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSsp(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

        case 2:
            if(method == int1e_md)
                nuclear_attraction_MDpp(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSpp(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

        case 3:
            if(method == int1e_md)
                nuclear_attraction_MDsd(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSsd(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

        case 4:
            if(method == int1e_md)
                nuclear_attraction_MDpd(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSpd(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

        case 5:
            if(method == int1e_md)
                nuclear_attraction_MDdd(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSdd(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

#if INT1E_MAX_L >= 3

        // 6–8 : Hybrid = OS
        case 6:
            if(method == int1e_md)
                nuclear_attraction_MDsf(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSsf(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

        case 7:
            if(method == int1e_md)
                nuclear_attraction_MDpf(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSpf(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

        case 8:
            if(method == int1e_md)
                nuclear_attraction_MDdf(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSdf(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

        // 9 : Hybrid = MD
        case 9:
            if(method == int1e_md || method == int1e_hybrid)
                nuclear_attraction_MDff(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSff(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

#endif

#if INT1E_MAX_L >= 4

        // 10–11 : Hybrid = OS
        case 10:
            if(method == int1e_md)
                nuclear_attraction_MDsg(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSsg(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

        case 11:
            if(method == int1e_md)
                nuclear_attraction_MDpg(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSpg(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

        // 12–14 : Hybrid = MD
        case 12:
            if(method == int1e_md || method == int1e_hybrid)
                nuclear_attraction_MDdg(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSdg(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

        case 13:
            if(method == int1e_md || method == int1e_hybrid)
                nuclear_attraction_MDfg(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSfg(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

        case 14:
            if(method == int1e_md || method == int1e_hybrid)
                nuclear_attraction_MDgg(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            else
                nuclear_attraction_OSgg(item_ct1,g_nucattr,g_shell,g_cgto_normalization_factors,
                                        g_atom,num_atoms,shell_s0,shell_s1,
                                        num_threads,num_basis,g_boys_grid);
            break;

#endif

        default:
            compute_nuclear_attraction_integral(
                item_ct1, g_nucattr, g_shell,
                g_cgto_normalization_factors,
                g_atom, num_atoms,
                shell_s0, shell_s1,
                num_threads, num_basis,
                g_boys_grid);
            break;
    }

/*


    int flag=0;

    if(flag){
        if (a == 0 && b == 0) nuclear_attraction_MDss(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 0 && b == 1) nuclear_attraction_MDsp(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 1 && b == 1) nuclear_attraction_MDpp(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 0 && b == 2) nuclear_attraction_MDsd(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 1 && b == 2) nuclear_attraction_MDpd(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 2 && b == 2) nuclear_attraction_MDdd(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 0 && b == 3) nuclear_attraction_MDsf(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 1 && b == 3) nuclear_attraction_MDpf(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 2 && b == 3) nuclear_attraction_MDdf(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 3 && b == 3) nuclear_attraction_MDff(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else compute_nuclear_attraction_integral(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        // else THROW_EXCEPTION("Only up to f-orbitals are supported in MD method for nuclear attraction integrals");
    }
    else{
        if (a == 0 && b == 0) nuclear_attraction_OSss(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 0 && b == 1) nuclear_attraction_OSsp(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 1 && b == 1) nuclear_attraction_OSpp(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 0 && b == 2) nuclear_attraction_OSsd(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 1 && b == 2) nuclear_attraction_OSpd(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 2 && b == 2) nuclear_attraction_OSdd(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 0 && b == 3) nuclear_attraction_OSsf(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 1 && b == 3) nuclear_attraction_OSpf(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 2 && b == 3) nuclear_attraction_OSdf(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else if (a == 3 && b == 3) nuclear_attraction_OSff(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        else compute_nuclear_attraction_integral(item_ct1, g_nucattr,
                        g_shell, g_cgto_normalization_factors, g_atom, num_atoms,
                        shell_s0, shell_s1, num_threads, num_basis, g_boys_grid);
        // else THROW_EXCEPTION("Only up to f-orbitals are supported in OS method for nuclear attraction integrals");
    }
*/
}



} // namespace gansu::gpu

#endif
