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


/**
 * @file int3c2e.hpp This file contains the functions for computing the two-center two-electron repulsion integrals.
 */


 #ifndef INT3C2E_CPP
 #define INT3C2E_CPP

 #define N_ORBITAL_TYPE_BASIS 3   // (s,1), (p,2), ...

 #ifndef N_ORBITAL_TYPE_AUX
 #define N_ORBITAL_TYPE_AUX 5
 #endif

#include <sycl/sycl.hpp>
#include "int2c2e.hpp"
#include "boys.hpp"
#include "types.hpp"
#include "utils_cuda.hpp"
#include "parameters.h"
#include "compile_flag.hpp"

namespace gansu::gpu{

__inline__ double calcNormsWOFact2_3center(double alpha, double beta, double gamma, int sum_ang1,  int sum_ang2,  int sum_ang3){
    return sycl::pow(2.0, sum_ang1 + sum_ang2 + sum_ang3)
           // * pow(factorial2_gpu(2.0*i1-1.0)*factorial2_gpu(2.0*j1-1.0)*factorial2_gpu(2.0*k1-1.0)*factorial2_gpu(2.0*l1-1.0)*factorial2_gpu(2.0*m1-1.0)*factorial2_gpu(2.0*n1-1.0)*factorial2_gpu(2.0*i2-1.0)*factorial2_gpu(2.0*k2-1.0)*factorial2_gpu(2.0*m2-1.0), -0.5)
           * sycl::pow(2.0 / M_PI, 2.25) *
           sycl::pow(alpha, (2.0 * (sum_ang1) + 3.0) / 4.0) *
           sycl::pow(beta, (2.0 * (sum_ang2) + 3.0) / 4.0) *
           sycl::pow(gamma, (2.0 * (sum_ang3) + 3.0) / 4.0);
}
    

__inline__ void addToResult_3center(double res, double *g_result, int p, int q, int r, int nCGTO, int nAux, bool is_prim_id_not_equal, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors){
    res *= d_cgto_normalization_factors[p] * d_cgto_normalization_factors[q] * d_auxiliary_cgto_normalization_factors[r];

//    double* addr = &g_result[r * nCGTO * nCGTO + p * nCGTO + q];
    double* addr = &g_result[(size_t)r * nCGTO * nCGTO + p * nCGTO + q];
    sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>
         atomic_val(*addr);
    atomic_val.fetch_add(res);
    if (is_prim_id_not_equal)
    {
//    double* addr2 = &g_result[r * nCGTO * nCGTO + q * nCGTO + p];
    double* addr2 = &g_result[(size_t)r * nCGTO * nCGTO + q * nCGTO + p];
    sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>
        atomic_val2(*addr2);
    atomic_val2.fetch_add(res);
    }
}


//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
// 3-center integrals [ss|s]~[pp|d]
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//


/* (ss|s) */
/*
DPCT1110:84: The total declared local variable size in device function
calc_sss_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_sss_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {

//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
//        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
//                       item_ct1.get_local_id(2);
//        size_t idx = item_ct1.get_global_id(2);
        uint64_t idx = item_ct1.get_global_linear_id();



        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[1];

                getIncrementalBoys(0, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);

                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 0, 0) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

        bool is_prim_id_not_equal = a!=b;

                #include "../src/integral_RI/int3c2e/orig_sss.txt"
        }
}


/* (ss|p) */
/*
DPCT1110:85: The total declared local variable size in device function
calc_ssp_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_ssp_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[2];
                getIncrementalBoys(1, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 0, 1) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

        bool is_prim_id_not_equal = a!=b;

                #include "../src/integral_RI/int3c2e/orig_ssp.txt"
        }
}


/* (ss|d) */
/*
DPCT1110:86: The total declared local variable size in device function
calc_ssd_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_ssd_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

        // printf("ssd| %d: %d %d %d\n",threadIdx.x, (int)primitive_index_a,(int)primitive_index_b,(int)primitive_index_c);

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[3];
                getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 0, 2) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
        bool is_prim_id_not_equal = a!=b;

                #include "../src/integral_RI/int3c2e/orig_ssd.txt"
        }
}


/* (ss|f) */
/*
DPCT1110:87: The total declared local variable size in device function
calc_ssf_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_ssf_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

        // printf("ssf| %d: %d %d %d\n",threadIdx.x, (int)primitive_index_a,(int)primitive_index_b,(int)primitive_index_c);

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[4];
                getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 0, 3) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

        bool is_prim_id_not_equal = a!=b;

                #include "../src/integral_RI/int3c2e/orig_ssf.txt"
        }
}


/* (sp|s) */
/*
DPCT1110:88: The total declared local variable size in device function
calc_sps_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_sps_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[2];
                getIncrementalBoys(1, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 1, 0) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_sps.txt"
        }
}


/* (sp|p) */
/*
DPCT1110:89: The total declared local variable size in device function
calc_spp_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_spp_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[3];
                getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 1, 1) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_spp.txt"
        }
}


/* (sp|d) */
/*
DPCT1110:90: The total declared local variable size in device function
calc_spd_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_spd_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[4];
                getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 1, 2) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;

                #include "../src/integral_RI/int3c2e/orig_spd.txt"
        }
}


/* (sp|f) */
/*
DPCT1110:91: The total declared local variable size in device function
calc_spf_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_spf_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[5];
                getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 1, 3) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_spf.txt"
        }
}


/* (pp|s) */
/*
DPCT1110:92: The total declared local variable size in device function
calc_pps_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_pps_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[3];
                getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 1, 1, 0) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

        bool is_prim_id_not_equal = a!=b;

                #include "../src/integral_RI/int3c2e/orig_pps.txt"
        }
}


/* (pp|p) */
/*
DPCT1110:93: The total declared local variable size in device function
calc_ppp_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_ppp_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[4];
                getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 1, 1, 1) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

        bool is_prim_id_not_equal = a!=b;

                #include "../src/integral_RI/int3c2e/orig_ppp.txt"
        }
}


/* (pp|d) */
/*
DPCT1110:94: The total declared local variable size in device function
calc_ppd_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_ppd_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[5];
                getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 1, 1, 2) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

        bool is_prim_id_not_equal = a!=b;

                #include "../src/integral_RI/int3c2e/orig_ppd.txt"
        }
}


/* (pp|f) */
/*
DPCT1110:95: The total declared local variable size in device function
calc_ppf_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_ppf_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[6];
                getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 1, 1, 3) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

        bool is_prim_id_not_equal = a!=b;

                #include "../src/integral_RI/int3c2e/orig_ppf.txt"
        }
}


#if defined(COMPUTE_D_BASIS)
/* (sd|s) */
/*
DPCT1110:12: The total declared local variable size in device function
calc_sds_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_sds_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
//        size_t idx = item_ct1.get_global_id(2);
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[3];
                getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 2, 0) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_sds.txt"
        }
}


/* (sd|p) */
/*
DPCT1110:13: The total declared local variable size in device function
calc_sdp_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_sdp_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[4];
                getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 2, 1) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_sdp.txt"
        }
}



/* (sd|d) */
/*
DPCT1110:14: The total declared local variable size in device function
calc_sdd_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_sdd_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[5];
                getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 2, 2) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_sdd.txt"
        }
}



/* (sd|f) */
/*
DPCT1110:15: The total declared local variable size in device function
calc_sdf_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_sdf_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[6];
                getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 2, 3) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_sdf.txt"
        }
}

/* (pd|s) */
/*
DPCT1110:16: The total declared local variable size in device function
calc_pds_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_pds_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[4];
                getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 1, 2, 0) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_pds.txt"
        }
}


/* (pd|p) */
/*
DPCT1110:17: The total declared local variable size in device function
calc_pdp_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_pdp_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[5];
                getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 1, 2, 1) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_pdp.txt"
        }
}



/* (pd|d) */
/*
DPCT1110:18: The total declared local variable size in device function
calc_pdd_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_pdd_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[6];
                getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 1, 2, 2) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_pdd.txt"
        }
}



/* (pd|f) */
/*
DPCT1110:19: The total declared local variable size in device function
calc_pdf_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_pdf_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[7];
                getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 1, 2, 3) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_pdf.txt"
        }
}


/* (dd|s) */
/*
DPCT1110:20: The total declared local variable size in device function
calc_dds_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_dds_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[5];
                getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 2, 2, 0) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

       bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_dds.txt"
        }
}



/* (dd|p) */
/*
DPCT1110:21: The total declared local variable size in device function
calc_ddp_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_ddp_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[6];
                getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 2, 2, 1) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

       bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_ddp.txt"
        }
}



/* (dd|d) */
/*
DPCT1110:22: The total declared local variable size in device function
calc_ddd_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_ddd_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[7];
                getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 2, 2, 2) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

       bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_ddd.txt"
        }
}


/* (dd|f) */
/*
DPCT1110:23: The total declared local variable size in device function
calc_ddf_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_ddf_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[8];
                getIncrementalBoys(7, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 2, 2, 3) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

        bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_ddf.txt"
        }
}

#else
/* (dd|f) */
inline
void calc_ddf_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_ddd_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_ddp_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_dds_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_pdf_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_pdd_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_pdp_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_pds_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_sdf_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_sdd_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_sdp_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_sds_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}
#endif




#if defined(COMPUTE_D_BASIS) && defined(COMPUTE_G_AUX)
/* (sd|g) */
/*
DPCT1110:24: The total declared local variable size in device function
calc_sdg_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_sdg_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
                const size_t primitive_index_c = abc.y + shell_s2.start_index;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];
                // screening (suzuki)
                if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[7];
                getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 2, 4) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_sdg.txt"
        }
}

/* (pd|g) */
/*
DPCT1110:25: The total declared local variable size in device function
calc_pdg_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_pdg_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
                const size_t primitive_index_c = abc.y + shell_s2.start_index;
                // screening (suzuki)
                if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[8];
                getIncrementalBoys(7, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 1, 2, 4) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_pdg.txt"
        }
}


/* (dd|g) */
/*
DPCT1110:26: The total declared local variable size in device function
calc_ddg_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_ddg_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
                const size_t primitive_index_c = abc.y + shell_s2.start_index;
                // screening (suzuki)
                if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[9];
                getIncrementalBoys(8, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 2, 2, 4) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_ddg.txt"
        }
}

#else
inline
void calc_sdg_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_ddg_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_pdg_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}
#endif



#if defined(COMPUTE_G_AUX)
/* (ss|g) */
/*
DPCT1110:12: The total declared local variable size in device function
calc_ssg_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_ssg_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[5];
                getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 0, 4) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

        bool is_prim_id_not_equal = a!=b;

                #include "../src/integral_RI/int3c2e/orig_ssg.txt"
        }
}

/* (sp|g) */
/*
DPCT1110:13: The total declared local variable size in device function
calc_spg_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_spg_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[6];
                getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 0, 1, 4) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);
                bool is_prim_id_not_equal = a!=b;
                #include "../src/integral_RI/int3c2e/orig_spg.txt"
        }
}

/* (pp|g) */
/*
DPCT1110:14: The total declared local variable size in device function
calc_ppg_gpu exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void calc_ppg_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
//        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        size_t idx = item_ct1.get_global_linear_id();

        if(idx < num_tasks){
                const size_t2 abc = index1to2(idx, false, shell_s2.count);

                const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
                const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
            // screening (suzuki)
            if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

                const PrimitiveShell *a = &g_pshell[primitive_index_a];
                const PrimitiveShell *b = &g_pshell[primitive_index_b];
                const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

                double p = a->exponent + b->exponent;
                double sum_exp = p + c->exponent;
                double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
                double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
                double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
                double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
                double Boys[7];
                getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
                double coefAndNorm =
                    a->coefficient * b->coefficient * c->coefficient *
                    calcNormsWOFact2_3center(a->exponent, b->exponent,
                                             c->exponent, 1, 1, 4) *
                    TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                    (p * c->exponent * sycl::sqrt(p + c->exponent)) *
                    sycl::exp(-(a->exponent * b->exponent) *
                              ((a->coordinate.x - b->coordinate.x) *
                                   (a->coordinate.x - b->coordinate.x) +
                               (a->coordinate.y - b->coordinate.y) *
                                   (a->coordinate.y - b->coordinate.y) +
                               (a->coordinate.z - b->coordinate.z) *
                                   (a->coordinate.z - b->coordinate.z)) /
                              p);

        bool is_prim_id_not_equal = a!=b;

                #include "../src/integral_RI/int3c2e/orig_ppg.txt"
        }
}

#else
/* (ss|g) */
inline
void calc_ssg_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_spg_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_ppg_gpu(sycl::nd_item<1>& item_ct1, real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_normalization_factors,
                  const real_t *d_auxiliary_cgto_normalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}
#endif







inline int calcIdx_triangular(int a, int b, int N){
    return (int)(a*N - (a*(a-1))/2) + (b-a);
}












/*
DPCT1110:96: The total declared local variable size in device function
MD_int3c2e_1T1SP exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void MD_int3c2e_1T1SP(sycl::nd_item<1>& item_ct1,real_t *g_result, const PrimitiveShell *g_pshell,
                      const PrimitiveShell *g_pshell_aux,
                      const real_t *d_cgto_normalization_factors,
                      const real_t *d_auxiliary_cgto_normalization_factors,
                      ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                      ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                      const size_t2 *d_primitive_shell_pair_indices,
                      const double *g_upper_bound_factors,
                      const double *g_auxiliary_upper_bound_factors,
                      const double schwarz_screening_threshold,
                      int num_auxiliary_basis, const double *g_boys_grid){
//                      dpct::accessor<int, dpct::constant, 3> loop_to_ang_RI,
//                      dpct::accessor<int, dpct::constant, 2> tuv_list,
//                      double (*MD_EtArray[])(double, double, double, double,
//                                             double) *
//                          *MD_EtArray) {
//auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
{
    // 通し番号indexの計算
//    const size_t id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
//                      item_ct1.get_local_id(2);
    const size_t id = item_ct1.get_global_linear_id();


    if (id >= num_tasks) return;

    const double size_Rmid=1377;

    //使い捨ての中間体R_mid
    double R_mid[3*1377];

    //解を格納する配列R
    double R[2925];

    //thread内で結果を保持するメモリ
    double thread_val=0.0;

    const size_t2 abc = index1to2(id, false, shell_s2.count);
    const size_t2 ab =  index1to2(abc.x, (shell_s0.start_index == shell_s1.start_index), shell_s1.count);


    // Obtain primitive shells [ab|c]
        const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
        const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
    const size_t primitive_index_c = abc.y + shell_s2.start_index;

        // screening (suzuki)
        if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


    const PrimitiveShell a = g_pshell[primitive_index_a];
    const PrimitiveShell b = g_pshell[primitive_index_b];
    const PrimitiveShell c = g_pshell_aux[primitive_index_c];


    // Obtain basis index (ij|k)
    const size_t size_a = a.basis_index;
    const size_t size_b = b.basis_index;
    const size_t size_c = c.basis_index;


    bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);


    //使用データを取得，レジスタに書き込み

    //指数部
    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double gamma = c.exponent;
    const double p = alpha+beta;
    const double xi = p*gamma / (p+gamma);

    //係数部
    const double coef_a = a.coefficient;
    const double coef_b = b.coefficient;
    const double coef_c = c.coefficient;

    //座標
    const double pos_A[3] = {a.coordinate.x, a.coordinate.y, a.coordinate.z};
    const double pos_B[3] = {b.coordinate.x, b.coordinate.y, b.coordinate.z};
    const double pos_C[3] = {c.coordinate.x, c.coordinate.y, c.coordinate.z};
    const double pos_P[3] = {(alpha*pos_A[0]+beta*pos_B[0])/(alpha+beta), (alpha*pos_A[1]+beta*pos_B[1])/(alpha+beta), (alpha*pos_A[2]+beta*pos_B[2])/(alpha+beta)};


    //角運動量の総和
    const int orbital_A = a.shell_type;
    const int orbital_B = b.shell_type;
    const int orbital_C = c.shell_type;


    //軌道間距離の二乗
    const double dist = ((pos_P[0]-pos_C[0])*(pos_P[0]-pos_C[0]) + (pos_P[1]-pos_C[1])*(pos_P[1]-pos_C[1]) + (pos_P[2]-pos_C[2])*(pos_P[2]-pos_C[2]));


    const int K = orbital_A + orbital_B + orbital_C;

    double Boys[25];
    getIncrementalBoys(K, xi*dist, g_boys_grid, Boys);

    //Boys関数の値を計算(Single)
    for(int i=0; i <= K; i++){
        Boys[i] *= (right2left_binary_woif((-2*xi), i));
    }

    //各ERIを計算
    //事前計算⇒実際のERI計算の順に実行
    //p軌道の場合lmn_aが0:px, 1:py, 2:pz軌道のように対応付け



    double Norm_A, Norm_B, Norm_C;
    double Norm;

    int t,u,v;
    int t_max;
    int u_max;
    int v_max;
    int tau_max;
    int nu_max;
    int phi_max;

    // int tid=0;


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


                                Norm = Norm_A * Norm_B * Norm_C;
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
                                tau_max = l3+1;
                                nu_max = m3+1;
                                phi_max = n3+1;

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
                                                                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[0] - pos_C[0])*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_Rmid)];
                                                                }
                                                                else if(u >= 1){
                                                                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[1] - pos_C[1])*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_Rmid)];
                                                                }
                                                                else{
                                                                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[2] - pos_C[2])*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_Rmid)];
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

                                        double my_val = 0.0;
                    // ERI計算部
                                        double Et, Eu, Ev, Etau, Enu, Ephi;
                    for(int t=0; t<l1+l2+1; t++){
                        Et = MD_Et_NonRecursion(l1, l2, t, alpha, beta,
                                                (pos_A[0] - pos_B[0]));
                        for(int u=0; u<m1+m2+1; u++){
                            Eu = MD_Et_NonRecursion(m1, m2, u, alpha, beta,
                                                    (pos_A[1] - pos_B[1]));
                            for(int v=0; v<n1+n2+1; v++){
                                Ev = MD_Et_NonRecursion(n1, n2, v, alpha, beta,
                                                        (pos_A[2] - pos_B[2]));
                                for(int tau=0; tau<l3+1; tau++){
                                    Etau = MD_Et_NonRecursion(l3, 0, tau, gamma,
                                                              0.0, 0.0);
                                    for(int nu=0; nu<m3+1; nu++){
                                        Enu = MD_Et_NonRecursion(
                                            m3, 0, nu, gamma, 0.0, 0.0);
                                        for(int phi=0; phi<n3+1; phi++){
                                            Ephi = MD_Et_NonRecursion(
                                                n3, 0, phi, gamma, 0.0, 0.0);
                                            int k=t+u+v+tau+nu+phi;
                                            my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k,u+nu,v+phi,0,0,0)];
                                            // my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R_GPU_Recursion(0, t+tau, u+nu, v+phi, P, Q, Boys);
                                        }
                                    }
                                }
                            }
                        }
                    }
                                        thread_val = my_val * 2 * M_PI_2_5 /
                                                     (p * gamma *
                                                      sycl::sqrt((p + gamma))) *
                                                     coef_a * coef_b * coef_c;

                    // 書き込み部

                    // Global Memoryへ書き込み
                    // 汎用カーネルでは全要素判定(case1)
                    addToResult_3center(
                        Norm*thread_val,
                        g_result,
                        size_a+lmn_a, size_b+lmn_b, size_c+lmn_c,
                        num_basis, num_auxiliary_basis,
                        is_prim_id_not_equal,
                                                d_cgto_normalization_factors, d_auxiliary_cgto_normalization_factors
                    );
                }
            }
        }
    }
    return;
}

inline void launch_3center_kernel(sycl::nd_item<1>& item_ct1, int a, int b, int c, real_t* args, const PrimitiveShell* shell1, const PrimitiveShell* shell2, const real_t* param1, const real_t* param2, ShellTypeInfo info1, ShellTypeInfo info2, ShellTypeInfo info3, int64_t var1, int var2, const size_t2* dp_ind, const double* g_u, const double* g_a, const double s_s_th,
int var3, const double* param3) {
#if !defined(COMPUTE_D_BASIS)
    if (a >= 2 || b >= 2) {
        MD_int3c2e_1T1SP(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
        dp_ind, g_u, g_a, s_s_th, var3, param3);
        return;
    }
#endif
#if !defined(COMPUTE_G_AUX)
    if (c >= 4) {
        MD_int3c2e_1T1SP(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
        dp_ind, g_u, g_a, s_s_th, var3, param3);
        return;
    }
#endif

    if (a < N_ORBITAL_TYPE_BASIS && b < N_ORBITAL_TYPE_BASIS && c < N_ORBITAL_TYPE_AUX) {
        int idx = calcIdx_triangular(a, b, N_ORBITAL_TYPE_BASIS) * N_ORBITAL_TYPE_AUX + c;
        switch (idx) {
            case 0: calc_sss_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 1: calc_ssp_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 2: calc_ssd_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 3: calc_ssf_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 4: calc_ssg_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 5: calc_sps_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 6: calc_spp_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 7: calc_spd_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 8: calc_spf_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 9: calc_spg_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 10: calc_sds_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 11: calc_sdp_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 12: calc_sdd_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 13: calc_sdf_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 14: calc_sdg_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 15: calc_pps_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 16: calc_ppp_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 17: calc_ppd_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 18: calc_ppf_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 19: calc_ppg_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 20: calc_pds_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 21: calc_pdp_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 22: calc_pdd_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 23: calc_pdf_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 24: calc_pdg_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 25: calc_dds_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 26: calc_ddp_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 27: calc_ddd_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 28: calc_ddf_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 29: calc_ddg_gpu(item_ct1, args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            default: ;
//              throw std::runtime_error("Invalid kernel index.\n");
        }
//    } else {
//        throw std::runtime_error("Invalid call for 3center eri.\n");
    }
}

  

    /*----------------------------------------------- int3c2e kernels for Direct-RI-RHF computation -----------------------------------------------*/
    using direct_ri_w_kernel_t = void (*)(real_t*, real_t*, const PrimitiveShell* , const PrimitiveShell* , const real_t*, const real_t*, ShellTypeInfo, ShellTypeInfo, ShellTypeInfo, int64_t, int, const size_t2*, const double*, const double*, const double, int, int, const double*);
    using direct_ri_c_J_kernel_t = void (*)(real_t*, const real_t*, const PrimitiveShell* , const PrimitiveShell* , const real_t*, const real_t*, ShellTypeInfo, ShellTypeInfo, ShellTypeInfo, int64_t, int, const size_t2*, const double*, const double*, const double, int, const double*);

SYCL_EXTERNAL void compute_RI_Direct_c_kernel(real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2,  int64_t num_tasks, int num_basis,  const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold,  int num_auxiliary_basis,  const double* g_boys_grid);
SYCL_EXTERNAL void compute_RI_Direct_J_kernel(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux,const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors,ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2,int64_t num_tasks, int num_basis,const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors,const double* g_auxiliary_upper_bound_factors,const double schwarz_screening_threshold,int num_auxiliary_basis,const double* g_boys_grid);
SYCL_EXTERNAL void compute_RI_Direct_W_kernel(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2,  int64_t num_tasks, int num_basis,  const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold,  int num_auxiliary_basis, int iter, const double* g_boys_grid);


    /* (ss|s) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_sss(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (ss|p) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_ssp(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (ss|d) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_ssd(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (ss|f) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_ssf(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (sp|s) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_sps(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (sp|p) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_spp(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (sp|d) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_spd(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (sp|f) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_spf(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (sd|s) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_sds(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (sd|p) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_sdp(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (sd|d) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_sdd(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (sd|f) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_sdf(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (pp|s) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_pps(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (pp|p) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_ppp(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (pp|d) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_ppd(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (pp|f) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_ppf(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (pd|s) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_pds(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (pd|p) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_pdp(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (pd|d) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_pdd(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (pd|f) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_pdf(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (dd|s) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_dds(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (dd|p) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_ddp(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (dd|d) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_ddd(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    /* (dd|f) */ SYCL_EXTERNAL void compute_RI_Direct_W_kernel_ddf(real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);
    
    /* (ss|s) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_sss(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (ss|p) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_ssp(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (ss|d) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_ssd(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (ss|f) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_ssf(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|s) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_sps(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|p) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_spp(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|d) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_spd(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|f) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_spf(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|s) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_sds(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|p) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_sdp(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|d) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_sdd(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|f) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_sdf(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|s) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_pps(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|p) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_ppp(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|d) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_ppd(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|f) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_ppf(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|s) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_pds(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|p) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_pdp(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|d) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_pdd(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|f) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_pdf(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (dd|s) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_dds(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (dd|p) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_ddp(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (dd|d) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_ddd(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (dd|f) */ SYCL_EXTERNAL void compute_RI_Direct_c_kernel_ddf(real_t* d_W_diff, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);

    /* (ss|s) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_sss(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (ss|p) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_ssp(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (ss|d) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_ssd(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (ss|f) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_ssf(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|s) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_sps(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|p) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_spp(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|d) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_spd(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|f) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_spf(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|s) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_sds(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|p) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_sdp(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|d) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_sdd(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|f) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_sdf(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|s) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_pps(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|p) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_ppp(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|d) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_ppd(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|f) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_ppf(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|s) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_pds(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|p) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_pdp(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|d) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_pdd(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|f) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_pdf(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (dd|s) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_dds(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (dd|p) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_ddp(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (dd|d) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_ddd(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (dd|f) */ SYCL_EXTERNAL void compute_RI_Direct_J_kernel_ddf(real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);



void compute_RI_Direct_Z_kernel(real_t* d_Z, const real_t* d_C, const real_t* d_L_inv, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices,const double* g_upper_bound_factors, const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, int iter,const double* g_boys_grid);


	/*----------------------------------------------- int3c2e kernels for Direct-RI-RHF computation -----------------------------------------------*/
// atomicADD is used in "../src/integral_RI/direct_ri_c/.." so wrapp them.
template <typename T>
inline T atomic_add(T* addr, T value) noexcept
{
    sycl::atomic_ref<
        T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        atom(*addr);

    return atom.fetch_add(value);
}

#define atomicAdd atomic_add

	/* (ss|s) */
//	__global__ void compute_RI_Direct_c_kernel_sss(real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
inline void compute_RI_Direct_c_kernel_sss(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
//		const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[1];
		getIncrementalBoys(0, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_sss.txt"

	}
		

	/* (ss|p) */
inline void compute_RI_Direct_c_kernel_ssp(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[2];
		getIncrementalBoys(1, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_ssp.txt"
	}
		

	/* (ss|d) */
inline void compute_RI_Direct_c_kernel_ssd(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_ssd.txt"
	}
		

	/* (ss|f) */
inline void compute_RI_Direct_c_kernel_ssf(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_ssf.txt"
	}
		

	/* (sp|s) */
inline void compute_RI_Direct_c_kernel_sps(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[2];
		getIncrementalBoys(1, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_sps.txt"
	}
		

	/* (sp|p) */
inline void compute_RI_Direct_c_kernel_spp(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_spp.txt"
	}
		

	/* (sp|d) */
inline void compute_RI_Direct_c_kernel_spd(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_spd.txt"
	}
		

	/* (sp|f) */
inline void compute_RI_Direct_c_kernel_spf(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_spf.txt"
	}
		

	/* (sd|s) */
inline void compute_RI_Direct_c_kernel_sds(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        #if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;
        bool is_prim_id_neq = (primitive_index_a != primitive_index_b);


		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_sds.txt"
		#endif
	}
		

	/* (sd|p) */
inline void compute_RI_Direct_c_kernel_sdp(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        #if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;
        bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_sdp.txt"
		#endif
	}
		

	/* (sd|d) */
inline void compute_RI_Direct_c_kernel_sdd(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        #if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;
        bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_sdd.txt"
		#endif
	}
		

	/* (sd|f) */
inline void compute_RI_Direct_c_kernel_sdf(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        #if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

        bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_sdf.txt"
		#endif
	}
		

	/* (pp|s) */
inline void compute_RI_Direct_c_kernel_pps(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_pps.txt"
	}
		

	/* (pp|p) */
inline void compute_RI_Direct_c_kernel_ppp(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_ppp.txt"
	}
		

	/* (pp|d) */
inline void compute_RI_Direct_c_kernel_ppd(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_ppd.txt"
	}
		

	/* (pp|f) */
inline void compute_RI_Direct_c_kernel_ppf(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_ppf.txt"
	}
		

	/* (pd|s) */
inline void compute_RI_Direct_c_kernel_pds(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        #if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_pds.txt"
		#endif
	}
		

	/* (pd|p) */
inline void compute_RI_Direct_c_kernel_pdp(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_pdp.txt"
		#endif
	}
		

	/* (pd|d) */
inline void compute_RI_Direct_c_kernel_pdd(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_pdd.txt"
		#endif
	}
		

	/* (pd|f) */
inline void compute_RI_Direct_c_kernel_pdf(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_pdf.txt"
		#endif
	}
		

	/* (dd|s) */
inline void compute_RI_Direct_c_kernel_dds(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_dds.txt"
		#endif
	}
		

	/* (dd|p) */
inline void compute_RI_Direct_c_kernel_ddp(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_ddp.txt"
		#endif
	}
		

	/* (dd|d) */
inline void compute_RI_Direct_c_kernel_ddd(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_ddd.txt"
		#endif
	}
		

	/* (dd|f) */
inline void compute_RI_Direct_c_kernel_ddf(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[8];
		getIncrementalBoys(7, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_c/orig_ddf.txt"
		#endif
	}
		




	/* (ss|s) */
inline void compute_RI_Direct_J_kernel_sss(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[1];
		getIncrementalBoys(0, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_sss.txt"
	}
		

	/* (ss|p) */
inline void compute_RI_Direct_J_kernel_ssp(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[2];
		getIncrementalBoys(1, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_ssp.txt"
	}
		

	/* (ss|d) */
inline void compute_RI_Direct_J_kernel_ssd(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_ssd.txt"
	}
		

	/* (ss|f) */
inline void compute_RI_Direct_J_kernel_ssf(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_ssf.txt"
	}
		

	/* (sp|s) */
inline void compute_RI_Direct_J_kernel_sps(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[2];
		getIncrementalBoys(1, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_sps.txt"
	}
		

	/* (sp|p) */
inline void compute_RI_Direct_J_kernel_spp(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_spp.txt"
	}
		

	/* (sp|d) */
inline void compute_RI_Direct_J_kernel_spd(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_spd.txt"
	}
		

	/* (sp|f) */
inline void compute_RI_Direct_J_kernel_spf(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_spf.txt"
	}
		

	/* (sd|s) */
inline void compute_RI_Direct_J_kernel_sds(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_sds.txt"
		#endif
	}
		

	/* (sd|p) */
inline void compute_RI_Direct_J_kernel_sdp(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_sdp.txt"
		#endif
	}
		

	/* (sd|d) */
inline void compute_RI_Direct_J_kernel_sdd(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_sdd.txt"
		#endif
	}
		

	/* (sd|f) */
inline void compute_RI_Direct_J_kernel_sdf(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_sdf.txt"
		#endif
	}
		

	/* (pp|s) */
inline void compute_RI_Direct_J_kernel_pps(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_pps.txt"
	}
		

	/* (pp|p) */
inline void compute_RI_Direct_J_kernel_ppp(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_ppp.txt"
	}
		

	/* (pp|d) */
inline void compute_RI_Direct_J_kernel_ppd(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_ppd.txt"
	}
		

	/* (pp|f) */
inline void compute_RI_Direct_J_kernel_ppf(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_ppf.txt"
	}
		

	/* (pd|s) */
inline void compute_RI_Direct_J_kernel_pds(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_pds.txt"
		#endif
	}
		

	/* (pd|p) */
inline void compute_RI_Direct_J_kernel_pdp(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_pdp.txt"
		#endif
	}
		

	/* (pd|d) */
inline void compute_RI_Direct_J_kernel_pdd(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_pdd.txt"
		#endif
	}
		

	/* (pd|f) */
inline void compute_RI_Direct_J_kernel_pdf(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_pdf.txt"
		#endif
	}
		

	/* (dd|s) */
inline void compute_RI_Direct_J_kernel_dds(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_dds.txt"
		#endif
	}
		

	/* (dd|p) */
inline void compute_RI_Direct_J_kernel_ddp(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_ddp.txt"
		#endif
	}
		

	/* (dd|d) */
inline void compute_RI_Direct_J_kernel_ddd(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_ddd.txt"
		#endif
	}
		

	/* (dd|f) */
inline void compute_RI_Direct_J_kernel_ddf(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[8];
		getIncrementalBoys(7, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include "../src/integral_RI/direct_ri_J/orig_ddf.txt"
		#endif
	}
    





	/* (ss|s) */
inline void compute_RI_Direct_W_kernel_sss(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[1];
		getIncrementalBoys(0, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		
		#include "../src/integral_RI/direct_ri_w/orig_sss.txt"
	}
		

	/* (ss|p) */
inline void compute_RI_Direct_W_kernel_ssp(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[2];
		getIncrementalBoys(1, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		
		#include "../src/integral_RI/direct_ri_w/orig_ssp.txt"
	}
		

	/* (ss|d) */
inline void compute_RI_Direct_W_kernel_ssd(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		
		#include "../src/integral_RI/direct_ri_w/orig_ssd.txt"
	}
		

	/* (ss|f) */
inline void compute_RI_Direct_W_kernel_ssf(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		
		#include "../src/integral_RI/direct_ri_w/orig_ssf.txt"
	}
		

	/* (sp|s) */
inline void compute_RI_Direct_W_kernel_sps(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[2];
		getIncrementalBoys(1, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		
		#include "../src/integral_RI/direct_ri_w/orig_sps.txt"
	}
		

	/* (sp|p) */
inline void compute_RI_Direct_W_kernel_spp(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		
		#include "../src/integral_RI/direct_ri_w/orig_spp.txt"
	}
		

	/* (sp|d) */
inline void compute_RI_Direct_W_kernel_spd(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		
		#include "../src/integral_RI/direct_ri_w/orig_spd.txt"
	}
		

	/* (sp|f) */
inline void compute_RI_Direct_W_kernel_spf(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		
		#include "../src/integral_RI/direct_ri_w/orig_spf.txt"
	}

	/* (sd|s) */
inline void compute_RI_Direct_W_kernel_sds(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include  "../src/integral_RI/direct_ri_w/orig_sds.txt"
		#endif
	}
		

	/* (sd|p) */
inline void compute_RI_Direct_W_kernel_sdp(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include  "../src/integral_RI/direct_ri_w/orig_sdp.txt"
		#endif
	}
		

	/* (sd|d) */
inline void compute_RI_Direct_W_kernel_sdd(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include  "../src/integral_RI/direct_ri_w/orig_sdd.txt"
		#endif
	}
		

	/* (sd|f) */
inline void compute_RI_Direct_W_kernel_sdf(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include  "../src/integral_RI/direct_ri_w/orig_sdf.txt"
		#endif
	}
		

	/* (pp|s) */
inline void compute_RI_Direct_W_kernel_pps(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		
		#include "../src/integral_RI/direct_ri_w/orig_pps.txt"
	}
		

	/* (pp|p) */
inline void compute_RI_Direct_W_kernel_ppp(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		
		#include "../src/integral_RI/direct_ri_w/orig_ppp.txt"
	}
		

	/* (pp|d) */
inline void compute_RI_Direct_W_kernel_ppd(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		
		#include "../src/integral_RI/direct_ri_w/orig_ppd.txt"
	}
		

	/* (pp|f) */
inline void compute_RI_Direct_W_kernel_ppf(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		
		#include "../src/integral_RI/direct_ri_w/orig_ppf.txt"
	}
		

	/* (pd|s) */
inline void compute_RI_Direct_W_kernel_pds(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include  "../src/integral_RI/direct_ri_w/orig_pds.txt"
		#endif
	}
		

	/* (pd|p) */
inline void compute_RI_Direct_W_kernel_pdp(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include  "../src/integral_RI/direct_ri_w/orig_pdp.txt"
		#endif
	}
		

	/* (pd|d) */
inline void compute_RI_Direct_W_kernel_pdd(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include  "../src/integral_RI/direct_ri_w/orig_pdd.txt"
		#endif
	}
		

	/* (pd|f) */
inline void compute_RI_Direct_W_kernel_pdf(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include  "../src/integral_RI/direct_ri_w/orig_pdf.txt"
		#endif
	}
		

	/* (dd|s) */
inline void compute_RI_Direct_W_kernel_dds(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include  "../src/integral_RI/direct_ri_w/orig_dds.txt"
		#endif
	}
		

	/* (dd|p) */
inline void compute_RI_Direct_W_kernel_ddp(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include  "../src/integral_RI/direct_ri_w/orig_ddp.txt"
		#endif
	}
		

	/* (dd|d) */
inline void compute_RI_Direct_W_kernel_ddd(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include  "../src/integral_RI/direct_ri_w/orig_ddd.txt"
		#endif
	}
		

	/* (dd|f) */
inline void compute_RI_Direct_W_kernel_ddf(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
													const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
													ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
													int64_t num_tasks, int num_basis, 
													const size_t2* d_primitive_shell_pair_indices,
													const double* g_upper_bound_factors, 
													const double* g_auxiliary_upper_bound_factors, 
													const double schwarz_screening_threshold, 
													int num_auxiliary_basis, 
													int iter,
													const double* g_boys_grid){

		// 通し番号indexの計算
		#if defined(COMPUTE_D_BASIS)
        const size_t id = item.get_global_linear_id(); 
		if (id >= num_tasks) return;


		// Obtain primitive shells [ab|c]
		const size_t2 abc = index1to2(id, false, shell_s2.count);
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];


		real_t max_coefficient_value = 0.0;
		real_t tmp;
		for(int lmn_b=0; lmn_b<comb_max(b->shell_type); lmn_b++){
			tmp = fabs(d_C_diff_vector[(b->basis_index + lmn_b)]);
			if(max_coefficient_value < tmp) max_coefficient_value = tmp;
		}

		if (is_prim_id_neq) {
			for(int lmn_a=0; lmn_a<comb_max(a->shell_type); lmn_a++){
				tmp = fabs(d_C_diff_vector[(a->basis_index + lmn_a)]);
				if(max_coefficient_value < tmp) max_coefficient_value = tmp;
			}
		}
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;


		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3]  = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[8];
		getIncrementalBoys(7, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p); 

		#include  "../src/integral_RI/direct_ri_w/orig_ddf.txt"
		#endif
	}











inline void compute_RI_Direct_c_kernel(sycl::nd_item<1> item,
real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
										    const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
											ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
											int64_t num_tasks, int num_basis, 
											const size_t2* d_primitive_shell_pair_indices,
											const double* g_upper_bound_factors, 
											const double* g_auxiliary_upper_bound_factors, 
											const double schwarz_screening_threshold, 
											int num_auxiliary_basis, 
											const double* g_boys_grid){
{
    // 通し番号indexの計算
    const size_t id = item.get_global_linear_id(); 

    if (id >= num_tasks) return;

    const double size_Rmid=1377;

    //使い捨ての中間体R_mid
    double R_mid[3*1377];

    //解を格納する配列R
    double R[2925];

    //thread内で結果を保持するメモリ
    double thread_val=0.0;

    const size_t2 abc = index1to2(id, false, shell_s2.count);
    


    // Obtain primitive shells [ab|c]
	const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
	const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
    const size_t primitive_index_c = abc.y + shell_s2.start_index;


	// screening (suzuki)
	if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


    const PrimitiveShell a = g_pshell[primitive_index_a];
    const PrimitiveShell b = g_pshell[primitive_index_b];
    const PrimitiveShell c = g_pshell_aux[primitive_index_c];


    bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

    
    //使用データを取得，レジスタに書き込み

    //指数部
    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double gamma = c.exponent;
    const double p = alpha+beta;
    const double xi = p*gamma / (p+gamma);

    //係数部
    const double coef_a = a.coefficient;
    const double coef_b = b.coefficient;
    const double coef_c = c.coefficient;

    //座標
    const double pos_A[3] = {a.coordinate.x, a.coordinate.y, a.coordinate.z};
    const double pos_B[3] = {b.coordinate.x, b.coordinate.y, b.coordinate.z};
    const double pos_C[3] = {c.coordinate.x, c.coordinate.y, c.coordinate.z};
    const double pos_P[3] = {(alpha*pos_A[0]+beta*pos_B[0])/(alpha+beta), (alpha*pos_A[1]+beta*pos_B[1])/(alpha+beta), (alpha*pos_A[2]+beta*pos_B[2])/(alpha+beta)};


    //角運動量の総和
    const int orbital_A = a.shell_type;
    const int orbital_B = b.shell_type;
    const int orbital_C = c.shell_type;


    //軌道間距離の二乗
    const double dist = ((pos_P[0]-pos_C[0])*(pos_P[0]-pos_C[0]) + (pos_P[1]-pos_C[1])*(pos_P[1]-pos_C[1]) + (pos_P[2]-pos_C[2])*(pos_P[2]-pos_C[2]));


    const int K = orbital_A + orbital_B + orbital_C;
    
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

    
    
    double Norm_A, Norm_B, Norm_C;
    double Norm;

    int t,u,v;
    int t_max;
    int u_max;
    int v_max;
    int tau_max;
    int nu_max;
    int phi_max;

    // int tid=0;


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


				Norm = Norm_A * Norm_B * Norm_C;
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
				tau_max = l3+1;
				nu_max = m3+1;
				phi_max = n3+1;

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
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[0] - pos_C[0])*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_Rmid)];
								}
								else if(u >= 1){
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[1] - pos_C[1])*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_Rmid)];
								}
								else{
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[2] - pos_C[2])*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_Rmid)];
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
                    
					double my_val = 0.0;
                    // ERI計算部
					double Et, Eu, Ev, Etau, Enu, Ephi;
                    for(int t=0; t<l1+l2+1; t++){
                        Et =  MD_Et_NonRecursion(l1, l2, t, alpha, beta, (pos_A[0]-pos_B[0]));
                        for(int u=0; u<m1+m2+1; u++){
                            Eu = MD_Et_NonRecursion(m1, m2, u, alpha, beta, (pos_A[1]-pos_B[1]));
                            for(int v=0; v<n1+n2+1; v++){
                                Ev = MD_Et_NonRecursion(n1, n2, v, alpha, beta, (pos_A[2]-pos_B[2]));
                                for(int tau=0; tau<l3+1; tau++){
                                    Etau = MD_Et_NonRecursion(l3, 0, tau, gamma, 0.0, 0.0);
                                    for(int nu=0; nu<m3+1; nu++){
                                        Enu = MD_Et_NonRecursion(m3, 0, nu, gamma, 0.0, 0.0);
                                        for(int phi=0; phi<n3+1; phi++){
                                            Ephi = MD_Et_NonRecursion(n3, 0, phi, gamma, 0.0, 0.0);
                                            int k=t+u+v+tau+nu+phi;
                                            my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k,u+nu,v+phi,0,0,0)];
                                            // my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R_GPU_Recursion(0, t+tau, u+nu, v+phi, P, Q, Boys);
                                        }
                                    }
                                }
                            }
                        }
                    }
					thread_val = Norm * my_val*2 * M_PI_2_5 /(p*gamma * sqrt((p+gamma))) * coef_a*coef_b*coef_c * d_cgto_normalization_factors[a.basis_index + lmn_a] * d_cgto_normalization_factors[b.basis_index + lmn_b] * d_auxiliary_cgto_normalization_factors[c.basis_index + lmn_c];

                    // 書き込み部
                    thread_val *= (is_prim_id_neq) ? (d_density_matrix[(a.basis_index+lmn_a)*num_basis + b.basis_index+lmn_b] + d_density_matrix[(b.basis_index+lmn_b)*num_basis + a.basis_index+lmn_a])
                                                         : d_density_matrix[(a.basis_index+lmn_a)*num_basis + b.basis_index+lmn_b];
                    
                    atomicAdd(&d_c[c.basis_index+lmn_c], thread_val);
                }
            }
        }
    }
    return;
}



inline void compute_RI_Direct_J_kernel(sycl::nd_item<1> item,
real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
										    const real_t* d_cgto_normalization_factors, const real_t* d_auxiliary_cgto_normalization_factors, 
											ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
											int64_t num_tasks, int num_basis, 
											const size_t2* d_primitive_shell_pair_indices,
											const double* g_upper_bound_factors, 
											const double* g_auxiliary_upper_bound_factors, 
											const double schwarz_screening_threshold, 
											int num_auxiliary_basis, 
											const double* g_boys_grid){
{
    // 通し番号indexの計算
    const size_t id = item.get_global_linear_id(); 


    if (id >= num_tasks) return;

    const double size_Rmid=1377;

    //使い捨ての中間体R_mid
    double R_mid[3*1377];

    //解を格納する配列R
    double R[2925];

    //thread内で結果を保持するメモリ
    double thread_val=0.0;

    const size_t2 abc = index1to2(id, false, shell_s2.count);
    


    // Obtain primitive shells [ab|c]
	const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
	const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
    const size_t primitive_index_c = abc.y + shell_s2.start_index;

	// screening (suzuki)
	if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


    const PrimitiveShell a = g_pshell[primitive_index_a];
    const PrimitiveShell b = g_pshell[primitive_index_b];
    const PrimitiveShell c = g_pshell_aux[primitive_index_c];


    bool is_prim_id_neq = (primitive_index_a != primitive_index_b);

    
    //使用データを取得，レジスタに書き込み

    //指数部
    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double gamma = c.exponent;
    const double p = alpha+beta;
    const double xi = p*gamma / (p+gamma);

    //係数部
    const double coef_a = a.coefficient;
    const double coef_b = b.coefficient;
    const double coef_c = c.coefficient;

    //座標
    const double pos_A[3] = {a.coordinate.x, a.coordinate.y, a.coordinate.z};
    const double pos_B[3] = {b.coordinate.x, b.coordinate.y, b.coordinate.z};
    const double pos_C[3] = {c.coordinate.x, c.coordinate.y, c.coordinate.z};
    const double pos_P[3] = {(alpha*pos_A[0]+beta*pos_B[0])/(alpha+beta), (alpha*pos_A[1]+beta*pos_B[1])/(alpha+beta), (alpha*pos_A[2]+beta*pos_B[2])/(alpha+beta)};


    //角運動量の総和
    const int orbital_A = a.shell_type;
    const int orbital_B = b.shell_type;
    const int orbital_C = c.shell_type;


    //軌道間距離の二乗
    const double dist = ((pos_P[0]-pos_C[0])*(pos_P[0]-pos_C[0]) + (pos_P[1]-pos_C[1])*(pos_P[1]-pos_C[1]) + (pos_P[2]-pos_C[2])*(pos_P[2]-pos_C[2]));


    const int K = orbital_A + orbital_B + orbital_C;
    
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

    
    
    double Norm_A, Norm_B, Norm_C;
    double Norm;

    int t,u,v;
    int t_max;
    int u_max;
    int v_max;
    int tau_max;
    int nu_max;
    int phi_max;

    // int tid=0;


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


				Norm = Norm_A * Norm_B * Norm_C;
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
				tau_max = l3+1;
				nu_max = m3+1;
				phi_max = n3+1;

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
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[0] - pos_C[0])*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_Rmid)];
								}
								else if(u >= 1){
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[1] - pos_C[1])*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_Rmid)];
								}
								else{
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[2] - pos_C[2])*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_Rmid)];
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
                    
					double my_val = 0.0;
                    // ERI計算部
					double Et, Eu, Ev, Etau, Enu, Ephi;
                    for(int t=0; t<l1+l2+1; t++){
                        Et =  MD_Et_NonRecursion(l1, l2, t, alpha, beta, (pos_A[0]-pos_B[0]));
                        for(int u=0; u<m1+m2+1; u++){
                            Eu = MD_Et_NonRecursion(m1, m2, u, alpha, beta, (pos_A[1]-pos_B[1]));
                            for(int v=0; v<n1+n2+1; v++){
                                Ev = MD_Et_NonRecursion(n1, n2, v, alpha, beta, (pos_A[2]-pos_B[2]));
                                for(int tau=0; tau<l3+1; tau++){
                                    Etau = MD_Et_NonRecursion(l3, 0, tau, gamma, 0.0, 0.0);
                                    for(int nu=0; nu<m3+1; nu++){
                                        Enu = MD_Et_NonRecursion(m3, 0, nu, gamma, 0.0, 0.0);
                                        for(int phi=0; phi<n3+1; phi++){
                                            Ephi = MD_Et_NonRecursion(n3, 0, phi, gamma, 0.0, 0.0);
                                            int k=t+u+v+tau+nu+phi;
                                            my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k,u+nu,v+phi,0,0,0)];
                                            // my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R_GPU_Recursion(0, t+tau, u+nu, v+phi, P, Q, Boys);
                                        }
                                    }
                                }
                            }
                        }
                    }
					thread_val = Norm * my_val*2 * M_PI_2_5 /(p*gamma * sqrt((p+gamma)))  *coef_a*coef_b*coef_c * d_cgto_normalization_factors[a.basis_index + lmn_a] * d_cgto_normalization_factors[b.basis_index + lmn_b] * d_auxiliary_cgto_normalization_factors[c.basis_index + lmn_c];

                    // 書き込み部
                    thread_val *= d_t[c.basis_index+lmn_c];

                    atomicAdd(&d_J[(a.basis_index+lmn_a)*num_basis + b.basis_index+lmn_b], thread_val);
                    if(is_prim_id_neq) atomicAdd(&d_J[(b.basis_index+lmn_b)*num_basis + a.basis_index+lmn_a], thread_val);
                }
            }
        }
    }
    return;
}











inline void compute_RI_Direct_Z_kernel(sycl::nd_item<1> item,
real_t* d_Z, const real_t* d_C, const real_t* d_L_inv, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
										    const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, 
											ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
											int64_t num_tasks, int num_basis, 
											const size_t2* d_primitive_shell_pair_indices,
											const double* g_upper_bound_factors, 
											const double* g_auxiliary_upper_bound_factors, 
											const double schwarz_screening_threshold, 
											int num_auxiliary_basis, 
                                            int iter,
											const double* g_boys_grid){
{
    // __shared__ int sh_head_idx[2];

    // __shared__ real_t sh_val[128];


    // 通し番号indexの計算
    const size_t id = item.get_global_linear_id(); 


    if (id >= num_tasks) return;

    const double size_Rmid=1377;

    //使い捨ての中間体R_mid
    double R_mid[3*1377];

    //解を格納する配列R
    double R[2925];

    //thread内で結果を保持するメモリ
    double thread_val=0.0;

    

    const size_t2 abc = index1to2(id, false, shell_s2.count);
    


    // Obtain primitive shells [ab|c]
	const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
	const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
    const size_t primitive_index_c = abc.y + shell_s2.start_index;

	// screening (suzuki)
	if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


    const PrimitiveShell a = g_pshell[primitive_index_a];
    const PrimitiveShell b = g_pshell[primitive_index_b];
    const PrimitiveShell c = g_pshell_aux[primitive_index_c];


    bool is_prim_id_neq = (primitive_index_a != primitive_index_b);


    // suzuki
    // if(!threadIdx.x) {
    //     sh_head_idx[0] = a.basis_index;
    //     sh_head_idx[1] = b.basis_index;
    // }

    
    //使用データを取得，レジスタに書き込み

    //指数部
    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double gamma = c.exponent;
    const double p = alpha+beta;
    const double xi = p*gamma / (p+gamma);

    //係数部
    const double coef_a = a.coefficient;
    const double coef_b = b.coefficient;
    const double coef_c = c.coefficient;

    //座標
    const double pos_A[3] = {a.coordinate.x, a.coordinate.y, a.coordinate.z};
    const double pos_B[3] = {b.coordinate.x, b.coordinate.y, b.coordinate.z};
    const double pos_C[3] = {c.coordinate.x, c.coordinate.y, c.coordinate.z};
    const double pos_P[3] = {(alpha*pos_A[0]+beta*pos_B[0])/(alpha+beta), (alpha*pos_A[1]+beta*pos_B[1])/(alpha+beta), (alpha*pos_A[2]+beta*pos_B[2])/(alpha+beta)};


    //角運動量の総和
    const int orbital_A = a.shell_type;
    const int orbital_B = b.shell_type;
    const int orbital_C = c.shell_type;


    //軌道間距離の二乗
    const double dist = ((pos_P[0]-pos_C[0])*(pos_P[0]-pos_C[0]) + (pos_P[1]-pos_C[1])*(pos_P[1]-pos_C[1]) + (pos_P[2]-pos_C[2])*(pos_P[2]-pos_C[2]));


    const int K = orbital_A + orbital_B + orbital_C;
    
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

    
    
    double Norm_A, Norm_B, Norm_C;
    double Norm;

    int t,u,v;
    int t_max;
    int u_max;
    int v_max;
    int tau_max;
    int nu_max;
    int phi_max;

    // int tid=0;


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


				Norm = Norm_A * Norm_B * Norm_C;
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
				tau_max = l3+1;
				nu_max = m3+1;
				phi_max = n3+1;

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
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[0] - pos_C[0])*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_Rmid)];
								}
								else if(u >= 1){
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[1] - pos_C[1])*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_Rmid)];
								}
								else{
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[2] - pos_C[2])*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_Rmid)];
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
                    
					double my_val = 0.0;
                    // ERI計算部
					double Et, Eu, Ev, Etau, Enu, Ephi;
                    for(int t=0; t<l1+l2+1; t++){
                        Et =  MD_Et_NonRecursion(l1, l2, t, alpha, beta, (pos_A[0]-pos_B[0]));
                        for(int u=0; u<m1+m2+1; u++){
                            Eu = MD_Et_NonRecursion(m1, m2, u, alpha, beta, (pos_A[1]-pos_B[1]));
                            for(int v=0; v<n1+n2+1; v++){
                                Ev = MD_Et_NonRecursion(n1, n2, v, alpha, beta, (pos_A[2]-pos_B[2]));
                                for(int tau=0; tau<l3+1; tau++){
                                    Etau = MD_Et_NonRecursion(l3, 0, tau, gamma, 0.0, 0.0);
                                    for(int nu=0; nu<m3+1; nu++){
                                        Enu = MD_Et_NonRecursion(m3, 0, nu, gamma, 0.0, 0.0);
                                        for(int phi=0; phi<n3+1; phi++){
                                            Ephi = MD_Et_NonRecursion(n3, 0, phi, gamma, 0.0, 0.0);
                                            int k=t+u+v+tau+nu+phi;
                                            my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k,u+nu,v+phi,0,0,0)];
                                            // my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R_GPU_Recursion(0, t+tau, u+nu, v+phi, P, Q, Boys);
                                        }
                                    }
                                }
                            }
                        }
                    }
					thread_val = Norm * my_val*2 * M_PI_2_5 /(p*gamma * sqrt((p+gamma))) * coef_a*coef_b*coef_c 
                               * d_cgto_nomalization_factors[a.basis_index + lmn_a] * d_cgto_nomalization_factors[b.basis_index + lmn_b] * d_auxiliary_cgto_nomalization_factors[c.basis_index + lmn_c];




                    // sh_val[threadIdx.x] = 0.0;
                    // __syncthreads();




                    // Global Memoryへ書き込み
                    // 汎用カーネルでは全要素判定(case1)
                    for(int r=0; r<num_auxiliary_basis; r++){
                                            // 書き込み部
                            // sharedへの集約
                            // sh_val[threadIdx.x] = 0.0;
                            // __syncthreads();

                            // int idx = (((int)a.basis_index - sh_head_idx[0]) >= 0) ? a.basis_index-sh_head_idx[0] : a.basis_index-sh_head_idx[0]+num_basis;

                            // atomicAdd(&sh_val[idx], thread_val * d_C[(b.basis_index + lmn_b)*num_basis + iter] * d_L_inv[r*num_auxiliary_basis + (c.basis_index + lmn_c)]);
                            // __syncthreads();

                            // idx = (threadIdx.x+sh_head_idx[0])%num_basis + lmn_a;

                            // if (sh_val[threadIdx.x] != 0.0) atomicAdd(&d_Z[idx * num_auxiliary_basis + r], sh_val[threadIdx.x]);
                            // __syncthreads();

                            atomicAdd(&d_Z[(a.basis_index+lmn_a) * num_auxiliary_basis + r], thread_val * d_C[(b.basis_index + lmn_b)*num_basis + iter] * d_L_inv[r*num_auxiliary_basis + (c.basis_index + lmn_c)]);
                            if(is_prim_id_neq) atomicAdd(&d_Z[(b.basis_index+lmn_b) * num_auxiliary_basis + r], thread_val * d_C[(a.basis_index + lmn_a)*num_basis + iter] * d_L_inv[r*num_auxiliary_basis + (c.basis_index + lmn_c)]);
                            // d_Z[(a.basis_index+lmn_a) * num_auxiliary_basis + r] += thread_val * d_C[(b.basis_index + lmn_b)*num_basis + iter] * d_L_inv[r*num_auxiliary_basis + (c.basis_index + lmn_c)];
                            // if(is_prim_id_neq) d_Z[(b.basis_index+lmn_b) * num_auxiliary_basis + r] += thread_val * d_C[(a.basis_index + lmn_a)*num_basis + iter] * d_L_inv[r*num_auxiliary_basis + (c.basis_index + lmn_c)];
                    }
                }
            }
        }
    }
    return;
}



inline void compute_RI_Direct_W_kernel(sycl::nd_item<1> item,
real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
										    const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, 
											ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
											int64_t num_tasks, int num_basis, 
											const size_t2* d_primitive_shell_pair_indices,
											const double* g_upper_bound_factors, 
                                            // const double* g_upper_bound_factors_unsorted, 
											const double* g_auxiliary_upper_bound_factors, 
											const double schwarz_screening_threshold, 
											int num_auxiliary_basis, 
                                            int iter,
											const double* g_boys_grid){
{
    // __shared__ int sh_head_idx[2];

    // __shared__ real_t sh_val[128];


    // 通し番号indexの計算
    const size_t id = item.get_global_linear_id(); 


    if (id >= num_tasks) return;

    const double size_Rmid=1377;

    //使い捨ての中間体R_mid
    double R_mid[3*1377];

    //解を格納する配列R
    double R[2925];

    //thread内で結果を保持するメモリ
    double thread_val=0.0;

    

    const size_t2 abc = index1to2(id, false, shell_s2.count);
    


    // Obtain primitive shells [ab|c]
	const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x;
	const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y;
    const size_t primitive_index_c = abc.y + shell_s2.start_index;

    bool is_prim_id_neq = (primitive_index_a != primitive_index_b);



    const PrimitiveShell a = g_pshell[primitive_index_a];
    const PrimitiveShell b = g_pshell[primitive_index_b];
    const PrimitiveShell c = g_pshell_aux[primitive_index_c];






    
    //使用データを取得，レジスタに書き込み

    //指数部
    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double gamma = c.exponent;
    const double p = alpha+beta;
    const double xi = p*gamma / (p+gamma);

    //係数部
    const double coef_a = a.coefficient;
    const double coef_b = b.coefficient;
    const double coef_c = c.coefficient;

    //座標
    const double pos_A[3] = {a.coordinate.x, a.coordinate.y, a.coordinate.z};
    const double pos_B[3] = {b.coordinate.x, b.coordinate.y, b.coordinate.z};
    const double pos_C[3] = {c.coordinate.x, c.coordinate.y, c.coordinate.z};
    const double pos_P[3] = {(alpha*pos_A[0]+beta*pos_B[0])/(alpha+beta), (alpha*pos_A[1]+beta*pos_B[1])/(alpha+beta), (alpha*pos_A[2]+beta*pos_B[2])/(alpha+beta)};


    //角運動量の総和
    const int orbital_A = a.shell_type;
    const int orbital_B = b.shell_type;
    const int orbital_C = c.shell_type;


    real_t max_coefficient_value = 0.0;
    real_t tmp;
    for(int lmn_b=0; lmn_b<comb_max(orbital_B); lmn_b++){   
        tmp = fabs(d_C_diff_vector[(b.basis_index + lmn_b)]);
        if(max_coefficient_value < tmp) max_coefficient_value = tmp;
    }

    if (is_prim_id_neq) {
        for(int lmn_a=0; lmn_a<comb_max(orbital_A); lmn_a++){
            tmp = fabs(d_C_diff_vector[(a.basis_index + lmn_a)]);
            if(max_coefficient_value < tmp) max_coefficient_value = tmp;
        }
    }
    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * max_coefficient_value < schwarz_screening_threshold) return;



	// screening (suzuki)
	// if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;



    //軌道間距離の二乗
    const double dist = ((pos_P[0]-pos_C[0])*(pos_P[0]-pos_C[0]) + (pos_P[1]-pos_C[1])*(pos_P[1]-pos_C[1]) + (pos_P[2]-pos_C[2])*(pos_P[2]-pos_C[2]));


    const int K = orbital_A + orbital_B + orbital_C;
    
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

    
    
    double Norm_A, Norm_B, Norm_C;
    double Norm;

    int t,u,v;
    int t_max;
    int u_max;
    int v_max;
    int tau_max;
    int nu_max;
    int phi_max;

    // int tid=0;


    // 方位量子数l,m,nの値をループ変数から導出

    for(int lmn_a=0; lmn_a<comb_max(orbital_A); lmn_a++){
        int l1=loop_to_ang[orbital_A][lmn_a][0]; int m1=loop_to_ang[orbital_A][lmn_a][1]; int n1=loop_to_ang[orbital_A][lmn_a][2];
        Norm_A = calcNorm(alpha, l1, m1, n1);

        for(int lmn_b=0; lmn_b<comb_max(orbital_B); lmn_b++){                  
            int l2=loop_to_ang[orbital_B][lmn_b][0]; int m2=loop_to_ang[orbital_B][lmn_b][1]; int n2=loop_to_ang[orbital_B][lmn_b][2];
            Norm_B = calcNorm(beta, l2, m2, n2);


            if ((g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * fabs(d_C_diff_vector[(b.basis_index + lmn_b)]) < schwarz_screening_threshold) && (!is_prim_id_neq || g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * fabs(d_C_diff_vector[(a.basis_index + lmn_a)]) < schwarz_screening_threshold)) continue;
            // if ((g_upper_bound_factors_unsorted[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) && (!is_prim_id_neq || g_upper_bound_factors_unsorted[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c]  < schwarz_screening_threshold)) continue;

            // if (is_prim_id_neq && g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] * fabs(d_C_diff_vector[(a.basis_index + lmn_a)]) < schwarz_screening_threshold) continue;


            for(int lmn_c=0; lmn_c<comb_max(orbital_C); lmn_c++){
                int l3=loop_to_ang[orbital_C][lmn_c][0]; int m3=loop_to_ang[orbital_C][lmn_c][1]; int n3=loop_to_ang[orbital_C][lmn_c][2];
                Norm_C = calcNorm(gamma, l3, m3, n3);




				Norm = Norm_A * Norm_B * Norm_C;
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
				tau_max = l3+1;
				nu_max = m3+1;
				phi_max = n3+1;

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
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[0] - pos_C[0])*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_Rmid)];
								}
								else if(u >= 1){
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[1] - pos_C[1])*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_Rmid)];
								}
								else{
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[2] - pos_C[2])*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_Rmid)];
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
                    
					double my_val = 0.0;
                    // ERI計算部
					double Et, Eu, Ev, Etau, Enu, Ephi;
                    for(int t=0; t<l1+l2+1; t++){
                        Et =  MD_Et_NonRecursion(l1, l2, t, alpha, beta, (pos_A[0]-pos_B[0]));
                        for(int u=0; u<m1+m2+1; u++){
                            Eu = MD_Et_NonRecursion(m1, m2, u, alpha, beta, (pos_A[1]-pos_B[1]));
                            for(int v=0; v<n1+n2+1; v++){
                                Ev = MD_Et_NonRecursion(n1, n2, v, alpha, beta, (pos_A[2]-pos_B[2]));
                                for(int tau=0; tau<l3+1; tau++){
                                    Etau = MD_Et_NonRecursion(l3, 0, tau, gamma, 0.0, 0.0);
                                    for(int nu=0; nu<m3+1; nu++){
                                        Enu = MD_Et_NonRecursion(m3, 0, nu, gamma, 0.0, 0.0);
                                        for(int phi=0; phi<n3+1; phi++){
                                            Ephi = MD_Et_NonRecursion(n3, 0, phi, gamma, 0.0, 0.0);
                                            int k=t+u+v+tau+nu+phi;
                                            my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k,u+nu,v+phi,0,0,0)];
                                            // my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R_GPU_Recursion(0, t+tau, u+nu, v+phi, P, Q, Boys);
                                        }
                                    }
                                }
                            }
                        }
                    }
					thread_val = Norm * my_val*2 * M_PI_2_5 /(p*gamma * sqrt((p+gamma))) * coef_a*coef_b*coef_c 
                               * d_cgto_nomalization_factors[a.basis_index + lmn_a] * d_cgto_nomalization_factors[b.basis_index + lmn_b] * d_auxiliary_cgto_nomalization_factors[c.basis_index + lmn_c];



                    // Global Memoryへ書き込み
                    // 汎用カーネルでは全要素判定(case1)
                    atomicAdd(&d_W_diff[(a.basis_index+lmn_a) * num_auxiliary_basis + c.basis_index+lmn_c], thread_val * d_C_diff_vector[(b.basis_index + lmn_b)]);


                    if(is_prim_id_neq) {
                        atomicAdd(&d_W_diff[(b.basis_index+lmn_b) * num_auxiliary_basis + c.basis_index+lmn_c], thread_val * d_C_diff_vector[(a.basis_index + lmn_a)]);
                    }

                    
                }
            }
        }
    }
    return;
}

void launch_c_kernel(sycl::nd_item<1> item,
    int s0, int s1, int s2,
    real_t* d_c, const real_t* d_density_matrix, const PrimitiveShell* g_shell,
    const PrimitiveShell* g_shell_aux, const real_t* d_cgto_nomalization_factors,
    const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2,
    int64_t num_tasks, int num_basis,
    const size_t2* d_primitive_shell_pair_indices,
    const double* g_upper_bound_factors,
    const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold,
    int num_auxiliary_basis, const double* d_boys_grid)
{
//        direct_ri_c_J_kernel_t c_kernel;

        if(s0==0 && s1==0 && s2==0) compute_RI_Direct_c_kernel_sss(item, 
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
//            &d_primitive_shell_pair_indices[shell_pair_type_infos[calcIdx_triangular_(s0, s1, shell_type_count)].start_index],
            d_primitive_shell_pair_indices,
//            &d_schwarz_upper_bound_factors[shell_pair_type_infos[calcIdx_triangular_(s0, s1, shell_type_count)].start_index],
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==0 && s1==0 && s2==1) compute_RI_Direct_c_kernel_ssp(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==0 && s1==0 && s2==2) compute_RI_Direct_c_kernel_ssd(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==0 && s1==0 && s2==3) compute_RI_Direct_c_kernel_ssf(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==0 && s1==1 && s2==0) compute_RI_Direct_c_kernel_sps(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==0 && s1==1 && s2==1) compute_RI_Direct_c_kernel_spp(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==0 && s1==1 && s2==2) compute_RI_Direct_c_kernel_spd(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==0 && s1==1 && s2==3) compute_RI_Direct_c_kernel_spf(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==1 && s1==1 && s2==0) compute_RI_Direct_c_kernel_pps(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==1 && s1==1 && s2==1) compute_RI_Direct_c_kernel_ppp(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==1 && s1==1 && s2==2) compute_RI_Direct_c_kernel_ppd(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==1 && s1==1 && s2==3) compute_RI_Direct_c_kernel_ppf(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        #if defined(COMPUTE_D_BASIS)
        else if(s0==0 && s1==2 && s2==0) compute_RI_Direct_c_kernel_sds(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==0 && s1==2 && s2==1) compute_RI_Direct_c_kernel_sdp(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==0 && s1==2 && s2==2) compute_RI_Direct_c_kernel_sdd(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==0 && s1==2 && s2==3) compute_RI_Direct_c_kernel_sdf(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==1 && s1==2 && s2==0) compute_RI_Direct_c_kernel_pds(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==1 && s1==2 && s2==1) compute_RI_Direct_c_kernel_pdp(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==1 && s1==2 && s2==2) compute_RI_Direct_c_kernel_pdd(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==1 && s1==2 && s2==3) compute_RI_Direct_c_kernel_pdf(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==2 && s1==2 && s2==0) compute_RI_Direct_c_kernel_dds(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==2 && s1==2 && s2==1) compute_RI_Direct_c_kernel_ddp(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==2 && s1==2 && s2==2) compute_RI_Direct_c_kernel_ddd(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        else if(s0==2 && s1==2 && s2==3) compute_RI_Direct_c_kernel_ddf(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
        #endif
        else compute_RI_Direct_c_kernel(item,
            d_c, d_density_matrix, g_shell, g_shell_aux, d_cgto_nomalization_factors,
            d_auxiliary_cgto_nomalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, d_boys_grid);
/*
3780         else if(s0==0 && s1==0 && s2==1) c_kernel = compute_RI_Direct_c_kernel_ssp;
3781         else if(s0==0 && s1==0 && s2==2) c_kernel = compute_RI_Direct_c_kernel_ssd;
3782         else if(s0==0 && s1==0 && s2==3) c_kernel = compute_RI_Direct_c_kernel_ssf;
3783         else if(s0==0 && s1==1 && s2==0) c_kernel = compute_RI_Direct_c_kernel_sps;
3784         else if(s0==0 && s1==1 && s2==1) c_kernel = compute_RI_Direct_c_kernel_spp;
3785         else if(s0==0 && s1==1 && s2==2) c_kernel = compute_RI_Direct_c_kernel_spd;
3786         else if(s0==0 && s1==1 && s2==3) c_kernel = compute_RI_Direct_c_kernel_spf;
3787         else if(s0==1 && s1==1 && s2==0) c_kernel = compute_RI_Direct_c_kernel_pps;
3788         else if(s0==1 && s1==1 && s2==1) c_kernel = compute_RI_Direct_c_kernel_ppp;
3789         else if(s0==1 && s1==1 && s2==2) c_kernel = compute_RI_Direct_c_kernel_ppd;
3790         else if(s0==1 && s1==1 && s2==3) c_kernel = compute_RI_Direct_c_kernel_ppf;
3791         #if defined(COMPUTE_D_BASIS)
3792         else if(s0==0 && s1==2 && s2==0) c_kernel = compute_RI_Direct_c_kernel_sds;
3793         else if(s0==0 && s1==2 && s2==1) c_kernel = compute_RI_Direct_c_kernel_sdp;
3794         else if(s0==0 && s1==2 && s2==2) c_kernel = compute_RI_Direct_c_kernel_sdd;
3795         else if(s0==0 && s1==2 && s2==3) c_kernel = compute_RI_Direct_c_kernel_sdf;
3796         else if(s0==1 && s1==2 && s2==0) c_kernel = compute_RI_Direct_c_kernel_pds;
3797         else if(s0==1 && s1==2 && s2==1) c_kernel = compute_RI_Direct_c_kernel_pdp;
3798         else if(s0==1 && s1==2 && s2==2) c_kernel = compute_RI_Direct_c_kernel_pdd;
3799         else if(s0==1 && s1==2 && s2==3) c_kernel = compute_RI_Direct_c_kernel_pdf;
3800         else if(s0==2 && s1==2 && s2==0) c_kernel = compute_RI_Direct_c_kernel_dds;
3801         else if(s0==2 && s1==2 && s2==1) c_kernel = compute_RI_Direct_c_kernel_ddp;
3802         else if(s0==2 && s1==2 && s2==2) c_kernel = compute_RI_Direct_c_kernel_ddd;
3803         else if(s0==2 && s1==2 && s2==3) c_kernel = compute_RI_Direct_c_kernel_ddf;
3804         #endif
3805         else c_kernel = compute_RI_Direct_c_kernel;
3806 
3807         // c_kernel = compute_RI_Direct_c_kernel;
3808 
3809 
3810         c_kernel<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_c, d_density_matrix, d_primitive_shells, d_auxiliary_pri     mitive_shells,                                                                                                                         
*/

}


void launch_J_kernel(sycl::nd_item<1> item,
    int s0, int s1, int s2,
    real_t* d_J, const real_t* d_t, const PrimitiveShell* g_pshell,
    const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors,
    const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2,
    int64_t num_tasks, int num_basis,
    const size_t2* d_primitive_shell_pair_indices,
    const double* g_upper_bound_factors,
    const double* g_auxiliary_upper_bound_factors, const double schwarz_screening_threshold,
    int num_auxiliary_basis, const double* g_boys_grid)
{
        if(s0==0 && s1==0 && s2==0) compute_RI_Direct_J_kernel_sss(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==0 && s1==0 && s2==1) compute_RI_Direct_J_kernel_ssp(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==0 && s1==0 && s2==2) compute_RI_Direct_J_kernel_ssd(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==0 && s1==0 && s2==3) compute_RI_Direct_J_kernel_ssf(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==0 && s1==1 && s2==0) compute_RI_Direct_J_kernel_sps(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==0 && s1==1 && s2==1) compute_RI_Direct_J_kernel_spp(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==0 && s1==1 && s2==2) compute_RI_Direct_J_kernel_spd(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==0 && s1==1 && s2==3) compute_RI_Direct_J_kernel_spf(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==1 && s1==1 && s2==0) compute_RI_Direct_J_kernel_pps(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==1 && s1==1 && s2==1) compute_RI_Direct_J_kernel_ppp(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==1 && s1==1 && s2==2) compute_RI_Direct_J_kernel_ppd(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==1 && s1==1 && s2==3) compute_RI_Direct_J_kernel_ppf(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        #if defined(COMPUTE_D_BASIS)
        else if(s0==0 && s1==2 && s2==0) compute_RI_Direct_J_kernel_sds(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==0 && s1==2 && s2==1) compute_RI_Direct_J_kernel_sdp(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==0 && s1==2 && s2==2) compute_RI_Direct_J_kernel_sdd(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==0 && s1==2 && s2==3) compute_RI_Direct_J_kernel_sdf(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==1 && s1==2 && s2==0) compute_RI_Direct_J_kernel_pds(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==1 && s1==2 && s2==1) compute_RI_Direct_J_kernel_pdp(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==1 && s1==2 && s2==2) compute_RI_Direct_J_kernel_pdd(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==1 && s1==2 && s2==3) compute_RI_Direct_J_kernel_pdf(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==2 && s1==2 && s2==0) compute_RI_Direct_J_kernel_dds(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==2 && s1==2 && s2==1) compute_RI_Direct_J_kernel_ddp(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==2 && s1==2 && s2==2) compute_RI_Direct_J_kernel_ddd(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        else if(s0==2 && s1==2 && s2==3) compute_RI_Direct_J_kernel_ddf(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);
        #endif
        else compute_RI_Direct_J_kernel(item,
            d_J, d_t, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2,
            num_tasks, num_basis,
            d_primitive_shell_pair_indices,
            g_upper_bound_factors,
            g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, g_boys_grid);


/*
3937         else if(s0==0 && s1==0 && s2==1) J_kernel = compute_RI_Direct_J_kernel_ssp;
3938         else if(s0==0 && s1==0 && s2==2) J_kernel = compute_RI_Direct_J_kernel_ssd;
3939         else if(s0==0 && s1==0 && s2==3) J_kernel = compute_RI_Direct_J_kernel_ssf;
3940         else if(s0==0 && s1==1 && s2==0) J_kernel = compute_RI_Direct_J_kernel_sps;
3941         else if(s0==0 && s1==1 && s2==1) J_kernel = compute_RI_Direct_J_kernel_spp;
3942         else if(s0==0 && s1==1 && s2==2) J_kernel = compute_RI_Direct_J_kernel_spd;
3943         else if(s0==0 && s1==1 && s2==3) J_kernel = compute_RI_Direct_J_kernel_spf;
3944         else if(s0==1 && s1==1 && s2==0) J_kernel = compute_RI_Direct_J_kernel_pps;
3945         else if(s0==1 && s1==1 && s2==1) J_kernel = compute_RI_Direct_J_kernel_ppp;
3946         else if(s0==1 && s1==1 && s2==2) J_kernel = compute_RI_Direct_J_kernel_ppd;
3947         else if(s0==1 && s1==1 && s2==3) J_kernel = compute_RI_Direct_J_kernel_ppf;
3948         #if defined(COMPUTE_D_BASIS)
3949         else if(s0==0 && s1==2 && s2==0) J_kernel = compute_RI_Direct_J_kernel_sds;
3950         else if(s0==0 && s1==2 && s2==1) J_kernel = compute_RI_Direct_J_kernel_sdp;
3951         else if(s0==0 && s1==2 && s2==2) J_kernel = compute_RI_Direct_J_kernel_sdd;
3952         else if(s0==0 && s1==2 && s2==3) J_kernel = compute_RI_Direct_J_kernel_sdf;
3953         else if(s0==1 && s1==2 && s2==0) J_kernel = compute_RI_Direct_J_kernel_pds;
3954         else if(s0==1 && s1==2 && s2==1) J_kernel = compute_RI_Direct_J_kernel_pdp;
3955         else if(s0==1 && s1==2 && s2==2) J_kernel = compute_RI_Direct_J_kernel_pdd;
3956         else if(s0==1 && s1==2 && s2==3) J_kernel = compute_RI_Direct_J_kernel_pdf;
3957         else if(s0==2 && s1==2 && s2==0) J_kernel = compute_RI_Direct_J_kernel_dds;
3958         else if(s0==2 && s1==2 && s2==1) J_kernel = compute_RI_Direct_J_kernel_ddp;
3959         else if(s0==2 && s1==2 && s2==2) J_kernel = compute_RI_Direct_J_kernel_ddd;
3960         else if(s0==2 && s1==2 && s2==3) J_kernel = compute_RI_Direct_J_kernel_ddf;
3961         #endif
3962         else J_kernel = compute_RI_Direct_J_kernel;
3963 
3964         // J_kernel = compute_RI_Direct_J_kernel;
3965 
*/
}


void launch_W_kernel(sycl::nd_item<1> item,
    int s0, int s1, int s2,
    real_t* d_W_diff, real_t* d_C_diff_vector, const PrimitiveShell* g_pshell,
    const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_normalization_factors,
    const real_t* d_auxiliary_cgto_normalization_factors,
    ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
	int64_t num_tasks, int num_basis, 
    const size_t2* d_primitive_shell_pair_indices,
	const double* g_upper_bound_factors, 
	const double* g_auxiliary_upper_bound_factors, 
	const double schwarz_screening_threshold, 
	int num_auxiliary_basis, 
    int iter,
	const double* g_boys_grid){

            if(s0==0 && s1==0 && s2==0) compute_RI_Direct_W_kernel_sss(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==0 && s1==0 && s2==1) compute_RI_Direct_W_kernel_ssp(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==0 && s1==0 && s2==1) compute_RI_Direct_W_kernel_ssp(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==0 && s1==0 && s2==2) compute_RI_Direct_W_kernel_ssd(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==0 && s1==0 && s2==3) compute_RI_Direct_W_kernel_ssf(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==0 && s1==1 && s2==0) compute_RI_Direct_W_kernel_sps(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==0 && s1==1 && s2==1) compute_RI_Direct_W_kernel_spp(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==0 && s1==1 && s2==2) compute_RI_Direct_W_kernel_spd(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==0 && s1==1 && s2==3) compute_RI_Direct_W_kernel_spf(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==1 && s1==1 && s2==0) compute_RI_Direct_W_kernel_pps(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==1 && s1==1 && s2==1) compute_RI_Direct_W_kernel_ppp(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==1 && s1==1 && s2==2) compute_RI_Direct_W_kernel_ppd(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==1 && s1==1 && s2==3) compute_RI_Direct_W_kernel_ppf(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            #if defined(COMPUTE_D_BASIS)
            else if(s0==0 && s1==2 && s2==0) compute_RI_Direct_W_kernel_sds(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==0 && s1==2 && s2==1) compute_RI_Direct_W_kernel_sdp(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==0 && s1==2 && s2==2) compute_RI_Direct_W_kernel_sdd(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==0 && s1==2 && s2==3) compute_RI_Direct_W_kernel_sdf(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==1 && s1==2 && s2==0) compute_RI_Direct_W_kernel_pds(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==1 && s1==2 && s2==1) compute_RI_Direct_W_kernel_pdp(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==1 && s1==2 && s2==2) compute_RI_Direct_W_kernel_pdd(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==1 && s1==2 && s2==3) compute_RI_Direct_W_kernel_pdf(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==2 && s1==2 && s2==0) compute_RI_Direct_W_kernel_dds(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==2 && s1==2 && s2==1) compute_RI_Direct_W_kernel_ddp(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==2 && s1==2 && s2==2) compute_RI_Direct_W_kernel_ddd(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            else if(s0==2 && s1==2 && s2==3) compute_RI_Direct_W_kernel_ddf(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
            #endif
            else compute_RI_Direct_W_kernel(item,
            d_W_diff, d_C_diff_vector, g_pshell, g_pshell_aux, d_cgto_normalization_factors,
            d_auxiliary_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, 
        	num_tasks, num_basis, d_primitive_shell_pair_indices, g_upper_bound_factors, 
        	g_auxiliary_upper_bound_factors, schwarz_screening_threshold, num_auxiliary_basis, 
            iter, g_boys_grid);
/*
            else if(s0==0 && s1==0 && s2==1) W_kernel = compute_RI_Direct_W_kernel_ssp;
4331             else if(s0==0 && s1==0 && s2==1) W_kernel = compute_RI_Direct_W_kernel_ssp;
4332             else if(s0==0 && s1==0 && s2==2) W_kernel = compute_RI_Direct_W_kernel_ssd;
4333             else if(s0==0 && s1==0 && s2==3) W_kernel = compute_RI_Direct_W_kernel_ssf;
4334             else if(s0==0 && s1==1 && s2==0) W_kernel = compute_RI_Direct_W_kernel_sps;
4335             else if(s0==0 && s1==1 && s2==1) W_kernel = compute_RI_Direct_W_kernel_spp;
4336             else if(s0==0 && s1==1 && s2==2) W_kernel = compute_RI_Direct_W_kernel_spd;
4337             else if(s0==0 && s1==1 && s2==3) W_kernel = compute_RI_Direct_W_kernel_spf;
4338             else if(s0==1 && s1==1 && s2==0) W_kernel = compute_RI_Direct_W_kernel_pps;
4339             else if(s0==1 && s1==1 && s2==1) W_kernel = compute_RI_Direct_W_kernel_ppp;
4340             else if(s0==1 && s1==1 && s2==2) W_kernel = compute_RI_Direct_W_kernel_ppd;
4341             else if(s0==1 && s1==1 && s2==3) W_kernel = compute_RI_Direct_W_kernel_ppf;
4342             #if defined(COMPUTE_D_BASIS)
4343             else if(s0==0 && s1==2 && s2==0) W_kernel = compute_RI_Direct_W_kernel_sds;
4344             else if(s0==0 && s1==2 && s2==1) W_kernel = compute_RI_Direct_W_kernel_sdp;
4345             else if(s0==0 && s1==2 && s2==2) W_kernel = compute_RI_Direct_W_kernel_sdd;
4346             else if(s0==0 && s1==2 && s2==3) W_kernel = compute_RI_Direct_W_kernel_sdf;
4347             else if(s0==1 && s1==2 && s2==0) W_kernel = compute_RI_Direct_W_kernel_pds;
4348             else if(s0==1 && s1==2 && s2==1) W_kernel = compute_RI_Direct_W_kernel_pdp;
4349             else if(s0==1 && s1==2 && s2==2) W_kernel = compute_RI_Direct_W_kernel_pdd;
4350             else if(s0==1 && s1==2 && s2==3) W_kernel = compute_RI_Direct_W_kernel_pdf;
4351             else if(s0==2 && s1==2 && s2==0) W_kernel = compute_RI_Direct_W_kernel_dds;
4352             else if(s0==2 && s1==2 && s2==1) W_kernel = compute_RI_Direct_W_kernel_ddp;
4353             else if(s0==2 && s1==2 && s2==2) W_kernel = compute_RI_Direct_W_kernel_ddd;
4354             else if(s0==2 && s1==2 && s2==3) W_kernel = compute_RI_Direct_W_kernel_ddf;
4355             #endif
4356             else W_kernel = compute_RI_Direct_W_kernel;
4357             // W_kernel = compute_RI_Direct_W_kernel;
*/
}

#undef atomicAdd

} // namespace gansu::gpu

#endif
