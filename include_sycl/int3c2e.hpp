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
#include <dpct/dpct.hpp>
#include "int2c2e.hpp"
#include "boys.hpp"
 #include "types.hpp"
 #include "utils_cuda.hpp"
 #include "parameters.h"
 #include "compile_flag.hpp"

 namespace gansu::gpu{

    __inline__ double calcNormsWOFact2_3center(double alpha, double beta, double gamma, int sum_ang1,  int sum_ang2,  int sum_ang3){
        return dpct::pow(2.0, sum_ang1 + sum_ang2 + sum_ang3)
               // * pow(factorial2_gpu(2.0*i1-1.0)*factorial2_gpu(2.0*j1-1.0)*factorial2_gpu(2.0*k1-1.0)*factorial2_gpu(2.0*l1-1.0)*factorial2_gpu(2.0*m1-1.0)*factorial2_gpu(2.0*n1-1.0)*factorial2_gpu(2.0*i2-1.0)*factorial2_gpu(2.0*k2-1.0)*factorial2_gpu(2.0*m2-1.0), -0.5)
               * dpct::pow(2.0 / M_PI, 2.25) *
               dpct::pow(alpha, (2.0 * (sum_ang1) + 3.0) / 4.0) *
               dpct::pow(beta, (2.0 * (sum_ang2) + 3.0) / 4.0) *
               dpct::pow(gamma, (2.0 * (sum_ang3) + 3.0) / 4.0);
    }
    

    __inline__ void addToResult_3center(double res, double *g_result, int p, int q, int r, int nCGTO, int nAux, bool is_prim_id_not_equal, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors){
        res *= d_cgto_nomalization_factors[p] * d_cgto_nomalization_factors[q] * d_auxiliary_cgto_nomalization_factors[r];
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &g_result[r * nCGTO * nCGTO + p * nCGTO + q], res);
        if (is_prim_id_not_equal)
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &g_result[r * nCGTO * nCGTO + q * nCGTO + p], res);
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
void calc_sss_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {

        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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

                #include "./integral_RI/int3c2e/orig_sss.txt"
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
void calc_ssp_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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

                #include "./integral_RI/int3c2e/orig_ssp.txt"
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
void calc_ssd_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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

                #include "./integral_RI/int3c2e/orig_ssd.txt"
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
void calc_ssf_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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

                #include "./integral_RI/int3c2e/orig_ssf.txt"
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
void calc_sps_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_sps.txt"
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
void calc_spp_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_spp.txt"
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
void calc_spd_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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

                #include "./integral_RI/int3c2e/orig_spd.txt"
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
void calc_spf_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_spf.txt"
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
void calc_pps_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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

                #include "./integral_RI/int3c2e/orig_pps.txt"
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
void calc_ppp_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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

                #include "./integral_RI/int3c2e/orig_ppp.txt"
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
void calc_ppd_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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

                #include "./integral_RI/int3c2e/orig_ppd.txt"
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
void calc_ppf_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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

                #include "./integral_RI/int3c2e/orig_ppf.txt"
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
void calc_sds_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_sds.txt"
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
void calc_sdp_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_sdp.txt"
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
void calc_sdd_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_sdd.txt"
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
void calc_sdf_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_sdf.txt"
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
void calc_pds_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_pds.txt"
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
void calc_pdp_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_pdp.txt"
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
void calc_pdd_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_pdd.txt"
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
void calc_pdf_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_pdf.txt"
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
void calc_dds_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_dds.txt"
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
void calc_ddp_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_ddp.txt"
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
void calc_ddd_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_ddd.txt"
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
void calc_ddf_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_ddf.txt"
        }
}

#else
/* (dd|f) */
inline
void calc_ddf_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_ddd_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_ddp_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_dds_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_pdf_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_pdd_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_pdp_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_pds_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_sdf_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_sdd_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_sdp_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_sds_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
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
void calc_sdg_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_sdg.txt"
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
void calc_pdg_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_pdg.txt"
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
void calc_ddg_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_ddg.txt"
        }
}

#else
inline
void calc_sdg_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_ddg_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_pdg_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
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
void calc_ssg_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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

                #include "./integral_RI/int3c2e/orig_ssg.txt"
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
void calc_spg_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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
                #include "./integral_RI/int3c2e/orig_spg.txt"
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
void calc_ppg_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        uint64_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

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

                #include "./integral_RI/int3c2e/orig_ppg.txt"
        }
}

#else
/* (ss|g) */
inline
void calc_ssg_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_spg_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}

inline
void calc_ppg_gpu(real_t *g_result, const PrimitiveShell *g_pshell,
                  const PrimitiveShell *g_pshell_aux,
                  const real_t *d_cgto_nomalization_factors,
                  const real_t *d_auxiliary_cgto_nomalization_factors,
                  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                  ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                  const size_t2 *d_primitive_shell_pair_indices,
                  const double *g_upper_bound_factors,
                  const double *g_auxiliary_upper_bound_factors,
                  const double schwarz_screening_threshold,
                  int num_auxiliary_basis, const double *g_boys_grid) {}
#endif





















  

    /* (dd|g) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_ddg_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (dd|f) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_ddf_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (dd|d) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_ddd_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (dd|p) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_ddp_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (dd|s) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_dds_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (pd|g) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_pdg_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (pd|f) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_pdf_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (pd|d) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_pdd_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (pd|p) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_pdp_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (pd|s) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_pds_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (pp|g) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_ppg_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (pp|f) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_ppf_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (pp|d) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_ppd_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (pp|p) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_ppp_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (pp|s) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_pps_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (sd|g) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_sdg_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (sd|f) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_sdf_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (sd|d) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_sdd_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (sd|p) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_sdp_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (sd|s) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_sds_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (sp|g) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_spg_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (sp|f) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_spf_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (sp|d) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_spd_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (sp|p) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_spp_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (sp|s) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_sps_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (ss|g) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_ssg_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (ss|f) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_ssf_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (ss|d) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_ssd_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (ss|p) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_ssp_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

    /* (ss|s) */
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void calc_sss_gpu_wrapper(real_t *g_result, const PrimitiveShell *g_pshell,
                          const PrimitiveShell *g_pshell_aux,
                          const real_t *d_cgto_nomalization_factors,
                          const real_t *d_auxiliary_cgto_nomalization_factors,
                          ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                          ShellTypeInfo shell_s2, int64_t num_tasks,
                          int num_basis,
                          const size_t2 *d_primitive_shell_pair_indices,
                          const double *g_upper_bound_factors,
                          const double *g_auxiliary_upper_bound_factors,
                          const double schwarz_screening_threshold,
                          int num_auxiliary_basis, const double *g_boys_grid);

/*
DPCT1110:96: The total declared local variable size in device function
MD_int3c2e_1T1SP exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline
void MD_int3c2e_1T1SP(real_t *g_result, const PrimitiveShell *g_pshell,
                      const PrimitiveShell *g_pshell_aux,
                      const real_t *d_cgto_nomalization_factors,
                      const real_t *d_auxiliary_cgto_nomalization_factors,
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
auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
{
    // 通し番号indexの計算
    const size_t id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                      item_ct1.get_local_id(2);

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
    //d以上はconstant配列のloop_to_ang_RIを参照



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
        int l1=loop_to_ang_RI[orbital_A][lmn_a][0]; int m1=loop_to_ang_RI[orbital_A][lmn_a][1]; int n1=loop_to_ang_RI[orbital_A][lmn_a][2];
        Norm_A = calcNorm(alpha, l1, m1, n1);

        for(int lmn_b=0; lmn_b<comb_max(orbital_B); lmn_b++){
            int l2=loop_to_ang_RI[orbital_B][lmn_b][0]; int m2=loop_to_ang_RI[orbital_B][lmn_b][1]; int n2=loop_to_ang_RI[orbital_B][lmn_b][2];
            Norm_B = calcNorm(beta, l2, m2, n2);


            for(int lmn_c=0; lmn_c<comb_max(orbital_C); lmn_c++){
                int l3=loop_to_ang_RI[orbital_C][lmn_c][0]; int m3=loop_to_ang_RI[orbital_C][lmn_c][1]; int n3=loop_to_ang_RI[orbital_C][lmn_c][2];
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
                                                d_cgto_nomalization_factors, d_auxiliary_cgto_nomalization_factors
                    );
                }
            }
        }
    }
    return;
}

inline void launch_3center_kernel(int a, int b, int c, real_t* args, const PrimitiveShell* shell1, const PrimitiveShell* shell2, const real_t* param1, const real_t* param2, ShellTypeInfo info1, ShellTypeInfo info2, ShellTypeInfo info3, int64_t var1, int var2, const size_t2* dp_ind, const double* g_u, const double* g_a, const double s_s_th,
int var3, const double* param3) {
#if !defined(COMPUTE_D_BASIS)
    if (a >= 2 || b >= 2) {
        MD_int3c2e_1T1SP(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
        dp_ind, g_u, g_a, s_s_th, var3, param3);
        return;
    }
#endif
#if !defined(COMPUTE_G_AUX)
    if (c >= 4) {
        MD_int3c2e_1T1SP(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
        dp_ind, g_u, g_a, s_s_th, var3, param3);
        return;
    }
#endif

    if (a < N_ORBITAL_TYPE_BASIS && b < N_ORBITAL_TYPE_BASIS && c < N_ORBITAL_TYPE_AUX) {
        int idx = calcIdx_triangular(a, b, N_ORBITAL_TYPE_BASIS) * N_ORBITAL_TYPE_AUX + c;
        switch (idx) {
            case 0: calc_sss_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 1: calc_ssp_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 2: calc_ssd_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 3: calc_ssf_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 4: calc_ssg_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 5: calc_sps_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 6: calc_spp_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 7: calc_spd_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 8: calc_spf_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 9: calc_spg_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 10: calc_sds_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 11: calc_sdp_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 12: calc_sdd_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 13: calc_sdf_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 14: calc_sdg_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 15: calc_pps_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 16: calc_ppp_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 17: calc_ppd_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 18: calc_ppf_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 19: calc_ppg_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 20: calc_pds_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 21: calc_pdp_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 22: calc_pdd_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 23: calc_pdf_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 24: calc_pdg_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 25: calc_dds_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 26: calc_ddp_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 27: calc_ddd_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 28: calc_ddf_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            case 29: calc_ddg_gpu(args, shell1, shell2, param1, param2, info1, info2, info3, var1, var2,
                    dp_ind, g_u, g_a, s_s_th, var3, param3); break;
            default: ;
//              throw std::runtime_error("Invalid kernel index.\n");
        }
//    } else {
//        throw std::runtime_error("Invalid call for 3center eri.\n");
    }
}

  
 } // namespace gansu::gpu
  
  #endif
