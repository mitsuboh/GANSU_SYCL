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



#ifndef GRADIENTS
#define GRADIENTS

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "int2e.hpp"
#include "Et_functions.hpp"
#include "Et_grad_functions.hpp"
#include "parameters.h"

// 1電子部分・・・i軌道の場合，13(6+6+1)
#ifndef boys_one_size
    #define boys_one_size 13
#endif

#ifndef size_one_R
    #define size_one_R 455
#endif

#ifndef size_one_Rmid
    #define size_one_Rmid 225
#endif


// 2電子部分・・・i軌道の場合，25(12+12+1)
#ifndef boys_size
    #define boys_size 25
#endif

#ifndef size_Rmid
    #define size_Rmid 1377
#endif


#ifndef size_R
    #define size_R 2925
#endif


namespace gansu::gpu{

void printGradientMatrix_Kernel(const double* g_grad, int num_atoms);

// define the kernel to calculate nuclear repulsion gradient
void compute_nuclear_repulsion_gradient_kernel(double* g_grad, const Atom* g_atom, const int num_atoms);


// define the kernel to calculate W matrix for overlap gradient
void compute_W_Matrix_kernel(real_t* d_W_matrix, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const int num_electron, const int num_basis);


// define the kernel to calculate the gradients of moliucular integrals
void compute_gradients_overlap(double *g_gradients, const real_t *g_W_matrix,
                               const PrimitiveShell *g_shell,
                               const real_t *g_cgto_normalization_factors,
                               const int num_basis, ShellTypeInfo shell_s0,
                               ShellTypeInfo shell_s1,
                               const size_t num_threads);

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void compute_gradients_overlap_wrapper(
    double *g_gradients, const real_t *g_W_matrix,
    const PrimitiveShell *g_shell, const real_t *g_cgto_normalization_factors,
    const int num_basis, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
    const size_t num_threads);

void compute_gradients_kinetic(double *g_gradients,
                               const real_t *g_density_matrix,
                               const PrimitiveShell *g_shell,
                               const real_t *g_cgto_normalization_factors,
                               const int num_basis, ShellTypeInfo shell_s0,
                               ShellTypeInfo shell_s1,
                               const size_t num_threads);

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void compute_gradients_kinetic_wrapper(
    double *g_gradients, const real_t *g_density_matrix,
    const PrimitiveShell *g_shell, const real_t *g_cgto_normalization_factors,
    const int num_basis, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
    const size_t num_threads);

void compute_gradients_nuclear(double *g_gradients,
                               const real_t *g_density_matrix,
                               const PrimitiveShell *g_shell,
                               const real_t *g_cgto_normalization_factors,
                               const Atom *g_atom, const int num_atoms,
                               const int num_basis, ShellTypeInfo shell_s0,
                               ShellTypeInfo shell_s1, const size_t num_threads,
                               const real_t *g_boys_grid);

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void compute_gradients_nuclear_wrapper(
    double *g_gradients, const real_t *g_density_matrix,
    const PrimitiveShell *g_shell, const real_t *g_cgto_normalization_factors,
    const Atom *g_atom, const int num_atoms, const int num_basis,
    ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, const size_t num_threads,
    const real_t *g_boys_grid);

// define the kernel to calculate the gradient of the two-electron part
void compute_gradients_two_electron(
    double *g_gradients, const real_t *g_density_matrix,
    const PrimitiveShell *g_shell, const real_t *g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads, const int num_basis, const double *g_boys_grid);

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void compute_gradients_two_electron_wrapper(
    double *g_gradients, const real_t *g_density_matrix,
    const PrimitiveShell *g_shell, const real_t *g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads, const int num_basis, const double *g_boys_grid);

// define the kernel functions as function poconst inters for one electron const integrals
using compute_basis_deriv_overlap = void (*)(double*, const real_t*, const PrimitiveShell*, const real_t*, const int, ShellTypeInfo, ShellTypeInfo, const size_t);
using compute_basis_deriv_kinetic = void (*)(double*, const real_t*, const PrimitiveShell*, const real_t*, const int, ShellTypeInfo, ShellTypeInfo, const size_t);
using compute_basis_deriv_nuclear = void (*)(double*, const real_t*, const PrimitiveShell*, const real_t*, const Atom*, const int, const int, ShellTypeInfo, ShellTypeInfo, const size_t, const real_t*);

using compute_basis_deriv_repulsion = void (*)(double*, const real_t*, const PrimitiveShell*, const real_t*, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const size_t, const int, const double*);



#ifndef ND_DIST_GPU
// 2点間の距離を求める関数（2乗済み）・・・関数のオーバーロードを使用
struct double3 {
    double x, y, z;
};

inline double3 make_double3(double x, double y, double z) {
    return double3{x, y, z};
}

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
inline double calc_dist_GPU(const double3 &coord1,
                                     const double3 &coord2) {
    return (coord1.x - coord2.x) * (coord1.x - coord2.x) +
           (coord1.y - coord2.y) * (coord1.y - coord2.y) +
           (coord1.z - coord2.z) * (coord1.z - coord2.z);
}


// 該当箇所に排他的に加算する関数
inline 
void AddToResult(double* g_result, size_t index, double result, bool flag) {
    double val = flag ? 2.0 * result : result;
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &g_result[index], val);
}


// TEI・・・Two Electron Integral
inline 
void AddToResult_TEI(double* g_result, size_t index, double result, bool sym_bra, bool sym_ket, bool sym_braket) {
    int f = 1 + static_cast<int>(!sym_bra) + static_cast<int>(!sym_ket) + static_cast<int>(!sym_bra && !sym_ket) + static_cast<int>(!sym_braket) * ( 1 + static_cast<int>(!sym_bra) + static_cast<int>(!sym_ket) + static_cast<int>(!sym_bra && !sym_ket) );
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &g_result[index], result * f);
}

// // 係数部E_t(i,l)を計算する関数
inline double Et_GPU_Iterative( int I, int L, int T, double alpha, double beta, double dist)
{
    constexpr int MAX_T = 32;
    assert(I + L < MAX_T);
    const int t_max = I + L;

    // rolling buffers
    double prev[MAX_T];
    double curr[MAX_T];

    // initialize
    for(int t=0; t<=t_max; ++t){
        prev[t] = 0.0;
        curr[t] = 0.0;
    }
    const double inv_ab2 = 1.0 / (2*(alpha+beta));
    const double coef_i  = -beta*dist/(alpha+beta);
    const double coef_l  =  alpha*dist/(alpha+beta);

    prev[0] = sycl::exp(-(alpha*beta/(alpha+beta))*dist*dist);

    // build along i direction first
    for(int i=1; i<=I; ++i){
        for(int t=0; t<=i; ++t){
            double v = 0.0;
            if(t-1 >= 0) v += inv_ab2 * prev[t-1];
            v += coef_i * prev[t];
            if(t+1 <= t_max) v += (t+1) * prev[t+1];
            curr[t] = v;
        }
        for(int t=0; t<=t_max; ++t) prev[t] = curr[t];
    }

    // then along l direction
    for(int l=1; l<=L; ++l){
        for(int t=0; t<=I+l; ++t){
            double v = 0.0;
            if(t-1 >= 0) v += inv_ab2 * prev[t-1];
            v += coef_l * prev[t];
            if(t+1 <= t_max) v += (t+1)*prev[t+1];
            curr[t] = v;
        }
        for(int t=0; t<=t_max; ++t) prev[t] = curr[t];
    }
    if(T < 0 || T > t_max) return 0.0;
    return prev[T];
}

/*
DPCT1109:4: Recursive functions cannot be called in SYCL device code. You need
to adjust the code.
*/
/*
inline double Et_GPU_Recursion(int i, int l, int t, const double alpha,
                               const double beta, const double dist) {
    if(t==0 && i==0 && l==0){
        // return 1.0;
        return sycl::exp(-(alpha * beta / (alpha + beta)) * dist * dist);
    }else if(t<0 || i+l<t){ // 範囲外の処理
        return 0.0;
    }else if(i>0){ // iに関して求める
        return 1 / (2 * (alpha + beta)) *
                   Et_GPU_Recursion(i - 1, l, t - 1, alpha, beta, dist) -
               (beta * dist / (alpha + beta)) *
                   Et_GPU_Recursion(i - 1, l, t, alpha, beta, dist) +
               (t + 1) * Et_GPU_Recursion(i - 1, l, t + 1, alpha, beta, dist);
    }else{ // lに関して求める
        return 1 / (2 * (alpha + beta)) *
                   Et_GPU_Recursion(i, l - 1, t - 1, alpha, beta, dist) +
               (alpha * dist / (alpha + beta)) *
                   Et_GPU_Recursion(i, l - 1, t, alpha, beta, dist) +
               (t + 1) * Et_GPU_Recursion(i, l - 1, t + 1, alpha, beta, dist);
    }
}
*/

// // 係数部E_t(i,l)の微分に関する影響を計算する関数
inline double Et_GPU_grad( int I, int L, int T, double alpha, double beta, double dist)
{
    constexpr int MAX_T = 32;
    if(I + L >= MAX_T) return 0.0;

    const int t_max = I + L;
    double prev[MAX_T];
    double curr[MAX_T];

    // initialize
    for(int t=0;t<=t_max+1;++t) {
        prev[t] = 0.0;
        curr[t] = 0.0;
    }

    // base case
    prev[0] = -2*alpha*beta/(alpha+beta)*dist * sycl::exp(-(alpha*beta/(alpha+beta))*dist*dist);

    // build along i
    for(int i=1;i<=I;++i){
        for(int t=0;t<=i;++t){
            double val = 0.0;
            if(t-1 >= 0) val += 1.0/(2*(alpha+beta))*prev[t-1];
            val -= (beta*dist/(alpha+beta))*prev[t];
            val -= (beta/(alpha+beta))*Et_GPU_Iterative(i-1,L,t,alpha,beta,dist);
            if(t+1 <= t_max) val += (t+1)*prev[t+1];
            curr[t] = val;
        }
        for(int t=0;t<=t_max;++t) prev[t] = curr[t];
    }

    // build along l
    for(int l=1;l<=L;++l){
        for(int t=0;t<=I+l;++t){
            double val = 0.0;
            if(t-1 >= 0) val += 1.0/(2*(alpha+beta))*prev[t-1];
            val += (alpha*dist/(alpha+beta))*prev[t];
            val += (alpha/(alpha+beta))*Et_GPU_Iterative(I,l-1,t,alpha,beta,dist);
            if(t+1 <= t_max) val += (t+1)*prev[t+1];
            curr[t] = val;
        }
        for(int t=0;t<=t_max;++t) prev[t] = curr[t];
    }

    if(T<0 || T>t_max) return 0.0;
    return prev[T];
}

/*
DPCT1109:7: Recursive functions cannot be called in SYCL device code. You need
to adjust the code.
*/
/*
inline double Et_GPU_gradients(int i, int l, int t, const double alpha,
                               const double beta, const double dist) {
    if(t==0 && i==0 && l==0){
        return -2 * alpha * beta / (alpha + beta) * dist *
               sycl::exp(-(alpha * beta / (alpha + beta)) * dist * dist);
    }else if(t<0 || i+l<t){ // 範囲外の処理
        return 0.0;
    }else if(i>0){ // iに関して求める
        return 1 / (2 * (alpha + beta)) *
                   Et_GPU_gradients(i - 1, l, t - 1, alpha, beta, dist) -
               (beta * dist / (alpha + beta)) *
                   Et_GPU_gradients(i - 1, l, t, alpha, beta, dist) -
               (beta / (alpha + beta)) *
                   Et_GPU_NR(i - 1, l, t, alpha, beta, dist) +
               (t + 1) * Et_GPU_gradients(i - 1, l, t + 1, alpha, beta, dist);
    }else{ // lに関して求める
        return 1 / (2 * (alpha + beta)) *
                   Et_GPU_gradients(i, l - 1, t - 1, alpha, beta, dist) +
               (alpha * dist / (alpha + beta)) *
                   Et_GPU_gradients(i, l - 1, t, alpha, beta, dist) +
               (alpha / (alpha + beta)) *
                   Et_GPU_NR(i, l - 1, t, alpha, beta, dist) +
               (t + 1) * Et_GPU_gradients(i, l - 1, t + 1, alpha, beta, dist);
    }
}
*/

// R(t,u,v)の計算
inline double R_GPU_Iterative( int n, int t_max, int u_max, int v_max, const double3 &P, const Coordinate &atom_pos, const double* Boys) 
{
    constexpr int MAX_TUV = 16; // 安全マージン
    if(t_max > MAX_TUV || u_max > MAX_TUV || v_max > MAX_TUV) return 0.0;

    auto idx = [=](int t,int u,int v,int Umax,int Vmax){
        return t*(Umax+1)*(Vmax+1) + u*(Vmax+1) + v;
    };

    // 2つのバッファをローリング
    double R_curr[(MAX_TUV+1)*(MAX_TUV+1)*(MAX_TUV+1)] = {0.0};
    double R_next[(MAX_TUV+1)*(MAX_TUV+1)*(MAX_TUV+1)] = {0.0};

    // 基底ケース: t=u=v=0
    R_next[idx(0,0,0,u_max,v_max)] = Boys[n];

    // nのインデックスを増やしながら反復
    int n_max = n + t_max + u_max + v_max; // 必要なBoysの最大インデックス

    for(int nn = n_max; nn >= n; --nn){

        // t,u,v方向を反復
        for(int tt = 0; tt <= t_max; ++tt){
            for(int uu = 0; uu <= u_max; ++uu){
                for(int vv = 0; vv <= v_max; ++vv){
                    if(tt==0 && uu==0 && vv==0){
                        R_curr[idx(tt,uu,vv,u_max,v_max)] = Boys[nn];
                        continue;
                    }
                    double val = 0.0;

                    if(tt>1)
                        val += (tt-1)*R_next[idx(tt-2,uu,vv,u_max,v_max)];
                    if(tt>0)
                        val += (P.x - atom_pos.x) * R_next[idx(tt-1,uu,vv,u_max,v_max)];
                    if(uu>1)
                        val += (uu-1)*R_next[idx(tt,uu-2,vv,u_max,v_max)];
                    if(uu>0)
                        val += (P.y - atom_pos.y) * R_next[idx(tt,uu-1,vv,u_max,v_max)];
                    if(vv>1)
                        val += (vv-1)*R_next[idx(tt,uu,vv-2,u_max,v_max)];
                    if(vv>0)
                        val += (P.z - atom_pos.z) * R_next[idx(tt,uu,vv-1,u_max,v_max)];
                    R_curr[idx(tt,uu,vv,u_max,v_max)] = val;
                }
            }
        }
        // 次の n に移行する前にローリング
        for(int i = 0; i <= t_max; ++i)
            for(int j = 0; j <= u_max; ++j)
                for(int k = 0; k <= v_max; ++k)
                    R_next[idx(i,j,k,u_max,v_max)] = R_curr[idx(i,j,k,u_max,v_max)];
    }
    return R_curr[idx(t_max,u_max,v_max,u_max,v_max)];
}

/*
DPCT1109:10: Recursive functions cannot be called in SYCL device code. You need
to adjust the code.
*/
/*
inline double R_GPU_Recursion(int n, int t, int u, int v,
                              const sycl::double3 &P,
                              const Coordinate &atom_pos, double *Boys) {
    if(t==0 and u==0 and v==0){
        return Boys[n];
    }else if(t>0){
        return ((t - 1) *
                    R_GPU_Recursion(n + 1, t - 2, u, v, P, atom_pos, Boys) +
                (P.x() - atom_pos.x) *
                    R_GPU_Recursion(n + 1, t - 1, u, v, P, atom_pos, Boys));
    }else if(u>0){
        return ((u - 1) *
                    R_GPU_Recursion(n + 1, t, u - 2, v, P, atom_pos, Boys) +
                (P.y() - atom_pos.y) *
                    R_GPU_Recursion(n + 1, t, u - 1, v, P, atom_pos, Boys));
    }else if(v>0){
        return ((v - 1) *
                    R_GPU_Recursion(n + 1, t, u, v - 2, P, atom_pos, Boys) +
                (P.z() - atom_pos.z) *
                    R_GPU_Recursion(n + 1, t, u, v - 1, P, atom_pos, Boys));
    }else{
        return 0.0;
    }
}
*/

inline double R_GPU_Iterative( int n, int t_max, int u_max, int v_max, const double3 &P, const double3 &Q, const double* Boys)
{
    constexpr int MAX_TUV = 16;
    if(t_max > MAX_TUV || u_max > MAX_TUV || v_max > MAX_TUV) return 0.0;

    auto idx = [=](int t,int u,int v,int Umax,int Vmax){
        return t*(Umax+1)*(Vmax+1) + u*(Vmax+1) + v;
    };

    double R_curr[(MAX_TUV+1)*(MAX_TUV+1)*(MAX_TUV+1)] = {0.0};
    double R_next[(MAX_TUV+1)*(MAX_TUV+1)*(MAX_TUV+1)] = {0.0};

    // 基底ケース
    R_next[idx(0,0,0,u_max,v_max)] = Boys[n];
    int n_max = n + t_max + u_max + v_max;
    for(int nn = n_max; nn >= n; --nn){
        for(int tt = 0; tt <= t_max; ++tt){
            for(int uu = 0; uu <= u_max; ++uu){
                for(int vv = 0; vv <= v_max; ++vv){
                    if(tt==0 && uu==0 && vv==0){
                        R_curr[idx(tt,uu,vv,u_max,v_max)] = Boys[nn];
                        continue;
                    }
                    double val = 0.0;
                    if(tt>1)
                        val += (tt-1)*R_next[idx(tt-2,uu,vv,u_max,v_max)];
                    if(tt>0)
                        val += (P.x - Q.x) * R_next[idx(tt-1,uu,vv,u_max,v_max)];
                    if(uu>1)
                        val += (uu-1)*R_next[idx(tt,uu-2,vv,u_max,v_max)];
                    if(uu>0)
                        val += (P.y - Q.y) * R_next[idx(tt,uu-1,vv,u_max,v_max)];
                    if(vv>1)
                        val += (vv-1)*R_next[idx(tt,uu,vv-2,u_max,v_max)];
                    if(vv>0)
                        val += (P.z - Q.z) * R_next[idx(tt,uu,vv-1,u_max,v_max)];
                    R_curr[idx(tt,uu,vv,u_max,v_max)] = val;
                }
            }
        }
        // ローリング
        for(int i=0;i<=t_max;++i)
            for(int j=0;j<=u_max;++j)
                for(int k=0;k<=v_max;++k)
                    R_next[idx(i,j,k,u_max,v_max)] = R_curr[idx(i,j,k,u_max,v_max)];
    }
    return R_curr[idx(t_max,u_max,v_max,u_max,v_max)];
}

/*
DPCT1109:14: Recursive functions cannot be called in SYCL device code. You need
to adjust the code.
*/
/*
inline double R_GPU_Recursion(int n, int t, int u, int v,
                              const sycl::double3 &P, const sycl::double3 &Q,
                              double *Boys) {
    if(t==0 and u==0 and v==0){
        return Boys[n];
    }else if(t>0){
        return ((t - 1) * R_GPU_Recursion(n + 1, t - 2, u, v, P, Q, Boys) +
                (P.x() - Q.x()) *
                    R_GPU_Recursion(n + 1, t - 1, u, v, P, Q, Boys));
    }else if(u>0){
        return ((u - 1) * R_GPU_Recursion(n + 1, t, u - 2, v, P, Q, Boys) +
                (P.y() - Q.y()) *
                    R_GPU_Recursion(n + 1, t, u - 1, v, P, Q, Boys));
    }else if(v>0){
        return ((v - 1) * R_GPU_Recursion(n + 1, t, u, v - 2, P, Q, Boys) +
                (P.z() - Q.z()) *
                    R_GPU_Recursion(n + 1, t, u, v - 1, P, Q, Boys));
    }else{
        return 0.0;
    }
}
*/

// MD法のRの再帰関係をトリプルバッファリングで計算
inline void compute_R_TripleBuffer( real_t *R, real_t *R_mid, const real_t *Boys, const double3 &P,
    const Coordinate &coord, const int K, const int t_max, const int u_max,
    const int v_max) {
    //Step 0: Boys関数評価
    R[0]=Boys[0];
    for(int i=0; i <= K; i++){
        R_mid[i]=Boys[i];
    } 
    //Step 1~Kの計算
    for(int k=1; k <= K; k++){
        for(int z=0; z<=(K+1)*comb_max(k); z++){
            int i = z/comb_max(k);
            if(i <= K-k){
                int t = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][0];
                int u = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][1];
                int v = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][2];
                if((t <= t_max) && (u <= u_max) && (v <= v_max)){
                    if(t >= 1){
                        R_mid[calc_Idx_Rmid(k, u, v, i, comb_max(k), size_one_Rmid)] =
                            (P.x - coord.x) * R_mid[calc_Idx_Rmid(k - 1, u, v, i + 1, comb_max(k - 1), size_one_Rmid)] +
                            (t - 1) * R_mid[calc_Idx_Rmid(k - 2, u, v, i + 1, comb_max(k - 2), size_one_Rmid)];
                    }
                    else if(u >= 1){
                        R_mid[calc_Idx_Rmid(k, u, v, i, comb_max(k), size_one_Rmid)] =
                            (P.y - coord.y) * R_mid[calc_Idx_Rmid(k - 1, u - 1, v, i + 1, comb_max(k - 1), size_one_Rmid)] +
                            (u - 1) * R_mid[calc_Idx_Rmid( k - 2, u - 2, v, i + 1, comb_max(k - 2), size_one_Rmid)];
                    }
                    else{
                        R_mid[calc_Idx_Rmid(k, u, v, i, comb_max(k), size_one_Rmid)] =
                            (P.z - coord.z) * R_mid[calc_Idx_Rmid(k - 1, u, v - 1, i + 1, comb_max(k - 1), size_one_Rmid)] +
                            (v - 1) * R_mid[calc_Idx_Rmid( k - 2, u, v - 2, i + 1, comb_max(k - 2), size_one_Rmid)];
                    }
                }
            }
        }
        //必要な結果を配列Rに書き込み
        for(int i=0; i<=comb_max(k); i++){
            R[static_cast<int>(k*(k+1)*(k+2)/6) + i] = R_mid[(k%3)*static_cast<int>(size_one_Rmid) + i];
        }
    }
}

inline void compute_R_TripleBuffer(
    double *R, double *R_mid, const double *Boys, const double3 &P,
    const double3 &Q, const int K, const int t_max, const int u_max,
    const int v_max) {
//    const int v_max, dpct::accessor<int, dpct::constant, 2> tuv_list) {
    //Step 0: Boys関数評価
    R[0]=Boys[0];
    for(int i=0; i <= K; i++){
        R_mid[i]=Boys[i];
    }

    //Step 1~Kの計算
    for(int k=1; k <= K; k++){
        for(int z=0; z<=(K+1)*comb_max(k); z++){
            int i = z/comb_max(k);
            if(i <= K-k){
                int t = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][0];
                int u = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][1];
                int v = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][2];
                if((t <= t_max) && (u <= u_max) && (v <= v_max)){
                    if(t >= 1){
                        R_mid[calc_Idx_Rmid(k, u, v, i, comb_max(k), size_Rmid)] =
                            (P.x - Q.x) * R_mid[calc_Idx_Rmid(k - 1, u, v, i + 1, comb_max(k - 1), size_Rmid)] +
                            (t - 1) * R_mid[calc_Idx_Rmid(k - 2, u, v, i + 1, comb_max(k - 2), size_Rmid)];
                    }
                    else if(u >= 1){
                        R_mid[calc_Idx_Rmid(k, u, v, i, comb_max(k), size_Rmid)] =
                            (P.y - Q.y) * R_mid[calc_Idx_Rmid(k - 1, u - 1, v, i + 1, comb_max(k - 1), size_Rmid)] +
                            (u - 1) * R_mid[calc_Idx_Rmid(k - 2, u - 2, v, i + 1, comb_max(k - 2), size_Rmid)];
                    }
                    else{
                        R_mid[calc_Idx_Rmid(k, u, v, i, comb_max(k), size_Rmid)] =
                            (P.z - Q.z) * R_mid[calc_Idx_Rmid(k - 1, u, v - 1, i + 1, comb_max(k - 1), size_Rmid)] +
                            (v - 1) * R_mid[calc_Idx_Rmid( k - 2, u, v - 2, i + 1, comb_max(k - 2), size_Rmid)];
                    }
                }
            }
        }
        //必要な結果を配列Rに書き込み
        for(int i=0; i<=comb_max(k); i++){
            R[static_cast<int>(k*(k+1)*(k+2)/6) + i] = R_mid[(k%3)*static_cast<int>(size_Rmid) + i];
        }

    }
}





inline compute_basis_deriv_overlap get_compute_gradients_overlap() {
    return compute_gradients_overlap_wrapper;
}

inline compute_basis_deriv_kinetic get_compute_gradients_kinetic() {
    return compute_gradients_kinetic_wrapper;
}

inline compute_basis_deriv_nuclear get_compute_gradients_nuclear() {
    return compute_gradients_nuclear_wrapper;
}

inline compute_basis_deriv_repulsion get_compute_gradients_repulsion() {
    return compute_gradients_two_electron_wrapper;
}


} // namespace gansu::gpu

#endif
