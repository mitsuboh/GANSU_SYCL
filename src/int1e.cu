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


#include <cuda.h>
#include <cmath>

#include "boys.hpp"
#include "types.hpp"
#include "utils_cuda.hpp"

namespace gansu::gpu{

/* not used???
#define A_TR 0.352905920120321
#define B_TR 0.015532762923351
#define A_RS 0.064048916778075
#define B_RS 28.487431543672

#define AA 0.03768724  
#define BB 0.60549623
#define CC 6.32743473
#define DD 10.350421
*/




// 二条階乗を返す関数
__device__ int factorial2gpu(int i){
    if(i<1){
        return 1;
    }else{
        return i*factorial2gpu(i-2);
    }
}

// 2点間の距離を求める関数（2乗済み）
__device__ double calc_dist_GPU2(const Coordinate& coord1, const Coordinate& coord2){
    return (coord1.x-coord2.x)*(coord1.x-coord2.x) + (coord1.y-coord2.y)*(coord1.y-coord2.y) + (coord1.z-coord2.z)*(coord1.z-coord2.z);
}
__device__ double calc_dist_GPU3(const double3& coord1, const Coordinate& coord2){
    return (coord1.x-coord2.x)*(coord1.x-coord2.x) + (coord1.y-coord2.y)*(coord1.y-coord2.y) + (coord1.z-coord2.z)*(coord1.z-coord2.z);
}

// /* PGTOの規格化定数を算出(2つのPGTOまとめて) */
// __device__ double calc_Norms(double alpha, double beta, int i, int j, int k, int l, int m, int n){
//     return pow(2.0, i+j+k+l+m+n) 
//         * pow(factorial2gpu(2.0*i-1.0)*factorial2gpu(2.0*j-1.0)*factorial2gpu(2.0*k-1.0)*factorial2gpu(2.0*l-1.0)*factorial2gpu(2.0*m-1.0)*factorial2gpu(2.0*n-1.0), -0.5) 
//         * pow(2.0/M_PI, 1.5)
//         * pow(alpha, (2.0*(i+j+k)+3.0)/4.0)
//         * pow(beta, (2.0*(l+m+n)+3.0)/4.0);
// }

/* PGTOの規格化定数を算出(2つのPGTOまとめて) */
__device__ double calc_Norms(double alpha, double beta, int i, int j, int k, int l, int m, int n){
    return pow(2.0, i+j+k+l+m+n) 
        * pow(2.0/M_PI, 1.5)
        * pow(alpha, (2.0*(i+j+k)+3.0)/4.0)
        * pow(beta, (2.0*(l+m+n)+3.0)/4.0);
}





// 該当箇所に排他的に加算する関数
__device__ void AddToResult(double result, double *g_nucattr, int y, int x, int num_basis, bool is_symmetric){
    atomicAdd(&g_nucattr[y*num_basis + x], result);
    if(!is_symmetric){
        // printf("%d, %d\n",y,x);
        atomicAdd(&g_nucattr[x*num_basis + y], result);
    }
}

// f軌道の場合における、基底関数の順番を調整する多に使用
__constant__ int f_index[10] = {0, 1, 2, 4, 5, 3, 6, 8, 7, 9};

// MD method
#include "MD_OandK_kernel.txt"
#include "MD_V_kernel.txt"



// OS method
#include "OS_OandK_kernel.txt"
#include "OS_V_kernel.txt"

} // namespace gansu::gpu