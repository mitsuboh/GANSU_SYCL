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



__device__ double calc_Norms(double alpha, double beta, int ijk, int lmn){
    return pow(2.0, ijk+lmn) 
        * pow(2.0/M_PI, 1.5)
        * pow(alpha, (2.0*ijk+3.0)/4.0)
        * pow(beta, (2.0*lmn+3.0)/4.0);
}


__global__ void Matrix_Symmetrization(double* matrix, int n){
    __shared__ double sh_mem[32][33];

    if(blockIdx.y > blockIdx.x) return;

    int src_block = blockIdx.y*32*n + blockIdx.x*32;
    int dst_block = blockIdx.x*32*n + blockIdx.y*32;

    if(blockIdx.x*32+threadIdx.x < n || blockIdx.y*32+threadIdx.y < n){
        sh_mem[threadIdx.y][threadIdx.x] = matrix[src_block + threadIdx.y*n+threadIdx.x];
    }
    __syncthreads();

    if (blockIdx.y==blockIdx.x && threadIdx.y <= threadIdx.x || (dst_block + threadIdx.y*n+threadIdx.x >=n*n) ) return;

    matrix[dst_block + threadIdx.y*n+threadIdx.x] = sh_mem[threadIdx.x][threadIdx.y];
}


__device__ int calc_result_index(int y, int x, int sumCGTO){
    return (y<=x) ? y*sumCGTO + x : x*sumCGTO + y;
}

// MD method
#include "MD_kernel.txt"
// OS method
#include "OS_kernel.txt"

} // namespace gansu::gpu
