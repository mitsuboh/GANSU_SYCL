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



#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "types.hpp"

namespace gansu::gpu {
inline size_t2 index1to2_one_electron(const size_t index, bool is_symmetric, size_t num_basis=0){
//    assert(is_symmetric or num_basis > 0);
    if(is_symmetric){
        /*
        DPCT1013:111: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        const size_t x =
            sycl::vec<double, 1>{
                ((sycl::sqrt((double)(8 * index + 1)) - 1) / 2)}
                .convert<long long, sycl::rounding_mode::rtn>()[0];
        const size_t y = index - x * (x + 1) / 2;
        return {x, y};
    }else{
        const size_t x = index % num_basis;
        const size_t y = index / num_basis;
        return {x, y};
    }
}

inline size_t2 index1to2(const size_t index, bool is_symmetric, size_t num_basis=0){
//    assert(is_symmetric or num_basis > 0);
    if(is_symmetric){
        /*
        DPCT1013:112: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        const size_t r2 =
            sycl::vec<double, 1>{
                ((sycl::sqrt((double)(8 * index + 1)) - 1) / 2)}
                .convert<long long, sycl::rounding_mode::rtn>()[0];
        const size_t r1 = index - r2 * (r2 + 1) / 2;
        return {r1, r2};
    }else{
        return {index / num_basis, index % num_basis};
    }
}

inline size_t4 sort_eri_index(size_t a, size_t b, size_t c, size_t d){
    if(a > b){
        size_t tmp = a;
        a = b;
        b = tmp;
    }
    if(c > d){
        size_t tmp = c;
        c = d;
        d = tmp;
    }
    if(a > c || (a == c && b > d)){
        size_t tmp = a;
        a = c;
        c = tmp;
        tmp = b;
        b = d;
        d = tmp;
    }
    return {a, b, c, d};
}

inline size_t get_index_2to1(const size_t i, const size_t j, const size_t n)
{
    return j - static_cast<size_t>(i*(i-2*n+1)/2);
}

inline size_t get_1d_index(const size_t i, const size_t j, const size_t k, const size_t l, const size_t num_basis)
{
    size_t4 sorted = sort_eri_index(i,j,k,l);
    size_t bra = get_index_2to1(sorted.x, sorted.y, num_basis);
    size_t ket = get_index_2to1(sorted.z, sorted.w, num_basis);
    return get_index_2to1(bra, ket, static_cast<size_t>(num_basis*(num_basis+1)/2));
}

inline size_t get_1d_indexM4(const size_t i, const size_t j, const size_t k, const size_t l, const size_t num_basis)
{
    return  num_basis * num_basis * num_basis * i + num_basis * num_basis * j + num_basis * k + l;
}





} // namespace gansu::gpu