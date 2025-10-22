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
 * @file env.hpp
 * @brief Display the environment information (e.g., versons of CUDA, cuBLAS, and cuSOLVER)
 */

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <dpct/blas_utils.hpp>

#include <dpct/lapack_utils.hpp>

#include <dpct/lib_common_utils.hpp>

namespace gansu{

/**
 * @brief Display the environment information
 */
inline void display_env_info(){
    // SYCL Platform Version
    sycl::queue q;
    auto platform = q.get_device().get_platform();
    std::cout << "Platform name: " << platform.get_info<sycl::info::platform::name>() << std::endl;
    std::cout << "Platform version: " << platform.get_info<sycl::info::platform::version>() << std::endl;

    // cuBLAS Version
    // MKL Version
    MKLVersion ver;
    mkl_get_version(&ver);
    std::cout << "MKL Version: " << ver.MajorVersion << "." << ver.MinorVersion << "." << ver.UpdateVersion << std::endl ;
}


} // namespace gansu
