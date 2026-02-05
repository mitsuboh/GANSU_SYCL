/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
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

#include <sycl/sycl.hpp>
#include "gpu_kernels.hpp"
#include "utils.hpp"

namespace gansu::gpu{

// Almost all functions moved to hpp.

void constructERIHash_kernel(const std::vector<ShellTypeInfo> shell_type_infos, const std::vector<ShellPairTypeInfo> shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_cgto_normalization_factors, /* Hash memoryへのポインタ, */ const bool verbose)
{
    // ここにERIを計算して、ハッシュテーブルに格納する処理を実装する
}

void computeFockMatrix_Hash_RHF_kernel(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, /* Hash memoryへのポインタ, */ real_t* d_fock_matrix, const int num_basis, const int verbose)
{
    // ハッシュテーブルを使用してFock行列を計算する処理を実装する
}

} // namespace gansu::gpu
