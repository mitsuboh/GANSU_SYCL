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


#include "eri.hpp"

namespace gansu{

ERI_Stored::ERI_Stored(const HF& hf): 
        hf_(hf),
        num_basis_(hf.get_num_basis()),
        eri_matrix_(num_basis_*num_basis_, num_basis_*num_basis_),
        schwarz_upper_bound_factors(hf.get_num_primitive_shell_pairs())
{
    // nothing to do
}


void ERI_Stored::precomputation() {
    // compute the electron repulsion integrals
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const DeviceHostMemory<real_t>& cgto_nomalization_factors = hf_.get_cgto_nomalization_factors();
    const real_t schwarz_screening_threshold = hf_.get_schwarz_screening_threshold();
    const int verbose = hf_.get_verbose();

    // Compute Schwarz Upper Bounds
    gpu::computeSchwarzUpperBounds(
        shell_type_infos,
        shell_pair_type_infos,
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(), 
        cgto_nomalization_factors.device_ptr(), 
        schwarz_upper_bound_factors.device_ptr(), 
        verbose
        );


    //gpu::computeERIMatrix(shell_type_infos, primitive_shells.device_ptr(), boys_grid.device_ptr(), cgto_nomalization_factors.device_ptr(), eri_matrix_.device_ptr(), schwarz_screening_threshold, num_basis_, verbose);

    gpu::computeERIMatrix(
        shell_type_infos, 
        shell_pair_type_infos, 
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(),
        cgto_nomalization_factors.device_ptr(),   
        eri_matrix_.device_ptr(), 
        schwarz_upper_bound_factors.device_ptr(),
        schwarz_screening_threshold, 
        num_basis_, 
        verbose
        );

    // print the eri matrix
    if(verbose){
        // copy the eri matrix to the host memory
        eri_matrix_.toHost();

        std::cout << "ERI matrix:" << std::endl;
        for(int l=0; l<num_basis_; l++){
            for(int k=0; k<=l; k++){
                for(int j=0; j<=l; j++){
                    const auto i_max = (l==j) ? k : j;
                    for(int i=0; i<=i_max; i++){
                        std::cout << "i: " << i << ", j: " << j << ", k: " << k << ", l: " << l << ": " << eri_matrix_(i*num_basis_+j, k*num_basis_+l) << std::endl;
                    }
                }
            }
        }
    }
}




ERI_RI::ERI_RI(const HF& hf, const Molecular& auxiliary_molecular): 
        hf_(hf),
        num_basis_(hf.get_num_basis()),
        num_auxiliary_basis_(auxiliary_molecular.get_num_basis()),
        auxiliary_shell_type_infos_(auxiliary_molecular.get_shell_type_infos()),
        auxiliary_primitive_shells_(auxiliary_molecular.get_primitive_shells()),
        auxiliary_cgto_nomalization_factors_(auxiliary_molecular.get_cgto_normalization_factors()),
        intermediate_matrix_B_(num_auxiliary_basis_, num_basis_*num_basis_)
{
    // to device memory
    auxiliary_primitive_shells_.toDevice();
    auxiliary_cgto_nomalization_factors_.toDevice();
}


void ERI_RI::precomputation() {
    // compute the intermediate matrix B of the auxiliary basis functions
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& cgto_nomalization_factors = hf_.get_cgto_nomalization_factors();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const int verbose = hf_.get_verbose();

    gpu::compute_RI_IntermediateMatrixB(
        shell_type_infos, 
        primitive_shells.device_ptr(), 
        cgto_nomalization_factors.device_ptr(), 
        auxiliary_shell_type_infos_, 
        auxiliary_primitive_shells_.device_ptr(), 
        auxiliary_cgto_nomalization_factors_.device_ptr(), 
        intermediate_matrix_B_.device_ptr(), 
        num_basis_, 
        num_auxiliary_basis_, 
        boys_grid.device_ptr(), 
        verbose
        );
    /*
    if(1){
        // copy the intermediate matrix B to the host memory
        intermediate_matrix_B_.toHost();

        std::cout << "Intermediate matrix B:" << std::endl;
        for(int i=0; i<num_auxiliary_basis_; i++){
            for(int j=0; j<num_basis_; j++){
                for(int k=0; k<num_basis_; k++){
                    auto value = intermediate_matrix_B_(i, j*num_basis_+k);
                    if (std::isnan(value)) {
                        std::cout << "NaN found at (" << i << "," << j << "): " << value << std::endl;
                    }
                }
                std::cout << std::endl;
            }
        }
    }
    */
}



ERI_Direct::ERI_Direct(const HF& hf):
    hf_(hf),
    num_basis_(hf.get_num_basis()),
    schwarz_upper_bound_factors(hf.get_num_primitive_shell_pairs())
{
    // nothing to do
}

void ERI_Direct::precomputation() {
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& cgto_nomalization_factors = hf_.get_cgto_nomalization_factors();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const int verbose = hf_.get_verbose();

    gpu::computeSchwarzUpperBounds(
        shell_type_infos,
        shell_pair_type_infos,
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(), 
        cgto_nomalization_factors.device_ptr(), 
        schwarz_upper_bound_factors.device_ptr(), 
        verbose
        );
}


} // namespace gansu