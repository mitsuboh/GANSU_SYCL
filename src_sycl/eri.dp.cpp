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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include "eri.hpp"
#include "utils_cuda.hpp"
//#include "device_host_memory.hpp"

namespace gansu{

inline size_t2 index1to2(const size_t index, bool is_symmetric, size_t num_basis=0){
//    assert(is_symmetric or num_basis > 0);
    if(is_symmetric){
        /*
        DPCT1013:274: The rounding mode could not be specified and the generated
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

void generatePrimitiveShellPairIndices(const sycl::nd_item<1>& item_ct1, size_t2* d_indices_array, size_t num_threads, bool is_symmetric, size_t num_basis){
//    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t id = item_ct1.get_global_linear_id();
    if (id >= num_threads) return;
    d_indices_array[id] = index1to2(id, is_symmetric, num_basis);
}


void generatePrimitiveShellPairIndices(const sycl::nd_item<1>& item_ct1, size_t2* d_indices_array, size_t num_threads,
        bool is_symmetric, size_t num_basis, bool if_full_range, size_t start_index_a, size_t start_index_b){
    const size_t id = item_ct1.get_global_linear_id();
    if (id >= num_threads) return;
    d_indices_array[id] = index1to2(id, is_symmetric, num_basis);


    d_indices_array[id].x += start_index_a;
    d_indices_array[id].y += start_index_b;
}

void initializePrimitiveShellPairIndices(const sycl::nd_item<1>& item_ct1, sycl::int2* d_indices_array, int num_threads,
        bool is_symmetric, int num_basis) {
    const size_t id = item_ct1.get_global_linear_id();
    if (id >= num_threads) return;
    size_t2 index_pair = index1to2(id, is_symmetric, num_basis);
    d_indices_array[id] = sycl::int2(static_cast<int>(index_pair.x), static_cast<int>(index_pair.y));

}


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
    const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
    const real_t schwarz_screening_threshold = hf_.get_schwarz_screening_threshold();
    const int verbose = hf_.get_verbose();

    // Compute Schwarz Upper Bounds
    gpu::computeSchwarzUpperBounds(
        shell_type_infos,
        shell_pair_type_infos,
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(), 
        cgto_normalization_factors.device_ptr(), 
        schwarz_upper_bound_factors.device_ptr(), 
        verbose
        );


    //gpu::computeERIMatrix(shell_type_infos, primitive_shells.device_ptr(), boys_grid.device_ptr(), cgto_normalization_factors.device_ptr(), eri_matrix_.device_ptr(), schwarz_screening_threshold, num_basis_, verbose);

    gpu::computeERIMatrix(
        shell_type_infos, 
        shell_pair_type_infos, 
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(),
        cgto_normalization_factors.device_ptr(),   
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
        auxiliary_cgto_normalization_factors_(auxiliary_molecular.get_cgto_normalization_factors()),
        intermediate_matrix_B_(num_auxiliary_basis_, num_basis_*num_basis_),
        d_J_(num_basis_, num_basis_),
        d_K_(num_basis_, num_basis_),
        d_W_tmp_(num_auxiliary_basis_),
        d_T_tmp_(num_auxiliary_basis_, num_basis_*num_basis_),
        d_V_tmp_(num_auxiliary_basis_, num_basis_*num_basis_),
        schwarz_upper_bound_factors(hf.get_num_primitive_shell_pairs()),
        auxiliary_schwarz_upper_bound_factors(auxiliary_molecular.get_primitive_shells().size())
{
    // to device memory
    auxiliary_primitive_shells_.toDevice();
    auxiliary_cgto_normalization_factors_.toDevice();
}

void ERI_RI::precomputation() {
    sycl::queue& q_ct1 = gpu::GPUHandle::syclqueue();
    // compute the intermediate matrix B of the auxiliary basis functions
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const int verbose = hf_.get_verbose();

    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const real_t schwarz_screening_threshold = hf_.get_schwarz_screening_threshold();

    // compute upper bounds of primitive-shell-pair
    gpu::computeSchwarzUpperBounds(
        shell_type_infos,
        shell_pair_type_infos,
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(), 
        cgto_normalization_factors.device_ptr(), 
        schwarz_upper_bound_factors.device_ptr(),   // schwarz_upper_bound_factorsに√(pq|pq)の値がはいっている
        verbose
    );


    const size_t num_primitive_shell_pairs = primitive_shells.size() * (primitive_shells.size() + 1) / 2;
    size_t2* d_primitive_shell_pair_indices;
    d_primitive_shell_pair_indices = sycl::malloc_device<size_t2>(num_primitive_shell_pairs, q_ct1);
//    d_primitive_shell_pair_indices = tracked_syclMalloc<size_t2>(num_primitive_shell_pairs);

    int pair_idx = 0;
//    const int threads_per_block = 1024;
    size_t max_wg = q_ct1.get_device().get_info<sycl::info::device::max_work_group_size>();
    size_t threads_per_block = std::min<size_t>(1024, max_wg);
    for(int s0 = 0; s0 < shell_type_infos.size(); s0++){
        for(int s1 = s0; s1 < shell_type_infos.size(); s1++){
            const int num_blocks = (shell_pair_type_infos[pair_idx].count + threads_per_block - 1) / threads_per_block; // the number of blocks
            /*
            DPCT1049:83: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            q_ct1.submit([&](sycl::handler &cgh) {
                auto
                    d_primitive_shell_pair_indices_shell_pair_type_infos_pair_idx_start_index_ct0 =
                        &d_primitive_shell_pair_indices
                            [shell_pair_type_infos[pair_idx].start_index];
                auto shell_pair_type_infos_pair_idx_count_ct1 =
                    shell_pair_type_infos[pair_idx].count;
                auto s0_s1_ct2 = s0 == s1;
                size_t shell_type_infos_s1_count_ct3 =
                    shell_type_infos[s1].count;

                cgh.parallel_for(sycl::nd_range<1>(num_blocks * threads_per_block, threads_per_block),
                    [=](sycl::nd_item<1> item_ct1) {
                        generatePrimitiveShellPairIndices( item_ct1, 
                            d_primitive_shell_pair_indices_shell_pair_type_infos_pair_idx_start_index_ct0,
                            shell_pair_type_infos_pair_idx_count_ct1, s0_s1_ct2,
                            shell_type_infos_s1_count_ct3);
                    });
            });

            real_t* keys_begin =
                &schwarz_upper_bound_factors.device_ptr()
                    [shell_pair_type_infos[pair_idx].start_index];
            real_t* keys_end = 
                keys_begin + shell_pair_type_infos[pair_idx].count;

            size_t2* values_begin =
                &d_primitive_shell_pair_indices
                    [shell_pair_type_infos[pair_idx].start_index];

            size_t count = shell_pair_type_infos[pair_idx].count;

            // zip(keys, values)
            auto zipped_begin = dpl::make_zip_iterator(keys_begin, values_begin);
            auto zipped_end   = dpl::make_zip_iterator(keys_end,   values_begin + count);

            auto policy = dpl::execution::make_device_policy(q_ct1);

            // ソート（Schwarz bound の降順）
            dpl::sort(policy, zipped_begin, zipped_end,
                  [](const auto& a, const auto& b) {
                      return std::get<0>(a) > std::get<0>(b);
            });

            pair_idx++;
        }
    }
    q_ct1.wait_and_throw();

    // compute upper bounds of  aux-shell
    gpu::computeAuxiliarySchwarzUpperBounds(
        auxiliary_shell_type_infos_, 
        auxiliary_primitive_shells_.device_ptr(), 
        boys_grid.device_ptr(), 
        auxiliary_cgto_normalization_factors_.device_ptr(), 
        auxiliary_schwarz_upper_bound_factors.device_ptr(),   // auxiliary_schwarz_upper_bound_factorsに√(pq|pq)の値がはいっている
        verbose
    );

    for (const auto &s : auxiliary_shell_type_infos_) {
        real_t *keys_begin = auxiliary_schwarz_upper_bound_factors.device_ptr() + s.start_index;
        real_t *keys_end   = keys_begin + s.count;

        PrimitiveShell *values_begin =
            auxiliary_primitive_shells_.device_ptr() + s.start_index;

        oneapi::dpl::sort_by_key(
            oneapi::dpl::execution::make_device_policy(q_ct1),
            keys_begin,
            keys_end,
            values_begin,
            std::greater<real_t>()  // or custom lambda
        );
    }



    gpu::compute_RI_IntermediateMatrixB(
        shell_type_infos, 
        shell_pair_type_infos,
        primitive_shells.device_ptr(), 
        cgto_normalization_factors.device_ptr(), 
        auxiliary_shell_type_infos_, 
        auxiliary_primitive_shells_.device_ptr(), 
        auxiliary_cgto_normalization_factors_.device_ptr(), 
        intermediate_matrix_B_.device_ptr(), 
        d_primitive_shell_pair_indices,
        schwarz_upper_bound_factors.device_ptr(),
        auxiliary_schwarz_upper_bound_factors.device_ptr(),
        schwarz_screening_threshold,
        num_basis_, 
        num_auxiliary_basis_, 
        boys_grid.device_ptr(), 
        verbose
        );

//    tracked_syclFree(d_primitive_shell_pair_indices);
    sycl::free(d_primitive_shell_pair_indices, q_ct1);
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
    schwarz_upper_bound_factors(hf.get_num_primitive_shell_pairs()),
    primitive_shell_pair_indices(hf.get_num_primitive_shell_pairs()),
    num_fock_replicas_(8)
{
    sycl::queue& q_ct1 = gpu::GPUHandle::syclqueue();
    // for distributed atomicAdd operations
    //cudaMalloc(&fock_matrix_replicas_, sizeof(real_t) * num_basis_ * num_basis_ * num_fock_replicas_);
    fock_matrix_replicas_ = sycl::malloc_device<real_t>(num_basis_ * num_basis_ * num_fock_replicas_, q_ct1);
//    fock_matrix_replicas_ = tracked_syclMalloc<real_t>(num_basis_ * num_basis_ * num_fock_replicas_);
    //cudaMemset(fock_matrix_replicas_, 0.0, sizeof(real_t) * num_basis_ * num_basis_ * num_fock_replicas_);
}

ERI_Direct::~ERI_Direct() {
    sycl::queue& q_ct1 = gpu::GPUHandle::syclqueue();
    for (auto p : global_counters_) { if (p) sycl::free(p, q_ct1); }
    for (auto p : min_skipped_columns_) { if (p) sycl::free(p, q_ct1); }
    global_counters_.clear();
    min_skipped_columns_.clear();

    if (fock_matrix_replicas_) {
//        tracked_syclFree(fock_matrix_replicas_);
        sycl::free(fock_matrix_replicas_, q_ct1);
        fock_matrix_replicas_ = nullptr;
    }
}

void ERI_Direct::precomputation() {
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const int verbose = hf_.get_verbose();

    // for dynamic Schwarz screening
    const int shell_type_count = shell_type_infos.size();
    sycl::queue& q_ct1 = gpu::GPUHandle::syclqueue();
    std::vector<std::tuple<int, int, int, int>> shell_quadruples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < shell_type_count; ++c) {
                for (int d = c; d < shell_type_count; ++d) {
                    if (a < c || (a == c && b <= d)) {
                        shell_quadruples.emplace_back(a, b, c, d);
                    }
                }
            }
        }
    }
    const int task_group_size = 16;
    const int num_braket_types = shell_quadruples.size();
    global_counters_.resize(num_braket_types, nullptr);
    min_skipped_columns_.resize(num_braket_types, nullptr);
    int s0, s1, s2, s3;
    ShellTypeInfo shell_s0, shell_s1; //shell_s2, shell_s3;
    int num_bra, num_bra_groups;
    for (int idx = 0; idx < num_braket_types; ++idx) {
        std::tie(s0, s1, s2, s3) = shell_quadruples[idx];
        shell_s0 = shell_type_infos[s0];
        shell_s1 = shell_type_infos[s1];
        num_bra = (s0 == s1) ? shell_s0.count * (shell_s0.count + 1) / 2 : shell_s0.count * shell_s1.count;
        num_bra_groups = (num_bra + task_group_size - 1) / task_group_size;
//        cudaMalloc(&global_counters_[idx], sizeof(int) * num_bra_groups);
//        cudaMalloc(&min_skipped_columns_[idx], sizeof(int) * num_bra_groups);
        global_counters_[idx] = sycl::malloc_device<int>(num_bra_groups, q_ct1);
        min_skipped_columns_[idx] = sycl::malloc_device<int>(num_bra_groups, q_ct1);
//        global_counters_[idx] = tracked_syclMalloc<int>(num_bra_groups);
//        min_skipped_columns_[idx] = tracked_syclMalloc<int>(num_bra_groups);
    }

    gpu::computeSchwarzUpperBounds(
        shell_type_infos,
        shell_pair_type_infos,
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(), 
        cgto_normalization_factors.device_ptr(), 
        schwarz_upper_bound_factors.device_ptr(), 
        verbose
        );

    // Create an array for storing pairs of primitive shell indices
    const size_t num_primitive_shell_pairs = primitive_shells.size() * (primitive_shells.size() + 1) / 2;
    sycl::int2* d_primitive_shell_pair_indices = primitive_shell_pair_indices.device_ptr();

    // Store the pairs of primitive shell indices and sort them based on the Schwarz upper bound factors
    int pair_idx = 0;
//    const int threads_per_block = 1024;
    size_t max_wg = q_ct1.get_device().get_info<sycl::info::device::max_work_group_size>();
    size_t threads_per_block = std::min<size_t>(1024, max_wg);
    for(int s0 = 0; s0 < shell_type_infos.size(); s0++){
        for(int s1 = s0; s1 < shell_type_infos.size(); s1++){
            const int num_blocks = (shell_pair_type_infos[pair_idx].count + threads_per_block - 1) / threads_per_block; // the number of blocks
            /*
            DPCT1049:1: The work-group size passed to the SYCL kernel may exceed
            the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            q_ct1.submit([&](sycl::handler &cgh) {
                auto
                    d_primitive_shell_pair_indices_shell_pair_type_infos_pair_idx_start_index_ct0 =
                        &d_primitive_shell_pair_indices
                            [shell_pair_type_infos[pair_idx].start_index];
                int shell_pair_type_infos_pair_idx_count_ct1 =
                    shell_pair_type_infos[pair_idx].count;
                auto s0_s1_ct2 = s0 == s1;
                auto shell_type_infos_s1_count_ct3 = shell_type_infos[s1].count;

                cgh.parallel_for(sycl::nd_range<1>(num_blocks * threads_per_block, threads_per_block),
                    [=](sycl::nd_item<1> item_ct1) {
                        initializePrimitiveShellPairIndices( item_ct1,
                            d_primitive_shell_pair_indices_shell_pair_type_infos_pair_idx_start_index_ct0,
                            shell_pair_type_infos_pair_idx_count_ct1, s0_s1_ct2,
                            shell_type_infos_s1_count_ct3);
                    });
            });

            real_t* keys_begin =
                schwarz_upper_bound_factors.device_ptr() +
                    shell_pair_type_infos[pair_idx].start_index;
            real_t* keys_end = keys_begin + shell_pair_type_infos[pair_idx].count;

            sycl::int2* values_begin =
                  d_primitive_shell_pair_indices +
                  shell_pair_type_infos[pair_idx].start_index;

            oneapi::dpl::sort_by_key(
                oneapi::dpl::execution::make_device_policy(q_ct1),
                keys_begin, keys_end,
                values_begin,
                std::greater<real_t>()
            );

/*
            dpct::device_pointer<real_t> keys_begin(
                &schwarz_upper_bound_factors.device_ptr()
                     [shell_pair_type_infos[pair_idx].start_index]);
            dpct::device_pointer<real_t> keys_end(
                &schwarz_upper_bound_factors.device_ptr()
                     [shell_pair_type_infos[pair_idx].start_index] +
                shell_pair_type_infos[pair_idx].count);
            dpct::device_pointer<sycl::int2> values_begin(
                &d_primitive_shell_pair_indices[shell_pair_type_infos[pair_idx]
                                                    .start_index]);
            dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1),
                       keys_begin, keys_end, values_begin,
                       std::greater<real_t>());
*/

            pair_idx++;
        }
    q_ct1.wait_and_throw();
    }








}


ERI_Hash::ERI_Hash(const HF& hf):
    hf_(hf),
    num_basis_(hf.get_num_basis())
{
    // ここでHash memoryの初期化をおこなう
}

void ERI_Hash::precomputation() {
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const int verbose = hf_.get_verbose();

    gpu::constructERIHash(
        shell_type_infos,
        shell_pair_type_infos,
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(), 
        cgto_normalization_factors.device_ptr(), 
        // Hash memoryのポインタを渡す
        verbose
    );
}











// full_range
// All of CGTO Idx pair {a,b} satisfying a < b
inline void generatePrimitiveShellPairIndices_for_SAD_K_computation(const sycl::nd_item<1>& item_ct1, size_t2* d_primitive_shell_pair_indices_for_SAD_K_computation, const PrimitiveShell* d_primitive_shells, int num_primitive_shells, size_t num_threads){
	const size_t id = item_ct1.get_global_linear_id();
    if (id >= num_threads) return;
    
    size_t2 res = index1to2(id, false, num_primitive_shells);

    d_primitive_shell_pair_indices_for_SAD_K_computation[id] = res;
}


inline void copySchwarzUpperBoundFactors_for_SAD_K_computation(const sycl::nd_item<1>& item_ct1, real_t* d_schwarz_upper_bound_factors_for_SAD_K_computation, ShellPairSorter* d_shell_pair_sorter_for_SAD_K_computation, const size_t num_primitive_shells) {
	const size_t id = item_ct1.get_global_linear_id();
    if (id >= num_primitive_shells * num_primitive_shells) return;

    d_schwarz_upper_bound_factors_for_SAD_K_computation[id] = d_shell_pair_sorter_for_SAD_K_computation[id].schwarz_upper_bound_ab;
}
 
 

ERI_RI_Direct::ERI_RI_Direct(const HF& hf, const Molecular& auxiliary_molecular): 
    hf_(hf),
    num_basis_(hf.get_num_basis()),
    num_auxiliary_basis_(auxiliary_molecular.get_num_basis()),
    auxiliary_shell_type_infos_(auxiliary_molecular.get_shell_type_infos()),
    auxiliary_primitive_shells_(auxiliary_molecular.get_primitive_shells()),
    auxiliary_cgto_normalization_factors_(auxiliary_molecular.get_cgto_normalization_factors()),
    schwarz_upper_bound_factors(hf.get_num_primitive_shell_pairs()),
    auxiliary_schwarz_upper_bound_factors(auxiliary_molecular.get_primitive_shells().size()),
    two_center_eris(num_auxiliary_basis_ * num_auxiliary_basis_), 
    two_center_eris_inverse(num_auxiliary_basis_ * num_auxiliary_basis_), 
    primitive_shell_pair_indices(hf_.get_primitive_shells().size() * (hf_.get_primitive_shells().size() + 1) / 2),
    schwarz_upper_bound_factors_for_SAD_K_computation((hf_.get_initial_guess_algorithm_name() == "sad") ? hf_.get_primitive_shells().size() * hf_.get_primitive_shells().size() : 0),
    primitive_shell_pair_indices_for_SAD_K_computation((hf_.get_initial_guess_algorithm_name() == "sad") ? hf_.get_primitive_shells().size() * hf_.get_primitive_shells().size() : 0)
{
    // to device memory
    auxiliary_primitive_shells_.toDevice();
    auxiliary_cgto_normalization_factors_.toDevice();
}


void ERI_RI_Direct::precomputation() {
    // compute the intermediate matrix B of the auxiliary basis functions
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const int verbose = hf_.get_verbose();

    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const real_t schwarz_screening_threshold = hf_.get_schwarz_screening_threshold();

//    const int threads_per_block = 1024;
    sycl::queue& q_ct1 = gpu::GPUHandle::syclqueue();
    size_t max_wg = q_ct1.get_device().get_info<sycl::info::device::max_work_group_size>();
    size_t threads_per_block = std::min<size_t>(1024, max_wg);

    // K 計算用のソートに使用
    const size_t num_primitive_shells = primitive_shells.size();


    // K計算用のペア配列生成
    if(schwarz_upper_bound_factors_for_SAD_K_computation.size() > 0){
        size_t num_tasks = num_primitive_shells*num_primitive_shells;
        size_t num_blocks = (num_tasks + threads_per_block - 1) / threads_per_block;
//        generatePrimitiveShellPairIndices_for_SAD_K_computation<<<num_blocks, threads_per_block>>>(primitive_shell_pair_indices_for_SAD_K_computation.device_ptr(), primitive_shells.device_ptr(), num_primitive_shells, num_tasks);
        q_ct1.submit([&](sycl::handler& cgh) {
            auto* primitive_SAD_ptr = primitive_shell_pair_indices_for_SAD_K_computation.device_ptr();
            auto* primitive_ptr = primitive_shells.device_ptr();
            cgh.parallel_for(
                sycl::nd_range<1>(num_blocks * threads_per_block, threads_per_block),
                [=](sycl::nd_item<1> item) {
                    generatePrimitiveShellPairIndices_for_SAD_K_computation(
                        item,
                        primitive_SAD_ptr,
                        primitive_ptr,
                        num_primitive_shells,
                        num_tasks
                    );
                }
            );
        });
        
        // Sort用構造体配列
        ShellPairSorter* d_shell_pair_sorter_for_SAD_K_computation; 
        if(schwarz_upper_bound_factors_for_SAD_K_computation.size() > 0)
            d_shell_pair_sorter_for_SAD_K_computation = sycl::malloc_device<ShellPairSorter>(num_tasks, q_ct1);
//            cudaMalloc((void**)&d_shell_pair_sorter_for_SAD_K_computation, sizeof(ShellPairSorter)*num_tasks);
        
        
        // compute upper bounds of primitive-shell-pair
        // 通常のshell pairの上界計算も行う
        gpu::computeSchwarzUpperBounds_for_SAD_K_computation(
            shell_type_infos,
            shell_pair_type_infos,
            primitive_shells.device_ptr(),
            boys_grid.device_ptr(), 
            cgto_normalization_factors.device_ptr(),    
            schwarz_upper_bound_factors.device_ptr(),   // schwarz_upper_bound_factorsに√(pq|pq)の値がはいっている
            d_shell_pair_sorter_for_SAD_K_computation,
            num_primitive_shells,
            verbose
        );
        
        // K計算用のshell-pair配列ソート
        ShellPairSorter* keys_begin = d_shell_pair_sorter_for_SAD_K_computation;
        ShellPairSorter* keys_end   = keys_begin + num_tasks;
        size_t2* values_begin = primitive_shell_pair_indices_for_SAD_K_computation.device_ptr();

        oneapi::dpl::sort_by_key( oneapi::dpl::execution::make_device_policy(q_ct1), keys_begin, keys_end, values_begin);
/*
        thrust::device_ptr<ShellPairSorter> keys_begin(d_shell_pair_sorter_for_SAD_K_computation);  
        thrust::device_ptr<ShellPairSorter> keys_end(d_shell_pair_sorter_for_SAD_K_computation + num_tasks);
        thrust::device_ptr<size_t2> values_begin(primitive_shell_pair_indices_for_SAD_K_computation.device_ptr());
        thrust::sort_by_key(keys_begin, keys_end, values_begin);
*/        
//            size_t2* d_pairs = primitive_shell_pair_indices.device_ptr() + start_index;

            q_ct1.submit([&](sycl::handler& cgh) {
                auto* schwarz_ptr = schwarz_upper_bound_factors_for_SAD_K_computation.device_ptr();
                cgh.parallel_for(
                    sycl::nd_range<1>(num_blocks * threads_per_block, threads_per_block),
                    [=](sycl::nd_item<1> item) {
                        copySchwarzUpperBoundFactors_for_SAD_K_computation(item,
                            schwarz_ptr,
                            d_shell_pair_sorter_for_SAD_K_computation,
                            num_primitive_shells
                        );
                    }
                );
            });
//        copySchwarzUpperBoundFactors_for_SAD_K_computation<<<num_blocks, threads_per_block>>>(schwarz_upper_bound_factors_for_SAD_K_comp    utation.device_ptr(), d_shell_pair_sorter_for_SAD_K_computation, num_primitive_shells);
        
        primitive_shell_pair_indices_for_SAD_K_computation.toHost();
        sycl::free(d_shell_pair_sorter_for_SAD_K_computation,q_ct1);
//        cudaFree(d_shell_pair_sorter_for_SAD_K_computation);
    }else{
        gpu::computeSchwarzUpperBounds(
            shell_type_infos,
            shell_pair_type_infos,
            primitive_shells.device_ptr(), 
            boys_grid.device_ptr(), 
            cgto_normalization_factors.device_ptr(), 
            schwarz_upper_bound_factors.device_ptr(),   // schwarz_upper_bound_factorsに√(pq|pq)の値がはいっている
            verbose
        );
    }



    // shell-pair sort
    int pair_idx = 0;
    for(int s0 = 0; s0 < shell_type_infos.size(); s0++){
        for(int s1 = s0; s1 < shell_type_infos.size(); s1++){

            const int pair_idx_local = pair_idx;
            const int count = shell_pair_type_infos[pair_idx_local].count;
            if (count == 0) {
                pair_idx++;
                continue;
            }

            const int start_index = shell_pair_type_infos[pair_idx_local].start_index;
            const size_t num_blocks = ((count + threads_per_block - 1) / threads_per_block);

            const bool same_shell = (s0 == s1);
            const int s1_count = shell_type_infos[s1].count;
            const int s0_start = shell_type_infos[s0].start_index;
            const int s1_start = shell_type_infos[s1].start_index;

            size_t2* d_pairs = primitive_shell_pair_indices.device_ptr() + start_index;

            q_ct1.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(
                    sycl::nd_range<1>(num_blocks * threads_per_block, threads_per_block),
                    [=](sycl::nd_item<1> item) {
                        generatePrimitiveShellPairIndices(
                            item,
                            d_pairs,
                            count,
                            same_shell,
                            s1_count,
                            true,
                            s0_start,
                            s1_start
                        );
                    }
                );
            });

            real_t* keys_begin = schwarz_upper_bound_factors.device_ptr() + start_index;
            real_t* keys_end = keys_begin + count;

            size_t2* values_begin = primitive_shell_pair_indices.device_ptr() + start_index;

            oneapi::dpl::sort_by_key(
                oneapi::dpl::execution::make_device_policy(q_ct1),
                keys_begin,
                keys_end,
                values_begin,
                std::greater<real_t>()
            );

/*

            const int num_blocks = (shell_pair_type_infos[pair_idx].count + threads_per_block - 1) / threads_per_block; // the number of blocks
            generatePrimitiveShellPairIndices<<<num_blocks, threads_per_block>>>(&primitive_shell_pair_indices.device_ptr()[shell_pair_type_infos[pair_idx].start_index], shell_pair_type_infos[pair_idx].count, s0 == s1, shell_type_infos[s1].count, true, shell_type_infos[s0].start_index, shell_type_infos[s1].start_index);

            thrust::device_ptr<real_t> keys_begin(&schwarz_upper_bound_factors.device_ptr()[shell_pair_type_infos[pair_idx].start_index]);  
            thrust::device_ptr<real_t> keys_end(&schwarz_upper_bound_factors.device_ptr()[shell_pair_type_infos[pair_idx].start_index] + shell_pair_type_infos[pair_idx].count);
            thrust::device_ptr<size_t2> values_begin(&primitive_shell_pair_indices.device_ptr()[shell_pair_type_infos[pair_idx].start_index]);

            thrust::sort_by_key(keys_begin, keys_end, values_begin, thrust::greater<real_t>());
*/
            pair_idx++;
        }
    }
    q_ct1.wait();

    
    
    // compute upper bounds of  aux-shell
    gpu::computeAuxiliarySchwarzUpperBounds(
        auxiliary_shell_type_infos_, 
        auxiliary_primitive_shells_.device_ptr(), 
        boys_grid.device_ptr(), 
        auxiliary_cgto_normalization_factors_.device_ptr(), 
        auxiliary_schwarz_upper_bound_factors.device_ptr(),   // auxiliary_schwarz_upper_bound_factorsに√(pq|pq)の値がはいっている
        verbose
    );

    for (const auto& s : auxiliary_shell_type_infos_) {

        const int count = s.count;
        if (count == 0) continue;

        real_t* keys_begin = auxiliary_schwarz_upper_bound_factors.device_ptr() + s.start_index;
        real_t* keys_end = keys_begin + count;

        PrimitiveShell* values_begin = auxiliary_primitive_shells_.device_ptr() + s.start_index;

        oneapi::dpl::sort_by_key(
           oneapi::dpl::execution::make_device_policy(q_ct1),
            keys_begin,
            keys_end,
            values_begin,
            std::greater<real_t>()
        );
    }

    q_ct1.wait();
/*
    for(const auto& s : auxiliary_shell_type_infos_){
        thrust::device_ptr<real_t> keys_begin(&auxiliary_schwarz_upper_bound_factors.device_ptr()[s.start_index]);  
        thrust::device_ptr<real_t> keys_end(&auxiliary_schwarz_upper_bound_factors.device_ptr()[s.start_index] + s.count);
        thrust::device_ptr<PrimitiveShell> values_begin(&auxiliary_primitive_shells_.device_ptr()[s.start_index]);

        thrust::sort_by_key(keys_begin, keys_end, values_begin, thrust::greater<real_t>());
    }
*/

    gpu::computeTwoCenterERIs(
        auxiliary_shell_type_infos_, 
        auxiliary_primitive_shells_.device_ptr(), 
        auxiliary_cgto_normalization_factors_.device_ptr(), 
        two_center_eris.device_ptr(),
        num_auxiliary_basis_,
        boys_grid.device_ptr(),
        auxiliary_schwarz_upper_bound_factors.device_ptr(),
        schwarz_screening_threshold,
        verbose
    );


    gpu::choleskyDecomposition(two_center_eris.device_ptr(), num_auxiliary_basis_);
    gpu::computeInverseByDtrsm(two_center_eris.device_ptr(), two_center_eris_inverse.device_ptr(), num_auxiliary_basis_);
}


} // namespace gansu
