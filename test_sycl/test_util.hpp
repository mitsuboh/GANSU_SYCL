#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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

// Helper function: Copy device memory to host
inline void copyToHost(std::vector<double>& host_data, const double* device_data, size_t size) {
    dpct::get_in_order_queue()
        .memcpy(host_data.data(), device_data, size * sizeof(double))
        .wait();
}
