#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include "int2c2e.hpp"
#include "boys.hpp"
#include "parameters.h"
#include "types.hpp"
#include "utils_cuda.hpp"

#include "int2e.hpp"
#include "Et_functions.hpp"



namespace gansu::gpu{

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
// 2-center integrals [s|s]~[d|d]
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//



/* (s|s) */
/* (s|p) */
/* (s|d) */
/* (s|f) */
/* (p|p) */
/* (p|d) */
/* (p|f) */
/* (d|d) */
/* (d|f) */
/* (f|f) */
#if defined(COMPUTE_G_AUX)
/* (s|g) */
/* (p|g) */
/* (d|g) */
/* (f|g) */
/* (g|g) */
#endif

} // namespace gansu::gpu
