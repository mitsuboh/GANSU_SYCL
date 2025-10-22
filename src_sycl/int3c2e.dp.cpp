#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include "int3c2e.hpp"
#include "boys.hpp"
#include "parameters.h"
#include "types.hpp"
#include "utils_cuda.hpp"

#include "int2e.hpp"
#include "Et_functions.hpp"



namespace gansu::gpu{
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
// 3-center integrals [ss|s]~[pp|d]
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//


/* (ss|s) */
/* (ss|p) */
/* (ss|d) */
/* (ss|f) */
/* (sp|s) */
/* (sp|p) */
/* (sp|d) */
/* (sp|f) */
/* (pp|s) */
/* (pp|p) */
/* (pp|d) */
/* (pp|f) */
#if defined(COMPUTE_D_BASIS)
/* (sd|s) */
/* (sd|p) */
/* (sd|d) */
/* (sd|f) */
/* (pd|s) */
/* (pd|p) */
/* (pd|d) */
/* (pd|f) */
/* (dd|s) */
/* (dd|p) */
/* (dd|d) */
/* (dd|f) */
#else
/* (dd|f) */
/* (dd|d) */
/* (dd|p) */
/* (dd|s) */
/* (pd|f) */
/* (pd|d) */
/* (pd|p) */
/* (pd|s) */
/* (sd|f) */
/* (sd|d) */
/* (sd|p) */
/* (sd|s) */
#endif




#if defined(COMPUTE_D_BASIS) && defined(COMPUTE_G_AUX)
/* (sd|g) */
/* (pd|g) */
/* (dd|g) */
#else
/* (sd|g) */
/* (pd|g) */
/* (dd|g) */
#endif



#if defined(COMPUTE_G_AUX)
/* (ss|g) */
/* (sp|g) */
/* (pp|g) */
#else
/* (ss|g) */
/* (sp|g) */
/* (pp|g) */
#endif

/*
DPCT1110:96: The total declared local variable size in device function
MD_int3c2e_1T1SP exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
/*
void MD_int3c2e_1T1SP(real_t *g_result, const PrimitiveShell *g_pshell,
                      const PrimitiveShell *g_pshell_aux,
                      const real_t *d_cgto_nomalization_factors,
                      const real_t *d_auxiliary_cgto_nomalization_factors,
                      ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                      ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis,
                      const size_t2 *d_primitive_shell_pair_indices,
                      const double *g_upper_bound_factors,
                      const double *g_auxiliary_upper_bound_factors,
                      const double schwarz_screening_threshold,
                      int num_auxiliary_basis, const double *g_boys_grid){
//                      dpct::accessor<int, dpct::constant, 3> loop_to_ang_RI,
//                      dpct::accessor<int, dpct::constant, 2> tuv_list,
//                      double (*MD_EtArray[])(double, double, double, double,
//                                             double) *
//                          *MD_EtArray) {
*/


} // namespace gansu::gpu
