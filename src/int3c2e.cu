#include <cuda.h>

#include "int3c2e.hpp"
#include "boys.hpp"
#include "parameters.h"
#include "types.hpp"
#include "utils_cuda.hpp"




namespace gansu::gpu{
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
// 3-center integrals [ss|s]~[pp|d]
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//


/* (ss|s) */
__global__ void calc_sss_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
                             const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, 
                             ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
                             int64_t num_tasks, int num_basis, int num_auxiliary_basis, 
                             const double* g_boys_grid){
                                
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[1];

		getIncrementalBoys(0, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);

		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 0) 
		                     * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) 
							 * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_sss.txt"
	}
}



/* (ss|p) */
__global__ void calc_ssp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[2];
		getIncrementalBoys(1, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ssp.txt"
	}
}



/* (ss|d) */
__global__ void calc_ssd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

        // printf("ssd| %d: %d %d %d\n",threadIdx.x, (int)primitive_index_a,(int)primitive_index_b,(int)primitive_index_c);

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ssd.txt"
	}
}



/* (ss|f) */
__global__ void calc_ssf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

        // printf("ssf| %d: %d %d %d\n",threadIdx.x, (int)primitive_index_a,(int)primitive_index_b,(int)primitive_index_c);

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ssf.txt"
	}
}



/* (sp|s) */
__global__ void calc_sps_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[2];
		getIncrementalBoys(1, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_sps.txt"
	}
}



/* (sp|p) */
__global__ void calc_spp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_spp.txt"
	}
}



/* (sp|d) */
__global__ void calc_spd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		
		#include "./integral_RI/int3c2e/orig_spd.txt"
	}
}



/* (sp|f) */
__global__ void calc_spf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_spf.txt"
	}
}




/* (pp|s) */
__global__ void calc_pps_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_pps.txt"
	}
}



/* (pp|p) */
__global__ void calc_ppp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ppp.txt"
	}
}



/* (pp|d) */
__global__ void calc_ppd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ppd.txt"
	}
}



/* (pp|f) */
__global__ void calc_ppf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ppf.txt"
	}
}



#if defined(COMPUTE_D_BASIS)
/* (sd|s) */
__global__ void calc_sds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_sds.txt"
	}
}


/* (sd|p) */
__global__ void calc_sdp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_sdp.txt"
	}
}



/* (sd|d) */
__global__ void calc_sdd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_sdd.txt"
	}
}



/* (sd|f) */
__global__ void calc_sdf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_sdf.txt"
	}
}

/* (pd|s) */
__global__ void calc_pds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_pds.txt"
	}
}


/* (pd|p) */
__global__ void calc_pdp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_pdp.txt"
	}
}



/* (pd|d) */
__global__ void calc_pdd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_pdd.txt"
	}
}



/* (pd|f) */
__global__ void calc_pdf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_pdf.txt"
	}
}


/* (dd|s) */
__global__ void calc_dds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
       bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_dds.txt"
	}
}



/* (dd|p) */
__global__ void calc_ddp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
       bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_ddp.txt"
	}
}



/* (dd|d) */
__global__ void calc_ddd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
       bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_ddd.txt"
	}
}


/* (dd|f) */
__global__ void calc_ddf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[8];
		getIncrementalBoys(7, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
        bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_ddf.txt"
	}
}

#else
/* (dd|f) */
__global__ void calc_ddf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (dd|d) */
__global__ void calc_ddd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (dd|p) */
 __global__ void calc_ddp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (dd|s) */
 __global__ void calc_dds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (pd|f) */
 __global__ void calc_pdf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (pd|d) */
 __global__ void calc_pdd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (pd|p) */
 __global__ void calc_pdp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (pd|s) */
 __global__ void calc_pds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (sd|f) */
 __global__ void calc_sdf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (sd|d) */
 __global__ void calc_sdd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (sd|p) */
 __global__ void calc_sdp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (sd|s) */
 __global__ void calc_sds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
#endif




#if defined(COMPUTE_D_BASIS) && defined(COMPUTE_G_AUX)
/* (sd|g) */
__global__ void calc_sdg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_sdg.txt"
	}
}

/* (pd|g) */
__global__ void calc_pdg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[8];
		getIncrementalBoys(7, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_pdg.txt"
	}
}


/* (dd|g) */
__global__ void calc_ddg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[9];
		getIncrementalBoys(8, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_ddg.txt"
	}
}

#else
__global__ void calc_sdg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (dd|g) */
__global__ void calc_ddg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (pd|g) */
__global__ void calc_pdg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
#endif



#if defined(COMPUTE_G_AUX)
/* (ss|g) */
__global__ void calc_ssg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ssg.txt"
	}
}

/* (sp|g) */
__global__ void calc_spg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, false, shell_s1.count);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_spg.txt"
	}
}

/* (pp|g) */
__global__ void calc_ppg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		const size_t2 ab =  index1to2(abc.x, true);
		const size_t primitive_index_a = ab.x + shell_s0.start_index;
		const size_t primitive_index_b = ab.y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		// if(a!=b && a->cgtoIdx==b->cgtoIdx) coefAndNorm *= 2.0;
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ppg.txt"
	}
}

#else
/* (ss|g) */
__global__ void calc_ssg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (sp|g) */	
__global__ void calc_spg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
/* (pp|g) */
__global__ void calc_ppg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, int num_auxiliary_basis, const double* g_boys_grid){}
#endif


} // namespace gansu::gpu