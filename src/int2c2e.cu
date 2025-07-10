#include <cuda.h>

#include "int2c2e.hpp"
#include "boys.hpp"
#include "parameters.h"
#include "types.hpp"
#include "utils_cuda.hpp"


namespace gansu::gpu{

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
// 2-center integrals [s|s]~[d|d]
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//



/* (s|s) */
__global__ void calc_ss_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		const size_t2 ab = index1to2(idx, true);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		// printf("[s|s] %d: %d %d\n", threadIdx.x, (int)primitive_index_a, (int)primitive_index_b);

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[1];
		getIncrementalBoys(0, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 0, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_ss.txt"
	}
}


/* (s|p) */
__global__ void calc_sp_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		// printf("[s|p] %d: %d %d\n", threadIdx.x, (int)primitive_index_a, (int)primitive_index_b);
		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[2];
		getIncrementalBoys(1, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 0, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_sp.txt"
	}
}


/* (s|d) */
__global__ void calc_sd_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		// printf("[s|d] %d: %d %d\n", threadIdx.x, (int)primitive_index_a, (int)primitive_index_b);

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[3];
		getIncrementalBoys(2, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 0, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_sd.txt"
	}
}


/* (s|f) */
__global__ void calc_sf_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[4];
		getIncrementalBoys(3, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 0, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_sf.txt"
	}
}


/* (p|p) */
__global__ void calc_pp_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		const size_t2 ab = index1to2(idx, true);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		// printf("[p|p] %d: %d %d\n", threadIdx.x, (int)primitive_index_a, (int)primitive_index_b);

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[3];
		getIncrementalBoys(2, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 1, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);
		#include "./integral_RI/int2c2e/orig_pp.txt"
	}
}


/* (p|d) */
__global__ void calc_pd_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		// printf("[p|d] %d: %d %d\n", threadIdx.x, (int)primitive_index_a, (int)primitive_index_b);

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[4];
		getIncrementalBoys(3, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 1, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_pd.txt"
	}
}


/* (p|f) */
__global__ void calc_pf_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[5];
		getIncrementalBoys(4, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);

		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 1, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_pf.txt"
	}
}



/* (d|d) */
__global__ void calc_dd_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		const size_t2 ab = index1to2(idx, true);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		// printf("[d|d] %d: %d %d\n", threadIdx.x, (int)primitive_index_a, (int)primitive_index_b);

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[5];
		getIncrementalBoys(4, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_dd.txt"
	}
}


/* (d|f) */
__global__ void calc_df_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[6];
		getIncrementalBoys(5, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_df.txt"
	}
}


/* (f|f) */
__global__ void calc_ff_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		const size_t2 ab = index1to2(idx, true);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[7];
		getIncrementalBoys(6, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 3, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_ff.txt"
	}
}







#if defined(COMPUTE_G_AUX)
/* (s|g) */
__global__ void calc_sg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[5];
		getIncrementalBoys(4, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 0, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_sg.txt"
	}
}

/* (p|g) */
__global__ void calc_pg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[6];
		getIncrementalBoys(5, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 1, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_pg.txt"
	}
}

/* (d|g) */
__global__ void calc_dg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[7];
		getIncrementalBoys(6, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 2, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_dg.txt"
	}
}

/* (f|g) */
__global__ void calc_fg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[8];
		getIncrementalBoys(7, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 3, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_fg.txt"
	}
}


/* (g|g) */
__global__ void calc_gg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		const size_t2 ab = index1to2(idx, true);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[9];
		getIncrementalBoys(8, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 4, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_gg.txt"
	}
}
#else
	__global__ void calc_sg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){}
	__global__ void calc_pg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){}
	__global__ void calc_dg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){}
	__global__ void calc_fg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){}
	__global__ void calc_gg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, int num_auxiliary_basis, const double* g_boys_grid){}
#endif

} // namespace gansu::gpu