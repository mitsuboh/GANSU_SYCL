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



#ifndef INT1E_CUH
#define INT1E_CUH

#include "types.hpp"
#include "utils_cuda.hpp"

namespace gansu::gpu{

// MD method(overlap and kinetic integral)
__global__
void overlap_kinetic_MDss(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__ 
void overlap_kinetic_MDsp(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__ 
void overlap_kinetic_MDpp(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_MDsd(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_MDpd(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_MDdd(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_MDsf(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_MDpf(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_MDdf(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_MDff(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

// OS method(overlap and kinetic integral)
__global__
void overlap_kinetic_OSss(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__ 
void overlap_kinetic_OSsp(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__ 
void overlap_kinetic_OSpp(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_OSsd(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_OSpd(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_OSdd(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_OSsf(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_OSpf(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_OSdf(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);

__global__
void overlap_kinetic_OSff(double* g_overlap, double* g_kinetic, 
                        PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                        size_t num_threads,
                        int num_basis);



// MD method(nuclear attraction integral)
__global__ 
void nuclear_attraction_MDss(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_MDsp(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_MDpp(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_MDsd(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_MDpd(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_MDdd(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_MDsf(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_MDpf(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_MDdf(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_MDff(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);


// OS method(nuclear attraction integral)
__global__ 
void nuclear_attraction_OSss(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_OSsp(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_OSpp(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_OSsd(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_OSpd(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_OSdd(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_OSsf(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_OSpf(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_OSdf(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

__global__ 
void nuclear_attraction_OSff(double* g_nucattr, PrimitiveShell *g_shell, real_t* g_cgto_normalization_factors,  
                           Atom* g_atom, int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                           size_t num_threads,
                           int num_basis, double* g_boys_grid);

// 2025-05-26 define a function to target matrices
__global__ void Matrix_Symmetrization(double* matrix, int n);

// define the kernel functions as function pointers for one electron integrals
using overlap_kinect_kernel_t     = void (*)(real_t*, real_t*, PrimitiveShell*, real_t*, ShellTypeInfo, ShellTypeInfo, size_t, int);
using nuclear_attraction_kernel_t = void (*)(real_t*, PrimitiveShell*, real_t*, Atom*, int, ShellTypeInfo, ShellTypeInfo, size_t, int, real_t*);

inline overlap_kinect_kernel_t get_overlap_kinetic_kernel(int a, int b){
    int flag=0;

    if(flag){
        if(a==0 && b==0)      return overlap_kinetic_MDss;
        else if(a==0 && b==1) return overlap_kinetic_MDsp;
        else if(a==1 && b==1) return overlap_kinetic_MDpp;
        else if(a==0 && b==2) return overlap_kinetic_MDsd;
        else if(a==1 && b==2) return overlap_kinetic_MDpd;
        else if(a==2 && b==2) return overlap_kinetic_MDdd;
        else if(a==0 && b==3) return overlap_kinetic_MDsf;
        else if(a==1 && b==3) return overlap_kinetic_MDpf;
        else if(a==2 && b==3) return overlap_kinetic_MDdf;
        else if(a==3 && b==3) return overlap_kinetic_MDff;
        else throw std::runtime_error("Invalid shell type");
    }
    else{
        if(a==0 && b==0)      return overlap_kinetic_OSss;
        else if(a==0 && b==1) return overlap_kinetic_OSsp;
        else if(a==1 && b==1) return overlap_kinetic_OSpp;
        else if(a==0 && b==2) return overlap_kinetic_OSsd;
        else if(a==1 && b==2) return overlap_kinetic_OSpd;
        else if(a==2 && b==2) return overlap_kinetic_OSdd;
        else if(a==0 && b==3) return overlap_kinetic_OSsf;
        else if(a==1 && b==3) return overlap_kinetic_OSpf;
        else if(a==2 && b==3) return overlap_kinetic_OSdf;
        else if(a==3 && b==3) return overlap_kinetic_OSff;
        else throw std::runtime_error("Invalid shell type");
    }
}

inline nuclear_attraction_kernel_t get_nuclear_attraction_kernel(int a, int b){
    int flag=0;

    if(flag){
        if(a==0 && b==0)      return nuclear_attraction_MDss;
        else if(a==0 && b==1) return nuclear_attraction_MDsp;
        else if(a==1 && b==1) return nuclear_attraction_MDpp;
        else if(a==0 && b==2) return nuclear_attraction_MDsd;
        else if(a==1 && b==2) return nuclear_attraction_MDpd;
        else if(a==2 && b==2) return nuclear_attraction_MDdd;
        else if(a==0 && b==3) return nuclear_attraction_MDsf;
        else if(a==1 && b==3) return nuclear_attraction_MDpf;
        else if(a==2 && b==3) return nuclear_attraction_MDdf;
        else if(a==3 && b==3) return nuclear_attraction_MDff;
        else throw std::runtime_error("Invalid shell type");
    }
    else{
        if(a==0 && b==0)      return nuclear_attraction_OSss;
        else if(a==0 && b==1) return nuclear_attraction_OSsp;
        else if(a==1 && b==1) return nuclear_attraction_OSpp;
        else if(a==0 && b==2) return nuclear_attraction_OSsd;
        else if(a==1 && b==2) return nuclear_attraction_OSpd;
        else if(a==2 && b==2) return nuclear_attraction_OSdd;
        else if(a==0 && b==3) return nuclear_attraction_OSsf;
        else if(a==1 && b==3) return nuclear_attraction_OSpf;
        else if(a==2 && b==3) return nuclear_attraction_OSdf;
        else if(a==3 && b==3) return nuclear_attraction_OSff;
        else throw std::runtime_error("Invalid shell type");
    }

}

} // namespace gansu::gpu

#endif
