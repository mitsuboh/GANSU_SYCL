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

#pragma once
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <vector>
#include<cmath>
#include<string>
#include<fstream>
#include<iostream>
#include<sstream>
#include<stdlib.h>
#include<sys/time.h>
#include<algorithm>
#include<random>


namespace gansu::gpu{

inline double MD_Et_GPU000(double a, double b, double p, double d, double s){
	return s;
}

inline double MD_Et_GPU010(double a, double b, double p, double d, double s){
	return (a*d/p)*s;
}

inline double MD_Et_GPU011(double a, double b, double p, double d, double s){
	return (0.5/p)*s;
}

inline double MD_Et_GPU020(double a, double b, double p, double d, double s){
	return (((a*a)*(d*d) + 0.5*p)/(p*p))*s;
}

inline double MD_Et_GPU021(double a, double b, double p, double d, double s){
	return (a*d/(p*p))*s;
}

inline double MD_Et_GPU022(double a, double b, double p, double d, double s){
	return (0.25/(p*p))*s;
}

inline double MD_Et_GPU030(double a, double b, double p, double d, double s){
	return (a*d*((a*a)*(d*d) + 1.5*p)/(p*p*p))*s;
}

inline double MD_Et_GPU031(double a, double b, double p, double d, double s){
	return ((1.5*(a*a)*(d*d) + 0.75*p)/(p*p*p))*s;
}

inline double MD_Et_GPU032(double a, double b, double p, double d, double s){
	return (0.75*a*d/(p*p*p))*s;
}

inline double MD_Et_GPU033(double a, double b, double p, double d, double s){
	return (0.125/(p*p*p))*s;
}

inline double MD_Et_GPU040(double a, double b, double p, double d, double s){
	return (((a*a*a*a)*(d*d*d*d) + 3.0*(a*a)*(d*d)*p + 0.75*(p*p))/(p*p*p*p))*s;
}

inline double MD_Et_GPU041(double a, double b, double p, double d, double s){
	return (a*d*(2.0*(a*a)*(d*d) + 3.0*p)/(p*p*p*p))*s;
}

inline double MD_Et_GPU042(double a, double b, double p, double d, double s){
	return ((1.5*(a*a)*(d*d) + 0.75*p)/(p*p*p*p))*s;
}

inline double MD_Et_GPU043(double a, double b, double p, double d, double s){
	return (0.5*a*d/(p*p*p*p))*s;
}

inline double MD_Et_GPU044(double a, double b, double p, double d, double s){
	return (0.0625/(p*p*p*p))*s;
}

inline double MD_Et_GPU050(double a, double b, double p, double d, double s){
	return (a*d*((a*a*a*a)*(d*d*d*d) + 5.0*(a*a)*(d*d)*p + 3.75*(p*p))/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU051(double a, double b, double p, double d, double s){
	return ((2.5*(a*a*a*a)*(d*d*d*d) + 7.5*(a*a)*(d*d)*p + 1.875*(p*p))/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU052(double a, double b, double p, double d, double s){
	return (a*d*(2.5*(a*a)*(d*d) + 3.75*p)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU053(double a, double b, double p, double d, double s){
	return ((1.25*(a*a)*(d*d) + 0.625*p)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU054(double a, double b, double p, double d, double s){
	return (0.3125*a*d/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU055(double a, double b, double p, double d, double s){
	return (0.03125/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU060(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a)*(d*d*d*d*d*d) + 7.5*(a*a*a*a)*(d*d*d*d)*p + 11.25*(a*a)*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU061(double a, double b, double p, double d, double s){
	return (a*d*(3.0*(a*a*a*a)*(d*d*d*d) + 15.0*(a*a)*(d*d)*p + 11.25*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU062(double a, double b, double p, double d, double s){
	return ((3.75*(a*a*a*a)*(d*d*d*d) + 11.25*(a*a)*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU063(double a, double b, double p, double d, double s){
	return (a*d*(2.5*(a*a)*(d*d) + 3.75*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU064(double a, double b, double p, double d, double s){
	return ((0.9375*(a*a)*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU065(double a, double b, double p, double d, double s){
	return (0.1875*a*d/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU066(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU070(double a, double b, double p, double d, double s){
	return (a*d*((a*a*a*a*a*a)*(d*d*d*d*d*d) + 10.5*(a*a*a*a)*(d*d*d*d)*p + 26.25*(a*a)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU071(double a, double b, double p, double d, double s){
	return ((3.5*(a*a*a*a*a*a)*(d*d*d*d*d*d) + 26.25*(a*a*a*a)*(d*d*d*d)*p + 39.375*(a*a)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU072(double a, double b, double p, double d, double s){
	return (a*d*(5.25*(a*a*a*a)*(d*d*d*d) + 26.25*(a*a)*(d*d)*p + 19.6875*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU073(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a)*(d*d*d*d) + 13.125*(a*a)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU074(double a, double b, double p, double d, double s){
	return (a*d*(2.1875*(a*a)*(d*d) + 3.28125*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU075(double a, double b, double p, double d, double s){
	return ((0.65625*(a*a)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU076(double a, double b, double p, double d, double s){
	return (0.109375*a*d/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU077(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU080(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) + 14.0*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p + 52.5*(a*a*a*a)*(d*d*d*d)*(p*p) + 52.5*(a*a)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU081(double a, double b, double p, double d, double s){
	return (a*d*(4.0*(a*a*a*a*a*a)*(d*d*d*d*d*d) + 42.0*(a*a*a*a)*(d*d*d*d)*p + 105.0*(a*a)*(d*d)*(p*p) + 52.5*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU082(double a, double b, double p, double d, double s){
	return ((7.0*(a*a*a*a*a*a)*(d*d*d*d*d*d) + 52.5*(a*a*a*a)*(d*d*d*d)*p + 78.75*(a*a)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU083(double a, double b, double p, double d, double s){
	return (a*d*(7.0*(a*a*a*a)*(d*d*d*d) + 35.0*(a*a)*(d*d)*p + 26.25*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU084(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a)*(d*d*d*d) + 13.125*(a*a)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU085(double a, double b, double p, double d, double s){
	return (a*d*(1.75*(a*a)*(d*d) + 2.625*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU086(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU087(double a, double b, double p, double d, double s){
	return (0.0625*a*d/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU088(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU100(double a, double b, double p, double d, double s){
	return (-b*d/p)*s;
}

inline double MD_Et_GPU101(double a, double b, double p, double d, double s){
	return (0.5/p)*s;
}

inline double MD_Et_GPU110(double a, double b, double p, double d, double s){
	return ((-a*b*(d*d) + 0.5*p)/(p*p))*s;
}

inline double MD_Et_GPU111(double a, double b, double p, double d, double s){
	return (0.5*d*(a - b)/(p*p))*s;
}

inline double MD_Et_GPU112(double a, double b, double p, double d, double s){
	return (0.25/(p*p))*s;
}

inline double MD_Et_GPU120(double a, double b, double p, double d, double s){
	return (d*(-(a*a)*b*(d*d) + a*p - 0.5*b*p)/(p*p*p))*s;
}

inline double MD_Et_GPU121(double a, double b, double p, double d, double s){
	return ((0.5*(a*a)*(d*d) - a*b*(d*d) + 0.75*p)/(p*p*p))*s;
}

inline double MD_Et_GPU122(double a, double b, double p, double d, double s){
	return (d*(0.5*a - 0.25*b)/(p*p*p))*s;
}

inline double MD_Et_GPU123(double a, double b, double p, double d, double s){
	return (0.125/(p*p*p))*s;
}

inline double MD_Et_GPU130(double a, double b, double p, double d, double s){
	return ((-(a*a*a)*b*(d*d*d*d) + 1.5*(a*a)*(d*d)*p - 1.5*a*b*(d*d)*p + 0.75*(p*p))/(p*p*p*p))*s;
}

inline double MD_Et_GPU131(double a, double b, double p, double d, double s){
	return (d*(0.5*(a*a*a)*(d*d) - 1.5*(a*a)*b*(d*d) + 2.25*a*p - 0.75*b*p)/(p*p*p*p))*s;
}

inline double MD_Et_GPU132(double a, double b, double p, double d, double s){
	return (0.75*((a*a)*(d*d) - a*b*(d*d) + p)/(p*p*p*p))*s;
}

inline double MD_Et_GPU133(double a, double b, double p, double d, double s){
	return (d*(0.375*a - 0.125*b)/(p*p*p*p))*s;
}

inline double MD_Et_GPU134(double a, double b, double p, double d, double s){
	return (0.0625/(p*p*p*p))*s;
}

inline double MD_Et_GPU140(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a)*b*(d*d*d*d) + 2.0*(a*a*a)*(d*d)*p - 3.0*(a*a)*b*(d*d)*p + 3.0*a*(p*p) - 0.75*b*(p*p))/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU141(double a, double b, double p, double d, double s){
	return ((0.5*(a*a*a*a)*(d*d*d*d) - 2.0*(a*a*a)*b*(d*d*d*d) + 4.5*(a*a)*(d*d)*p - 3.0*a*b*(d*d)*p + 1.875*(p*p))/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU142(double a, double b, double p, double d, double s){
	return (d*((a*a*a)*(d*d) - 1.5*(a*a)*b*(d*d) + 3.0*a*p - 0.75*b*p)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU143(double a, double b, double p, double d, double s){
	return ((0.75*(a*a)*(d*d) - 0.5*a*b*(d*d) + 0.625*p)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU144(double a, double b, double p, double d, double s){
	return (d*(0.25*a - 0.0625*b)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU145(double a, double b, double p, double d, double s){
	return (0.03125/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU150(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.5*(a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a)*b*(d*d*d*d)*p + 7.5*(a*a)*(d*d)*(p*p) - 3.75*a*b*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU151(double a, double b, double p, double d, double s){
	return (d*(0.5*(a*a*a*a*a)*(d*d*d*d) - 2.5*(a*a*a*a)*b*(d*d*d*d) + 7.5*(a*a*a)*(d*d)*p - 7.5*(a*a)*b*(d*d)*p + 9.375*a*(p*p) - 1.875*b*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU152(double a, double b, double p, double d, double s){
	return ((1.25*(a*a*a*a)*(d*d*d*d) - 2.5*(a*a*a)*b*(d*d*d*d) + 7.5*(a*a)*(d*d)*p - 3.75*a*b*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU153(double a, double b, double p, double d, double s){
	return (d*(1.25*(a*a*a)*(d*d) - 1.25*(a*a)*b*(d*d) + 3.125*a*p - 0.625*b*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU154(double a, double b, double p, double d, double s){
	return ((0.625*(a*a)*(d*d) - 0.3125*a*b*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU155(double a, double b, double p, double d, double s){
	return (d*(0.15625*a - 0.03125*b)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU156(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU160(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.0*(a*a*a*a*a)*(d*d*d*d)*p - 7.5*(a*a*a*a)*b*(d*d*d*d)*p + 15.0*(a*a*a)*(d*d)*(p*p) - 11.25*(a*a)*b*(d*d)*(p*p) + 11.25*a*(p*p*p) - 1.875*b*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU161(double a, double b, double p, double d, double s){
	return ((0.5*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.0*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 11.25*(a*a*a*a)*(d*d*d*d)*p - 15.0*(a*a*a)*b*(d*d*d*d)*p + 28.125*(a*a)*(d*d)*(p*p) - 11.25*a*b*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU162(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a*a*a)*(d*d*d*d) - 3.75*(a*a*a*a)*b*(d*d*d*d) + 15.0*(a*a*a)*(d*d)*p - 11.25*(a*a)*b*(d*d)*p + 16.875*a*(p*p) - 2.8125*b*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU163(double a, double b, double p, double d, double s){
	return ((1.875*(a*a*a*a)*(d*d*d*d) - 2.5*(a*a*a)*b*(d*d*d*d) + 9.375*(a*a)*(d*d)*p - 3.75*a*b*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU164(double a, double b, double p, double d, double s){
	return (d*(1.25*(a*a*a)*(d*d) - 0.9375*(a*a)*b*(d*d) + 2.8125*a*p - 0.46875*b*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU165(double a, double b, double p, double d, double s){
	return ((0.46875*(a*a)*(d*d) - 0.1875*a*b*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU166(double a, double b, double p, double d, double s){
	return (d*(0.09375*a - 0.015625*b)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU167(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU170(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 3.5*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 10.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 26.25*(a*a*a*a)*(d*d*d*d)*(p*p) - 26.25*(a*a*a)*b*(d*d*d*d)*(p*p) + 39.375*(a*a)*(d*d)*(p*p*p) - 13.125*a*b*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU171(double a, double b, double p, double d, double s){
	return (d*(0.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 15.75*(a*a*a*a*a)*(d*d*d*d)*p - 26.25*(a*a*a*a)*b*(d*d*d*d)*p + 65.625*(a*a*a)*(d*d)*(p*p) - 39.375*(a*a)*b*(d*d)*(p*p) + 45.9375*a*(p*p*p) - 6.5625*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU172(double a, double b, double p, double d, double s){
	return ((1.75*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 5.25*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 26.25*(a*a*a*a)*(d*d*d*d)*p - 26.25*(a*a*a)*b*(d*d*d*d)*p + 59.0625*(a*a)*(d*d)*(p*p) - 19.6875*a*b*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU173(double a, double b, double p, double d, double s){
	return (d*(2.625*(a*a*a*a*a)*(d*d*d*d) - 4.375*(a*a*a*a)*b*(d*d*d*d) + 21.875*(a*a*a)*(d*d)*p - 13.125*(a*a)*b*(d*d)*p + 22.96875*a*(p*p) - 3.28125*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU174(double a, double b, double p, double d, double s){
	return ((2.1875*(a*a*a*a)*(d*d*d*d) - 2.1875*(a*a*a)*b*(d*d*d*d) + 9.84375*(a*a)*(d*d)*p - 3.28125*a*b*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU175(double a, double b, double p, double d, double s){
	return (d*(1.09375*(a*a*a)*(d*d) - 0.65625*(a*a)*b*(d*d) + 2.296875*a*p - 0.328125*b*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU176(double a, double b, double p, double d, double s){
	return ((0.328125*(a*a)*(d*d) - 0.109375*a*b*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU177(double a, double b, double p, double d, double s){
	return (d*(0.0546875*a - 0.0078125*b)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU178(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU180(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 4.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 14.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 42.0*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 52.5*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 105.0*(a*a*a)*(d*d)*(p*p*p) - 52.5*(a*a)*b*(d*d)*(p*p*p) + 52.5*a*(p*p*p*p) - 6.5625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU181(double a, double b, double p, double d, double s){
	return ((0.5*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 4.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 21.0 *(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 42.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 131.25*(a*a*a*a)*(d*d*d*d)*(p*p) - 105.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 183.75*(a*a)*(d*d)*(p*p*p) - 52.5*a*b*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU182(double a, double b, double p, double d, double s){
	return (d*(2.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 7.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 42.0*(a*a*a*a*a)*(d*d*d*d)*p - 52.5*(a*a*a*a)*b*(d*d*d*d)*p + 157.5*(a*a*a)*(d*d)*(p*p) - 78.75*(a*a)*b*(d*d)*(p*p) + 105.0*a*(p*p*p) - 13.125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU183(double a, double b, double p, double d, double s){
	return ((3.5*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 7.0*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 43.75*(a*a*a*a)*(d*d*d*d)*p - 35.0*(a*a*a)*b*(d*d*d*d)*p + 91.875*(a*a)*(d*d)*(p*p) - 26.25*a*b*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU184(double a, double b, double p, double d, double s){
	return (d*(3.5*(a*a*a*a*a)*(d*d*d*d) - 4.375*(a*a*a*a)*b*(d*d*d*d) + 26.25*(a*a*a)*(d*d)*p - 13.125*(a*a)*b*(d*d)*p + 26.25*a*(p*p) - 3.28125*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU185(double a, double b, double p, double d, double s){
	return ((2.1875*(a*a*a*a)*(d*d*d*d) - 1.75*(a*a*a)*b*(d*d*d*d) + 9.1875*(a*a)*(d*d)*p - 2.625*a*b*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU186(double a, double b, double p, double d, double s){
	return (d*(0.875*(a*a*a)*(d*d) - 0.4375*(a*a)*b*(d*d) + 1.75*a*p - 0.21875*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU187(double a, double b, double p, double d, double s){
	return ((0.21875*(a*a)*(d*d) - 0.0625*a*b*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU188(double a, double b, double p, double d, double s){
	return (d*(0.03125*a - 0.00390625*b)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU189(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU200(double a, double b, double p, double d, double s){
	return (((b*b)*(d*d) + 0.5*p)/(p*p))*s;
}

inline double MD_Et_GPU201(double a, double b, double p, double d, double s){
	return (-b*d/(p*p))*s;
}

inline double MD_Et_GPU202(double a, double b, double p, double d, double s){
	return (0.25/(p*p))*s;
}

inline double MD_Et_GPU210(double a, double b, double p, double d, double s){
	return (d*(a*(b*b)*(d*d) + 0.5*a*p - b*p)/(p*p*p))*s;
}

inline double MD_Et_GPU211(double a, double b, double p, double d, double s){
	return ((-a*b*(d*d) + 0.5*(b*b)*(d*d) + 0.75*p)/(p*p*p))*s;
}

inline double MD_Et_GPU212(double a, double b, double p, double d, double s){
	return (d*(0.25*a - 0.5*b)/(p*p*p))*s;
}

inline double MD_Et_GPU213(double a, double b, double p, double d, double s){
	return (0.125/(p*p*p))*s;
}

inline double MD_Et_GPU220(double a, double b, double p, double d, double s){
	return (((a*a)*(b*b)*(d*d*d*d) + 0.5*(a*a)*(d*d)*p - 2.0*a*b*(d*d)*p + 0.5*(b*b)*(d*d)*p + 0.75*(p*p))/(p*p*p*p))*s;
}

inline double MD_Et_GPU221(double a, double b, double p, double d, double s){
	return (d*(-(a*a)*b*(d*d) + a*(b*b)*(d*d) + 1.5*a*p - 1.5*b*p)/(p*p*p*p))*s;
}

inline double MD_Et_GPU222(double a, double b, double p, double d, double s){
	return ((0.25*(a*a)*(d*d) - a*b*(d*d) + 0.25*(b*b)*(d*d) + 0.75*p)/(p*p*p*p))*s;
}

inline double MD_Et_GPU223(double a, double b, double p, double d, double s){
	return (0.25*d*(a - b)/(p*p*p*p))*s;
}

inline double MD_Et_GPU224(double a, double b, double p, double d, double s){
	return (0.0625/(p*p*p*p))*s;
}

inline double MD_Et_GPU230(double a, double b, double p, double d, double s){
	return (d*((a*a*a)*(b*b)*(d*d*d*d) + 0.5*(a*a*a)*(d*d)*p - 3.0*(a*a)*b*(d*d)*p + 1.5*a*(b*b)*(d*d)*p + 2.25*a*(p*p) - 1.5*b*(p*p))/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU231(double a, double b, double p, double d, double s){
	return ((-(a*a*a)*b*(d*d*d*d) + 1.5*(a*a)*(b*b)*(d*d*d*d) + 2.25*(a*a)*(d*d)*p - 4.5*a*b*(d*d)*p + 0.75*(b*b)*(d*d)*p + 1.875*(p*p))/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU232(double a, double b, double p, double d, double s){
	return (d*(0.25*(a*a*a)*(d*d) - 1.5*(a*a)*b*(d*d) + 0.75*a*(b*b)*(d*d) + 2.25*a*p - 1.5*b*p)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU233(double a, double b, double p, double d, double s){
	return ((0.375*(a*a)*(d*d) - 0.75*a*b*(d*d) + 0.125*(b*b)*(d*d) + 0.625*p)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU234(double a, double b, double p, double d, double s){
	return (d*(0.1875*a - 0.125*b)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU235(double a, double b, double p, double d, double s){
	return (0.03125/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU240(double a, double b, double p, double d, double s){
	return (((a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.5*(a*a*a*a)*(d*d*d*d)*p - 4.0*(a*a*a)*b*(d*d*d*d)*p + 3.0*(a*a)*(b*b)*(d*d*d*d)*p + 4.5*(a*a)*(d*d)*(p*p) - 6.0*a*b*(d*d)*(p*p) + 0.75*(b*b)*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU241(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a)*b*(d*d*d*d) + 2.0*(a*a*a)*(b*b)*(d*d*d*d) + 3.0*(a*a*a)*(d*d)*p - 9.0*(a*a)*b*(d*d)*p + 3.0*a*(b*b)*(d*d)*p + 7.5*a*(p*p) - 3.75*b*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU242(double a, double b, double p, double d, double s){
	return ((0.25*(a*a*a*a)*(d*d*d*d) - 2.0*(a*a*a)*b*(d*d*d*d) + 1.5*(a*a)*(b*b)*(d*d*d*d) + 4.5*(a*a)*(d*d)*p - 6.0*a*b*(d*d)*p + 0.75*(b*b)*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU243(double a, double b, double p, double d, double s){
	return (d*(0.5*(a*a*a)*(d*d) - 1.5*(a*a)*b*(d*d) + 0.5*a*(b*b)*(d*d) + 2.5*a*p - 1.25*b*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU244(double a, double b, double p, double d, double s){
	return ((0.375*(a*a)*(d*d) - 0.5*a*b*(d*d) + 0.0625*(b*b)*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU245(double a, double b, double p, double d, double s){
	return (d*(0.125*a - 0.0625*b)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU246(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU250(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.5*(a*a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a*a)*b*(d*d*d*d)*p + 5.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 7.5*(a*a*a)*(d*d)*(p*p) - 15.0*(a*a)*b*(d*d)*(p*p) + 3.75*a*(b*b)*(d*d)*(p*p) + 9.375*a*(p*p*p) - 3.75*b*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU251(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.75*(a*a*a*a)*(d*d*d*d)*p - 15.0*(a*a*a)*b*(d*d*d*d)*p + 7.5*(a*a)*(b*b)*(d*d*d*d)*p + 18.75*(a*a)*(d*d)*(p*p) - 18.75*a*b*(d*d)*(p*p) + 1.875*(b*b)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU252(double a, double b, double p, double d, double s){
	return (d*(0.25*(a*a*a*a*a)*(d*d*d*d) - 2.5*(a*a*a*a)*b*(d*d*d*d) + 2.5*(a*a*a)*(b*b)*(d*d*d*d) + 7.5*(a*a*a)*(d*d)*p - 15.0*(a*a)*b*(d*d)*p + 3.75*a*(b*b)*(d*d)*p + 14.0625*a*(p*p) - 5.625*b*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU253(double a, double b, double p, double d, double s){
	return ((0.625*(a*a*a*a)*(d*d*d*d) - 2.5*(a*a*a)*b*(d*d*d*d) + 1.25*(a*a)*(b*b)*(d*d*d*d) + 6.25*(a*a)*(d*d)*p - 6.25*a*b*(d*d)*p + 0.625*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU254(double a, double b, double p, double d, double s){
	return (d*(0.625*(a*a*a)*(d*d) - 1.25*(a*a)*b*(d*d) + 0.3125*a*(b*b)*(d*d) + 2.34375*a*p - 0.9375*b*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU255(double a, double b, double p, double d, double s){
	return ((0.3125*(a*a)*(d*d) - 0.3125*a*b*(d*d) + 0.03125*(b*b)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU256(double a, double b, double p, double d, double s){
	return (d*(0.078125*a - 0.03125*b)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU257(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU260(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.5*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 6.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 7.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 11.25*(a*a*a*a)*(d*d*d*d)*(p*p) - 30.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 11.25*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 28.125*(a*a)*(d*d)*(p*p*p) - 22.5*a*b*(d*d)*(p*p*p) + 1.875*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU261(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 4.5*(a*a*a*a*a)*(d*d*d*d)*p - 22.5*(a*a*a*a)*b*(d*d*d*d)*p + 15.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 37.5*(a*a*a)*(d*d)*(p*p) - 56.25*(a*a)*b*(d*d)*(p*p) + 11.25*a*(b*b)*(d*d)*(p*p) + 39.375*a*(p*p*p) - 13.125*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU262(double a, double b, double p, double d, double s){
	return ((0.25*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.0*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 11.25*(a*a*a*a)*(d*d*d*d)*p - 30.0*(a*a*a)*b*(d*d*d*d)*p + 11.25*(a*a)*(b*b)*(d*d*d*d)*p + 42.1875*(a*a)*(d*d)*(p*p) - 33.75*a*b*(d*d)*(p*p) + 2.8125*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU263(double a, double b, double p, double d, double s){
	return (d*(0.75*(a*a*a*a*a)*(d*d*d*d) - 3.75*(a*a*a*a)*b*(d*d*d*d) + 2.5*(a*a*a)*(b*b)*(d*d*d*d) + 12.5*(a*a*a)*(d*d)*p - 18.75*(a*a)*b*(d*d)*p + 3.75*a*(b*b)*(d*d)*p + 19.6875*a*(p*p) - 6.5625*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU264(double a, double b, double p, double d, double s){
	return ((0.9375*(a*a*a*a)*(d*d*d*d) - 2.5*(a*a*a)*b*(d*d*d*d) + 0.9375*(a*a)*(b*b)*(d*d*d*d) + 7.03125*(a*a)*(d*d)*p - 5.625*a*b*(d*d)*p + 0.46875*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU265(double a, double b, double p, double d, double s){
	return (d*(0.625*(a*a*a)*(d*d) - 0.9375*(a*a)*b*(d*d) + 0.1875*a*(b*b)*(d*d) + 1.96875*a*p - 0.65625*b*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU266(double a, double b, double p, double d, double s){
	return ((0.234375*(a*a)*(d*d) - 0.1875*a*b*(d*d) + 0.015625*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU267(double a, double b, double p, double d, double s){
	return (d*(0.046875*a - 0.015625*b)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU268(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU270(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 10.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 15.75*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 52.5*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 26.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 65.625*(a*a*a)*(d*d)*(p*p*p) - 78.75*(a*a)*b*(d*d)*(p*p*p) + 13.125*a*(b*b)*(d*d)*(p*p*p) + 45.9375*a*(p*p*p*p) - 13.125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU271(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 3.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 5.25*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 31.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 26.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 65.625*(a*a*a*a)*(d*d*d*d)*(p*p) - 131.25*(a*a*a)*b*(d*d*d*d)*(p*p) + 39.375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 137.8125*(a*a)*(d*d)*(p*p*p) - 91.875*a*b*(d*d)*(p*p*p) + 6.5625*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU272(double a, double b, double p, double d, double s){
	return (d*(0.25*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 5.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 15.75*(a*a*a*a*a)*(d*d*d*d)*p - 52.5*(a*a*a*a)*b*(d*d*d*d)*p + 26.25*(a*a*a)*(b*b)*(d*d*d*d)*p + 98.4375*(a*a*a)*(d*d)*(p*p) - 118.125*(a*a)*b*(d*d)*(p*p) + 19.6875*a*(b*b)*(d*d)*(p*p) + 91.875*a*(p*p*p) - 26.25*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU273(double a, double b, double p, double d, double s){
	return ((0.875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 5.25*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 21.875*(a*a*a*a)*(d*d*d*d)*p - 43.75*(a*a*a)*b*(d*d*d*d)*p + 13.125*(a*a)*(b*b)*(d*d*d*d)*p + 68.90625*(a*a)*(d*d)*(p*p) - 45.9375*a*b*(d*d)*(p*p) + 3.28125*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU274(double a, double b, double p, double d, double s){
	return (d*(1.3125*(a*a*a*a*a)*(d*d*d*d) - 4.375*(a*a*a*a)*b*(d*d*d*d) + 2.1875*(a*a*a)*(b*b)*(d*d*d*d) + 16.40625*(a*a*a)*(d*d)*p - 19.6875*(a*a)*b*(d*d)*p + 3.28125*a*(b*b)*(d*d)*p + 22.96875*a*(p*p) - 6.5625*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU275(double a, double b, double p, double d, double s){
	return ((1.09375*(a*a*a*a)*(d*d*d*d) - 2.1875*(a*a*a)*b*(d*d*d*d) + 0.65625*(a*a)*(b*b)*(d*d*d*d) + 6.890625*(a*a)*(d*d)*p - 4.59375*a*b*(d*d)*p + 0.328125*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU276(double a, double b, double p, double d, double s){
	return (d*(0.546875*(a*a*a)*(d*d) - 0.65625*(a*a)*b*(d*d) + 0.109375*a*(b*b)*(d*d) + 1.53125*a*p - 0.4375*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU277(double a, double b, double p, double d, double s){
	return ((0.1640625*(a*a)*(d*d) - 0.109375*a*b*(d*d) + 0.0078125*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU278(double a, double b, double p, double d, double s){
	return (d*(0.02734375*a - 0.0078125*b)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU279(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU280(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.5*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 8.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 14.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 21.0 *(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 84.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 131.25*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 210.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 52.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 183.75*(a*a)*(d*d)*(p*p*p*p) - 105.0*a*b*(d*d)*(p*p*p*p) + 6.5625*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU281(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 4.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 6.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 42.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 42.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 105.0*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 262.5*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 105.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 367.5*(a*a*a)*(d*d)*(p*p*p) - 367.5*(a*a)*b*(d*d)*(p*p*p) + 52.5*a*(b*b)*(d*d)*(p*p*p) + 236.25*a*(p*p*p*p) - 59.0625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU282(double a, double b, double p, double d, double s){
	return ((0.25*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 4.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 7.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 21.0 *(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 84.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 52.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 196.875*(a*a*a*a)*(d*d*d*d)*(p*p) - 315.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 78.75*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 367.5*(a*a)*(d*d)*(p*p*p) - 210.0*a*b*(d*d)*(p*p*p) + 13.125*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU283(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 7.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 7.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 35.0*(a*a*a*a*a)*(d*d*d*d)*p - 87.5*(a*a*a*a)*b*(d*d*d*d)*p + 35.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 183.75*(a*a*a)*(d*d)*(p*p) - 183.75*(a*a)*b*(d*d)*(p*p) + 26.25*a*(b*b)*(d*d)*(p*p) + 157.5*a*(p*p*p) - 39.375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU284(double a, double b, double p, double d, double s){
	return ((1.75*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 7.0*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 32.8125*(a*a*a*a)*(d*d*d*d)*p - 52.5*(a*a*a)*b*(d*d*d*d)*p + 13.125*(a*a)*(b*b)*(d*d*d*d)*p + 91.875*(a*a)*(d*d)*(p*p) - 52.5*a*b*(d*d)*(p*p) + 3.28125*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU285(double a, double b, double p, double d, double s){
	return (d*(1.75*(a*a*a*a*a)*(d*d*d*d) - 4.375*(a*a*a*a)*b*(d*d*d*d) + 1.75*(a*a*a)*(b*b)*(d*d*d*d) + 18.375*(a*a*a)*(d*d)*p - 18.375*(a*a)*b*(d*d)*p + 2.625*a*(b*b)*(d*d)*p + 23.625*a*(p*p) - 5.90625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU286(double a, double b, double p, double d, double s){
	return ((1.09375*(a*a*a*a)*(d*d*d*d) - 1.75*(a*a*a)*b*(d*d*d*d) + 0.4375*(a*a)*(b*b)*(d*d*d*d) + 6.125*(a*a)*(d*d)*p - 3.5*a*b*(d*d)*p + 0.21875*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU287(double a, double b, double p, double d, double s){
	return (d*(0.4375*(a*a*a)*(d*d) - 0.4375*(a*a)*b*(d*d) + 0.0625*a*(b*b)*(d*d) + 1.125*a*p - 0.28125*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU288(double a, double b, double p, double d, double s){
	return ((0.109375*(a*a)*(d*d) - 0.0625*a*b*(d*d) + 0.00390625*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU289(double a, double b, double p, double d, double s){
	return (d*(0.015625*a - 0.00390625*b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU2810(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU300(double a, double b, double p, double d, double s){
	return (b*d*(-(b*b)*(d*d) - 1.5*p)/(p*p*p))*s;
}

inline double MD_Et_GPU301(double a, double b, double p, double d, double s){
	return ((1.5*(b*b)*(d*d) + 0.75*p)/(p*p*p))*s;
}

inline double MD_Et_GPU302(double a, double b, double p, double d, double s){
	return (-0.75*b*d/(p*p*p))*s;
}

inline double MD_Et_GPU303(double a, double b, double p, double d, double s){
	return (0.125/(p*p*p))*s;
}

inline double MD_Et_GPU310(double a, double b, double p, double d, double s){
	return ((-a*(b*b*b)*(d*d*d*d) - 1.5*a*b*(d*d)*p + 1.5*(b*b)*(d*d)*p + 0.75*(p*p))/(p*p*p*p))*s;
}

inline double MD_Et_GPU311(double a, double b, double p, double d, double s){
	return (d*(1.5*a*(b*b)*(d*d) + 0.75*a*p - 0.5*(b*b*b)*(d*d) - 2.25*b*p)/(p*p*p*p))*s;
}

inline double MD_Et_GPU312(double a, double b, double p, double d, double s){
	return (0.75*(-a*b*(d*d) + (b*b)*(d*d) + p)/(p*p*p*p))*s;
}

inline double MD_Et_GPU313(double a, double b, double p, double d, double s){
	return (d*(0.125*a - 0.375*b)/(p*p*p*p))*s;
}

inline double MD_Et_GPU314(double a, double b, double p, double d, double s){
	return (0.0625/(p*p*p*p))*s;
}

inline double MD_Et_GPU320(double a, double b, double p, double d, double s){
	return (d*(-(a*a)*(b*b*b)*(d*d*d*d) - 1.5*(a*a)*b*(d*d)*p + 3.0*a*(b*b)*(d*d)*p + 1.5*a*(p*p) - 0.5*(b*b*b)*(d*d)*p - 2.25*b*(p*p))/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU321(double a, double b, double p, double d, double s){
	return ((1.5*(a*a)*(b*b)*(d*d*d*d) + 0.75*(a*a)*(d*d)*p - a*(b*b*b)*(d*d*d*d) - 4.5*a*b*(d*d)*p + 2.25*(b*b)*(d*d)*p + 1.875*(p*p))/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU322(double a, double b, double p, double d, double s){
	return (d*(-0.75*(a*a)*b*(d*d) + 1.5*a*(b*b)*(d*d) + 1.5*a*p - 0.25*(b*b*b)*(d*d) - 2.25*b*p)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU323(double a, double b, double p, double d, double s){
	return ((0.125*(a*a)*(d*d) - 0.75*a*b*(d*d) + 0.375*(b*b)*(d*d) + 0.625*p)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU324(double a, double b, double p, double d, double s){
	return (d*(0.125*a - 0.1875*b)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU325(double a, double b, double p, double d, double s){
	return (0.03125/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU330(double a, double b, double p, double d, double s){
	return ((-(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 1.5*(a*a*a)*b*(d*d*d*d)*p + 4.5*(a*a)*(b*b)*(d*d*d*d)*p + 2.25*(a*a)*(d*d)*(p*p) - 1.5*a*(b*b*b)*(d*d*d*d)*p - 6.75*a*b*(d*d)*(p*p) + 2.25*(b*b)*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU331(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a)*(b*b)*(d*d*d*d) + 0.75*(a*a*a)*(d*d)*p - 1.5*(a*a)*(b*b*b)*(d*d*d*d) - 6.75*(a*a)*b*(d*d)*p + 6.75*a*(b*b)*(d*d)*p + 5.625*a*(p*p) - 0.75*(b*b*b)*(d*d)*p - 5.625*b*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU332(double a, double b, double p, double d, double s){
	return ((-0.75*(a*a*a)*b*(d*d*d*d) + 2.25*(a*a)*(b*b)*(d*d*d*d) + 2.25*(a*a)*(d*d)*p - 0.75*a*(b*b*b)*(d*d*d*d) - 6.75*a*b*(d*d)*p + 2.25*(b*b)*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU333(double a, double b, double p, double d, double s){
	return (d*(0.125*(a*a*a)*(d*d) - 1.125*(a*a)*b*(d*d) + 1.125*a*(b*b)*(d*d) + 1.875*a*p - 0.125*(b*b*b)*(d*d) - 1.875*b*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU334(double a, double b, double p, double d, double s){
	return ((0.1875*(a*a)*(d*d) - 0.5625*a*b*(d*d) + 0.1875*(b*b)*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU335(double a, double b, double p, double d, double s){
	return (0.09375*d*(a - b)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU336(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU340(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 1.5*(a*a*a*a)*b*(d*d*d*d)*p + 6.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 3.0*(a*a*a)*(d*d)*(p*p) - 3.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 13.5*(a*a)*b*(d*d)*(p*p) + 9.0*a*(b*b)*(d*d)*(p*p) + 7.5*a*(p*p*p) - 0.75*(b*b*b)*(d*d)*(p*p) - 5.625*b*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU341(double a, double b, double p, double d, double s){
	return ((1.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.75*(a*a*a*a)*(d*d*d*d)*p - 2.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 9.0*(a*a*a)*b*(d*d*d*d)*p + 13.5*(a*a)*(b*b)*(d*d*d*d)*p + 11.25*(a*a)*(d*d)*(p*p) - 3.0*a*(b*b*b)*(d*d*d*d)*p - 22.5*a*b*(d*d)*(p*p) + 5.625*(b*b)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU342(double a, double b, double p, double d, double s){
	return (d*(-0.75*(a*a*a*a)*b*(d*d*d*d) + 3.0*(a*a*a)*(b*b)*(d*d*d*d) + 3.0*(a*a*a)*(d*d)*p - 1.5*(a*a)*(b*b*b)*(d*d*d*d) - 13.5*(a*a)*b*(d*d)*p + 9.0*a*(b*b)*(d*d)*p + 11.25*a*(p*p) - 0.75*(b*b*b)*(d*d)*p - 8.4375*b*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU343(double a, double b, double p, double d, double s){
	return ((0.125*(a*a*a*a)*(d*d*d*d) - 1.5*(a*a*a)*b*(d*d*d*d) + 2.25*(a*a)*(b*b)*(d*d*d*d) + 3.75*(a*a)*(d*d)*p - 0.5*a*(b*b*b)*(d*d*d*d) - 7.5*a*b*(d*d)*p + 1.875*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU344(double a, double b, double p, double d, double s){
	return (d*(0.25*(a*a*a)*(d*d) - 1.125*(a*a)*b*(d*d) + 0.75*a*(b*b)*(d*d) + 1.875*a*p - 0.0625*(b*b*b)*(d*d) - 1.40625*b*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU345(double a, double b, double p, double d, double s){
	return ((0.1875*(a*a)*(d*d) - 0.375*a*b*(d*d) + 0.09375*(b*b)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU346(double a, double b, double p, double d, double s){
	return (d*(0.0625*a - 0.046875*b)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU347(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU350(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 1.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 7.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 3.75*(a*a*a*a)*(d*d*d*d)*(p*p) - 5.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 22.5*(a*a*a)*b*(d*d*d*d)*(p*p) + 22.5*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 18.75*(a*a)*(d*d)*(p*p*p) - 3.75*a*(b*b*b)*(d*d*d*d)*(p*p) - 28.125*a*b*(d*d)*(p*p*p) + 5.625*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU351(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.75*(a*a*a*a*a)*(d*d*d*d)*p - 2.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 11.25*(a*a*a*a)*b*(d*d*d*d)*p + 22.5*(a*a*a)*(b*b)*(d*d*d*d)*p + 18.75*(a*a*a)*(d*d)*(p*p) - 7.5*(a*a)*(b*b*b)*(d*d*d*d)*p - 56.25*(a*a)*b*(d*d)*(p*p) + 28.125*a*(b*b)*(d*d)*(p*p) + 32.8125*a*(p*p*p) - 1.875*(b*b*b)*(d*d)*(p*p) - 19.6875*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU352(double a, double b, double p, double d, double s){
	return ((-0.75*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.75*(a*a*a*a)*(d*d*d*d)*p - 2.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 22.5*(a*a*a)*b*(d*d*d*d)*p + 22.5*(a*a)*(b*b)*(d*d*d*d)*p + 28.125*(a*a)*(d*d)*(p*p) - 3.75*a*(b*b*b)*(d*d*d*d)*p - 42.1875*a*b*(d*d)*(p*p) + 8.4375*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU353(double a, double b, double p, double d, double s){
	return (d*(0.125*(a*a*a*a*a)*(d*d*d*d) - 1.875*(a*a*a*a)*b*(d*d*d*d) + 3.75*(a*a*a)*(b*b)*(d*d*d*d) + 6.25*(a*a*a)*(d*d)*p - 1.25*(a*a)*(b*b*b)*(d*d*d*d) - 18.75*(a*a)*b*(d*d)*p + 9.375*a*(b*b)*(d*d)*p + 16.40625*a*(p*p) - 0.625*(b*b*b)*(d*d)*p - 9.84375*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU354(double a, double b, double p, double d, double s){
	return ((0.3125*(a*a*a*a)*(d*d*d*d) - 1.875*(a*a*a)*b*(d*d*d*d) + 1.875*(a*a)*(b*b)*(d*d*d*d) + 4.6875*(a*a)*(d*d)*p - 0.3125*a*(b*b*b)*(d*d*d*d) - 7.03125*a*b*(d*d)*p + 1.40625*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU355(double a, double b, double p, double d, double s){
	return (d*(0.3125*(a*a*a)*(d*d) - 0.9375*(a*a)*b*(d*d) + 0.46875*a*(b*b)*(d*d) + 1.640625*a*p - 0.03125*(b*b*b)*(d*d) - 0.984375*b*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU356(double a, double b, double p, double d, double s){
	return ((0.15625*(a*a)*(d*d) - 0.234375*a*b*(d*d) + 0.046875*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU357(double a, double b, double p, double d, double s){
	return (d*(0.0390625*a - 0.0234375*b)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU358(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU360(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 1.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 9.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 4.5*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 7.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 33.75*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 45.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 37.5*(a*a*a)*(d*d)*(p*p*p) - 11.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 84.375*(a*a)*b*(d*d)*(p*p*p) + 33.75*a*(b*b)*(d*d)*(p*p*p) + 39.375*a*(p*p*p*p) - 1.875*(b*b*b)*(d*d)*(p*p*p) - 19.6875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU361(double a, double b, double p, double d, double s){
	return ((1.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.75*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 3.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 13.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 33.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 28.125*(a*a*a*a)*(d*d*d*d)*(p*p) - 15.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 112.5*(a*a*a)*b*(d*d*d*d)*(p*p) + 84.375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 98.4375*(a*a)*(d*d)*(p*p*p) - 11.25*a*(b*b*b)*(d*d*d*d)*(p*p) - 118.125*a*b*(d*d)*(p*p*p) + 19.6875*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU362(double a, double b, double p, double d, double s){
	return (d*(-0.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 4.5*(a*a*a*a*a)*(d*d*d*d)*p - 3.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 33.75*(a*a*a*a)*b*(d*d*d*d)*p + 45.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 56.25*(a*a*a)*(d*d)*(p*p) - 11.25*(a*a)*(b*b*b)*(d*d*d*d)*p - 126.5625*(a*a)*b*(d*d)*(p*p) + 50.625*a*(b*b)*(d*d)*(p*p) + 78.75*a*(p*p*p) - 2.8125*(b*b*b)*(d*d)*(p*p) - 39.375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU363(double a, double b, double p, double d, double s){
	return ((0.125*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 2.25*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 5.625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 9.375*(a*a*a*a)*(d*d*d*d)*p - 2.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 37.5*(a*a*a)*b*(d*d*d*d)*p + 28.125*(a*a)*(b*b)*(d*d*d*d)*p + 49.21875*(a*a)*(d*d)*(p*p) - 3.75*a*(b*b*b)*(d*d*d*d)*p - 59.0625*a*b*(d*d)*(p*p) + 9.84375*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU364(double a, double b, double p, double d, double s){
	return (d*(0.375*(a*a*a*a*a)*(d*d*d*d) - 2.8125*(a*a*a*a)*b*(d*d*d*d) + 3.75*(a*a*a)*(b*b)*(d*d*d*d) + 9.375*(a*a*a)*(d*d)*p - 0.9375*(a*a)*(b*b*b)*(d*d*d*d) - 21.09375*(a*a)*b*(d*d)*p + 8.4375*a*(b*b)*(d*d)*p + 19.6875*a*(p*p) - 0.46875*(b*b*b)*(d*d)*p - 9.84375*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU365(double a, double b, double p, double d, double s){
	return ((0.46875*(a*a*a*a)*(d*d*d*d) - 1.875*(a*a*a)*b*(d*d*d*d) + 1.40625*(a*a)*(b*b)*(d*d*d*d) + 4.921875*(a*a)*(d*d)*p - 0.1875*a*(b*b*b)*(d*d*d*d) - 5.90625*a*b*(d*d)*p + 0.984375*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU366(double a, double b, double p, double d, double s){
	return (d*(0.3125*(a*a*a)*(d*d) - 0.703125*(a*a)*b*(d*d) + 0.28125*a*(b*b)*(d*d) + 1.3125*a*p - 0.015625*(b*b*b)*(d*d) - 0.65625*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU367(double a, double b, double p, double d, double s){
	return ((0.1171875*(a*a)*(d*d) - 0.140625*a*b*(d*d) + 0.0234375*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU368(double a, double b, double p, double d, double s){
	return (d*(0.0234375*a - 0.01171875*b)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU369(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU370(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 1.5*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 10.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 5.25*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 10.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 47.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 78.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 65.625*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 26.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 196.875*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 118.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 137.8125*(a*a)*(d*d)*(p*p*p*p) - 13.125*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 137.8125*a*b*(d*d)*(p*p*p*p) + 19.6875*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU371(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.75*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 3.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 15.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 47.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 39.375*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 26.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 196.875*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 196.875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 229.6875*(a*a*a)*(d*d)*(p*p*p) - 39.375*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 413.4375*(a*a)*b*(d*d)*(p*p*p) + 137.8125*a*(b*b)*(d*d)*(p*p*p) + 206.71875*a*(p*p*p*p) - 6.5625*(b*b*b)*(d*d)*(p*p*p) - 88.59375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU372(double a, double b, double p, double d, double s){
	return ((-0.75*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 5.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 5.25*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 5.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 47.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 78.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 98.4375*(a*a*a*a)*(d*d*d*d)*(p*p) - 26.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 295.3125*(a*a*a)*b*(d*d*d*d)*(p*p) + 177.1875*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 275.625*(a*a)*(d*d)*(p*p*p) - 19.6875*a*(b*b*b)*(d*d*d*d)*(p*p) - 275.625*a*b*(d*d)*(p*p*p) + 39.375*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU373(double a, double b, double p, double d, double s){
	return (d*(0.125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 2.625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 7.875*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 13.125*(a*a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 65.625*(a*a*a*a)*b*(d*d*d*d)*p + 65.625*(a*a*a)*(b*b)*(d*d*d*d)*p + 114.84375*(a*a*a)*(d*d)*(p*p) - 13.125*(a*a)*(b*b*b)*(d*d*d*d)*p - 206.71875*(a*a)*b*(d*d)*(p*p) + 68.90625*a*(b*b)*(d*d)*(p*p) + 137.8125*a*(p*p*p) - 3.28125*(b*b*b)*(d*d)*(p*p) - 59.0625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU374(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.9375*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 6.5625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 16.40625*(a*a*a*a)*(d*d*d*d)*p - 2.1875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 49.21875*(a*a*a)*b*(d*d*d*d)*p + 29.53125*(a*a)*(b*b)*(d*d*d*d)*p + 68.90625*(a*a)*(d*d)*(p*p) - 3.28125*a*(b*b*b)*(d*d*d*d)*p - 68.90625*a*b*(d*d)*(p*p) + 9.84375*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU375(double a, double b, double p, double d, double s){
	return (d*(0.65625*(a*a*a*a*a)*(d*d*d*d) - 3.28125*(a*a*a*a)*b*(d*d*d*d) + 3.28125*(a*a*a)*(b*b)*(d*d*d*d) + 11.484375*(a*a*a)*(d*d)*p - 0.65625*(a*a)*(b*b*b)*(d*d*d*d) - 20.671875*(a*a)*b*(d*d)*p + 6.890625*a*(b*b)*(d*d)*p + 20.671875*a*(p*p) - 0.328125*(b*b*b)*(d*d)*p - 8.859375*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU376(double a, double b, double p, double d, double s){
	return ((0.546875*(a*a*a*a)*(d*d*d*d) - 1.640625*(a*a*a)*b*(d*d*d*d) + 0.984375*(a*a)*(b*b)*(d*d*d*d) + 4.59375*(a*a)*(d*d)*p - 0.109375*a*(b*b*b)*(d*d*d*d) - 4.59375*a*b*(d*d)*p + 0.65625*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU377(double a, double b, double p, double d, double s){
	return (d*(0.2734375*(a*a*a)*(d*d) - 0.4921875*(a*a)*b*(d*d) + 0.1640625*a*(b*b)*(d*d) + 0.984375*a*p - 0.0078125*(b*b*b)*(d*d) - 0.421875*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU378(double a, double b, double p, double d, double s){
	return ((0.08203125*(a*a)*(d*d) - 0.08203125*a*b*(d*d) + 0.01171875*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU379(double a, double b, double p, double d, double s){
	return (d*(0.013671875*a - 0.005859375*b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU3710(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU380(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 1.5*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 12.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 6.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 14.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 63.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 126.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 105.0*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 52.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 393.75*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 315.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 367.5*(a*a*a)*(d*d)*(p*p*p*p) - 52.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 551.25*(a*a)*b*(d*d)*(p*p*p*p) + 157.5*a*(b*b)*(d*d)*(p*p*p*p) + 236.25*a*(p*p*p*p*p) - 6.5625*(b*b*b)*(d*d)*(p*p*p*p) - 88.59375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU381(double a, double b, double p, double d, double s){
	return ((1.5*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.75*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 4.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 18.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 63.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 42.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 315.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 393.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 459.375*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 105.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 1102.5*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 551.25*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 826.875*(a*a)*(d*d)*(p*p*p*p) - 52.5*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 708.75*a*b*(d*d)*(p*p*p*p) + 88.59375*(b*b)*(d*d)*(p*p*p*p) + 162.421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU382(double a, double b, double p, double d, double s){
	return (d*(-0.75*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 6.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 6.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 63.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 126.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 157.5*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 52.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 590.625*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 472.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 735.0*(a*a*a)*(d*d)*(p*p*p) - 78.75*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 1102.5*(a*a)*b*(d*d)*(p*p*p) + 315.0*a*(b*b)*(d*d)*(p*p*p) + 590.625*a*(p*p*p*p) - 13.125*(b*b*b)*(d*d)*(p*p*p) - 221.484375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU383(double a, double b, double p, double d, double s){
	return ((0.125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 3.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 10.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 17.5*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 105.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 131.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 229.6875*(a*a*a*a)*(d*d*d*d)*(p*p) - 35.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 551.25*(a*a*a)*b*(d*d*d*d)*(p*p) + 275.625*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 551.25*(a*a)*(d*d)*(p*p*p) - 26.25*a*(b*b*b)*(d*d*d*d)*(p*p) - 472.5*a*b*(d*d)*(p*p*p) + 59.0625*(b*b)*(d*d)*(p*p*p) + 135.3515625*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU384(double a, double b, double p, double d, double s){
	return (d*(0.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 5.25*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 10.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 26.25*(a*a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 98.4375*(a*a*a*a)*b*(d*d*d*d)*p + 78.75*(a*a*a)*(b*b)*(d*d*d*d)*p + 183.75*(a*a*a)*(d*d)*(p*p) - 13.125*(a*a)*(b*b*b)*(d*d*d*d)*p - 275.625*(a*a)*b*(d*d)*(p*p) + 78.75*a*(b*b)*(d*d)*(p*p) + 196.875*a*(p*p*p) - 3.28125*(b*b*b)*(d*d)*(p*p) - 73.828125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU385(double a, double b, double p, double d, double s){
	return ((0.875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 5.25*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 6.5625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 22.96875*(a*a*a*a)*(d*d*d*d)*p - 1.75*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 55.125*(a*a*a)*b*(d*d*d*d)*p + 27.5625*(a*a)*(b*b)*(d*d*d*d)*p + 82.6875*(a*a)*(d*d)*(p*p) - 2.625*a*(b*b*b)*(d*d*d*d)*p - 70.875*a*b*(d*d)*(p*p) + 8.859375*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU386(double a, double b, double p, double d, double s){
	return (d*(0.875*(a*a*a*a*a)*(d*d*d*d) - 3.28125*(a*a*a*a)*b*(d*d*d*d) + 2.625*(a*a*a)*(b*b)*(d*d*d*d) + 12.25*(a*a*a)*(d*d)*p - 0.4375*(a*a)*(b*b*b)*(d*d*d*d) - 18.375*(a*a)*b*(d*d)*p + 5.25*a*(b*b)*(d*d)*p + 19.6875*a*(p*p) - 0.21875*(b*b*b)*(d*d)*p - 7.3828125*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU387(double a, double b, double p, double d, double s){
	return ((0.546875*(a*a*a*a)*(d*d*d*d) - 1.3125*(a*a*a)*b*(d*d*d*d) + 0.65625*(a*a)*(b*b)*(d*d*d*d) + 3.9375*(a*a)*(d*d)*p - 0.0625*a*(b*b*b)*(d*d*d*d) - 3.375*a*b*(d*d)*p + 0.421875*(b*b)*(d*d)*p + 1.93359375*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU388(double a, double b, double p, double d, double s){
	return (d*(0.21875*(a*a*a)*(d*d) - 0.328125*(a*a)*b*(d*d) + 0.09375*a*(b*b)*(d*d) + 0.703125*a*p - 0.00390625*(b*b*b)*(d*d) - 0.263671875*b*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU389(double a, double b, double p, double d, double s){
	return ((0.0546875*(a*a)*(d*d) - 0.046875*a*b*(d*d) + 0.005859375*(b*b)*(d*d) + 0.0537109375*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU3810(double a, double b, double p, double d, double s){
	return (d*(0.0078125*a - 0.0029296875*b)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU3811(double a, double b, double p, double d, double s){
	return (0.00048828125/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU400(double a, double b, double p, double d, double s){
	return (((b*b*b*b)*(d*d*d*d) + 3.0*(b*b)*(d*d)*p + 0.75*(p*p))/(p*p*p*p))*s;
}

inline double MD_Et_GPU401(double a, double b, double p, double d, double s){
	return (b*d*(-2.0*(b*b)*(d*d) - 3.0*p)/(p*p*p*p))*s;
}

inline double MD_Et_GPU402(double a, double b, double p, double d, double s){
	return ((1.5*(b*b)*(d*d) + 0.75*p)/(p*p*p*p))*s;
}

inline double MD_Et_GPU403(double a, double b, double p, double d, double s){
	return (-0.5*b*d/(p*p*p*p))*s;
}

inline double MD_Et_GPU404(double a, double b, double p, double d, double s){
	return (0.0625/(p*p*p*p))*s;
}

inline double MD_Et_GPU410(double a, double b, double p, double d, double s){
	return (d*(a*(b*b*b*b)*(d*d*d*d) + 3.0*a*(b*b)*(d*d)*p + 0.75*a*(p*p) - 2.0*(b*b*b)*(d*d)*p - 3.0*b*(p*p))/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU411(double a, double b, double p, double d, double s){
	return ((-2.0*a*(b*b*b)*(d*d*d*d) - 3.0*a*b*(d*d)*p + 0.5*(b*b*b*b)*(d*d*d*d) + 4.5*(b*b)*(d*d)*p + 1.875*(p*p))/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU412(double a, double b, double p, double d, double s){
	return (d*(1.5*a*(b*b)*(d*d) + 0.75*a*p - (b*b*b)*(d*d) - 3.0*b*p)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU413(double a, double b, double p, double d, double s){
	return ((-0.5*a*b*(d*d) + 0.75*(b*b)*(d*d) + 0.625*p)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU414(double a, double b, double p, double d, double s){
	return (d*(0.0625*a - 0.25*b)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU415(double a, double b, double p, double d, double s){
	return (0.03125/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU420(double a, double b, double p, double d, double s){
	return (((a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 3.0*(a*a)*(b*b)*(d*d*d*d)*p + 0.75*(a*a)*(d*d)*(p*p) - 4.0*a*(b*b*b)*(d*d*d*d)*p - 6.0*a*b*(d*d)*(p*p) + 0.5*(b*b*b*b)*(d*d*d*d)*p + 4.5*(b*b)*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU421(double a, double b, double p, double d, double s){
	return (d*(-2.0*(a*a)*(b*b*b)*(d*d*d*d) - 3.0*(a*a)*b*(d*d)*p + a*(b*b*b*b)*(d*d*d*d) + 9.0*a*(b*b)*(d*d)*p + 3.75*a*(p*p) - 3.0*(b*b*b)*(d*d)*p - 7.5*b*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU422(double a, double b, double p, double d, double s){
	return ((1.5*(a*a)*(b*b)*(d*d*d*d) + 0.75*(a*a)*(d*d)*p - 2.0*a*(b*b*b)*(d*d*d*d) - 6.0*a*b*(d*d)*p + 0.25*(b*b*b*b)*(d*d*d*d) + 4.5*(b*b)*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU423(double a, double b, double p, double d, double s){
	return (d*(-0.5*(a*a)*b*(d*d) + 1.5*a*(b*b)*(d*d) + 1.25*a*p - 0.5*(b*b*b)*(d*d) - 2.5*b*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU424(double a, double b, double p, double d, double s){
	return ((0.0625*(a*a)*(d*d) - 0.5*a*b*(d*d) + 0.375*(b*b)*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU425(double a, double b, double p, double d, double s){
	return (d*(0.0625*a - 0.125*b)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU426(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU430(double a, double b, double p, double d, double s){
	return (d*((a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 3.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 0.75*(a*a*a)*(d*d)*(p*p) - 6.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 9.0*(a*a)*b*(d*d)*(p*p) + 1.5*a*(b*b*b*b)*(d*d*d*d)*p + 13.5*a*(b*b)*(d*d)*(p*p) + 5.625*a*(p*p*p) - 3.0*(b*b*b)*(d*d)*(p*p) - 7.5*b*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU431(double a, double b, double p, double d, double s){
	return ((-2.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.0*(a*a*a)*b*(d*d*d*d)*p + 1.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 13.5*(a*a)*(b*b)*(d*d*d*d)*p + 5.625*(a*a)*(d*d)*(p*p) - 9.0*a*(b*b*b)*(d*d*d*d)*p - 22.5*a*b*(d*d)*(p*p) + 0.75*(b*b*b*b)*(d*d*d*d)*p + 11.25*(b*b)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU432(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a)*(b*b)*(d*d*d*d) + 0.75*(a*a*a)*(d*d)*p - 3.0*(a*a)*(b*b*b)*(d*d*d*d) - 9.0*(a*a)*b*(d*d)*p + 0.75*a*(b*b*b*b)*(d*d*d*d) + 13.5*a*(b*b)*(d*d)*p + 8.4375*a*(p*p) - 3.0*(b*b*b)*(d*d)*p - 11.25*b*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU433(double a, double b, double p, double d, double s){
	return ((-0.5*(a*a*a)*b*(d*d*d*d) + 2.25*(a*a)*(b*b)*(d*d*d*d) + 1.875*(a*a)*(d*d)*p - 1.5*a*(b*b*b)*(d*d*d*d) - 7.5*a*b*(d*d)*p + 0.125*(b*b*b*b)*(d*d*d*d) + 3.75*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU434(double a, double b, double p, double d, double s){
	return (d*(0.0625*(a*a*a)*(d*d) - 0.75*(a*a)*b*(d*d) + 1.125*a*(b*b)*(d*d) + 1.40625*a*p - 0.25*(b*b*b)*(d*d) - 1.875*b*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU435(double a, double b, double p, double d, double s){
	return ((0.09375*(a*a)*(d*d) - 0.375*a*b*(d*d) + 0.1875*(b*b)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU436(double a, double b, double p, double d, double s){
	return (d*(0.046875*a - 0.0625*b)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU437(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU440(double a, double b, double p, double d, double s){
	return (((a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 3.0*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 0.75*(a*a*a*a)*(d*d*d*d)*(p*p) - 8.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 12.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 3.0*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 27.0*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 11.25*(a*a)*(d*d)*(p*p*p) - 12.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 30.0*a*b*(d*d)*(p*p*p) + 0.75*(b*b*b*b)*(d*d*d*d)*(p*p) + 11.25*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU441(double a, double b, double p, double d, double s){
	return (d*(-2.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.0*(a*a*a*a)*b*(d*d*d*d)*p + 2.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 18.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 7.5*(a*a*a)*(d*d)*(p*p) - 18.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 45.0*(a*a)*b*(d*d)*(p*p) + 3.0*a*(b*b*b*b)*(d*d*d*d)*p + 45.0*a*(b*b)*(d*d)*(p*p) + 26.25*a*(p*p*p) - 7.5*(b*b*b)*(d*d)*(p*p) - 26.25*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU442(double a, double b, double p, double d, double s){
	return ((1.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.75*(a*a*a*a)*(d*d*d*d)*p - 4.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 12.0*(a*a*a)*b*(d*d*d*d)*p + 1.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 27.0*(a*a)*(b*b)*(d*d*d*d)*p + 16.875*(a*a)*(d*d)*(p*p) - 12.0*a*(b*b*b)*(d*d*d*d)*p - 45.0*a*b*(d*d)*(p*p) + 0.75*(b*b*b*b)*(d*d*d*d)*p + 16.875*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU443(double a, double b, double p, double d, double s){
	return (d*(-0.5*(a*a*a*a)*b*(d*d*d*d) + 3.0*(a*a*a)*(b*b)*(d*d*d*d) + 2.5*(a*a*a)*(d*d)*p - 3.0*(a*a)*(b*b*b)*(d*d*d*d) - 15.0*(a*a)*b*(d*d)*p + 0.5*a*(b*b*b*b)*(d*d*d*d) + 15.0*a*(b*b)*(d*d)*p + 13.125*a*(p*p) - 2.5*(b*b*b)*(d*d)*p - 13.125*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU444(double a, double b, double p, double d, double s){
	return ((0.0625*(a*a*a*a)*(d*d*d*d) - (a*a*a)*b*(d*d*d*d) + 2.25*(a*a)*(b*b)*(d*d*d*d) + 2.8125*(a*a)*(d*d)*p - a*(b*b*b)*(d*d*d*d) - 7.5*a*b*(d*d)*p + 0.0625*(b*b*b*b)*(d*d*d*d) + 2.8125*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU445(double a, double b, double p, double d, double s){
	return (d*(0.125*(a*a*a)*(d*d) - 0.75*(a*a)*b*(d*d) + 0.75*a*(b*b)*(d*d) + 1.3125*a*p - 0.125*(b*b*b)*(d*d) - 1.3125*b*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU446(double a, double b, double p, double d, double s){
	return ((0.09375*(a*a)*(d*d) - 0.25*a*b*(d*d) + 0.09375*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU447(double a, double b, double p, double d, double s){
	return (0.03125*d*(a - b)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU448(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU450(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 3.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 0.75*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 10.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 15.0*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 5.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 45.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 18.75*(a*a*a)*(d*d)*(p*p*p) - 30.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 75.0*(a*a)*b*(d*d)*(p*p*p) + 3.75*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 56.25*a*(b*b)*(d*d)*(p*p*p) + 32.8125*a*(p*p*p*p) - 7.5*(b*b*b)*(d*d)*(p*p*p) - 26.25*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU451(double a, double b, double p, double d, double s){
	return ((-2.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 2.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 22.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 9.375*(a*a*a*a)*(d*d*d*d)*(p*p) - 30.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 75.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 7.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 112.5*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 65.625*(a*a)*(d*d)*(p*p*p) - 37.5*a*(b*b*b)*(d*d*d*d)*(p*p) - 131.25*a*b*(d*d)*(p*p*p) + 1.875*(b*b*b*b)*(d*d*d*d)*(p*p) + 39.375*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU452(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.75*(a*a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 15.0*(a*a*a*a)*b*(d*d*d*d)*p + 2.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 45.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 28.125*(a*a*a)*(d*d)*(p*p) - 30.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 112.5*(a*a)*b*(d*d)*(p*p) + 3.75*a*(b*b*b*b)*(d*d*d*d)*p + 84.375*a*(b*b)*(d*d)*(p*p) + 65.625*a*(p*p*p) - 11.25*(b*b*b)*(d*d)*(p*p) - 52.5*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU453(double a, double b, double p, double d, double s){
	return ((-0.5*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.125*(a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 25.0*(a*a*a)*b*(d*d*d*d)*p + 1.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 37.5*(a*a)*(b*b)*(d*d*d*d)*p + 32.8125*(a*a)*(d*d)*(p*p) - 12.5*a*(b*b*b)*(d*d*d*d)*p - 65.625*a*b*(d*d)*(p*p) + 0.625*(b*b*b*b)*(d*d*d*d)*p + 19.6875*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU454(double a, double b, double p, double d, double s){
	return (d*(0.0625*(a*a*a*a*a)*(d*d*d*d) - 1.25*(a*a*a*a)*b*(d*d*d*d) + 3.75*(a*a*a)*(b*b)*(d*d*d*d) + 4.6875*(a*a*a)*(d*d)*p - 2.5*(a*a)*(b*b*b)*(d*d*d*d) - 18.75*(a*a)*b*(d*d)*p + 0.3125*a*(b*b*b*b)*(d*d*d*d) + 14.0625*a*(b*b)*(d*d)*p + 16.40625*a*(p*p) - 1.875*(b*b*b)*(d*d)*p - 13.125*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU455(double a, double b, double p, double d, double s){
	return ((0.15625*(a*a*a*a)*(d*d*d*d) - 1.25*(a*a*a)*b*(d*d*d*d) + 1.875*(a*a)*(b*b)*(d*d*d*d) + 3.28125*(a*a)*(d*d)*p - 0.625*a*(b*b*b)*(d*d*d*d) - 6.5625*a*b*(d*d)*p + 0.03125*(b*b*b*b)*(d*d*d*d) + 1.96875*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU456(double a, double b, double p, double d, double s){
	return (d*(0.15625*(a*a*a)*(d*d) - 0.625*(a*a)*b*(d*d) + 0.46875*a*(b*b)*(d*d) + 1.09375*a*p - 0.0625*(b*b*b)*(d*d) - 0.875*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU457(double a, double b, double p, double d, double s){
	return ((0.078125*(a*a)*(d*d) - 0.15625*a*b*(d*d) + 0.046875*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU458(double a, double b, double p, double d, double s){
	return (d*(0.01953125*a - 0.015625*b)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU459(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU460(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 3.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 0.75*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 12.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 18.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 7.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 67.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 28.125*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 60.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 150.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 11.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 168.75*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 98.4375*(a*a)*(d*d)*(p*p*p*p) - 45.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 157.5*a*b*(d*d)*(p*p*p*p) + 1.875*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 39.375*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU461(double a, double b, double p, double d, double s){
	return (d*(-2.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 3.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 27.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 11.25*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 45.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 112.5*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 15.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 225.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 131.25*(a*a*a)*(d*d)*(p*p*p) - 112.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 393.75*(a*a)*b*(d*d)*(p*p*p) + 11.25*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 236.25*a*(b*b)*(d*d)*(p*p*p) + 177.1875*a*(p*p*p*p) - 26.25*(b*b*b)*(d*d)*(p*p*p) - 118.125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU462(double a, double b, double p, double d, double s){
	return ((1.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.75*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 6.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 18.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 3.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 67.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 42.1875*(a*a*a*a)*(d*d*d*d)*(p*p) - 60.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 225.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 11.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 253.125*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 196.875*(a*a)*(d*d)*(p*p*p) - 67.5*a*(b*b*b)*(d*d*d*d)*(p*p) - 315.0*a*b*(d*d)*(p*p*p) + 2.8125*(b*b*b*b)*(d*d*d*d)*(p*p) + 78.75*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU463(double a, double b, double p, double d, double s){
	return (d*(-0.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.75*(a*a*a*a*a)*(d*d*d*d)*p - 7.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 37.5*(a*a*a*a)*b*(d*d*d*d)*p + 2.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 75.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 65.625*(a*a*a)*(d*d)*(p*p) - 37.5*(a*a)*(b*b*b)*(d*d*d*d)*p - 196.875*(a*a)*b*(d*d)*(p*p) + 3.75*a*(b*b*b*b)*(d*d*d*d)*p + 118.125*a*(b*b)*(d*d)*(p*p) + 118.125*a*(p*p*p) - 13.125*(b*b*b)*(d*d)*(p*p) - 78.75*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU464(double a, double b, double p, double d, double s){
	return ((0.0625*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 1.5*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 5.625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 7.03125*(a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 37.5*(a*a*a)*b*(d*d*d*d)*p + 0.9375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 42.1875*(a*a)*(b*b)*(d*d*d*d)*p + 49.21875*(a*a)*(d*d)*(p*p) - 11.25*a*(b*b*b)*(d*d*d*d)*p - 78.75*a*b*(d*d)*(p*p) + 0.46875*(b*b*b*b)*(d*d*d*d)*p + 19.6875*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU465(double a, double b, double p, double d, double s){
	return (d*(0.1875*(a*a*a*a*a)*(d*d*d*d) - 1.875*(a*a*a*a)*b*(d*d*d*d) + 3.75*(a*a*a)*(b*b)*(d*d*d*d) + 6.5625*(a*a*a)*(d*d)*p - 1.875*(a*a)*(b*b*b)*(d*d*d*d) - 19.6875*(a*a)*b*(d*d)*p + 0.1875*a*(b*b*b*b)*(d*d*d*d) + 11.8125*a*(b*b)*(d*d)*p + 17.71875*a*(p*p) - 1.3125*(b*b*b)*(d*d)*p - 11.8125*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU466(double a, double b, double p, double d, double s){
	return ((0.234375*(a*a*a*a)*(d*d*d*d) - 1.25*(a*a*a)*b*(d*d*d*d) + 1.40625*(a*a)*(b*b)*(d*d*d*d) + 3.28125*(a*a)*(d*d)*p - 0.375*a*(b*b*b)*(d*d*d*d) - 5.25*a*b*(d*d)*p + 0.015625*(b*b*b*b)*(d*d*d*d) + 1.3125*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU467(double a, double b, double p, double d, double s){
	return (d*(0.15625*(a*a*a)*(d*d) - 0.46875*(a*a)*b*(d*d) + 0.28125*a*(b*b)*(d*d) + 0.84375*a*p - 0.03125*(b*b*b)*(d*d) - 0.5625*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU468(double a, double b, double p, double d, double s){
	return ((0.05859375*(a*a)*(d*d) - 0.09375*a*b*(d*d) + 0.0234375*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU469(double a, double b, double p, double d, double s){
	return (d*(0.01171875*a - 0.0078125*b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU4610(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU470(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 3.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 0.75*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 14.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 21.0 *(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 10.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 94.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 39.375*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 105.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 262.5*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 26.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 393.75*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 229.6875*(a*a*a)*(d*d)*(p*p*p*p) - 157.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 551.25*(a*a)*b*(d*d)*(p*p*p*p) + 13.125*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 275.625*a*(b*b)*(d*d)*(p*p*p*p) + 206.71875*a*(p*p*p*p*p) - 26.25*(b*b*b)*(d*d)*(p*p*p*p) - 118.125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU471(double a, double b, double p, double d, double s){
	return ((-2.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 3.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 31.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 13.125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 63.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 157.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 26.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 393.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 229.6875*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 262.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 918.75*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 39.375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 826.875*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 620.15625*(a*a)*(d*d)*(p*p*p*p) - 183.75*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 826.875*a*b*(d*d)*(p*p*p*p) + 6.5625*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 177.1875*(b*b)*(d*d)*(p*p*p*p) + 162.421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU472(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.75*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 21.0 *(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 5.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 94.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 59.0625*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 105.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 393.75*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 26.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 590.625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 459.375*(a*a*a)*(d*d)*(p*p*p) - 236.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 1102.5*(a*a)*b*(d*d)*(p*p*p) + 19.6875*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 551.25*a*(b*b)*(d*d)*(p*p*p) + 516.796875*a*(p*p*p*p) - 52.5*(b*b*b)*(d*d)*(p*p*p) - 295.3125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU473(double a, double b, double p, double d, double s){
	return ((-0.5*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 5.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 4.375*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 10.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 52.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 4.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 131.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 114.84375*(a*a*a*a)*(d*d*d*d)*(p*p) - 87.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 459.375*(a*a*a)*b*(d*d*d*d)*(p*p) + 13.125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 413.4375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 413.4375*(a*a)*(d*d)*(p*p*p) - 91.875*a*(b*b*b)*(d*d*d*d)*(p*p) - 551.25*a*b*(d*d)*(p*p*p) + 3.28125*(b*b*b*b)*(d*d*d*d)*(p*p) + 118.125*(b*b)*(d*d)*(p*p*p) + 135.3515625*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU474(double a, double b, double p, double d, double s){
	return (d*(0.0625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 1.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 7.875*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 9.84375*(a*a*a*a*a)*(d*d*d*d)*p - 8.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 65.625*(a*a*a*a)*b*(d*d*d*d)*p + 2.1875*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 98.4375*(a*a*a)*(b*b)*(d*d*d*d)*p + 114.84375*(a*a*a)*(d*d)*(p*p) - 39.375*(a*a)*(b*b*b)*(d*d*d*d)*p - 275.625*(a*a)*b*(d*d)*(p*p) + 3.28125*a*(b*b*b*b)*(d*d*d*d)*p + 137.8125*a*(b*b)*(d*d)*(p*p) + 172.265625*a*(p*p*p) - 13.125*(b*b*b)*(d*d)*(p*p) - 98.4375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU475(double a, double b, double p, double d, double s){
	return ((0.21875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 2.625*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 6.5625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 11.484375*(a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 45.9375*(a*a*a)*b*(d*d*d*d)*p + 0.65625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 41.34375*(a*a)*(b*b)*(d*d*d*d)*p + 62.015625*(a*a)*(d*d)*(p*p) - 9.1875*a*(b*b*b)*(d*d*d*d)*p - 82.6875*a*b*(d*d)*(p*p) + 0.328125*(b*b*b*b)*(d*d*d*d)*p + 17.71875*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU476(double a, double b, double p, double d, double s){
	return (d*(0.328125*(a*a*a*a*a)*(d*d*d*d) - 2.1875*(a*a*a*a)*b*(d*d*d*d) + 3.28125*(a*a*a)*(b*b)*(d*d*d*d) + 7.65625*(a*a*a)*(d*d)*p - 1.3125*(a*a)*(b*b*b)*(d*d*d*d) - 18.375*(a*a)*b*(d*d)*p + 0.109375*a*(b*b*b*b)*(d*d*d*d) + 9.1875*a*(b*b)*(d*d)*p + 17.2265625*a*(p*p) - 0.875*(b*b*b)*(d*d)*p - 9.84375*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU477(double a, double b, double p, double d, double s){
	return ((0.2734375*(a*a*a*a)*(d*d*d*d) - 1.09375*(a*a*a)*b*(d*d*d*d) + 0.984375*(a*a)*(b*b)*(d*d*d*d) + 2.953125*(a*a)*(d*d)*p - 0.21875*a*(b*b*b)*(d*d*d*d) - 3.9375*a*b*(d*d)*p + 0.0078125*(b*b*b*b)*(d*d*d*d) + 0.84375*(b*b)*(d*d)*p + 1.93359375*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU478(double a, double b, double p, double d, double s){
	return (d*(0.13671875*(a*a*a)*(d*d) - 0.328125*(a*a)*b*(d*d) + 0.1640625*a*(b*b)*(d*d) + 0.615234375*a*p - 0.015625*(b*b*b)*(d*d) - 0.3515625*b*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU479(double a, double b, double p, double d, double s){
	return ((0.041015625*(a*a)*(d*d) - 0.0546875*a*b*(d*d) + 0.01171875*(b*b)*(d*d) + 0.0537109375*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU4710(double a, double b, double p, double d, double s){
	return (d*(0.0068359375*a - 0.00390625*b)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU4711(double a, double b, double p, double d, double s){
	return (0.00048828125/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU480(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 3.0*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 0.75*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p) - 16.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 24.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 14.0*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 126.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 168.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 420.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 52.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 787.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 459.375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 420.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1470.0*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 52.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1102.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 826.875*(a*a)*(d*d)*(p*p*p*p*p) - 210.0*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 945.0*a*b*(d*d)*(p*p*p*p*p) + 6.5625*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 177.1875*(b*b)*(d*d)*(p*p*p*p*p) + 162.421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU481(double a, double b, double p, double d, double s){
	return (d*(-2.0*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.0*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 4.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 36.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 15.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 84.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 210.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 42.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 630.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 367.5*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 525.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 1837.5*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 105.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2205.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 1653.75*(a*a*a)*(d*d)*(p*p*p*p) - 735.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 3307.5*(a*a)*b*(d*d)*(p*p*p*p) + 52.5*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 1417.5*a*(b*b)*(d*d)*(p*p*p*p) + 1299.375*a*(p*p*p*p*p) - 118.125*(b*b*b)*(d*d)*(p*p*p*p) - 649.6875*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU482(double a, double b, double p, double d, double s){
	return ((1.5*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.75*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 8.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 24.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 7.0*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 126.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 78.75*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 168.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 630.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1181.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 918.75*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 630.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 2940.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 78.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2205.0*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 2067.1875*(a*a)*(d*d)*(p*p*p*p) - 420.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 2362.5*a*b*(d*d)*(p*p*p*p) + 13.125*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 442.96875*(b*b)*(d*d)*(p*p*p*p) + 487.265625*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU483(double a, double b, double p, double d, double s){
	return (d*(-0.5*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 6.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 5.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 14.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 70.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 7.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 210.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 183.75*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 175.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 918.75*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 35.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1102.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 1102.5*(a*a*a)*(d*d)*(p*p*p) - 367.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 2205.0*(a*a)*b*(d*d)*(p*p*p) + 26.25*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 945.0*a*(b*b)*(d*d)*(p*p*p) + 1082.8125*a*(p*p*p*p) - 78.75*(b*b*b)*(d*d)*(p*p*p) - 541.40625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU484(double a, double b, double p, double d, double s){
	return ((0.0625*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 2.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 10.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 14.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 105.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 4.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 196.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 229.6875*(a*a*a*a)*(d*d*d*d)*(p*p) - 105.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 735.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 13.125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 551.25*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 689.0625*(a*a)*(d*d)*(p*p*p) - 105.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 787.5*a*b*(d*d)*(p*p*p) + 3.28125*(b*b*b*b)*(d*d*d*d)*(p*p) + 147.65625*(b*b)*(d*d)*(p*p*p) + 203.02734375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU485(double a, double b, double p, double d, double s){
	return (d*(0.25*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 10.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 18.375*(a*a*a*a*a)*(d*d*d*d)*p - 8.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 91.875*(a*a*a*a)*b*(d*d*d*d)*p + 1.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 110.25*(a*a*a)*(b*b)*(d*d*d*d)*p + 165.375*(a*a*a)*(d*d)*(p*p) - 36.75*(a*a)*(b*b*b)*(d*d*d*d)*p - 330.75*(a*a)*b*(d*d)*(p*p) + 2.625*a*(b*b*b*b)*(d*d*d*d)*p + 141.75*a*(b*b)*(d*d)*(p*p) + 216.5625*a*(p*p*p) - 11.8125*(b*b*b)*(d*d)*(p*p) - 108.28125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU486(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.5*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 6.5625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 15.3125*(a*a*a*a)*(d*d*d*d)*p - 3.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 49.0*(a*a*a)*b*(d*d*d*d)*p + 0.4375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 36.75*(a*a)*(b*b)*(d*d*d*d)*p + 68.90625*(a*a)*(d*d)*(p*p) - 7.0*a*(b*b*b)*(d*d*d*d)*p - 78.75*a*b*(d*d)*(p*p) + 0.21875*(b*b*b*b)*(d*d*d*d)*p + 14.765625*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU487(double a, double b, double p, double d, double s){
	return (d*(0.4375*(a*a*a*a*a)*(d*d*d*d) - 2.1875*(a*a*a*a)*b*(d*d*d*d) + 2.625*(a*a*a)*(b*b)*(d*d*d*d) + 7.875*(a*a*a)*(d*d)*p - 0.875*(a*a)*(b*b*b)*(d*d*d*d) - 15.75*(a*a)*b*(d*d)*p + 0.0625*a*(b*b*b*b)*(d*d*d*d) + 6.75*a*(b*b)*(d*d)*p + 15.46875*a*(p*p) - 0.5625*(b*b*b)*(d*d)*p - 7.734375*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU488(double a, double b, double p, double d, double s){
	return ((0.2734375*(a*a*a*a)*(d*d*d*d) - 0.875*(a*a*a)*b*(d*d*d*d) + 0.65625*(a*a)*(b*b)*(d*d*d*d) + 2.4609375*(a*a)*(d*d)*p - 0.125*a*(b*b*b)*(d*d*d*d) - 2.8125*a*b*(d*d)*p + 0.00390625*(b*b*b*b)*(d*d*d*d) + 0.52734375*(b*b)*(d*d)*p + 1.4501953125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU489(double a, double b, double p, double d, double s){
	return (d*(0.109375*(a*a*a)*(d*d) - 0.21875*(a*a)*b*(d*d) + 0.09375*a*(b*b)*(d*d) + 0.4296875*a*p - 0.0078125*(b*b*b)*(d*d) - 0.21484375*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU4810(double a, double b, double p, double d, double s){
	return ((0.02734375*(a*a)*(d*d) - 0.03125*a*b*(d*d) + 0.005859375*(b*b)*(d*d) + 0.0322265625*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU4811(double a, double b, double p, double d, double s){
	return (d*(0.00390625*a - 0.001953125*b)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU4812(double a, double b, double p, double d, double s){
	return (0.000244140625/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU500(double a, double b, double p, double d, double s){
	return (b*d*(-(b*b*b*b)*(d*d*d*d) - 5.0*(b*b)*(d*d)*p - 3.75*(p*p))/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU501(double a, double b, double p, double d, double s){
	return ((2.5*(b*b*b*b)*(d*d*d*d) + 7.5*(b*b)*(d*d)*p + 1.875*(p*p))/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU502(double a, double b, double p, double d, double s){
	return (b*d*(-2.5*(b*b)*(d*d) - 3.75*p)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU503(double a, double b, double p, double d, double s){
	return ((1.25*(b*b)*(d*d) + 0.625*p)/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU504(double a, double b, double p, double d, double s){
	return (-0.3125*b*d/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU505(double a, double b, double p, double d, double s){
	return (0.03125/(p*p*p*p*p))*s;
}

inline double MD_Et_GPU510(double a, double b, double p, double d, double s){
	return ((-a*(b*b*b*b*b)*(d*d*d*d*d*d) - 5.0*a*(b*b*b)*(d*d*d*d)*p - 3.75*a*b*(d*d)*(p*p) + 2.5*(b*b*b*b)*(d*d*d*d)*p + 7.5*(b*b)*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU511(double a, double b, double p, double d, double s){
	return (d*(2.5*a*(b*b*b*b)*(d*d*d*d) + 7.5*a*(b*b)*(d*d)*p + 1.875*a*(p*p) - 0.5*(b*b*b*b*b)*(d*d*d*d) - 7.5*(b*b*b)*(d*d)*p - 9.375*b*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU512(double a, double b, double p, double d, double s){
	return ((-2.5*a*(b*b*b)*(d*d*d*d) - 3.75*a*b*(d*d)*p + 1.25*(b*b*b*b)*(d*d*d*d) + 7.5*(b*b)*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU513(double a, double b, double p, double d, double s){
	return (d*(1.25*a*(b*b)*(d*d) + 0.625*a*p - 1.25*(b*b*b)*(d*d) - 3.125*b*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU514(double a, double b, double p, double d, double s){
	return ((-0.3125*a*b*(d*d) + 0.625*(b*b)*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU515(double a, double b, double p, double d, double s){
	return (d*(0.03125*a - 0.15625*b)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU516(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU520(double a, double b, double p, double d, double s){
	return (d*(-(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 5.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 3.75*(a*a)*b*(d*d)*(p*p) + 5.0*a*(b*b*b*b)*(d*d*d*d)*p + 15.0*a*(b*b)*(d*d)*(p*p) + 3.75*a*(p*p*p) - 0.5*(b*b*b*b*b)*(d*d*d*d)*p - 7.5*(b*b*b)*(d*d)*(p*p) - 9.375*b*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU521(double a, double b, double p, double d, double s){
	return ((2.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 7.5*(a*a)*(b*b)*(d*d*d*d)*p + 1.875*(a*a)*(d*d)*(p*p) - a*(b*b*b*b*b)*(d*d*d*d*d*d) - 15.0*a*(b*b*b)*(d*d*d*d)*p - 18.75*a*b*(d*d)*(p*p) + 3.75*(b*b*b*b)*(d*d*d*d)*p + 18.75*(b*b)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU522(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a)*(b*b*b)*(d*d*d*d) - 3.75*(a*a)*b*(d*d)*p + 2.5*a*(b*b*b*b)*(d*d*d*d) + 15.0*a*(b*b)*(d*d)*p + 5.625*a*(p*p) - 0.25*(b*b*b*b*b)*(d*d*d*d) - 7.5*(b*b*b)*(d*d)*p - 14.0625*b*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU523(double a, double b, double p, double d, double s){
	return ((1.25*(a*a)*(b*b)*(d*d*d*d) + 0.625*(a*a)*(d*d)*p - 2.5*a*(b*b*b)*(d*d*d*d) - 6.25*a*b*(d*d)*p + 0.625*(b*b*b*b)*(d*d*d*d) + 6.25*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU524(double a, double b, double p, double d, double s){
	return (d*(-0.3125*(a*a)*b*(d*d) + 1.25*a*(b*b)*(d*d) + 0.9375*a*p - 0.625*(b*b*b)*(d*d) - 2.34375*b*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU525(double a, double b, double p, double d, double s){
	return ((0.03125*(a*a)*(d*d) - 0.3125*a*b*(d*d) + 0.3125*(b*b)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU526(double a, double b, double p, double d, double s){
	return (d*(0.03125*a - 0.078125*b)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU527(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU530(double a, double b, double p, double d, double s){
	return ((-(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 5.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 3.75*(a*a*a)*b*(d*d*d*d)*(p*p) + 7.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 22.5*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 5.625*(a*a)*(d*d)*(p*p*p) - 1.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 22.5*a*(b*b*b)*(d*d*d*d)*(p*p) - 28.125*a*b*(d*d)*(p*p*p) + 3.75*(b*b*b*b)*(d*d*d*d)*(p*p) + 18.75*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU531(double a, double b, double p, double d, double s){
	return (d*(2.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 7.5*(a*a*a)*(b*b)*(d*d*d*d)*p + 1.875*(a*a*a)*(d*d)*(p*p) - 1.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 22.5*(a*a)*(b*b*b)*(d*d*d*d)*p - 28.125*(a*a)*b*(d*d)*(p*p) + 11.25*a*(b*b*b*b)*(d*d*d*d)*p + 56.25*a*(b*b)*(d*d)*(p*p) + 19.6875*a*(p*p*p) - 0.75*(b*b*b*b*b)*(d*d*d*d)*p - 18.75*(b*b*b)*(d*d)*(p*p) - 32.8125*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU532(double a, double b, double p, double d, double s){
	return ((-2.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.75*(a*a*a)*b*(d*d*d*d)*p + 3.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 22.5*(a*a)*(b*b)*(d*d*d*d)*p + 8.4375*(a*a)*(d*d)*(p*p) - 0.75*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 22.5*a*(b*b*b)*(d*d*d*d)*p - 42.1875*a*b*(d*d)*(p*p) + 3.75*(b*b*b*b)*(d*d*d*d)*p + 28.125*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU533(double a, double b, double p, double d, double s){
	return (d*(1.25*(a*a*a)*(b*b)*(d*d*d*d) + 0.625*(a*a*a)*(d*d)*p - 3.75*(a*a)*(b*b*b)*(d*d*d*d) - 9.375*(a*a)*b*(d*d)*p + 1.875*a*(b*b*b*b)*(d*d*d*d) + 18.75*a*(b*b)*(d*d)*p + 9.84375*a*(p*p) - 0.125*(b*b*b*b*b)*(d*d*d*d) - 6.25*(b*b*b)*(d*d)*p - 16.40625*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU534(double a, double b, double p, double d, double s){
	return ((-0.3125*(a*a*a)*b*(d*d*d*d) + 1.875*(a*a)*(b*b)*(d*d*d*d) + 1.40625*(a*a)*(d*d)*p - 1.875*a*(b*b*b)*(d*d*d*d) - 7.03125*a*b*(d*d)*p + 0.3125*(b*b*b*b)*(d*d*d*d) + 4.6875*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU535(double a, double b, double p, double d, double s){
	return (d*(0.03125*(a*a*a)*(d*d) - 0.46875*(a*a)*b*(d*d) + 0.9375*a*(b*b)*(d*d) + 0.984375*a*p - 0.3125*(b*b*b)*(d*d) - 1.640625*b*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU536(double a, double b, double p, double d, double s){
	return ((0.046875*(a*a)*(d*d) - 0.234375*a*b*(d*d) + 0.15625*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU537(double a, double b, double p, double d, double s){
	return (d*(0.0234375*a - 0.0390625*b)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU538(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU540(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 5.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 3.75*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 10.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 30.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 7.5*(a*a*a)*(d*d)*(p*p*p) - 3.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 45.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 56.25*(a*a)*b*(d*d)*(p*p*p) + 15.0*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 75.0*a*(b*b)*(d*d)*(p*p*p) + 26.25*a*(p*p*p*p) - 0.75*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 18.75*(b*b*b)*(d*d)*(p*p*p) - 32.8125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU541(double a, double b, double p, double d, double s){
	return ((2.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 1.875*(a*a*a*a)*(d*d*d*d)*(p*p) - 2.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 30.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 37.5*(a*a*a)*b*(d*d*d*d)*(p*p) + 22.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 112.5*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 39.375*(a*a)*(d*d)*(p*p*p) - 3.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 75.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 131.25*a*b*(d*d)*(p*p*p) + 9.375*(b*b*b*b)*(d*d*d*d)*(p*p) + 65.625*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU542(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.75*(a*a*a*a)*b*(d*d*d*d)*p + 5.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 30.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 11.25*(a*a*a)*(d*d)*(p*p) - 1.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 45.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 84.375*(a*a)*b*(d*d)*(p*p) + 15.0*a*(b*b*b*b)*(d*d*d*d)*p + 112.5*a*(b*b)*(d*d)*(p*p) + 52.5*a*(p*p*p) - 0.75*(b*b*b*b*b)*(d*d*d*d)*p - 28.125*(b*b*b)*(d*d)*(p*p) - 65.625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU543(double a, double b, double p, double d, double s){
	return ((1.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.625*(a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 12.5*(a*a*a)*b*(d*d*d*d)*p + 3.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 37.5*(a*a)*(b*b)*(d*d*d*d)*p + 19.6875*(a*a)*(d*d)*(p*p) - 0.5*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 25.0*a*(b*b*b)*(d*d*d*d)*p - 65.625*a*b*(d*d)*(p*p) + 3.125*(b*b*b*b)*(d*d*d*d)*p + 32.8125*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU544(double a, double b, double p, double d, double s){
	return (d*(-0.3125*(a*a*a*a)*b*(d*d*d*d) + 2.5*(a*a*a)*(b*b)*(d*d*d*d) + 1.875*(a*a*a)*(d*d)*p - 3.75*(a*a)*(b*b*b)*(d*d*d*d) - 14.0625*(a*a)*b*(d*d)*p + 1.25*a*(b*b*b*b)*(d*d*d*d) + 18.75*a*(b*b)*(d*d)*p + 13.125*a*(p*p) - 0.0625*(b*b*b*b*b)*(d*d*d*d) - 4.6875*(b*b*b)*(d*d)*p - 16.40625*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU545(double a, double b, double p, double d, double s){
	return ((0.03125*(a*a*a*a)*(d*d*d*d) - 0.625*(a*a*a)*b*(d*d*d*d) + 1.875*(a*a)*(b*b)*(d*d*d*d) + 1.96875*(a*a)*(d*d)*p - 1.25*a*(b*b*b)*(d*d*d*d) - 6.5625*a*b*(d*d)*p + 0.15625*(b*b*b*b)*(d*d*d*d) + 3.28125*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU546(double a, double b, double p, double d, double s){
	return (d*(0.0625*(a*a*a)*(d*d) - 0.46875*(a*a)*b*(d*d) + 0.625*a*(b*b)*(d*d) + 0.875*a*p - 0.15625*(b*b*b)*(d*d) - 1.09375*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU547(double a, double b, double p, double d, double s){
	return ((0.046875*(a*a)*(d*d) - 0.15625*a*b*(d*d) + 0.078125*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU548(double a, double b, double p, double d, double s){
	return (d*(0.015625*a - 0.01953125*b)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU549(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU550(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 5.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 3.75*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 12.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 37.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 9.375*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 5.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 75.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 93.75*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 37.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 187.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 65.625*(a*a)*(d*d)*(p*p*p*p) - 3.75*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 93.75*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 164.0625*a*b*(d*d)*(p*p*p*p) + 9.375*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 65.625*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU551(double a, double b, double p, double d, double s){
	return (d*(2.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 1.875*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 2.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 37.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 46.875*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 37.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 187.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 65.625*(a*a*a)*(d*d)*(p*p*p) - 7.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 187.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 328.125*(a*a)*b*(d*d)*(p*p*p) + 46.875*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 328.125*a*(b*b)*(d*d)*(p*p*p) + 147.65625*a*(p*p*p*p) - 1.875*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 65.625*(b*b*b)*(d*d)*(p*p*p) - 147.65625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU552(double a, double b, double p, double d, double s){
	return ((-2.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 6.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 37.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 14.0625*(a*a*a*a)*(d*d*d*d)*(p*p) - 2.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 75.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 140.625*(a*a*a)*b*(d*d*d*d)*(p*p) + 37.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 281.25*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 131.25*(a*a)*(d*d)*(p*p*p) - 3.75*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 140.625*a*(b*b*b)*(d*d*d*d)*(p*p) - 328.125*a*b*(d*d)*(p*p*p) + 14.0625*(b*b*b*b)*(d*d*d*d)*(p*p) + 131.25*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU553(double a, double b, double p, double d, double s){
	return (d*(1.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.625*(a*a*a*a*a)*(d*d*d*d)*p - 6.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 15.625*(a*a*a*a)*b*(d*d*d*d)*p + 6.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 62.5*(a*a*a)*(b*b)*(d*d*d*d)*p + 32.8125*(a*a*a)*(d*d)*(p*p) - 1.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 62.5*(a*a)*(b*b*b)*(d*d*d*d)*p - 164.0625*(a*a)*b*(d*d)*(p*p) + 15.625*a*(b*b*b*b)*(d*d*d*d)*p + 164.0625*a*(b*b)*(d*d)*(p*p) + 98.4375*a*(p*p*p) - 0.625*(b*b*b*b*b)*(d*d*d*d)*p - 32.8125*(b*b*b)*(d*d)*(p*p) - 98.4375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU554(double a, double b, double p, double d, double s){
	return ((-0.3125*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 2.34375*(a*a*a*a)*(d*d*d*d)*p - 6.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 23.4375*(a*a*a)*b*(d*d*d*d)*p + 3.125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 46.875*(a*a)*(b*b)*(d*d*d*d)*p + 32.8125*(a*a)*(d*d)*(p*p) - 0.3125*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 23.4375*a*(b*b*b)*(d*d*d*d)*p - 82.03125*a*b*(d*d)*(p*p) + 2.34375*(b*b*b*b)*(d*d*d*d)*p + 32.8125*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU555(double a, double b, double p, double d, double s){
	return (d*(0.03125*(a*a*a*a*a)*(d*d*d*d) - 0.78125*(a*a*a*a)*b*(d*d*d*d) + 3.125*(a*a*a)*(b*b)*(d*d*d*d) + 3.28125*(a*a*a)*(d*d)*p - 3.125*(a*a)*(b*b*b)*(d*d*d*d) - 16.40625*(a*a)*b*(d*d)*p + 0.78125*a*(b*b*b*b)*(d*d*d*d) + 16.40625*a*(b*b)*(d*d)*p + 14.765625*a*(p*p) - 0.03125*(b*b*b*b*b)*(d*d*d*d) - 3.28125*(b*b*b)*(d*d)*p - 14.765625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU556(double a, double b, double p, double d, double s){
	return ((0.078125*(a*a*a*a)*(d*d*d*d) - 0.78125*(a*a*a)*b*(d*d*d*d) + 1.5625*(a*a)*(b*b)*(d*d*d*d) + 2.1875*(a*a)*(d*d)*p - 0.78125*a*(b*b*b)*(d*d*d*d) - 5.46875*a*b*(d*d)*p + 0.078125*(b*b*b*b)*(d*d*d*d) + 2.1875*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU557(double a, double b, double p, double d, double s){
	return (d*(0.078125*(a*a*a)*(d*d) - 0.390625*(a*a)*b*(d*d) + 0.390625*a*(b*b)*(d*d) + 0.703125*a*p - 0.078125*(b*b*b)*(d*d) - 0.703125*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU558(double a, double b, double p, double d, double s){
	return ((0.0390625*(a*a)*(d*d) - 0.09765625*a*b*(d*d) + 0.0390625*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU559(double a, double b, double p, double d, double s){
	return (0.009765625*d*(a - b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU5510(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU560(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 5.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 3.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 15.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 45.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 11.25*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 7.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 112.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 140.625*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 75.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 375.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 131.25*(a*a*a)*(d*d)*(p*p*p*p) - 11.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 281.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 492.1875*(a*a)*b*(d*d)*(p*p*p*p) + 56.25*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 393.75*a*(b*b)*(d*d)*(p*p*p*p) + 177.1875*a*(p*p*p*p*p) - 1.875*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 65.625*(b*b*b)*(d*d)*(p*p*p*p) - 147.65625*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU561(double a, double b, double p, double d, double s){
	return ((2.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 1.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 3.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 45.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 56.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 56.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 281.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 98.4375*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 15.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 375.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 656.25*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 140.625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 984.375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 442.96875*(a*a)*(d*d)*(p*p*p*p) - 11.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 393.75*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 885.9375*a*b*(d*d)*(p*p*p*p) + 32.8125*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 295.3125*(b*b)*(d*d)*(p*p*p*p) + 162.421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU562(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 7.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 45.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 16.875*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 3.75*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 112.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 210.9375*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 75.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 562.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 262.5*(a*a*a)*(d*d)*(p*p*p) - 11.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 421.875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 984.375*(a*a)*b*(d*d)*(p*p*p) + 84.375*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 787.5*a*(b*b)*(d*d)*(p*p*p) + 442.96875*a*(p*p*p*p) - 2.8125*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 131.25*(b*b*b)*(d*d)*(p*p*p) - 369.140625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU563(double a, double b, double p, double d, double s){
	return ((1.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.625*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 18.75*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 9.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 93.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 49.21875*(a*a*a*a)*(d*d*d*d)*(p*p) - 2.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 125.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 328.125*(a*a*a)*b*(d*d*d*d)*(p*p) + 46.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 492.1875*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 295.3125*(a*a)*(d*d)*(p*p*p) - 3.75*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 196.875*a*(b*b*b)*(d*d*d*d)*(p*p) - 590.625*a*b*(d*d)*(p*p*p) + 16.40625*(b*b*b*b)*(d*d*d*d)*(p*p) + 196.875*(b*b)*(d*d)*(p*p*p) + 135.3515625*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU564(double a, double b, double p, double d, double s){
	return (d*(-0.3125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 2.8125*(a*a*a*a*a)*(d*d*d*d)*p - 9.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 35.15625*(a*a*a*a)*b*(d*d*d*d)*p + 6.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 93.75*(a*a*a)*(b*b)*(d*d*d*d)*p + 65.625*(a*a*a)*(d*d)*(p*p) - 0.9375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 70.3125*(a*a)*(b*b*b)*(d*d*d*d)*p - 246.09375*(a*a)*b*(d*d)*(p*p) + 14.0625*a*(b*b*b*b)*(d*d*d*d)*p + 196.875*a*(b*b)*(d*d)*(p*p) + 147.65625*a*(p*p*p) - 0.46875*(b*b*b*b*b)*(d*d*d*d)*p - 32.8125*(b*b*b)*(d*d)*(p*p) - 123.046875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU565(double a, double b, double p, double d, double s){
	return ((0.03125*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.9375*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.6875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 4.921875*(a*a*a*a)*(d*d*d*d)*p - 6.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 32.8125*(a*a*a)*b*(d*d*d*d)*p + 2.34375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 49.21875*(a*a)*(b*b)*(d*d*d*d)*p + 44.296875*(a*a)*(d*d)*(p*p) - 0.1875*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 19.6875*a*(b*b*b)*(d*d*d*d)*p - 88.59375*a*b*(d*d)*(p*p) + 1.640625*(b*b*b*b)*(d*d*d*d)*p + 29.53125*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU566(double a, double b, double p, double d, double s){
	return (d*(0.09375*(a*a*a*a*a)*(d*d*d*d) - 1.171875*(a*a*a*a)*b*(d*d*d*d) + 3.125*(a*a*a)*(b*b)*(d*d*d*d) + 4.375*(a*a*a)*(d*d)*p - 2.34375*(a*a)*(b*b*b)*(d*d*d*d) - 16.40625*(a*a)*b*(d*d)*p + 0.46875*a*(b*b*b*b)*(d*d*d*d) + 13.125*a*(b*b)*(d*d)*p + 14.765625*a*(p*p) - 0.015625*(b*b*b*b*b)*(d*d*d*d) - 2.1875*(b*b*b)*(d*d)*p - 12.3046875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU567(double a, double b, double p, double d, double s){
	return ((0.1171875*(a*a*a*a)*(d*d*d*d) - 0.78125*(a*a*a)*b*(d*d*d*d) + 1.171875*(a*a)*(b*b)*(d*d*d*d) + 2.109375*(a*a)*(d*d)*p - 0.46875*a*(b*b*b)*(d*d*d*d) - 4.21875*a*b*(d*d)*p + 0.0390625*(b*b*b*b)*(d*d*d*d) + 1.40625*(b*b)*(d*d)*p + 1.93359375*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU568(double a, double b, double p, double d, double s){
	return (d*(0.078125*(a*a*a)*(d*d) - 0.29296875*(a*a)*b*(d*d) + 0.234375*a*(b*b)*(d*d) + 0.52734375*a*p - 0.0390625*(b*b*b)*(d*d) - 0.439453125*b*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU569(double a, double b, double p, double d, double s){
	return ((0.029296875*(a*a)*(d*d) - 0.05859375*a*b*(d*d) + 0.01953125*(b*b)*(d*d) + 0.0537109375*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU5610(double a, double b, double p, double d, double s){
	return (d*(0.005859375*a - 0.0048828125*b)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU5611(double a, double b, double p, double d, double s){
	return (0.00048828125/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU570(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 5.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 3.75*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 17.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 10.5*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 157.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 196.875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 131.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 656.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 229.6875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 26.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 656.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1148.4375*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 196.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1378.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 620.15625*(a*a)*(d*d)*(p*p*p*p*p) - 13.125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 459.375*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 1033.59375*a*b*(d*d)*(p*p*p*p*p) + 32.8125*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 295.3125*(b*b)*(d*d)*(p*p*p*p*p) + 162.421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU571(double a, double b, double p, double d, double s){
	return (d*(2.5*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 1.875*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 3.5*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 52.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 65.625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 78.75*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 393.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 137.8125*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 26.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 656.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 1148.4375*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 328.125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2296.875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 1033.59375*(a*a*a)*(d*d)*(p*p*p*p) - 39.375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1378.125*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 3100.78125*(a*a)*b*(d*d)*(p*p*p*p) + 229.6875*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 2067.1875*a*(b*b)*(d*d)*(p*p*p*p) + 1136.953125*a*(p*p*p*p*p) - 6.5625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 295.3125*(b*b*b)*(d*d)*(p*p*p*p) - 812.109375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU572(double a, double b, double p, double d, double s){
	return ((-2.5*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 8.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 52.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 19.6875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 5.25*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 157.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 295.3125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 131.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 984.375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 459.375*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 26.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 984.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 2296.875*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 295.3125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2756.25*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 1550.390625*(a*a)*(d*d)*(p*p*p*p) - 19.6875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 918.75*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 2583.984375*a*b*(d*d)*(p*p*p*p) + 65.625*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 738.28125*(b*b)*(d*d)*(p*p*p*p) + 487.265625*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU573(double a, double b, double p, double d, double s){
	return (d*(1.25*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 8.75*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 21.875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 13.125*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 131.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 68.90625*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 4.375*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 218.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 574.21875*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 109.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1148.4375*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 689.0625*(a*a*a)*(d*d)*(p*p*p) - 13.125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 689.0625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 2067.1875*(a*a)*b*(d*d)*(p*p*p) + 114.84375*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 1378.125*a*(b*b)*(d*d)*(p*p*p) + 947.4609375*a*(p*p*p*p) - 3.28125*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 196.875*(b*b*b)*(d*d)*(p*p*p) - 676.7578125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU574(double a, double b, double p, double d, double s){
	return ((-0.3125*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 4.375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 3.28125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 13.125*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 49.21875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 10.9375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 164.0625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 114.84375*(a*a*a*a)*(d*d*d*d)*(p*p) - 2.1875*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 164.0625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 574.21875*(a*a*a)*b*(d*d*d*d)*(p*p) + 49.21875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 689.0625*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 516.796875*(a*a)*(d*d)*(p*p*p) - 3.28125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 229.6875*a*(b*b*b)*(d*d*d*d)*(p*p) - 861.328125*a*b*(d*d)*(p*p*p) + 16.40625*(b*b*b*b)*(d*d*d*d)*(p*p) + 246.09375*(b*b)*(d*d)*(p*p*p) + 203.02734375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU575(double a, double b, double p, double d, double s){
	return (d*(0.03125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 1.09375*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 6.5625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 6.890625*(a*a*a*a*a)*(d*d*d*d)*p - 10.9375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 57.421875*(a*a*a*a)*b*(d*d*d*d)*p + 5.46875*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 114.84375*(a*a*a)*(b*b)*(d*d*d*d)*p + 103.359375*(a*a*a)*(d*d)*(p*p) - 0.65625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 68.90625*(a*a)*(b*b*b)*(d*d*d*d)*p - 310.078125*(a*a)*b*(d*d)*(p*p) + 11.484375*a*(b*b*b*b)*(d*d*d*d)*p + 206.71875*a*(b*b)*(d*d)*(p*p) + 189.4921875*a*(p*p*p) - 0.328125*(b*b*b*b*b)*(d*d*d*d)*p - 29.53125*(b*b*b)*(d*d)*(p*p) - 135.3515625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU576(double a, double b, double p, double d, double s){
	return ((0.109375*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 1.640625*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 5.46875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 7.65625*(a*a*a*a)*(d*d*d*d)*p - 5.46875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 38.28125*(a*a*a)*b*(d*d*d*d)*p + 1.640625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 45.9375*(a*a)*(b*b)*(d*d*d*d)*p + 51.6796875*(a*a)*(d*d)*(p*p) - 0.109375*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 15.3125*a*(b*b*b)*(d*d*d*d)*p - 86.1328125*a*b*(d*d)*(p*p) + 1.09375*(b*b*b*b)*(d*d*d*d)*p + 24.609375*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU577(double a, double b, double p, double d, double s){
	return (d*(0.1640625*(a*a*a*a*a)*(d*d*d*d) - 1.3671875*(a*a*a*a)*b*(d*d*d*d) + 2.734375*(a*a*a)*(b*b)*(d*d*d*d) + 4.921875*(a*a*a)*(d*d)*p - 1.640625*(a*a)*(b*b*b)*(d*d*d*d) - 14.765625*(a*a)*b*(d*d)*p + 0.2734375*a*(b*b*b*b)*(d*d*d*d) + 9.84375*a*(b*b)*(d*d)*p + 13.53515625*a*(p*p) - 0.0078125*(b*b*b*b*b)*(d*d*d*d) - 1.40625*(b*b*b)*(d*d)*p - 9.66796875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU578(double a, double b, double p, double d, double s){
	return ((0.13671875*(a*a*a*a)*(d*d*d*d) - 0.68359375*(a*a*a)*b*(d*d*d*d) + 0.8203125*(a*a)*(b*b)*(d*d*d*d) + 1.845703125*(a*a)*(d*d)*p - 0.2734375*a*(b*b*b)*(d*d*d*d) - 3.076171875*a*b*(d*d)*p + 0.01953125*(b*b*b*b)*(d*d*d*d) + 0.87890625*(b*b)*(d*d)*p + 1.4501953125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU579(double a, double b, double p, double d, double s){
	return (d*(0.068359375*(a*a*a)*(d*d) - 0.205078125*(a*a)*b*(d*d) + 0.13671875*a*(b*b)*(d*d) + 0.3759765625*a*p - 0.01953125*(b*b*b)*(d*d) - 0.2685546875*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU5710(double a, double b, double p, double d, double s){
	return ((0.0205078125*(a*a)*(d*d) - 0.0341796875*a*b*(d*d) + 0.009765625*(b*b)*(d*d) + 0.0322265625*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU5711(double a, double b, double p, double d, double s){
	return (d*(0.00341796875*a - 0.00244140625*b)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU5712(double a, double b, double p, double d, double s){
	return (0.000244140625/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU580(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 5.0*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 3.75*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 20.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 60.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 15.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 14.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 210.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 262.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 210.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1050.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 367.5*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 52.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1312.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 2296.875*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 525.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 3675.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 1653.75*(a*a*a)*(d*d)*(p*p*p*p*p) - 52.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1837.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 4134.375*(a*a)*b*(d*d)*(p*p*p*p*p) + 262.5*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 2362.5*a*(b*b)*(d*d)*(p*p*p*p*p) + 1299.375*a*(p*p*p*p*p*p) - 6.5625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 295.3125*(b*b*b)*(d*d)*(p*p*p*p*p) - 812.109375*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU581(double a, double b, double p, double d, double s){
	return ((2.5*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1.875*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p) - 4.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 60.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 75.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 105.0*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 525.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 183.75*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 42.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 1050.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1837.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 656.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 4593.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 2067.1875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 105.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 3675.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 8268.75*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 918.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 8268.75*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 4547.8125*(a*a)*(d*d)*(p*p*p*p*p) - 52.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 2362.5*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 6496.875*a*b*(d*d)*(p*p*p*p*p) + 147.65625*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 1624.21875*(b*b)*(d*d)*(p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU582(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 10.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 60.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 22.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 7.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 210.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 393.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 210.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1575.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 735.0*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 52.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1968.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 4593.75*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 787.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 7350.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 4134.375*(a*a*a)*(d*d)*(p*p*p*p) - 78.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 3675.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 10335.9375*(a*a)*b*(d*d)*(p*p*p*p) + 525.0*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 5906.25*a*(b*b)*(d*d)*(p*p*p*p) + 3898.125*a*(p*p*p*p*p) - 13.125*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 738.28125*(b*b*b)*(d*d)*(p*p*p*p) - 2436.328125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU583(double a, double b, double p, double d, double s){
	return ((1.25*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.625*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 10.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 25.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 17.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 175.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 91.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 7.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 350.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 918.75*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 218.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2296.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1378.125*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 35.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1837.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 5512.5*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 459.375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 5512.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 3789.84375*(a*a)*(d*d)*(p*p*p*p) - 26.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1575.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 5414.0625*a*b*(d*d)*(p*p*p*p) + 98.4375*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 1353.515625*(b*b)*(d*d)*(p*p*p*p) + 1055.7421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU584(double a, double b, double p, double d, double s){
	return (d*(-0.3125*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 5.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 3.75*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 17.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 65.625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 17.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 262.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 183.75*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 4.375*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 328.125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 1148.4375*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 131.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1837.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 1378.125*(a*a*a)*(d*d)*(p*p*p) - 13.125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 918.75*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 3445.3125*(a*a)*b*(d*d)*(p*p*p) + 131.25*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 1968.75*a*(b*b)*(d*d)*(p*p*p) + 1624.21875*a*(p*p*p*p) - 3.28125*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 246.09375*(b*b*b)*(d*d)*(p*p*p) - 1015.13671875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU585(double a, double b, double p, double d, double s){
	return ((0.03125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 1.25*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 8.75*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 9.1875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 17.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 91.875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 10.9375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 229.6875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 206.71875*(a*a*a*a)*(d*d*d*d)*(p*p) - 1.75*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 183.75*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 826.875*(a*a*a)*b*(d*d*d*d)*(p*p) + 45.9375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 826.875*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 757.96875*(a*a)*(d*d)*(p*p*p) - 2.625*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 236.25*a*(b*b*b)*(d*d*d*d)*(p*p) - 1082.8125*a*b*(d*d)*(p*p*p) + 14.765625*(b*b*b*b)*(d*d*d*d)*(p*p) + 270.703125*(b*b)*(d*d)*(p*p*p) + 263.935546875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU586(double a, double b, double p, double d, double s){
	return (d*(0.125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 2.1875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 8.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 12.25*(a*a*a*a*a)*(d*d*d*d)*p - 10.9375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 76.5625*(a*a*a*a)*b*(d*d*d*d)*p + 4.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 122.5*(a*a*a)*(b*b)*(d*d*d*d)*p + 137.8125*(a*a*a)*(d*d)*(p*p) - 0.4375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 61.25*(a*a)*(b*b*b)*(d*d*d*d)*p - 344.53125*(a*a)*b*(d*d)*(p*p) + 8.75*a*(b*b*b*b)*(d*d*d*d)*p + 196.875*a*(b*b)*(d*d)*(p*p) + 216.5625*a*(p*p*p) - 0.21875*(b*b*b*b*b)*(d*d*d*d)*p - 24.609375*(b*b*b)*(d*d)*(p*p) - 135.3515625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU587(double a, double b, double p, double d, double s){
	return ((0.21875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 2.1875*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 5.46875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 9.84375*(a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 39.375*(a*a*a)*b*(d*d*d*d)*p + 1.09375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 39.375*(a*a)*(b*b)*(d*d*d*d)*p + 54.140625*(a*a)*(d*d)*(p*p) - 0.0625*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 11.25*a*(b*b*b)*(d*d*d*d)*p - 77.34375*a*b*(d*d)*(p*p) + 0.703125*(b*b*b*b)*(d*d*d*d)*p + 19.3359375*(b*b)*(d*d)*(p*p) + 25.13671875*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU588(double a, double b, double p, double d, double s){
	return (d*(0.21875*(a*a*a*a*a)*(d*d*d*d) - 1.3671875*(a*a*a*a)*b*(d*d*d*d) + 2.1875*(a*a*a)*(b*b)*(d*d*d*d) + 4.921875*(a*a*a)*(d*d)*p - 1.09375*(a*a)*(b*b*b)*(d*d*d*d) - 12.3046875*(a*a)*b*(d*d)*p + 0.15625*a*(b*b*b*b)*(d*d*d*d) + 7.03125*a*(b*b)*(d*d)*p + 11.6015625*a*(p*p) - 0.00390625*(b*b*b*b*b)*(d*d*d*d) - 0.87890625*(b*b*b)*(d*d)*p - 7.2509765625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU589(double a, double b, double p, double d, double s){
	return ((0.13671875*(a*a*a*a)*(d*d*d*d) - 0.546875*(a*a*a)*b*(d*d*d*d) + 0.546875*(a*a)*(b*b)*(d*d*d*d) + 1.50390625*(a*a)*(d*d)*p - 0.15625*a*(b*b*b)*(d*d*d*d) - 2.1484375*a*b*(d*d)*p + 0.009765625*(b*b*b*b)*(d*d*d*d) + 0.537109375*(b*b)*(d*d)*p + 1.04736328125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU5810(double a, double b, double p, double d, double s){
	return (d*(0.0546875*(a*a*a)*(d*d) - 0.13671875*(a*a)*b*(d*d) + 0.078125*a*(b*b)*(d*d) + 0.2578125*a*p - 0.009765625*(b*b*b)*(d*d) - 0.1611328125*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU5811(double a, double b, double p, double d, double s){
	return ((0.013671875*(a*a)*(d*d) - 0.01953125*a*b*(d*d) + 0.0048828125*(b*b)*(d*d) + 0.01904296875*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU5812(double a, double b, double p, double d, double s){
	return (d*(0.001953125*a - 0.001220703125*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU5813(double a, double b, double p, double d, double s){
	return (0.0001220703125/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU600(double a, double b, double p, double d, double s){
	return (((b*b*b*b*b*b)*(d*d*d*d*d*d) + 7.5*(b*b*b*b)*(d*d*d*d)*p + 11.25*(b*b)*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU601(double a, double b, double p, double d, double s){
	return (b*d*(-3.0*(b*b*b*b)*(d*d*d*d) - 15.0*(b*b)*(d*d)*p - 11.25*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU602(double a, double b, double p, double d, double s){
	return ((3.75*(b*b*b*b)*(d*d*d*d) + 11.25*(b*b)*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU603(double a, double b, double p, double d, double s){
	return (b*d*(-2.5*(b*b)*(d*d) - 3.75*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU604(double a, double b, double p, double d, double s){
	return ((0.9375*(b*b)*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU605(double a, double b, double p, double d, double s){
	return (-0.1875*b*d/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU606(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU610(double a, double b, double p, double d, double s){
	return (d*(a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 7.5*a*(b*b*b*b)*(d*d*d*d)*p + 11.25*a*(b*b)*(d*d)*(p*p) + 1.875*a*(p*p*p) - 3.0*(b*b*b*b*b)*(d*d*d*d)*p - 15.0*(b*b*b)*(d*d)*(p*p) - 11.25*b*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU611(double a, double b, double p, double d, double s){
	return ((-3.0*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 15.0*a*(b*b*b)*(d*d*d*d)*p - 11.25*a*b*(d*d)*(p*p) + 0.5*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 11.25*(b*b*b*b)*(d*d*d*d)*p + 28.125*(b*b)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU612(double a, double b, double p, double d, double s){
	return (d*(3.75*a*(b*b*b*b)*(d*d*d*d) + 11.25*a*(b*b)*(d*d)*p + 2.8125*a*(p*p) - 1.5*(b*b*b*b*b)*(d*d*d*d) - 15.0*(b*b*b)*(d*d)*p - 16.875*b*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU613(double a, double b, double p, double d, double s){
	return ((-2.5*a*(b*b*b)*(d*d*d*d) - 3.75*a*b*(d*d)*p + 1.875*(b*b*b*b)*(d*d*d*d) + 9.375*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU614(double a, double b, double p, double d, double s){
	return (d*(0.9375*a*(b*b)*(d*d) + 0.46875*a*p - 1.25*(b*b*b)*(d*d) - 2.8125*b*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU615(double a, double b, double p, double d, double s){
	return ((-0.1875*a*b*(d*d) + 0.46875*(b*b)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU616(double a, double b, double p, double d, double s){
	return (d*(0.015625*a - 0.09375*b)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU617(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU620(double a, double b, double p, double d, double s){
	return (((a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 7.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 11.25*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 1.875*(a*a)*(d*d)*(p*p*p) - 6.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 30.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 22.5*a*b*(d*d)*(p*p*p) + 0.5*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 11.25*(b*b*b*b)*(d*d*d*d)*(p*p) + 28.125*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU621(double a, double b, double p, double d, double s){
	return (d*(-3.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 15.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 11.25*(a*a)*b*(d*d)*(p*p) + a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 22.5*a*(b*b*b*b)*(d*d*d*d)*p + 56.25*a*(b*b)*(d*d)*(p*p) + 13.125*a*(p*p*p) - 4.5*(b*b*b*b*b)*(d*d*d*d)*p - 37.5*(b*b*b)*(d*d)*(p*p) - 39.375*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU622(double a, double b, double p, double d, double s){
	return ((3.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 11.25*(a*a)*(b*b)*(d*d*d*d)*p + 2.8125*(a*a)*(d*d)*(p*p) - 3.0*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 30.0*a*(b*b*b)*(d*d*d*d)*p - 33.75*a*b*(d*d)*(p*p) + 0.25*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 11.25*(b*b*b*b)*(d*d*d*d)*p + 42.1875*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU623(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a)*(b*b*b)*(d*d*d*d) - 3.75*(a*a)*b*(d*d)*p + 3.75*a*(b*b*b*b)*(d*d*d*d) + 18.75*a*(b*b)*(d*d)*p + 6.5625*a*(p*p) - 0.75*(b*b*b*b*b)*(d*d*d*d) - 12.5*(b*b*b)*(d*d)*p - 19.6875*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU624(double a, double b, double p, double d, double s){
	return ((0.9375*(a*a)*(b*b)*(d*d*d*d) + 0.46875*(a*a)*(d*d)*p - 2.5*a*(b*b*b)*(d*d*d*d) - 5.625*a*b*(d*d)*p + 0.9375*(b*b*b*b)*(d*d*d*d) + 7.03125*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU625(double a, double b, double p, double d, double s){
	return (d*(-0.1875*(a*a)*b*(d*d) + 0.9375*a*(b*b)*(d*d) + 0.65625*a*p - 0.625*(b*b*b)*(d*d) - 1.96875*b*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU626(double a, double b, double p, double d, double s){
	return ((0.015625*(a*a)*(d*d) - 0.1875*a*b*(d*d) + 0.234375*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU627(double a, double b, double p, double d, double s){
	return (d*(0.015625*a - 0.046875*b)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU628(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU630(double a, double b, double p, double d, double s){
	return (d*((a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 7.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 11.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 1.875*(a*a*a)*(d*d)*(p*p*p) - 9.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 45.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 33.75*(a*a)*b*(d*d)*(p*p*p) + 1.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 33.75*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 84.375*a*(b*b)*(d*d)*(p*p*p) + 19.6875*a*(p*p*p*p) - 4.5*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 37.5*(b*b*b)*(d*d)*(p*p*p) - 39.375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU631(double a, double b, double p, double d, double s){
	return ((-3.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 15.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 11.25*(a*a*a)*b*(d*d*d*d)*(p*p) + 1.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 33.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 84.375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 19.6875*(a*a)*(d*d)*(p*p*p) - 13.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 112.5*a*(b*b*b)*(d*d*d*d)*(p*p) - 118.125*a*b*(d*d)*(p*p*p) + 0.75*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 28.125*(b*b*b*b)*(d*d*d*d)*(p*p) + 98.4375*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU632(double a, double b, double p, double d, double s){
	return (d*(3.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 11.25*(a*a*a)*(b*b)*(d*d*d*d)*p + 2.8125*(a*a*a)*(d*d)*(p*p) - 4.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 45.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 50.625*(a*a)*b*(d*d)*(p*p) + 0.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 33.75*a*(b*b*b*b)*(d*d*d*d)*p + 126.5625*a*(b*b)*(d*d)*(p*p) + 39.375*a*(p*p*p) - 4.5*(b*b*b*b*b)*(d*d*d*d)*p - 56.25*(b*b*b)*(d*d)*(p*p) - 78.75*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU633(double a, double b, double p, double d, double s){
	return ((-2.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.75*(a*a*a)*b*(d*d*d*d)*p + 5.625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 28.125*(a*a)*(b*b)*(d*d*d*d)*p + 9.84375*(a*a)*(d*d)*(p*p) - 2.25*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 37.5*a*(b*b*b)*(d*d*d*d)*p - 59.0625*a*b*(d*d)*(p*p) + 0.125*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 9.375*(b*b*b*b)*(d*d*d*d)*p + 49.21875*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU634(double a, double b, double p, double d, double s){
	return (d*(0.9375*(a*a*a)*(b*b)*(d*d*d*d) + 0.46875*(a*a*a)*(d*d)*p - 3.75*(a*a)*(b*b*b)*(d*d*d*d) - 8.4375*(a*a)*b*(d*d)*p + 2.8125*a*(b*b*b*b)*(d*d*d*d) + 21.09375*a*(b*b)*(d*d)*p + 9.84375*a*(p*p) - 0.375*(b*b*b*b*b)*(d*d*d*d) - 9.375*(b*b*b)*(d*d)*p - 19.6875*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU635(double a, double b, double p, double d, double s){
	return ((-0.1875*(a*a*a)*b*(d*d*d*d) + 1.40625*(a*a)*(b*b)*(d*d*d*d) + 0.984375*(a*a)*(d*d)*p - 1.875*a*(b*b*b)*(d*d*d*d) - 5.90625*a*b*(d*d)*p + 0.46875*(b*b*b*b)*(d*d*d*d) + 4.921875*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU636(double a, double b, double p, double d, double s){
	return (d*(0.015625*(a*a*a)*(d*d) - 0.28125*(a*a)*b*(d*d) + 0.703125*a*(b*b)*(d*d) + 0.65625*a*p - 0.3125*(b*b*b)*(d*d) - 1.3125*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU637(double a, double b, double p, double d, double s){
	return ((0.0234375*(a*a)*(d*d) - 0.140625*a*b*(d*d) + 0.1171875*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU638(double a, double b, double p, double d, double s){
	return (d*(0.01171875*a - 0.0234375*b)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU639(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU640(double a, double b, double p, double d, double s){
	return (((a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 11.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1.875*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 12.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 60.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 45.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 3.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 67.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 168.75*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 39.375*(a*a)*(d*d)*(p*p*p*p) - 18.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 150.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 157.5*a*b*(d*d)*(p*p*p*p) + 0.75*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 28.125*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 98.4375*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU641(double a, double b, double p, double d, double s){
	return (d*(-3.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 15.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 11.25*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 2.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 45.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 112.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 26.25*(a*a*a)*(d*d)*(p*p*p) - 27.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 225.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 236.25*(a*a)*b*(d*d)*(p*p*p) + 3.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 112.5*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 393.75*a*(b*b)*(d*d)*(p*p*p) + 118.125*a*(p*p*p*p) - 11.25*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 131.25*(b*b*b)*(d*d)*(p*p*p) - 177.1875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU642(double a, double b, double p, double d, double s){
	return ((3.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 11.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 2.8125*(a*a*a*a)*(d*d*d*d)*(p*p) - 6.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 60.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 67.5*(a*a*a)*b*(d*d*d*d)*(p*p) + 1.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 67.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 253.125*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 78.75*(a*a)*(d*d)*(p*p*p) - 18.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 225.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 315.0*a*b*(d*d)*(p*p*p) + 0.75*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 42.1875*(b*b*b*b)*(d*d*d*d)*(p*p) + 196.875*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU643(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.75*(a*a*a*a)*b*(d*d*d*d)*p + 7.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 37.5*(a*a*a)*(b*b)*(d*d*d*d)*p + 13.125*(a*a*a)*(d*d)*(p*p) - 4.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 75.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 118.125*(a*a)*b*(d*d)*(p*p) + 0.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 37.5*a*(b*b*b*b)*(d*d*d*d)*p + 196.875*a*(b*b)*(d*d)*(p*p) + 78.75*a*(p*p*p) - 3.75*(b*b*b*b*b)*(d*d*d*d)*p - 65.625*(b*b*b)*(d*d)*(p*p) - 118.125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU644(double a, double b, double p, double d, double s){
	return ((0.9375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.46875*(a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 11.25*(a*a*a)*b*(d*d*d*d)*p + 5.625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 42.1875*(a*a)*(b*b)*(d*d*d*d)*p + 19.6875*(a*a)*(d*d)*(p*p) - 1.5*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 37.5*a*(b*b*b)*(d*d*d*d)*p - 78.75*a*b*(d*d)*(p*p) + 0.0625*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 7.03125*(b*b*b*b)*(d*d*d*d)*p + 49.21875*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU645(double a, double b, double p, double d, double s){
	return (d*(-0.1875*(a*a*a*a)*b*(d*d*d*d) + 1.875*(a*a*a)*(b*b)*(d*d*d*d) + 1.3125*(a*a*a)*(d*d)*p - 3.75*(a*a)*(b*b*b)*(d*d*d*d) - 11.8125*(a*a)*b*(d*d)*p + 1.875*a*(b*b*b*b)*(d*d*d*d) + 19.6875*a*(b*b)*(d*d)*p + 11.8125*a*(p*p) - 0.1875*(b*b*b*b*b)*(d*d*d*d) - 6.5625*(b*b*b)*(d*d)*p - 17.71875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU646(double a, double b, double p, double d, double s){
	return ((0.015625*(a*a*a*a)*(d*d*d*d) - 0.375*(a*a*a)*b*(d*d*d*d) + 1.40625*(a*a)*(b*b)*(d*d*d*d) + 1.3125*(a*a)*(d*d)*p - 1.25*a*(b*b*b)*(d*d*d*d) - 5.25*a*b*(d*d)*p + 0.234375*(b*b*b*b)*(d*d*d*d) + 3.28125*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU647(double a, double b, double p, double d, double s){
	return (d*(0.03125*(a*a*a)*(d*d) - 0.28125*(a*a)*b*(d*d) + 0.46875*a*(b*b)*(d*d) + 0.5625*a*p - 0.15625*(b*b*b)*(d*d) - 0.84375*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU648(double a, double b, double p, double d, double s){
	return ((0.0234375*(a*a)*(d*d) - 0.09375*a*b*(d*d) + 0.05859375*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU649(double a, double b, double p, double d, double s){
	return (d*(0.0078125*a - 0.01171875*b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6410(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU650(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 11.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1.875*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 15.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 75.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 56.25*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 5.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 112.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 281.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 65.625*(a*a*a)*(d*d)*(p*p*p*p) - 45.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 375.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 393.75*(a*a)*b*(d*d)*(p*p*p*p) + 3.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 140.625*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 492.1875*a*(b*b)*(d*d)*(p*p*p*p) + 147.65625*a*(p*p*p*p*p) - 11.25*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 131.25*(b*b*b)*(d*d)*(p*p*p*p) - 177.1875*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU651(double a, double b, double p, double d, double s){
	return ((-3.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 15.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 11.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 2.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 56.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 140.625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 32.8125*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 45.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 375.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 393.75*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 7.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 281.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 984.375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 295.3125*(a*a)*(d*d)*(p*p*p*p) - 56.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 656.25*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 885.9375*a*b*(d*d)*(p*p*p*p) + 1.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 98.4375*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 442.96875*(b*b)*(d*d)*(p*p*p*p) + 162.421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU652(double a, double b, double p, double d, double s){
	return (d*(3.75*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 11.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 2.8125*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 7.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 75.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 84.375*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 2.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 112.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 421.875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 131.25*(a*a*a)*(d*d)*(p*p*p) - 45.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 562.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 787.5*(a*a)*b*(d*d)*(p*p*p) + 3.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 210.9375*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 984.375*a*(b*b)*(d*d)*(p*p*p) + 369.140625*a*(p*p*p*p) - 16.875*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 262.5*(b*b*b)*(d*d)*(p*p*p) - 442.96875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU653(double a, double b, double p, double d, double s){
	return ((-2.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 9.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 46.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 16.40625*(a*a*a*a)*(d*d*d*d)*(p*p) - 7.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 125.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 196.875*(a*a*a)*b*(d*d*d*d)*(p*p) + 1.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 93.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 492.1875*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 196.875*(a*a)*(d*d)*(p*p*p) - 18.75*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 328.125*a*(b*b*b)*(d*d*d*d)*(p*p) - 590.625*a*b*(d*d)*(p*p*p) + 0.625*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 49.21875*(b*b*b*b)*(d*d*d*d)*(p*p) + 295.3125*(b*b)*(d*d)*(p*p*p) + 135.3515625*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU654(double a, double b, double p, double d, double s){
	return (d*(0.9375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.46875*(a*a*a*a*a)*(d*d*d*d)*p - 6.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 14.0625*(a*a*a*a)*b*(d*d*d*d)*p + 9.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 70.3125*(a*a*a)*(b*b)*(d*d*d*d)*p + 32.8125*(a*a*a)*(d*d)*(p*p) - 3.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 93.75*(a*a)*(b*b*b)*(d*d*d*d)*p - 196.875*(a*a)*b*(d*d)*(p*p) + 0.3125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 35.15625*a*(b*b*b*b)*(d*d*d*d)*p + 246.09375*a*(b*b)*(d*d)*(p*p) + 123.046875*a*(p*p*p) - 2.8125*(b*b*b*b*b)*(d*d*d*d)*p - 65.625*(b*b*b)*(d*d)*(p*p) - 147.65625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU655(double a, double b, double p, double d, double s){
	return ((-0.1875*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.34375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.640625*(a*a*a*a)*(d*d*d*d)*p - 6.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 19.6875*(a*a*a)*b*(d*d*d*d)*p + 4.6875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 49.21875*(a*a)*(b*b)*(d*d*d*d)*p + 29.53125*(a*a)*(d*d)*(p*p) - 0.9375*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 32.8125*a*(b*b*b)*(d*d*d*d)*p - 88.59375*a*b*(d*d)*(p*p) + 0.03125*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 4.921875*(b*b*b*b)*(d*d*d*d)*p + 44.296875*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU656(double a, double b, double p, double d, double s){
	return (d*(0.015625*(a*a*a*a*a)*(d*d*d*d) - 0.46875*(a*a*a*a)*b*(d*d*d*d) + 2.34375*(a*a*a)*(b*b)*(d*d*d*d) + 2.1875*(a*a*a)*(d*d)*p - 3.125*(a*a)*(b*b*b)*(d*d*d*d) - 13.125*(a*a)*b*(d*d)*p + 1.171875*a*(b*b*b*b)*(d*d*d*d) + 16.40625*a*(b*b)*(d*d)*p + 12.3046875*a*(p*p) - 0.09375*(b*b*b*b*b)*(d*d*d*d) - 4.375*(b*b*b)*(d*d)*p - 14.765625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU657(double a, double b, double p, double d, double s){
	return ((0.0390625*(a*a*a*a)*(d*d*d*d) - 0.46875*(a*a*a)*b*(d*d*d*d) + 1.171875*(a*a)*(b*b)*(d*d*d*d) + 1.40625*(a*a)*(d*d)*p - 0.78125*a*(b*b*b)*(d*d*d*d) - 4.21875*a*b*(d*d)*p + 0.1171875*(b*b*b*b)*(d*d*d*d) + 2.109375*(b*b)*(d*d)*p + 1.93359375*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU658(double a, double b, double p, double d, double s){
	return (d*(0.0390625*(a*a*a)*(d*d) - 0.234375*(a*a)*b*(d*d) + 0.29296875*a*(b*b)*(d*d) + 0.439453125*a*p - 0.078125*(b*b*b)*(d*d) - 0.52734375*b*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU659(double a, double b, double p, double d, double s){
	return ((0.01953125*(a*a)*(d*d) - 0.05859375*a*b*(d*d) + 0.029296875*(b*b)*(d*d) + 0.0537109375*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6510(double a, double b, double p, double d, double s){
	return (d*(0.0048828125*a - 0.005859375*b)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6511(double a, double b, double p, double d, double s){
	return (0.00048828125/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU660(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 11.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 18.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 90.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 67.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 7.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 168.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 421.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 98.4375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 90.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 750.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 787.5*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 11.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 421.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1476.5625*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 442.96875*(a*a)*(d*d)*(p*p*p*p*p) - 67.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 787.5*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 1063.125*a*b*(d*d)*(p*p*p*p*p) + 1.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 98.4375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 442.96875*(b*b)*(d*d)*(p*p*p*p*p) + 162.421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU661(double a, double b, double p, double d, double s){
	return (d*(-3.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 15.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 11.25*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 3.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 67.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 168.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 39.375*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 67.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 562.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 590.625*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 15.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 562.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1968.75*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 590.625*(a*a*a)*(d*d)*(p*p*p*p) - 168.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1968.75*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 2657.8125*(a*a)*b*(d*d)*(p*p*p*p) + 11.25*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 590.625*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 2657.8125*a*(b*b)*(d*d)*(p*p*p*p) + 974.53125*a*(p*p*p*p*p) - 39.375*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 590.625*(b*b*b)*(d*d)*(p*p*p*p) - 974.53125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU662(double a, double b, double p, double d, double s){
	return ((3.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 11.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 2.8125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 9.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 90.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 101.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 3.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 168.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 632.8125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 196.875*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 90.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1125.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 1575.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 11.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 632.8125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2953.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 1107.421875*(a*a)*(d*d)*(p*p*p*p) - 101.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1575.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 2657.8125*a*b*(d*d)*(p*p*p*p) + 2.8125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 196.875*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 1107.421875*(b*b)*(d*d)*(p*p*p*p) + 487.265625*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU663(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 11.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 56.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 19.6875*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 11.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 187.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 295.3125*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 2.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 187.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 984.375*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 393.75*(a*a*a)*(d*d)*(p*p*p) - 56.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 984.375*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 1771.875*(a*a)*b*(d*d)*(p*p*p) + 3.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 295.3125*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 1771.875*a*(b*b)*(d*d)*(p*p*p) + 812.109375*a*(p*p*p*p) - 19.6875*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 393.75*(b*b*b)*(d*d)*(p*p*p) - 812.109375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU664(double a, double b, double p, double d, double s){
	return ((0.9375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.46875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 16.875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 14.0625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 105.46875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 49.21875*(a*a*a*a)*(d*d*d*d)*(p*p) - 7.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 187.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 393.75*(a*a*a)*b*(d*d*d*d)*(p*p) + 0.9375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 105.46875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 738.28125*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 369.140625*(a*a)*(d*d)*(p*p*p) - 16.875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 393.75*a*(b*b*b)*(d*d*d*d)*(p*p) - 885.9375*a*b*(d*d)*(p*p*p) + 0.46875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 49.21875*(b*b*b*b)*(d*d*d*d)*(p*p) + 369.140625*(b*b)*(d*d)*(p*p*p) + 203.02734375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU665(double a, double b, double p, double d, double s){
	return (d*(-0.1875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.8125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.96875*(a*a*a*a*a)*(d*d*d*d)*p - 9.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 29.53125*(a*a*a*a)*b*(d*d*d*d)*p + 9.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 98.4375*(a*a*a)*(b*b)*(d*d*d*d)*p + 59.0625*(a*a*a)*(d*d)*(p*p) - 2.8125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 98.4375*(a*a)*(b*b*b)*(d*d*d*d)*p - 265.78125*(a*a)*b*(d*d)*(p*p) + 0.1875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 29.53125*a*(b*b*b*b)*(d*d*d*d)*p + 265.78125*a*(b*b)*(d*d)*(p*p) + 162.421875*a*(p*p*p) - 1.96875*(b*b*b*b*b)*(d*d*d*d)*p - 59.0625*(b*b*b)*(d*d)*(p*p) - 162.421875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU666(double a, double b, double p, double d, double s){
	return ((0.015625*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.5625*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.515625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.28125*(a*a*a*a)*(d*d*d*d)*p - 6.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 26.25*(a*a*a)*b*(d*d*d*d)*p + 3.515625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 49.21875*(a*a)*(b*b)*(d*d*d*d)*p + 36.9140625*(a*a)*(d*d)*(p*p) - 0.5625*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 26.25*a*(b*b*b)*(d*d*d*d)*p - 88.59375*a*b*(d*d)*(p*p) + 0.015625*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 3.28125*(b*b*b*b)*(d*d*d*d)*p + 36.9140625*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU667(double a, double b, double p, double d, double s){
	return (d*(0.046875*(a*a*a*a*a)*(d*d*d*d) - 0.703125*(a*a*a*a)*b*(d*d*d*d) + 2.34375*(a*a*a)*(b*b)*(d*d*d*d) + 2.8125*(a*a*a)*(d*d)*p - 2.34375*(a*a)*(b*b*b)*(d*d*d*d) - 12.65625*(a*a)*b*(d*d)*p + 0.703125*a*(b*b*b*b)*(d*d*d*d) + 12.65625*a*(b*b)*(d*d)*p + 11.6015625*a*(p*p) - 0.046875*(b*b*b*b*b)*(d*d*d*d) - 2.8125*(b*b*b)*(d*d)*p - 11.6015625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU668(double a, double b, double p, double d, double s){
	return ((0.05859375*(a*a*a*a)*(d*d*d*d) - 0.46875*(a*a*a)*b*(d*d*d*d) + 0.87890625*(a*a)*(b*b)*(d*d*d*d) + 1.318359375*(a*a)*(d*d)*p - 0.46875*a*(b*b*b)*(d*d*d*d) - 3.1640625*a*b*(d*d)*p + 0.05859375*(b*b*b*b)*(d*d*d*d) + 1.318359375*(b*b)*(d*d)*p + 1.4501953125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU669(double a, double b, double p, double d, double s){
	return (d*(0.0390625*(a*a*a)*(d*d) - 0.17578125*(a*a)*b*(d*d) + 0.17578125*a*(b*b)*(d*d) + 0.322265625*a*p - 0.0390625*(b*b*b)*(d*d) - 0.322265625*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6610(double a, double b, double p, double d, double s){
	return ((0.0146484375*(a*a)*(d*d) - 0.03515625*a*b*(d*d) + 0.0146484375*(b*b)*(d*d) + 0.0322265625*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6611(double a, double b, double p, double d, double s){
	return (0.0029296875*d*(a - b)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6612(double a, double b, double p, double d, double s){
	return (0.000244140625/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU670(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 11.25*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1.875*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 21.0 *(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 78.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 10.5*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 236.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 590.625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 137.8125*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 157.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1312.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1378.125*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 26.25*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 984.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 3445.3125*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 1033.59375*(a*a*a)*(d*d)*(p*p*p*p*p) - 236.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 2756.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 3720.9375*(a*a)*b*(d*d)*(p*p*p*p*p) + 13.125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 689.0625*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 3100.78125*a*(b*b)*(d*d)*(p*p*p*p*p) + 1136.953125*a*(p*p*p*p*p*p) - 39.375*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 590.625*(b*b*b)*(d*d)*(p*p*p*p*p) - 974.53125*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU671(double a, double b, double p, double d, double s){
	return ((-3.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 15.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 11.25*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 3.5*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 78.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 196.875*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 45.9375*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 94.5*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 787.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 826.875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 26.25*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 984.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 3445.3125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 1033.59375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 393.75*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 4593.75*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 6201.5625*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 39.375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 2067.1875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 9302.34375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 3410.859375*(a*a)*(d*d)*(p*p*p*p*p) - 275.625*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 4134.375*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 6821.71875*a*b*(d*d)*(p*p*p*p*p) + 6.5625*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 442.96875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 2436.328125*(b*b)*(d*d)*(p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU672(double a, double b, double p, double d, double s){
	return (d*(3.75*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 11.25*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 2.8125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 10.5*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 105.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 118.125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 5.25*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 236.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 885.9375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 275.625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 157.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1968.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 2756.25*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 26.25*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1476.5625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 6890.625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 2583.984375*(a*a*a)*(d*d)*(p*p*p*p) - 354.375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 5512.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 9302.34375*(a*a)*b*(d*d)*(p*p*p*p) + 19.6875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1378.125*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 7751.953125*a*(b*b)*(d*d)*(p*p*p*p) + 3410.859375*a*(p*p*p*p*p) - 78.75*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 1476.5625*(b*b*b)*(d*d)*(p*p*p*p) - 2923.59375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU673(double a, double b, double p, double d, double s){
	return ((-2.5*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 13.125*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 65.625*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 22.96875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 15.75*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 262.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 413.4375*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 4.375*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 328.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1722.65625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 689.0625*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 131.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 2296.875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 4134.375*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 13.125*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1033.59375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 6201.5625*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 2842.3828125*(a*a)*(d*d)*(p*p*p*p) - 137.8125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 2756.25*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 5684.765625*a*b*(d*d)*(p*p*p*p) + 3.28125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 295.3125*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 2030.2734375*(b*b)*(d*d)*(p*p*p*p) + 1055.7421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU674(double a, double b, double p, double d, double s){
	return (d*(0.9375*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.46875*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 8.75*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 19.6875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 19.6875*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 147.65625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 68.90625*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 13.125*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 328.125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 689.0625*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 2.1875*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 246.09375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1722.65625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 861.328125*(a*a*a)*(d*d)*(p*p*p) - 59.0625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 1378.125*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 3100.78125*(a*a)*b*(d*d)*(p*p*p) + 3.28125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 344.53125*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 2583.984375*a*(b*b)*(d*d)*(p*p*p) + 1421.19140625*a*(p*p*p*p) - 19.6875*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 492.1875*(b*b*b)*(d*d)*(p*p*p) - 1218.1640625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU675(double a, double b, double p, double d, double s){
	return ((-0.1875*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 3.28125*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 2.296875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 13.125*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 41.34375*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 16.40625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 172.265625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 103.359375*(a*a*a*a)*(d*d*d*d)*(p*p) - 6.5625*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 229.6875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 620.15625*(a*a*a)*b*(d*d*d*d)*(p*p) + 0.65625*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 103.359375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 930.234375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 568.4765625*(a*a)*(d*d)*(p*p*p) - 13.78125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 413.4375*a*(b*b*b)*(d*d*d*d)*(p*p) - 1136.953125*a*b*(d*d)*(p*p*p) + 0.328125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 44.296875*(b*b*b*b)*(d*d*d*d)*(p*p) + 406.0546875*(b*b)*(d*d)*(p*p*p) + 263.935546875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU676(double a, double b, double p, double d, double s){
	return (d*(0.015625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.65625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.921875*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 4.59375*(a*a*a*a*a)*(d*d*d*d)*p - 10.9375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 45.9375*(a*a*a*a)*b*(d*d*d*d)*p + 8.203125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 114.84375*(a*a*a)*(b*b)*(d*d*d*d)*p + 86.1328125*(a*a*a)*(d*d)*(p*p) - 1.96875*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 91.875*(a*a)*(b*b*b)*(d*d*d*d)*p - 310.078125*(a*a)*b*(d*d)*(p*p) + 0.109375*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 22.96875*a*(b*b*b*b)*(d*d*d*d)*p + 258.3984375*a*(b*b)*(d*d)*(p*p) + 189.4921875*a*(p*p*p) - 1.3125*(b*b*b*b*b)*(d*d*d*d)*p - 49.21875*(b*b*b)*(d*d)*(p*p) - 162.421875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU677(double a, double b, double p, double d, double s){
	return ((0.0546875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.984375*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.1015625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 4.921875*(a*a*a*a)*(d*d*d*d)*p - 5.46875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 29.53125*(a*a*a)*b*(d*d*d*d)*p + 2.4609375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 44.296875*(a*a)*(b*b)*(d*d*d*d)*p + 40.60546875*(a*a)*(d*d)*(p*p) - 0.328125*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 19.6875*a*(b*b*b)*(d*d*d*d)*p - 81.2109375*a*b*(d*d)*(p*p) + 0.0078125*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 2.109375*(b*b*b*b)*(d*d*d*d)*p + 29.00390625*(b*b)*(d*d)*(p*p) + 25.13671875*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU678(double a, double b, double p, double d, double s){
	return (d*(0.08203125*(a*a*a*a*a)*(d*d*d*d) - 0.8203125*(a*a*a*a)*b*(d*d*d*d) + 2.05078125*(a*a*a)*(b*b)*(d*d*d*d) + 3.076171875*(a*a*a)*(d*d)*p - 1.640625*(a*a)*(b*b*b)*(d*d*d*d) - 11.07421875*(a*a)*b*(d*d)*p + 0.41015625*a*(b*b*b*b)*(d*d*d*d) + 9.228515625*a*(b*b)*(d*d)*p + 10.1513671875*a*(p*p) - 0.0234375*(b*b*b*b*b)*(d*d*d*d) - 1.7578125*(b*b*b)*(d*d)*p - 8.701171875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU679(double a, double b, double p, double d, double s){
	return ((0.068359375*(a*a*a*a)*(d*d*d*d) - 0.41015625*(a*a*a)*b*(d*d*d*d) + 0.615234375*(a*a)*(b*b)*(d*d*d*d) + 1.1279296875*(a*a)*(d*d)*p - 0.2734375*a*(b*b*b)*(d*d*d*d) - 2.255859375*a*b*(d*d)*p + 0.029296875*(b*b*b*b)*(d*d*d*d) + 0.8056640625*(b*b)*(d*d)*p + 1.04736328125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6710(double a, double b, double p, double d, double s){
	return (d*(0.0341796875*(a*a*a)*(d*d) - 0.123046875*(a*a)*b*(d*d) + 0.1025390625*a*(b*b)*(d*d) + 0.2255859375*a*p - 0.01953125*(b*b*b)*(d*d) - 0.193359375*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6711(double a, double b, double p, double d, double s){
	return ((0.01025390625*(a*a)*(d*d) - 0.0205078125*a*b*(d*d) + 0.00732421875*(b*b)*(d*d) + 0.01904296875*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6712(double a, double b, double p, double d, double s){
	return (d*(0.001708984375*a - 0.00146484375*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6713(double a, double b, double p, double d, double s){
	return (0.0001220703125/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU680(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 11.25*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 1.875*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p*p) - 24.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 120.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 90.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 14.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 315.0*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 787.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 183.75*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 252.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 2100.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 2205.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 52.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 1968.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 6890.625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 2067.1875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 630.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 7350.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 9922.5*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 52.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 2756.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 12403.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 4547.8125*(a*a)*(d*d)*(p*p*p*p*p*p) - 315.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 4725.0*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 7796.25*a*b*(d*d)*(p*p*p*p*p*p) + 6.5625*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 442.96875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 2436.328125*(b*b)*(d*d)*(p*p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU681(double a, double b, double p, double d, double s){
	return (d*(-3.0*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 15.0*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 11.25*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 4.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 90.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 225.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 126.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 1050.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1102.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 42.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1575.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 5512.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 1653.75*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 787.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 9187.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 12403.125*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 105.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 5512.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 24806.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 9095.625*(a*a*a)*(d*d)*(p*p*p*p*p) - 1102.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 16537.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 27286.875*(a*a)*b*(d*d)*(p*p*p*p*p) + 52.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 3543.75*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 19490.625*a*(b*b)*(d*d)*(p*p*p*p*p) + 8445.9375*a*(p*p*p*p*p*p) - 177.1875*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 3248.4375*(b*b*b)*(d*d)*(p*p*p*p*p) - 6334.453125*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU682(double a, double b, double p, double d, double s){
	return ((3.75*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 11.25*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 2.8125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p) - 12.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 120.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 135.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 7.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 315.0*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1181.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 367.5*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 252.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 3150.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 4410.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 52.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 2953.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 13781.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 5167.96875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 945.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 14700.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 24806.25*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 78.75*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 5512.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 31007.8125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 13643.4375*(a*a)*(d*d)*(p*p*p*p*p) - 630.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 11812.5*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 23388.75*a*b*(d*d)*(p*p*p*p*p) + 13.125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1107.421875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 7308.984375*(b*b)*(d*d)*(p*p*p*p*p) + 3695.09765625*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU683(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 15.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 75.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 26.25*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 21.0 *(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 350.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 551.25*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 7.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 525.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2756.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1102.5*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 262.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 4593.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 8268.75*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 35.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2756.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 16537.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 7579.6875*(a*a*a)*(d*d)*(p*p*p*p) - 551.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 11025.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 22739.0625*(a*a)*b*(d*d)*(p*p*p*p) + 26.25*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2362.5*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 16242.1875*a*(b*b)*(d*d)*(p*p*p*p) + 8445.9375*a*(p*p*p*p*p) - 118.125*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 2707.03125*(b*b*b)*(d*d)*(p*p*p*p) - 6334.453125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU684(double a, double b, double p, double d, double s){
	return ((0.9375*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.46875*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 10.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 22.5*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 26.25*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 196.875*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 91.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 21.0 *(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 525.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 1102.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 4.375*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 492.1875*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 3445.3125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1722.65625*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 157.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 3675.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 8268.75*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 13.125*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1378.125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 10335.9375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 5684.765625*(a*a)*(d*d)*(p*p*p*p) - 157.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 3937.5*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 9745.3125*a*b*(d*d)*(p*p*p*p) + 3.28125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 369.140625*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 3045.41015625*(b*b)*(d*d)*(p*p*p*p) + 1847.548828125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU685(double a, double b, double p, double d, double s){
	return (d*(-0.1875*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 3.75*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 2.625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 17.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 55.125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 26.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 275.625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 165.375*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 13.125*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 459.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 1240.3125*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 1.75*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 275.625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 2480.625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 1515.9375*(a*a*a)*(d*d)*(p*p*p) - 55.125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 1653.75*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 4547.8125*(a*a)*b*(d*d)*(p*p*p) + 2.625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 354.375*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 3248.4375*a*(b*b)*(d*d)*(p*p*p) + 2111.484375*a*(p*p*p*p) - 17.71875*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 541.40625*(b*b*b)*(d*d)*(p*p*p) - 1583.61328125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU686(double a, double b, double p, double d, double s){
	return ((0.015625*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 0.75*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 6.5625*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 6.125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 17.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 73.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 16.40625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 229.6875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 172.265625*(a*a*a*a)*(d*d*d*d)*(p*p) - 5.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 245.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 826.875*(a*a*a)*b*(d*d*d*d)*(p*p) + 0.4375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 91.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1033.59375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 757.96875*(a*a)*(d*d)*(p*p*p) - 10.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 393.75*a*(b*b*b)*(d*d*d*d)*(p*p) - 1299.375*a*b*(d*d)*(p*p*p) + 0.21875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 36.9140625*(b*b*b*b)*(d*d*d*d)*(p*p) + 406.0546875*(b*b)*(d*d)*(p*p*p) + 307.9248046875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU687(double a, double b, double p, double d, double s){
	return (d*(0.0625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 1.3125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 6.5625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 7.875*(a*a*a*a*a)*(d*d*d*d)*p - 10.9375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 59.0625*(a*a*a*a)*b*(d*d*d*d)*p + 6.5625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 118.125*(a*a*a)*(b*b)*(d*d*d*d)*p + 108.28125*(a*a*a)*(d*d)*(p*p) - 1.3125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 78.75*(a*a)*(b*b*b)*(d*d*d*d)*p - 324.84375*(a*a)*b*(d*d)*(p*p) + 0.0625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 16.875*a*(b*b*b*b)*(d*d*d*d)*p + 232.03125*a*(b*b)*(d*d)*(p*p) + 201.09375*a*(p*p*p) - 0.84375*(b*b*b*b*b)*(d*d*d*d)*p - 38.671875*(b*b*b)*(d*d)*(p*p) - 150.8203125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU688(double a, double b, double p, double d, double s){
	return ((0.109375*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 1.3125*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.1015625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 6.15234375*(a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 29.53125*(a*a*a)*b*(d*d*d*d)*p + 1.640625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 36.9140625*(a*a)*(b*b)*(d*d*d*d)*p + 40.60546875*(a*a)*(d*d)*(p*p) - 0.1875*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 14.0625*a*(b*b*b)*(d*d*d*d)*p - 69.609375*a*b*(d*d)*(p*p) + 0.00390625*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 1.318359375*(b*b*b*b)*(d*d*d*d)*p + 21.7529296875*(b*b)*(d*d)*(p*p) + 21.99462890625*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU689(double a, double b, double p, double d, double s){
	return (d*(0.109375*(a*a*a*a*a)*(d*d*d*d) - 0.8203125*(a*a*a*a)*b*(d*d*d*d) + 1.640625*(a*a*a)*(b*b)*(d*d*d*d) + 3.0078125*(a*a*a)*(d*d)*p - 1.09375*(a*a)*(b*b*b)*(d*d*d*d) - 9.0234375*(a*a)*b*(d*d)*p + 0.234375*a*(b*b*b*b)*(d*d*d*d) + 6.4453125*a*(b*b)*(d*d)*p + 8.37890625*a*(p*p) - 0.01171875*(b*b*b*b*b)*(d*d*d*d) - 1.07421875*(b*b*b)*(d*d)*p - 6.2841796875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6810(double a, double b, double p, double d, double s){
	return ((0.068359375*(a*a*a*a)*(d*d*d*d) - 0.328125*(a*a*a)*b*(d*d*d*d) + 0.41015625*(a*a)*(b*b)*(d*d*d*d) + 0.90234375*(a*a)*(d*d)*p - 0.15625*a*(b*b*b)*(d*d*d*d) - 1.546875*a*b*(d*d)*p + 0.0146484375*(b*b*b*b)*(d*d*d*d) + 0.4833984375*(b*b)*(d*d)*p + 0.733154296875*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6811(double a, double b, double p, double d, double s){
	return (d*(0.02734375*(a*a*a)*(d*d) - 0.08203125*(a*a)*b*(d*d) + 0.05859375*a*(b*b)*(d*d) + 0.15234375*a*p - 0.009765625*(b*b*b)*(d*d) - 0.1142578125*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6812(double a, double b, double p, double d, double s){
	return ((0.0068359375*(a*a)*(d*d) - 0.01171875*a*b*(d*d) + 0.003662109375*(b*b)*(d*d) + 0.0111083984375*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6813(double a, double b, double p, double d, double s){
	return (d*(0.0009765625*a - 0.000732421875*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU6814(double a, double b, double p, double d, double s){
	return (6.103515625e-5/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU700(double a, double b, double p, double d, double s){
	return (b*d*(-(b*b*b*b*b*b)*(d*d*d*d*d*d) - 10.5*(b*b*b*b)*(d*d*d*d)*p - 26.25*(b*b)*(d*d)*(p*p) - 13.125*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU701(double a, double b, double p, double d, double s){
	return ((3.5*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 26.25*(b*b*b*b)*(d*d*d*d)*p + 39.375*(b*b)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU702(double a, double b, double p, double d, double s){
	return (b*d*(-5.25*(b*b*b*b)*(d*d*d*d) - 26.25*(b*b)*(d*d)*p - 19.6875*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU703(double a, double b, double p, double d, double s){
	return ((4.375*(b*b*b*b)*(d*d*d*d) + 13.125*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU704(double a, double b, double p, double d, double s){
	return (b*d*(-2.1875*(b*b)*(d*d) - 3.28125*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU705(double a, double b, double p, double d, double s){
	return ((0.65625*(b*b)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU706(double a, double b, double p, double d, double s){
	return (-0.109375*b*d/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU707(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU710(double a, double b, double p, double d, double s){
	return ((-a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 10.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 26.25*a*(b*b*b)*(d*d*d*d)*(p*p) - 13.125*a*b*(d*d)*(p*p*p) + 3.5*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 26.25*(b*b*b*b)*(d*d*d*d)*(p*p) + 39.375*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU711(double a, double b, double p, double d, double s){
	return (d*(3.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 26.25*a*(b*b*b*b)*(d*d*d*d)*p + 39.375*a*(b*b)*(d*d)*(p*p) + 6.5625*a*(p*p*p) - 0.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 15.75*(b*b*b*b*b)*(d*d*d*d)*p - 65.625*(b*b*b)*(d*d)*(p*p) - 45.9375*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU712(double a, double b, double p, double d, double s){
	return ((-5.25*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 26.25*a*(b*b*b)*(d*d*d*d)*p - 19.6875*a*b*(d*d)*(p*p) + 1.75*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 26.25*(b*b*b*b)*(d*d*d*d)*p + 59.0625*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU713(double a, double b, double p, double d, double s){
	return (d*(4.375*a*(b*b*b*b)*(d*d*d*d) + 13.125*a*(b*b)*(d*d)*p + 3.28125*a*(p*p) - 2.625*(b*b*b*b*b)*(d*d*d*d) - 21.875*(b*b*b)*(d*d)*p - 22.96875*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU714(double a, double b, double p, double d, double s){
	return ((-2.1875*a*(b*b*b)*(d*d*d*d) - 3.28125*a*b*(d*d)*p + 2.1875*(b*b*b*b)*(d*d*d*d) + 9.84375*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU715(double a, double b, double p, double d, double s){
	return (d*(0.65625*a*(b*b)*(d*d) + 0.328125*a*p - 1.09375*(b*b*b)*(d*d) - 2.296875*b*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU716(double a, double b, double p, double d, double s){
	return ((-0.109375*a*b*(d*d) + 0.328125*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU717(double a, double b, double p, double d, double s){
	return (d*(0.0078125*a - 0.0546875*b)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU718(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU720(double a, double b, double p, double d, double s){
	return (d*(-(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 10.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 26.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 13.125*(a*a)*b*(d*d)*(p*p*p) + 7.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 52.5*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 78.75*a*(b*b)*(d*d)*(p*p*p) + 13.125*a*(p*p*p*p) - 0.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 15.75*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 65.625*(b*b*b)*(d*d)*(p*p*p) - 45.9375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU721(double a, double b, double p, double d, double s){
	return ((3.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 26.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 39.375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 6.5625*(a*a)*(d*d)*(p*p*p) - a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 31.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 131.25*a*(b*b*b)*(d*d*d*d)*(p*p) - 91.875*a*b*(d*d)*(p*p*p) + 5.25*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 65.625*(b*b*b*b)*(d*d*d*d)*(p*p) + 137.8125*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU722(double a, double b, double p, double d, double s){
	return (d*(-5.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 26.25*(a*a)*(b*b*b)*(d*d*d*d)*p - 19.6875*(a*a)*b*(d*d)*(p*p) + 3.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 52.5*a*(b*b*b*b)*(d*d*d*d)*p + 118.125*a*(b*b)*(d*d)*(p*p) + 26.25*a*(p*p*p) - 0.25*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 15.75*(b*b*b*b*b)*(d*d*d*d)*p - 98.4375*(b*b*b)*(d*d)*(p*p) - 91.875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU723(double a, double b, double p, double d, double s){
	return ((4.375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 13.125*(a*a)*(b*b)*(d*d*d*d)*p + 3.28125*(a*a)*(d*d)*(p*p) - 5.25*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 43.75*a*(b*b*b)*(d*d*d*d)*p - 45.9375*a*b*(d*d)*(p*p) + 0.875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 21.875*(b*b*b*b)*(d*d*d*d)*p + 68.90625*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU724(double a, double b, double p, double d, double s){
	return (d*(-2.1875*(a*a)*(b*b*b)*(d*d*d*d) - 3.28125*(a*a)*b*(d*d)*p + 4.375*a*(b*b*b*b)*(d*d*d*d) + 19.6875*a*(b*b)*(d*d)*p + 6.5625*a*(p*p) - 1.3125*(b*b*b*b*b)*(d*d*d*d) - 16.40625*(b*b*b)*(d*d)*p - 22.96875*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU725(double a, double b, double p, double d, double s){
	return ((0.65625*(a*a)*(b*b)*(d*d*d*d) + 0.328125*(a*a)*(d*d)*p - 2.1875*a*(b*b*b)*(d*d*d*d) - 4.59375*a*b*(d*d)*p + 1.09375*(b*b*b*b)*(d*d*d*d) + 6.890625*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU726(double a, double b, double p, double d, double s){
	return (d*(-0.109375*(a*a)*b*(d*d) + 0.65625*a*(b*b)*(d*d) + 0.4375*a*p - 0.546875*(b*b*b)*(d*d) - 1.53125*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU727(double a, double b, double p, double d, double s){
	return ((0.0078125*(a*a)*(d*d) - 0.109375*a*b*(d*d) + 0.1640625*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU728(double a, double b, double p, double d, double s){
	return (d*(0.0078125*a - 0.02734375*b)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU729(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU730(double a, double b, double p, double d, double s){
	return ((-(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 10.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 13.125*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 10.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 78.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 118.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 19.6875*(a*a)*(d*d)*(p*p*p*p) - 1.5*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 47.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 196.875*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 137.8125*a*b*(d*d)*(p*p*p*p) + 5.25*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 65.625*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 137.8125*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU731(double a, double b, double p, double d, double s){
	return (d*(3.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 26.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 39.375*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 6.5625*(a*a*a)*(d*d)*(p*p*p) - 1.5*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 47.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 196.875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 137.8125*(a*a)*b*(d*d)*(p*p*p) + 15.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 196.875*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 413.4375*a*(b*b)*(d*d)*(p*p*p) + 88.59375*a*(p*p*p*p) - 0.75*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 39.375*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 229.6875*(b*b*b)*(d*d)*(p*p*p) - 206.71875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU732(double a, double b, double p, double d, double s){
	return ((-5.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 26.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 19.6875*(a*a*a)*b*(d*d*d*d)*(p*p) + 5.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 78.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 177.1875*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 39.375*(a*a)*(d*d)*(p*p*p) - 0.75*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 47.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 295.3125*a*(b*b*b)*(d*d*d*d)*(p*p) - 275.625*a*b*(d*d)*(p*p*p) + 5.25*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 98.4375*(b*b*b*b)*(d*d*d*d)*(p*p) + 275.625*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU733(double a, double b, double p, double d, double s){
	return (d*(4.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 13.125*(a*a*a)*(b*b)*(d*d*d*d)*p + 3.28125*(a*a*a)*(d*d)*(p*p) - 7.875*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 65.625*(a*a)*(b*b*b)*(d*d*d*d)*p - 68.90625*(a*a)*b*(d*d)*(p*p) + 2.625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 65.625*a*(b*b*b*b)*(d*d*d*d)*p + 206.71875*a*(b*b)*(d*d)*(p*p) + 59.0625*a*(p*p*p) - 0.125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 13.125*(b*b*b*b*b)*(d*d*d*d)*p - 114.84375*(b*b*b)*(d*d)*(p*p) - 137.8125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU734(double a, double b, double p, double d, double s){
	return ((-2.1875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.28125*(a*a*a)*b*(d*d*d*d)*p + 6.5625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 29.53125*(a*a)*(b*b)*(d*d*d*d)*p + 9.84375*(a*a)*(d*d)*(p*p) - 3.9375*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 49.21875*a*(b*b*b)*(d*d*d*d)*p - 68.90625*a*b*(d*d)*(p*p) + 0.4375*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 16.40625*(b*b*b*b)*(d*d*d*d)*p + 68.90625*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU735(double a, double b, double p, double d, double s){
	return (d*(0.65625*(a*a*a)*(b*b)*(d*d*d*d) + 0.328125*(a*a*a)*(d*d)*p - 3.28125*(a*a)*(b*b*b)*(d*d*d*d) - 6.890625*(a*a)*b*(d*d)*p + 3.28125*a*(b*b*b*b)*(d*d*d*d) + 20.671875*a*(b*b)*(d*d)*p + 8.859375*a*(p*p) - 0.65625*(b*b*b*b*b)*(d*d*d*d) - 11.484375*(b*b*b)*(d*d)*p - 20.671875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU736(double a, double b, double p, double d, double s){
	return ((-0.109375*(a*a*a)*b*(d*d*d*d) + 0.984375*(a*a)*(b*b)*(d*d*d*d) + 0.65625*(a*a)*(d*d)*p - 1.640625*a*(b*b*b)*(d*d*d*d) - 4.59375*a*b*(d*d)*p + 0.546875*(b*b*b*b)*(d*d*d*d) + 4.59375*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU737(double a, double b, double p, double d, double s){
	return (d*(0.0078125*(a*a*a)*(d*d) - 0.1640625*(a*a)*b*(d*d) + 0.4921875*a*(b*b)*(d*d) + 0.421875*a*p - 0.2734375*(b*b*b)*(d*d) - 0.984375*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU738(double a, double b, double p, double d, double s){
	return ((0.01171875*(a*a)*(d*d) - 0.08203125*a*b*(d*d) + 0.08203125*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU739(double a, double b, double p, double d, double s){
	return (d*(0.005859375*a - 0.013671875*b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7310(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU740(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 10.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 13.125*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 14.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 105.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 157.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 26.25*(a*a*a)*(d*d)*(p*p*p*p) - 3.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 94.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 393.75*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 275.625*(a*a)*b*(d*d)*(p*p*p*p) + 21.0 *a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 262.5*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 551.25*a*(b*b)*(d*d)*(p*p*p*p) + 118.125*a*(p*p*p*p*p) - 0.75*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 39.375*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 229.6875*(b*b*b)*(d*d)*(p*p*p*p) - 206.71875*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU741(double a, double b, double p, double d, double s){
	return ((3.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 26.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 39.375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 6.5625*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 2.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 63.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 262.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 183.75*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 31.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 393.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 826.875*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 177.1875*(a*a)*(d*d)*(p*p*p*p) - 3.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 157.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 918.75*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 826.875*a*b*(d*d)*(p*p*p*p) + 13.125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 229.6875*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 620.15625*(b*b)*(d*d)*(p*p*p*p) + 162.421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU742(double a, double b, double p, double d, double s){
	return (d*(-5.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 26.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 19.6875*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 7.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 105.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 236.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 52.5*(a*a*a)*(d*d)*(p*p*p) - 1.5*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 94.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 590.625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 551.25*(a*a)*b*(d*d)*(p*p*p) + 21.0 *a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 393.75*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 1102.5*a*(b*b)*(d*d)*(p*p*p) + 295.3125*a*(p*p*p*p) - 0.75*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 59.0625*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 459.375*(b*b*b)*(d*d)*(p*p*p) - 516.796875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU743(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 3.28125*(a*a*a*a)*(d*d*d*d)*(p*p) - 10.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 87.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 91.875*(a*a*a)*b*(d*d*d*d)*(p*p) + 5.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 131.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 413.4375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 118.125*(a*a)*(d*d)*(p*p*p) - 0.5*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 52.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 459.375*a*(b*b*b)*(d*d*d*d)*(p*p) - 551.25*a*b*(d*d)*(p*p*p) + 4.375*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 114.84375*(b*b*b*b)*(d*d*d*d)*(p*p) + 413.4375*(b*b)*(d*d)*(p*p*p) + 135.3515625*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU744(double a, double b, double p, double d, double s){
	return (d*(-2.1875*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.28125*(a*a*a*a)*b*(d*d*d*d)*p + 8.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 39.375*(a*a*a)*(b*b)*(d*d*d*d)*p + 13.125*(a*a*a)*(d*d)*(p*p) - 7.875*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 98.4375*(a*a)*(b*b*b)*(d*d*d*d)*p - 137.8125*(a*a)*b*(d*d)*(p*p) + 1.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 65.625*a*(b*b*b*b)*(d*d*d*d)*p + 275.625*a*(b*b)*(d*d)*(p*p) + 98.4375*a*(p*p*p) - 0.0625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 9.84375*(b*b*b*b*b)*(d*d*d*d)*p - 114.84375*(b*b*b)*(d*d)*(p*p) - 172.265625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU745(double a, double b, double p, double d, double s){
	return ((0.65625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.328125*(a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 9.1875*(a*a*a)*b*(d*d*d*d)*p + 6.5625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 41.34375*(a*a)*(b*b)*(d*d*d*d)*p + 17.71875*(a*a)*(d*d)*(p*p) - 2.625*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 45.9375*a*(b*b*b)*(d*d*d*d)*p - 82.6875*a*b*(d*d)*(p*p) + 0.21875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 11.484375*(b*b*b*b)*(d*d*d*d)*p + 62.015625*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU746(double a, double b, double p, double d, double s){
	return (d*(-0.109375*(a*a*a*a)*b*(d*d*d*d) + 1.3125*(a*a*a)*(b*b)*(d*d*d*d) + 0.875*(a*a*a)*(d*d)*p - 3.28125*(a*a)*(b*b*b)*(d*d*d*d) - 9.1875*(a*a)*b*(d*d)*p + 2.1875*a*(b*b*b*b)*(d*d*d*d) + 18.375*a*(b*b)*(d*d)*p + 9.84375*a*(p*p) - 0.328125*(b*b*b*b*b)*(d*d*d*d) - 7.65625*(b*b*b)*(d*d)*p - 17.2265625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU747(double a, double b, double p, double d, double s){
	return ((0.0078125*(a*a*a*a)*(d*d*d*d) - 0.21875*(a*a*a)*b*(d*d*d*d) + 0.984375*(a*a)*(b*b)*(d*d*d*d) + 0.84375*(a*a)*(d*d)*p - 1.09375*a*(b*b*b)*(d*d*d*d) - 3.9375*a*b*(d*d)*p + 0.2734375*(b*b*b*b)*(d*d*d*d) + 2.953125*(b*b)*(d*d)*p + 1.93359375*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU748(double a, double b, double p, double d, double s){
	return (d*(0.015625*(a*a*a)*(d*d) - 0.1640625*(a*a)*b*(d*d) + 0.328125*a*(b*b)*(d*d) + 0.3515625*a*p - 0.13671875*(b*b*b)*(d*d) - 0.615234375*b*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU749(double a, double b, double p, double d, double s){
	return ((0.01171875*(a*a)*(d*d) - 0.0546875*a*b*(d*d) + 0.041015625*(b*b)*(d*d) + 0.0537109375*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7410(double a, double b, double p, double d, double s){
	return (d*(0.00390625*a - 0.0068359375*b)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7411(double a, double b, double p, double d, double s){
	return (0.00048828125/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU750(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 10.5*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 13.125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 17.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 131.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 196.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 32.8125*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 5.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 157.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 656.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 459.375*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 52.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 656.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1378.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 295.3125*(a*a)*(d*d)*(p*p*p*p*p) - 3.75*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 196.875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1148.4375*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 1033.59375*a*b*(d*d)*(p*p*p*p*p) + 13.125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 229.6875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 620.15625*(b*b)*(d*d)*(p*p*p*p*p) + 162.421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU751(double a, double b, double p, double d, double s){
	return (d*(3.5*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 26.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 39.375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 6.5625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 2.5*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 78.75*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 328.125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 229.6875*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 52.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 656.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1378.125*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 295.3125*(a*a*a)*(d*d)*(p*p*p*p) - 7.5*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 393.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 2296.875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 2067.1875*(a*a)*b*(d*d)*(p*p*p*p) + 65.625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1148.4375*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 3100.78125*a*(b*b)*(d*d)*(p*p*p*p) + 812.109375*a*(p*p*p*p*p) - 1.875*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 137.8125*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 1033.59375*(b*b*b)*(d*d)*(p*p*p*p) - 1136.953125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU752(double a, double b, double p, double d, double s){
	return ((-5.25*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 26.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 19.6875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 8.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 131.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 295.3125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 65.625*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 2.5*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 157.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 984.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 918.75*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 52.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 984.375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2756.25*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 738.28125*(a*a)*(d*d)*(p*p*p*p) - 3.75*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 295.3125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 2296.875*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 2583.984375*a*b*(d*d)*(p*p*p*p) + 19.6875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 459.375*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 1550.390625*(b*b)*(d*d)*(p*p*p*p) + 487.265625*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU753(double a, double b, double p, double d, double s){
	return (d*(4.375*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 13.125*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 109.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 114.84375*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 8.75*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 218.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 689.0625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 196.875*(a*a*a)*(d*d)*(p*p*p) - 1.25*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 131.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 1148.4375*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 1378.125*(a*a)*b*(d*d)*(p*p*p) + 21.875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 574.21875*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 2067.1875*a*(b*b)*(d*d)*(p*p*p) + 676.7578125*a*(p*p*p*p) - 0.625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 68.90625*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 689.0625*(b*b*b)*(d*d)*(p*p*p) - 947.4609375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU754(double a, double b, double p, double d, double s){
	return ((-2.1875*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.28125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 10.9375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 49.21875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 16.40625*(a*a*a*a)*(d*d*d*d)*(p*p) - 13.125*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 164.0625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 229.6875*(a*a*a)*b*(d*d*d*d)*(p*p) + 4.375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 164.0625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 689.0625*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 246.09375*(a*a)*(d*d)*(p*p*p) - 0.3125*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 49.21875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 574.21875*a*(b*b*b)*(d*d*d*d)*(p*p) - 861.328125*a*b*(d*d)*(p*p*p) + 3.28125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 114.84375*(b*b*b*b)*(d*d*d*d)*(p*p) + 516.796875*(b*b)*(d*d)*(p*p*p) + 203.02734375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU755(double a, double b, double p, double d, double s){
	return (d*(0.65625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.328125*(a*a*a*a*a)*(d*d*d*d)*p - 5.46875*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 11.484375*(a*a*a*a)*b*(d*d*d*d)*p + 10.9375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 68.90625*(a*a*a)*(b*b)*(d*d*d*d)*p + 29.53125*(a*a*a)*(d*d)*(p*p) - 6.5625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 114.84375*(a*a)*(b*b*b)*(d*d*d*d)*p - 206.71875*(a*a)*b*(d*d)*(p*p) + 1.09375*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 57.421875*a*(b*b*b*b)*(d*d*d*d)*p + 310.078125*a*(b*b)*(d*d)*(p*p) + 135.3515625*a*(p*p*p) - 0.03125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 6.890625*(b*b*b*b*b)*(d*d*d*d)*p - 103.359375*(b*b*b)*(d*d)*(p*p) - 189.4921875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU756(double a, double b, double p, double d, double s){
	return ((-0.109375*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.640625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.09375*(a*a*a*a)*(d*d*d*d)*p - 5.46875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 15.3125*(a*a*a)*b*(d*d*d*d)*p + 5.46875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 45.9375*(a*a)*(b*b)*(d*d*d*d)*p + 24.609375*(a*a)*(d*d)*(p*p) - 1.640625*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 38.28125*a*(b*b*b)*(d*d*d*d)*p - 86.1328125*a*b*(d*d)*(p*p) + 0.109375*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 7.65625*(b*b*b*b)*(d*d*d*d)*p + 51.6796875*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU757(double a, double b, double p, double d, double s){
	return (d*(0.0078125*(a*a*a*a*a)*(d*d*d*d) - 0.2734375*(a*a*a*a)*b*(d*d*d*d) + 1.640625*(a*a*a)*(b*b)*(d*d*d*d) + 1.40625*(a*a*a)*(d*d)*p - 2.734375*(a*a)*(b*b*b)*(d*d*d*d) - 9.84375*(a*a)*b*(d*d)*p + 1.3671875*a*(b*b*b*b)*(d*d*d*d) + 14.765625*a*(b*b)*(d*d)*p + 9.66796875*a*(p*p) - 0.1640625*(b*b*b*b*b)*(d*d*d*d) - 4.921875*(b*b*b)*(d*d)*p - 13.53515625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU758(double a, double b, double p, double d, double s){
	return ((0.01953125*(a*a*a*a)*(d*d*d*d) - 0.2734375*(a*a*a)*b*(d*d*d*d) + 0.8203125*(a*a)*(b*b)*(d*d*d*d) + 0.87890625*(a*a)*(d*d)*p - 0.68359375*a*(b*b*b)*(d*d*d*d) - 3.076171875*a*b*(d*d)*p + 0.13671875*(b*b*b*b)*(d*d*d*d) + 1.845703125*(b*b)*(d*d)*p + 1.4501953125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU759(double a, double b, double p, double d, double s){
	return (d*(0.01953125*(a*a*a)*(d*d) - 0.13671875*(a*a)*b*(d*d) + 0.205078125*a*(b*b)*(d*d) + 0.2685546875*a*p - 0.068359375*(b*b*b)*(d*d) - 0.3759765625*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7510(double a, double b, double p, double d, double s){
	return ((0.009765625*(a*a)*(d*d) - 0.0341796875*a*b*(d*d) + 0.0205078125*(b*b)*(d*d) + 0.0322265625*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7511(double a, double b, double p, double d, double s){
	return (d*(0.00244140625*a - 0.00341796875*b)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7512(double a, double b, double p, double d, double s){
	return (0.000244140625/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU760(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 10.5*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 13.125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 21.0 *(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 157.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 236.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 39.375*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 7.5*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 236.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 984.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 689.0625*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 105.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1312.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 2756.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 590.625*(a*a*a)*(d*d)*(p*p*p*p*p) - 11.25*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 590.625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 3445.3125*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 3100.78125*(a*a)*b*(d*d)*(p*p*p*p*p) + 78.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1378.125*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 3720.9375*a*(b*b)*(d*d)*(p*p*p*p*p) + 974.53125*a*(p*p*p*p*p*p) - 1.875*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 137.8125*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 1033.59375*(b*b*b)*(d*d)*(p*p*p*p*p) - 1136.953125*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU761(double a, double b, double p, double d, double s){
	return ((3.5*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 26.25*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 39.375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 6.5625*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 3.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 94.5*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 393.75*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 275.625*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 78.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 984.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 2067.1875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 442.96875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 15.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 787.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 4593.75*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 4134.375*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 196.875*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 3445.3125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 9302.34375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 2436.328125*(a*a)*(d*d)*(p*p*p*p*p) - 11.25*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 826.875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 6201.5625*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 6821.71875*a*b*(d*d)*(p*p*p*p*p) + 45.9375*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1033.59375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 3410.859375*(b*b)*(d*d)*(p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU762(double a, double b, double p, double d, double s){
	return (d*(-5.25*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 26.25*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 19.6875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 10.5*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 157.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 354.375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 78.75*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 3.75*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 236.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1476.5625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 1378.125*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 105.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1968.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 5512.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 1476.5625*(a*a*a)*(d*d)*(p*p*p*p) - 11.25*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 885.9375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 6890.625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 7751.953125*(a*a)*b*(d*d)*(p*p*p*p) + 118.125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2756.25*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 9302.34375*a*(b*b)*(d*d)*(p*p*p*p) + 2923.59375*a*(p*p*p*p*p) - 2.8125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 275.625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 2583.984375*(b*b*b)*(d*d)*(p*p*p*p) - 3410.859375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU763(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 15.75*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 131.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 137.8125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 328.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1033.59375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 295.3125*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 2.5*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 262.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 2296.875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 2756.25*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 65.625*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1722.65625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 6201.5625*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 2030.2734375*(a*a)*(d*d)*(p*p*p*p) - 3.75*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 413.4375*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 4134.375*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 5684.765625*a*b*(d*d)*(p*p*p*p) + 22.96875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 689.0625*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 2842.3828125*(b*b)*(d*d)*(p*p*p*p) + 1055.7421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU764(double a, double b, double p, double d, double s){
	return (d*(-2.1875*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.28125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 13.125*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 59.0625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 19.6875*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 19.6875*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 246.09375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 344.53125*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 8.75*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 328.125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1378.125*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 492.1875*(a*a*a)*(d*d)*(p*p*p) - 0.9375*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 147.65625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 1722.65625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 2583.984375*(a*a)*b*(d*d)*(p*p*p) + 19.6875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 689.0625*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 3100.78125*a*(b*b)*(d*d)*(p*p*p) + 1218.1640625*a*(p*p*p*p) - 0.46875*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 68.90625*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 861.328125*(b*b*b)*(d*d)*(p*p*p) - 1421.19140625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU765(double a, double b, double p, double d, double s){
	return ((0.65625*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.328125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 6.5625*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 13.78125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 16.40625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 103.359375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 44.296875*(a*a*a*a)*(d*d*d*d)*(p*p) - 13.125*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 229.6875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 413.4375*(a*a*a)*b*(d*d*d*d)*(p*p) + 3.28125*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 172.265625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 930.234375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 406.0546875*(a*a)*(d*d)*(p*p*p) - 0.1875*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 41.34375*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 620.15625*a*(b*b*b)*(d*d*d*d)*(p*p) - 1136.953125*a*b*(d*d)*(p*p*p) + 2.296875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 103.359375*(b*b*b*b)*(d*d*d*d)*(p*p) + 568.4765625*(b*b)*(d*d)*(p*p*p) + 263.935546875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU766(double a, double b, double p, double d, double s){
	return (d*(-0.109375*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.96875*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.3125*(a*a*a*a*a)*(d*d*d*d)*p - 8.203125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 22.96875*(a*a*a*a)*b*(d*d*d*d)*p + 10.9375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 91.875*(a*a*a)*(b*b)*(d*d*d*d)*p + 49.21875*(a*a*a)*(d*d)*(p*p) - 4.921875*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 114.84375*(a*a)*(b*b*b)*(d*d*d*d)*p - 258.3984375*(a*a)*b*(d*d)*(p*p) + 0.65625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 45.9375*a*(b*b*b*b)*(d*d*d*d)*p + 310.078125*a*(b*b)*(d*d)*(p*p) + 162.421875*a*(p*p*p) - 0.015625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 4.59375*(b*b*b*b*b)*(d*d*d*d)*p - 86.1328125*(b*b*b)*(d*d)*(p*p) - 189.4921875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU767(double a, double b, double p, double d, double s){
	return ((0.0078125*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.328125*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.4609375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 2.109375*(a*a*a*a)*(d*d*d*d)*p - 5.46875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 19.6875*(a*a*a)*b*(d*d*d*d)*p + 4.1015625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 44.296875*(a*a)*(b*b)*(d*d*d*d)*p + 29.00390625*(a*a)*(d*d)*(p*p) - 0.984375*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 29.53125*a*(b*b*b)*(d*d*d*d)*p - 81.2109375*a*b*(d*d)*(p*p) + 0.0546875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 4.921875*(b*b*b*b)*(d*d*d*d)*p + 40.60546875*(b*b)*(d*d)*(p*p) + 25.13671875*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU768(double a, double b, double p, double d, double s){
	return (d*(0.0234375*(a*a*a*a*a)*(d*d*d*d) - 0.41015625*(a*a*a*a)*b*(d*d*d*d) + 1.640625*(a*a*a)*(b*b)*(d*d*d*d) + 1.7578125*(a*a*a)*(d*d)*p - 2.05078125*(a*a)*(b*b*b)*(d*d*d*d) - 9.228515625*(a*a)*b*(d*d)*p + 0.8203125*a*(b*b*b*b)*(d*d*d*d) + 11.07421875*a*(b*b)*(d*d)*p + 8.701171875*a*(p*p) - 0.08203125*(b*b*b*b*b)*(d*d*d*d) - 3.076171875*(b*b*b)*(d*d)*p - 10.1513671875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU769(double a, double b, double p, double d, double s){
	return ((0.029296875*(a*a*a*a)*(d*d*d*d) - 0.2734375*(a*a*a)*b*(d*d*d*d) + 0.615234375*(a*a)*(b*b)*(d*d*d*d) + 0.8056640625*(a*a)*(d*d)*p - 0.41015625*a*(b*b*b)*(d*d*d*d) - 2.255859375*a*b*(d*d)*p + 0.068359375*(b*b*b*b)*(d*d*d*d) + 1.1279296875*(b*b)*(d*d)*p + 1.04736328125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7610(double a, double b, double p, double d, double s){
	return (d*(0.01953125*(a*a*a)*(d*d) - 0.1025390625*(a*a)*b*(d*d) + 0.123046875*a*(b*b)*(d*d) + 0.193359375*a*p - 0.0341796875*(b*b*b)*(d*d) - 0.2255859375*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7611(double a, double b, double p, double d, double s){
	return ((0.00732421875*(a*a)*(d*d) - 0.0205078125*a*b*(d*d) + 0.01025390625*(b*b)*(d*d) + 0.01904296875*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7612(double a, double b, double p, double d, double s){
	return (d*(0.00146484375*a - 0.001708984375*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7613(double a, double b, double p, double d, double s){
	return (0.0001220703125/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU770(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) - 10.5*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 13.125*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 24.5*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 183.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 275.625*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 45.9375*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 10.5*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 330.75*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 1378.125*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 964.6875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 183.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 2296.875*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 4823.4375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 1033.59375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 26.25*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 1378.125*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 8039.0625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 7235.15625*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 275.625*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 4823.4375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 13023.28125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 3410.859375*(a*a)*(d*d)*(p*p*p*p*p*p) - 13.125*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 964.6875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 7235.15625*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 7958.671875*a*b*(d*d)*(p*p*p*p*p*p) + 45.9375*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 1033.59375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 3410.859375*(b*b)*(d*d)*(p*p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU771(double a, double b, double p, double d, double s){
	return (d*(3.5*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 26.25*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 39.375*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 6.5625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 3.5*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 110.25*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 459.375*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 321.5625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 110.25*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1378.125*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 2894.0625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 620.15625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 26.25*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 1378.125*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 8039.0625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 7235.15625*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 459.375*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 8039.0625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 21705.46875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 5684.765625*(a*a*a)*(d*d)*(p*p*p*p*p) - 39.375*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 2894.0625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 21705.46875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 23876.015625*(a*a)*b*(d*d)*(p*p*p*p*p) + 321.5625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 7235.15625*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 23876.015625*a*(b*b)*(d*d)*(p*p*p*p*p) + 7390.1953125*a*(p*p*p*p*p*p) - 6.5625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 620.15625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 5684.765625*(b*b*b)*(d*d)*(p*p*p*p*p) - 7390.1953125*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU772(double a, double b, double p, double d, double s){
	return ((-5.25*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 26.25*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 19.6875*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 12.25*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 183.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 413.4375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 91.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 5.25*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 330.75*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 2067.1875*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1929.375*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 183.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 3445.3125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 9646.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 2583.984375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 26.25*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 2067.1875*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 16078.125*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 18087.890625*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 413.4375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 9646.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 32558.203125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 10232.578125*(a*a)*(d*d)*(p*p*p*p*p) - 19.6875*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1929.375*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 18087.890625*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 23876.015625*a*b*(d*d)*(p*p*p*p*p) + 91.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 2583.984375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 10232.578125*(b*b)*(d*d)*(p*p*p*p*p) + 3695.09765625*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU773(double a, double b, double p, double d, double s){
	return (d*(4.375*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 18.375*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 153.125*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 160.78125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 18.375*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 459.375*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1447.03125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 413.4375*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 4.375*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 459.375*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 4019.53125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 4823.4375*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 153.125*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4019.53125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 14470.3125*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 4737.3046875*(a*a*a)*(d*d)*(p*p*p*p) - 13.125*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1447.03125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 14470.3125*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 19896.6796875*(a*a)*b*(d*d)*(p*p*p*p) + 160.78125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 4823.4375*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 19896.6796875*a*(b*b)*(d*d)*(p*p*p*p) + 7390.1953125*a*(p*p*p*p*p) - 3.28125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 413.4375*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 4737.3046875*(b*b*b)*(d*d)*(p*p*p*p) - 7390.1953125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU774(double a, double b, double p, double d, double s){
	return ((-2.1875*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.28125*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 15.3125*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 68.90625*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 22.96875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 27.5625*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 344.53125*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 482.34375*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 15.3125*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 574.21875*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2411.71875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 861.328125*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 2.1875*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 344.53125*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 4019.53125*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 6029.296875*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 68.90625*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2411.71875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 10852.734375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 4263.57421875*(a*a)*(d*d)*(p*p*p*p) - 3.28125*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 482.34375*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 6029.296875*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 9948.33984375*a*b*(d*d)*(p*p*p*p) + 22.96875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 861.328125*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 4263.57421875*(b*b)*(d*d)*(p*p*p*p) + 1847.548828125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU775(double a, double b, double p, double d, double s){
	return (d*(0.65625*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.328125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.65625*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 16.078125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 22.96875*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 144.703125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 62.015625*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 22.96875*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 401.953125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 723.515625*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 7.65625*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 401.953125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 2170.546875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 947.4609375*(a*a*a)*(d*d)*(p*p*p) - 0.65625*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 144.703125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 2170.546875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 3979.3359375*(a*a)*b*(d*d)*(p*p*p) + 16.078125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 723.515625*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 3979.3359375*a*(b*b)*(d*d)*(p*p*p) + 1847.548828125*a*(p*p*p*p) - 0.328125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 62.015625*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 947.4609375*(b*b*b)*(d*d)*(p*p*p) - 1847.548828125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU776(double a, double b, double p, double d, double s){
	return ((-0.109375*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 2.296875*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 1.53125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 11.484375*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 32.15625*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 19.140625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 160.78125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 86.1328125*(a*a*a*a)*(d*d*d*d)*(p*p) - 11.484375*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 267.96875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 602.9296875*(a*a*a)*b*(d*d*d*d)*(p*p) + 2.296875*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 160.78125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1085.2734375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 568.4765625*(a*a)*(d*d)*(p*p*p) - 0.109375*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 32.15625*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 602.9296875*a*(b*b*b)*(d*d*d*d)*(p*p) - 1326.4453125*a*b*(d*d)*(p*p*p) + 1.53125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 86.1328125*(b*b*b*b)*(d*d*d*d)*(p*p) + 568.4765625*(b*b)*(d*d)*(p*p*p) + 307.9248046875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU777(double a, double b, double p, double d, double s){
	return (d*(0.0078125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.3828125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.4453125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 2.953125*(a*a*a*a*a)*(d*d*d*d)*p - 9.5703125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 34.453125*(a*a*a*a)*b*(d*d*d*d)*p + 9.5703125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 103.359375*(a*a*a)*(b*b)*(d*d*d*d)*p + 67.67578125*(a*a*a)*(d*d)*(p*p) - 3.4453125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 103.359375*(a*a)*(b*b*b)*(d*d*d*d)*p - 284.23828125*(a*a)*b*(d*d)*(p*p) + 0.3828125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 34.453125*a*(b*b*b*b)*(d*d*d*d)*p + 284.23828125*a*(b*b)*(d*d)*(p*p) + 175.95703125*a*(p*p*p) - 0.0078125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 2.953125*(b*b*b*b*b)*(d*d*d*d)*p - 67.67578125*(b*b*b)*(d*d)*(p*p) - 175.95703125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU778(double a, double b, double p, double d, double s){
	return ((0.02734375*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.57421875*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.87109375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.076171875*(a*a*a*a)*(d*d*d*d)*p - 4.78515625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 21.533203125*(a*a*a)*b*(d*d*d*d)*p + 2.87109375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 38.759765625*(a*a)*(b*b)*(d*d*d*d)*p + 30.4541015625*(a*a)*(d*d)*(p*p) - 0.57421875*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 21.533203125*a*(b*b*b)*(d*d*d*d)*p - 71.0595703125*a*b*(d*d)*(p*p) + 0.02734375*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 3.076171875*(b*b*b*b)*(d*d*d*d)*p + 30.4541015625*(b*b)*(d*d)*(p*p) + 21.99462890625*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU779(double a, double b, double p, double d, double s){
	return (d*(0.041015625*(a*a*a*a*a)*(d*d*d*d) - 0.478515625*(a*a*a*a)*b*(d*d*d*d) + 1.435546875*(a*a*a)*(b*b)*(d*d*d*d) + 1.8798828125*(a*a*a)*(d*d)*p - 1.435546875*(a*a)*(b*b*b)*(d*d*d*d) - 7.8955078125*(a*a)*b*(d*d)*p + 0.478515625*a*(b*b*b*b)*(d*d*d*d) + 7.8955078125*a*(b*b)*(d*d)*p + 7.33154296875*a*(p*p) - 0.041015625*(b*b*b*b*b)*(d*d*d*d) - 1.8798828125*(b*b*b)*(d*d)*p - 7.33154296875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7710(double a, double b, double p, double d, double s){
	return ((0.0341796875*(a*a*a*a)*(d*d*d*d) - 0.2392578125*(a*a*a)*b*(d*d*d*d) + 0.4306640625*(a*a)*(b*b)*(d*d*d*d) + 0.6767578125*(a*a)*(d*d)*p - 0.2392578125*a*(b*b*b)*(d*d*d*d) - 1.5791015625*a*b*(d*d)*p + 0.0341796875*(b*b*b*b)*(d*d*d*d) + 0.6767578125*(b*b)*(d*d)*p + 0.733154296875*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7711(double a, double b, double p, double d, double s){
	return (d*(0.01708984375*(a*a*a)*(d*d) - 0.07177734375*(a*a)*b*(d*d) + 0.07177734375*a*(b*b)*(d*d) + 0.13330078125*a*p - 0.01708984375*(b*b*b)*(d*d) - 0.13330078125*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7712(double a, double b, double p, double d, double s){
	return ((0.005126953125*(a*a)*(d*d) - 0.011962890625*a*b*(d*d) + 0.005126953125*(b*b)*(d*d) + 0.0111083984375*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7713(double a, double b, double p, double d, double s){
	return (0.0008544921875*d*(a - b)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7714(double a, double b, double p, double d, double s){
	return (6.103515625e-5/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU780(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) - 10.5*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 13.125*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 28.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 210.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 315.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 52.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 14.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 441.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 1837.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 1286.25*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 294.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 3675.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 7717.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 1653.75*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 52.5*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 2756.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 16078.125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 14470.3125*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 735.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 12862.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 34728.75*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 9095.625*(a*a*a)*(d*d)*(p*p*p*p*p*p) - 52.5*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 3858.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 28940.625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 31834.6875*(a*a)*b*(d*d)*(p*p*p*p*p*p) + 367.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 8268.75*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 27286.875*a*(b*b)*(d*d)*(p*p*p*p*p*p) + 8445.9375*a*(p*p*p*p*p*p*p) - 6.5625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 620.15625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 5684.765625*(b*b*b)*(d*d)*(p*p*p*p*p*p) - 7390.1953125*b*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU781(double a, double b, double p, double d, double s){
	return ((3.5*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 26.25*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 39.375*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 6.5625*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p*p) - 4.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) - 126.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 525.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 367.5*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 147.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 1837.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 3858.75*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 826.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 42.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 2205.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 12862.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 11576.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 918.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 16078.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 43410.9375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 11369.53125*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 105.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 7717.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 57881.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 63669.375*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 1286.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 28940.625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 95504.0625*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 29560.78125*(a*a)*(d*d)*(p*p*p*p*p*p) - 52.5*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 4961.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 45478.125*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 59121.5625*a*b*(d*d)*(p*p*p*p*p*p) + 206.71875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 5684.765625*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 22170.5859375*(b*b)*(d*d)*(p*p*p*p*p*p) + 7918.06640625*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU782(double a, double b, double p, double d, double s){
	return (d*(-5.25*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 26.25*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 19.6875*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 14.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 210.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 472.5*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 105.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 7.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 441.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 2756.25*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 2572.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 294.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 5512.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 15435.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 4134.375*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 52.5*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 4134.375*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 32156.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 36175.78125*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 1102.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 25725.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 86821.875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 27286.875*(a*a*a)*(d*d)*(p*p*p*p*p) - 78.75*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 7717.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 72351.5625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 95504.0625*(a*a)*b*(d*d)*(p*p*p*p*p) + 735.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 20671.875*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 81860.625*a*(b*b)*(d*d)*(p*p*p*p*p) + 29560.78125*a*(p*p*p*p*p*p) - 13.125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1550.390625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 17054.296875*(b*b*b)*(d*d)*(p*p*p*p*p) - 25865.68359375*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU783(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p) - 21.0 *(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 175.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 183.75*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 24.5*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 612.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1929.375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 551.25*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 7.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 735.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 6431.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 7717.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 306.25*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 8039.0625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 28940.625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 9474.609375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 35.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 3858.75*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 38587.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 53057.8125*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 643.125*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 19293.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 79586.71875*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 29560.78125*(a*a)*(d*d)*(p*p*p*p*p) - 26.25*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 3307.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 37898.4375*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 59121.5625*a*b*(d*d)*(p*p*p*p*p) + 137.8125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 4737.3046875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 22170.5859375*(b*b)*(d*d)*(p*p*p*p*p) + 9237.744140625*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU784(double a, double b, double p, double d, double s){
	return (d*(-2.1875*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.28125*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 17.5*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 78.75*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 26.25*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 36.75*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 459.375*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 643.125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 24.5*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 918.75*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 3858.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1378.125*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 4.375*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 689.0625*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 8039.0625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 12058.59375*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 183.75*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 6431.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 28940.625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 11369.53125*(a*a*a)*(d*d)*(p*p*p*p) - 13.125*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1929.375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 24117.1875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 39793.359375*(a*a)*b*(d*d)*(p*p*p*p) + 183.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 6890.625*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 34108.59375*a*(b*b)*(d*d)*(p*p*p*p) + 14780.390625*a*(p*p*p*p*p) - 3.28125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 516.796875*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 7105.95703125*(b*b*b)*(d*d)*(p*p*p*p) - 12932.841796875*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU785(double a, double b, double p, double d, double s){
	return ((0.65625*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.328125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 8.75*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 18.375*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 30.625*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 192.9375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 82.6875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 36.75*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 643.125*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 1157.625*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 15.3125*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 803.90625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4341.09375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1894.921875*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 1.75*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 385.875*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 5788.125*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 10611.5625*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 64.3125*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2894.0625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 15917.34375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 7390.1953125*(a*a)*(d*d)*(p*p*p*p) - 2.625*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 496.125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 7579.6875*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 14780.390625*a*b*(d*d)*(p*p*p*p) + 20.671875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 947.4609375*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 5542.646484375*(b*b)*(d*d)*(p*p*p*p) + 2771.3232421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU786(double a, double b, double p, double d, double s){
	return (d*(-0.109375*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 2.625*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 1.75*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 15.3125*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 42.875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 30.625*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 257.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 137.8125*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 22.96875*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 535.9375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 1205.859375*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 6.125*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 428.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 2894.0625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 1515.9375*(a*a*a)*(d*d)*(p*p*p) - 0.4375*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 128.625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 2411.71875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 5305.78125*(a*a)*b*(d*d)*(p*p*p) + 12.25*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 689.0625*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 4547.8125*a*(b*b)*(d*d)*(p*p*p) + 2463.3984375*a*(p*p*p*p) - 0.21875*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 51.6796875*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 947.4609375*(b*b*b)*(d*d)*(p*p*p) - 2155.4736328125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU787(double a, double b, double p, double d, double s){
	return ((0.0078125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 0.4375*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 4.59375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 3.9375*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 15.3125*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 55.125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 19.140625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 206.71875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 135.3515625*(a*a*a*a)*(d*d*d*d)*(p*p) - 9.1875*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 275.625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 757.96875*(a*a*a)*b*(d*d*d*d)*(p*p) + 1.53125*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 137.8125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1136.953125*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 703.828125*(a*a)*(d*d)*(p*p*p) - 0.0625*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 23.625*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 541.40625*a*(b*b*b)*(d*d*d*d)*(p*p) - 1407.65625*a*b*(d*d)*(p*p*p) + 0.984375*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 67.67578125*(b*b*b*b)*(d*d*d*d)*(p*p) + 527.87109375*(b*b)*(d*d)*(p*p*p) + 329.91943359375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU788(double a, double b, double p, double d, double s){
	return (d*(0.03125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.765625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.59375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 4.921875*(a*a*a*a*a)*(d*d*d*d)*p - 9.5703125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 43.06640625*(a*a*a*a)*b*(d*d*d*d)*p + 7.65625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 103.359375*(a*a*a)*(b*b)*(d*d*d*d)*p + 81.2109375*(a*a*a)*(d*d)*(p*p) - 2.296875*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 86.1328125*(a*a)*(b*b*b)*(d*d*d*d)*p - 284.23828125*(a*a)*b*(d*d)*(p*p) + 0.21875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 24.609375*a*(b*b*b*b)*(d*d*d*d)*p + 243.6328125*a*(b*b)*(d*d)*(p*p) + 175.95703125*a*(p*p*p) - 0.00390625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 1.845703125*(b*b*b*b*b)*(d*d*d*d)*p - 50.7568359375*(b*b*b)*(d*d)*(p*p) - 153.96240234375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU789(double a, double b, double p, double d, double s){
	return ((0.0546875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.765625*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.87109375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.759765625*(a*a*a*a)*(d*d*d*d)*p - 3.828125*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 21.0546875*(a*a*a)*b*(d*d*d*d)*p + 1.9140625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 31.58203125*(a*a)*(b*b)*(d*d*d*d)*p + 29.326171875*(a*a)*(d*d)*(p*p) - 0.328125*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 15.0390625*a*(b*b*b)*(d*d*d*d)*p - 58.65234375*a*b*(d*d)*(p*p) + 0.013671875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 1.8798828125*(b*b*b*b)*(d*d*d*d)*p + 21.99462890625*(b*b)*(d*d)*(p*p) + 18.328857421875*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7810(double a, double b, double p, double d, double s){
	return (d*(0.0546875*(a*a*a*a*a)*(d*d*d*d) - 0.478515625*(a*a*a*a)*b*(d*d*d*d) + 1.1484375*(a*a*a)*(b*b)*(d*d*d*d) + 1.8046875*(a*a*a)*(d*d)*p - 0.95703125*(a*a)*(b*b*b)*(d*d*d*d) - 6.31640625*(a*a)*b*(d*d)*p + 0.2734375*a*(b*b*b*b)*(d*d*d*d) + 5.4140625*a*(b*b)*(d*d)*p + 5.865234375*a*(p*p) - 0.0205078125*(b*b*b*b*b)*(d*d*d*d) - 1.1279296875*(b*b*b)*(d*d)*p - 5.132080078125*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7811(double a, double b, double p, double d, double s){
	return ((0.0341796875*(a*a*a*a)*(d*d*d*d) - 0.19140625*(a*a*a)*b*(d*d*d*d) + 0.287109375*(a*a)*(b*b)*(d*d*d*d) + 0.533203125*(a*a)*(d*d)*p - 0.13671875*a*(b*b*b)*(d*d*d*d) - 1.06640625*a*b*(d*d)*p + 0.01708984375*(b*b*b*b)*(d*d*d*d) + 0.39990234375*(b*b)*(d*d)*p + 0.4998779296875*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7812(double a, double b, double p, double d, double s){
	return (d*(0.013671875*(a*a*a)*(d*d) - 0.0478515625*(a*a)*b*(d*d) + 0.041015625*a*(b*b)*(d*d) + 0.0888671875*a*p - 0.008544921875*(b*b*b)*(d*d) - 0.0777587890625*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7813(double a, double b, double p, double d, double s){
	return ((0.00341796875*(a*a)*(d*d) - 0.0068359375*a*b*(d*d) + 0.0025634765625*(b*b)*(d*d) + 0.00640869140625*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7814(double a, double b, double p, double d, double s){
	return (d*(0.00048828125*a - 0.00042724609375*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU7815(double a, double b, double p, double d, double s){
	return (3.0517578125e-5/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU800(double a, double b, double p, double d, double s){
	return (((b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 14.0*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 52.5*(b*b*b*b)*(d*d*d*d)*(p*p) + 52.5*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU801(double a, double b, double p, double d, double s){
	return (b*d*(-4.0*(b*b*b*b*b*b)*(d*d*d*d*d*d) - 42.0*(b*b*b*b)*(d*d*d*d)*p - 105.0*(b*b)*(d*d)*(p*p) - 52.5*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU802(double a, double b, double p, double d, double s){
	return ((7.0*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 52.5*(b*b*b*b)*(d*d*d*d)*p + 78.75*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU803(double a, double b, double p, double d, double s){
	return (b*d*(-7.0*(b*b*b*b)*(d*d*d*d) - 35.0*(b*b)*(d*d)*p - 26.25*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU804(double a, double b, double p, double d, double s){
	return ((4.375*(b*b*b*b)*(d*d*d*d) + 13.125*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU805(double a, double b, double p, double d, double s){
	return (b*d*(-1.75*(b*b)*(d*d) - 2.625*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU806(double a, double b, double p, double d, double s){
	return ((0.4375*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU807(double a, double b, double p, double d, double s){
	return (-0.0625*b*d/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU808(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU810(double a, double b, double p, double d, double s){
	return (d*(a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 14.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 52.5*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 52.5*a*(b*b)*(d*d)*(p*p*p) + 6.5625*a*(p*p*p*p) - 4.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 42.0*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 105.0*(b*b*b)*(d*d)*(p*p*p) - 52.5*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU811(double a, double b, double p, double d, double s){
	return ((-4.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 42.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 105.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 52.5*a*b*(d*d)*(p*p*p) + 0.5*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 21.0 *(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 131.25*(b*b*b*b)*(d*d*d*d)*(p*p) + 183.75*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU812(double a, double b, double p, double d, double s){
	return (d*(7.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 52.5*a*(b*b*b*b)*(d*d*d*d)*p + 78.75*a*(b*b)*(d*d)*(p*p) + 13.125*a*(p*p*p) - 2.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 42.0*(b*b*b*b*b)*(d*d*d*d)*p - 157.5*(b*b*b)*(d*d)*(p*p) - 105.0*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU813(double a, double b, double p, double d, double s){
	return ((-7.0*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 35.0*a*(b*b*b)*(d*d*d*d)*p - 26.25*a*b*(d*d)*(p*p) + 3.5*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 43.75*(b*b*b*b)*(d*d*d*d)*p + 91.875*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU814(double a, double b, double p, double d, double s){
	return (d*(4.375*a*(b*b*b*b)*(d*d*d*d) + 13.125*a*(b*b)*(d*d)*p + 3.28125*a*(p*p) - 3.5*(b*b*b*b*b)*(d*d*d*d) - 26.25*(b*b*b)*(d*d)*p - 26.25*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU815(double a, double b, double p, double d, double s){
	return ((-1.75*a*(b*b*b)*(d*d*d*d) - 2.625*a*b*(d*d)*p + 2.1875*(b*b*b*b)*(d*d*d*d) + 9.1875*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU816(double a, double b, double p, double d, double s){
	return (d*(0.4375*a*(b*b)*(d*d) + 0.21875*a*p - 0.875*(b*b*b)*(d*d) - 1.75*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU817(double a, double b, double p, double d, double s){
	return ((-0.0625*a*b*(d*d) + 0.21875*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU818(double a, double b, double p, double d, double s){
	return (d*(0.00390625*a - 0.03125*b)/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU819(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU820(double a, double b, double p, double d, double s){
	return (((a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 52.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 52.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 6.5625*(a*a)*(d*d)*(p*p*p*p) - 8.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 84.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 210.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 105.0*a*b*(d*d)*(p*p*p*p) + 0.5*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 21.0 *(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 131.25*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 183.75*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU821(double a, double b, double p, double d, double s){
	return (d*(-4.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 42.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 105.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 52.5*(a*a)*b*(d*d)*(p*p*p) + a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 42.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 262.5*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 367.5*a*(b*b)*(d*d)*(p*p*p) + 59.0625*a*(p*p*p*p) - 6.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 105.0*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 367.5*(b*b*b)*(d*d)*(p*p*p) - 236.25*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU822(double a, double b, double p, double d, double s){
	return ((7.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 52.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 78.75*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 13.125*(a*a)*(d*d)*(p*p*p) - 4.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 84.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 315.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 210.0*a*b*(d*d)*(p*p*p) + 0.25*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 21.0 *(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 196.875*(b*b*b*b)*(d*d*d*d)*(p*p) + 367.5*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU823(double a, double b, double p, double d, double s){
	return (d*(-7.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 35.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 26.25*(a*a)*b*(d*d)*(p*p) + 7.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 87.5*a*(b*b*b*b)*(d*d*d*d)*p + 183.75*a*(b*b)*(d*d)*(p*p) + 39.375*a*(p*p*p) - (b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 35.0*(b*b*b*b*b)*(d*d*d*d)*p - 183.75*(b*b*b)*(d*d)*(p*p) - 157.5*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU824(double a, double b, double p, double d, double s){
	return ((4.375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 13.125*(a*a)*(b*b)*(d*d*d*d)*p + 3.28125*(a*a)*(d*d)*(p*p) - 7.0*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 52.5*a*(b*b*b)*(d*d*d*d)*p - 52.5*a*b*(d*d)*(p*p) + 1.75*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 32.8125*(b*b*b*b)*(d*d*d*d)*p + 91.875*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU825(double a, double b, double p, double d, double s){
	return (d*(-1.75*(a*a)*(b*b*b)*(d*d*d*d) - 2.625*(a*a)*b*(d*d)*p + 4.375*a*(b*b*b*b)*(d*d*d*d) + 18.375*a*(b*b)*(d*d)*p + 5.90625*a*(p*p) - 1.75*(b*b*b*b*b)*(d*d*d*d) - 18.375*(b*b*b)*(d*d)*p - 23.625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU826(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a)*(b*b)*(d*d*d*d) + 0.21875*(a*a)*(d*d)*p - 1.75*a*(b*b*b)*(d*d*d*d) - 3.5*a*b*(d*d)*p + 1.09375*(b*b*b*b)*(d*d*d*d) + 6.125*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU827(double a, double b, double p, double d, double s){
	return (d*(-0.0625*(a*a)*b*(d*d) + 0.4375*a*(b*b)*(d*d) + 0.28125*a*p - 0.4375*(b*b*b)*(d*d) - 1.125*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU828(double a, double b, double p, double d, double s){
	return ((0.00390625*(a*a)*(d*d) - 0.0625*a*b*(d*d) + 0.109375*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU829(double a, double b, double p, double d, double s){
	return (d*(0.00390625*a - 0.015625*b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8210(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU830(double a, double b, double p, double d, double s){
	return (d*((a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 6.5625*(a*a*a)*(d*d)*(p*p*p*p) - 12.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 126.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 315.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 157.5*(a*a)*b*(d*d)*(p*p*p*p) + 1.5*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 63.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 393.75*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 551.25*a*(b*b)*(d*d)*(p*p*p*p) + 88.59375*a*(p*p*p*p*p) - 6.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 105.0*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 367.5*(b*b*b)*(d*d)*(p*p*p*p) - 236.25*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU831(double a, double b, double p, double d, double s){
	return ((-4.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 42.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 52.5*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 1.5*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 63.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 393.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 551.25*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 88.59375*(a*a)*(d*d)*(p*p*p*p) - 18.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 315.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1102.5*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 708.75*a*b*(d*d)*(p*p*p*p) + 0.75*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 52.5*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 459.375*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 826.875*(b*b)*(d*d)*(p*p*p*p) + 162.421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU832(double a, double b, double p, double d, double s){
	return (d*(7.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 52.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 78.75*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 13.125*(a*a*a)*(d*d)*(p*p*p) - 6.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 126.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 472.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 315.0*(a*a)*b*(d*d)*(p*p*p) + 0.75*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 63.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 590.625*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 1102.5*a*(b*b)*(d*d)*(p*p*p) + 221.484375*a*(p*p*p*p) - 6.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 157.5*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 735.0*(b*b*b)*(d*d)*(p*p*p) - 590.625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU833(double a, double b, double p, double d, double s){
	return ((-7.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 35.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 26.25*(a*a*a)*b*(d*d*d*d)*(p*p) + 10.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 131.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 275.625*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 59.0625*(a*a)*(d*d)*(p*p*p) - 3.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 105.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 551.25*a*(b*b*b)*(d*d*d*d)*(p*p) - 472.5*a*b*(d*d)*(p*p*p) + 0.125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 17.5*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 229.6875*(b*b*b*b)*(d*d*d*d)*(p*p) + 551.25*(b*b)*(d*d)*(p*p*p) + 135.3515625*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU834(double a, double b, double p, double d, double s){
	return (d*(4.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 13.125*(a*a*a)*(b*b)*(d*d*d*d)*p + 3.28125*(a*a*a)*(d*d)*(p*p) - 10.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 78.75*(a*a)*(b*b*b)*(d*d*d*d)*p - 78.75*(a*a)*b*(d*d)*(p*p) + 5.25*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 98.4375*a*(b*b*b*b)*(d*d*d*d)*p + 275.625*a*(b*b)*(d*d)*(p*p) + 73.828125*a*(p*p*p) - 0.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 26.25*(b*b*b*b*b)*(d*d*d*d)*p - 183.75*(b*b*b)*(d*d)*(p*p) - 196.875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU835(double a, double b, double p, double d, double s){
	return ((-1.75*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 2.625*(a*a*a)*b*(d*d*d*d)*p + 6.5625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 27.5625*(a*a)*(b*b)*(d*d*d*d)*p + 8.859375*(a*a)*(d*d)*(p*p) - 5.25*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 55.125*a*(b*b*b)*(d*d*d*d)*p - 70.875*a*b*(d*d)*(p*p) + 0.875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 22.96875*(b*b*b*b)*(d*d*d*d)*p + 82.6875*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU836(double a, double b, double p, double d, double s){
	return (d*(0.4375*(a*a*a)*(b*b)*(d*d*d*d) + 0.21875*(a*a*a)*(d*d)*p - 2.625*(a*a)*(b*b*b)*(d*d*d*d) - 5.25*(a*a)*b*(d*d)*p + 3.28125*a*(b*b*b*b)*(d*d*d*d) + 18.375*a*(b*b)*(d*d)*p + 7.3828125*a*(p*p) - 0.875*(b*b*b*b*b)*(d*d*d*d) - 12.25*(b*b*b)*(d*d)*p - 19.6875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU837(double a, double b, double p, double d, double s){
	return ((-0.0625*(a*a*a)*b*(d*d*d*d) + 0.65625*(a*a)*(b*b)*(d*d*d*d) + 0.421875*(a*a)*(d*d)*p - 1.3125*a*(b*b*b)*(d*d*d*d) - 3.375*a*b*(d*d)*p + 0.546875*(b*b*b*b)*(d*d*d*d) + 3.9375*(b*b)*(d*d)*p + 1.93359375*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU838(double a, double b, double p, double d, double s){
	return (d*(0.00390625*(a*a*a)*(d*d) - 0.09375*(a*a)*b*(d*d) + 0.328125*a*(b*b)*(d*d) + 0.263671875*a*p - 0.21875*(b*b*b)*(d*d) - 0.703125*b*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU839(double a, double b, double p, double d, double s){
	return ((0.005859375*(a*a)*(d*d) - 0.046875*a*b*(d*d) + 0.0546875*(b*b)*(d*d) + 0.0537109375*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8310(double a, double b, double p, double d, double s){
	return (d*(0.0029296875*a - 0.0078125*b)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8311(double a, double b, double p, double d, double s){
	return (0.00048828125/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU840(double a, double b, double p, double d, double s){
	return (((a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 6.5625*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 16.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 168.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 420.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 210.0*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 3.0*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 126.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 787.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1102.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 177.1875*(a*a)*(d*d)*(p*p*p*p*p) - 24.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 420.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1470.0*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 945.0*a*b*(d*d)*(p*p*p*p*p) + 0.75*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 52.5*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 459.375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 826.875*(b*b)*(d*d)*(p*p*p*p*p) + 162.421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU841(double a, double b, double p, double d, double s){
	return (d*(-4.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 42.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 52.5*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 2.0*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 84.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 525.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 735.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 118.125*(a*a*a)*(d*d)*(p*p*p*p) - 36.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 630.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 2205.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 1417.5*(a*a)*b*(d*d)*(p*p*p*p) + 3.0*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 210.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1837.5*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 3307.5*a*(b*b)*(d*d)*(p*p*p*p) + 649.6875*a*(p*p*p*p*p) - 15.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 367.5*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 1653.75*(b*b*b)*(d*d)*(p*p*p*p) - 1299.375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU842(double a, double b, double p, double d, double s){
	return ((7.0*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 52.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 78.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 8.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 168.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 630.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 420.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 1.5*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 126.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1181.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2205.0*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 442.96875*(a*a)*(d*d)*(p*p*p*p) - 24.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 630.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 2940.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 2362.5*a*b*(d*d)*(p*p*p*p) + 0.75*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 78.75*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 918.75*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 2067.1875*(b*b)*(d*d)*(p*p*p*p) + 487.265625*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU843(double a, double b, double p, double d, double s){
	return (d*(-7.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 35.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 26.25*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 14.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 175.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 367.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 78.75*(a*a*a)*(d*d)*(p*p*p) - 6.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 210.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 1102.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 945.0*(a*a)*b*(d*d)*(p*p*p) + 0.5*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 70.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 918.75*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 2205.0*a*(b*b)*(d*d)*(p*p*p) + 541.40625*a*(p*p*p*p) - 5.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 183.75*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 1102.5*(b*b*b)*(d*d)*(p*p*p) - 1082.8125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU844(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 3.28125*(a*a*a*a)*(d*d*d*d)*(p*p) - 14.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 105.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 105.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 10.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 196.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 551.25*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 147.65625*(a*a)*(d*d)*(p*p*p) - 2.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 105.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 735.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 787.5*a*b*(d*d)*(p*p*p) + 0.0625*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 13.125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 229.6875*(b*b*b*b)*(d*d*d*d)*(p*p) + 689.0625*(b*b)*(d*d)*(p*p*p) + 203.02734375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU845(double a, double b, double p, double d, double s){
	return (d*(-1.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 2.625*(a*a*a*a)*b*(d*d*d*d)*p + 8.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 36.75*(a*a*a)*(b*b)*(d*d*d*d)*p + 11.8125*(a*a*a)*(d*d)*(p*p) - 10.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 110.25*(a*a)*(b*b*b)*(d*d*d*d)*p - 141.75*(a*a)*b*(d*d)*(p*p) + 3.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 91.875*a*(b*b*b*b)*(d*d*d*d)*p + 330.75*a*(b*b)*(d*d)*(p*p) + 108.28125*a*(p*p*p) - 0.25*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 18.375*(b*b*b*b*b)*(d*d*d*d)*p - 165.375*(b*b*b)*(d*d)*(p*p) - 216.5625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU846(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.21875*(a*a*a*a)*(d*d*d*d)*p - 3.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 7.0*(a*a*a)*b*(d*d*d*d)*p + 6.5625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 36.75*(a*a)*(b*b)*(d*d*d*d)*p + 14.765625*(a*a)*(d*d)*(p*p) - 3.5*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 49.0*a*(b*b*b)*(d*d*d*d)*p - 78.75*a*b*(d*d)*(p*p) + 0.4375*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 15.3125*(b*b*b*b)*(d*d*d*d)*p + 68.90625*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU847(double a, double b, double p, double d, double s){
	return (d*(-0.0625*(a*a*a*a)*b*(d*d*d*d) + 0.875*(a*a*a)*(b*b)*(d*d*d*d) + 0.5625*(a*a*a)*(d*d)*p - 2.625*(a*a)*(b*b*b)*(d*d*d*d) - 6.75*(a*a)*b*(d*d)*p + 2.1875*a*(b*b*b*b)*(d*d*d*d) + 15.75*a*(b*b)*(d*d)*p + 7.734375*a*(p*p) - 0.4375*(b*b*b*b*b)*(d*d*d*d) - 7.875*(b*b*b)*(d*d)*p - 15.46875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU848(double a, double b, double p, double d, double s){
	return ((0.00390625*(a*a*a*a)*(d*d*d*d) - 0.125*(a*a*a)*b*(d*d*d*d) + 0.65625*(a*a)*(b*b)*(d*d*d*d) + 0.52734375*(a*a)*(d*d)*p - 0.875*a*(b*b*b)*(d*d*d*d) - 2.8125*a*b*(d*d)*p + 0.2734375*(b*b*b*b)*(d*d*d*d) + 2.4609375*(b*b)*(d*d)*p + 1.4501953125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU849(double a, double b, double p, double d, double s){
	return (d*(0.0078125*(a*a*a)*(d*d) - 0.09375*(a*a)*b*(d*d) + 0.21875*a*(b*b)*(d*d) + 0.21484375*a*p - 0.109375*(b*b*b)*(d*d) - 0.4296875*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8410(double a, double b, double p, double d, double s){
	return ((0.005859375*(a*a)*(d*d) - 0.03125*a*b*(d*d) + 0.02734375*(b*b)*(d*d) + 0.0322265625*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8411(double a, double b, double p, double d, double s){
	return (d*(0.001953125*a - 0.00390625*b)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8412(double a, double b, double p, double d, double s){
	return (0.000244140625/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU850(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 6.5625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 20.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 210.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 525.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 262.5*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 5.0*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 210.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1312.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1837.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 295.3125*(a*a*a)*(d*d)*(p*p*p*p*p) - 60.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1050.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 3675.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 2362.5*(a*a)*b*(d*d)*(p*p*p*p*p) + 3.75*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 262.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 2296.875*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 4134.375*a*(b*b)*(d*d)*(p*p*p*p*p) + 812.109375*a*(p*p*p*p*p*p) - 15.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 367.5*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 1653.75*(b*b*b)*(d*d)*(p*p*p*p*p) - 1299.375*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU851(double a, double b, double p, double d, double s){
	return ((-4.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 42.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 52.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 2.5*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 105.0*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 656.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 918.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 147.65625*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 60.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 1050.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 3675.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 2362.5*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 7.5*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 525.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 4593.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 8268.75*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 1624.21875*(a*a)*(d*d)*(p*p*p*p*p) - 75.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1837.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 8268.75*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 6496.875*a*b*(d*d)*(p*p*p*p*p) + 1.875*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 183.75*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 2067.1875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 4547.8125*(b*b)*(d*d)*(p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU852(double a, double b, double p, double d, double s){
	return (d*(7.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 52.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 78.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 10.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 210.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 787.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 525.0*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 2.5*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 210.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1968.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 3675.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 738.28125*(a*a*a)*(d*d)*(p*p*p*p) - 60.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1575.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 7350.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 5906.25*(a*a)*b*(d*d)*(p*p*p*p) + 3.75*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 393.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 4593.75*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 10335.9375*a*(b*b)*(d*d)*(p*p*p*p) + 2436.328125*a*(p*p*p*p*p) - 22.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 735.0*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 4134.375*(b*b*b)*(d*d)*(p*p*p*p) - 3898.125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU853(double a, double b, double p, double d, double s){
	return ((-7.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 35.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 17.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 218.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 459.375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 98.4375*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 10.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 350.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1837.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 1575.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 1.25*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 175.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2296.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 5512.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 1353.515625*(a*a)*(d*d)*(p*p*p*p) - 25.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 918.75*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 5512.5*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 5414.0625*a*b*(d*d)*(p*p*p*p) + 0.625*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 91.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1378.125*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 3789.84375*(b*b)*(d*d)*(p*p*p*p) + 1055.7421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU854(double a, double b, double p, double d, double s){
	return (d*(4.375*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 17.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 131.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 131.25*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 17.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 328.125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 918.75*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 246.09375*(a*a*a)*(d*d)*(p*p*p) - 5.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 262.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 1837.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 1968.75*(a*a)*b*(d*d)*(p*p*p) + 0.3125*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 65.625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 1148.4375*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 3445.3125*a*(b*b)*(d*d)*(p*p*p) + 1015.13671875*a*(p*p*p*p) - 3.75*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 183.75*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 1378.125*(b*b*b)*(d*d)*(p*p*p) - 1624.21875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU855(double a, double b, double p, double d, double s){
	return ((-1.75*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 2.625*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 10.9375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 45.9375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 14.765625*(a*a*a*a)*(d*d*d*d)*(p*p) - 17.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 183.75*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 236.25*(a*a*a)*b*(d*d*d*d)*(p*p) + 8.75*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 229.6875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 826.875*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 270.703125*(a*a)*(d*d)*(p*p*p) - 1.25*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 91.875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 826.875*a*(b*b*b)*(d*d*d*d)*(p*p) - 1082.8125*a*b*(d*d)*(p*p*p) + 0.03125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 9.1875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 206.71875*(b*b*b*b)*(d*d*d*d)*(p*p) + 757.96875*(b*b)*(d*d)*(p*p*p) + 263.935546875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU856(double a, double b, double p, double d, double s){
	return (d*(0.4375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.21875*(a*a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 8.75*(a*a*a*a)*b*(d*d*d*d)*p + 10.9375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 61.25*(a*a*a)*(b*b)*(d*d*d*d)*p + 24.609375*(a*a*a)*(d*d)*(p*p) - 8.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 122.5*(a*a)*(b*b*b)*(d*d*d*d)*p - 196.875*(a*a)*b*(d*d)*(p*p) + 2.1875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 76.5625*a*(b*b*b*b)*(d*d*d*d)*p + 344.53125*a*(b*b)*(d*d)*(p*p) + 135.3515625*a*(p*p*p) - 0.125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 12.25*(b*b*b*b*b)*(d*d*d*d)*p - 137.8125*(b*b*b)*(d*d)*(p*p) - 216.5625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU857(double a, double b, double p, double d, double s){
	return ((-0.0625*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.09375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.703125*(a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 11.25*(a*a*a)*b*(d*d*d*d)*p + 5.46875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 39.375*(a*a)*(b*b)*(d*d*d*d)*p + 19.3359375*(a*a)*(d*d)*(p*p) - 2.1875*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 39.375*a*(b*b*b)*(d*d*d*d)*p - 77.34375*a*b*(d*d)*(p*p) + 0.21875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 9.84375*(b*b*b*b)*(d*d*d*d)*p + 54.140625*(b*b)*(d*d)*(p*p) + 25.13671875*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU858(double a, double b, double p, double d, double s){
	return (d*(0.00390625*(a*a*a*a*a)*(d*d*d*d) - 0.15625*(a*a*a*a)*b*(d*d*d*d) + 1.09375*(a*a*a)*(b*b)*(d*d*d*d) + 0.87890625*(a*a*a)*(d*d)*p - 2.1875*(a*a)*(b*b*b)*(d*d*d*d) - 7.03125*(a*a)*b*(d*d)*p + 1.3671875*a*(b*b*b*b)*(d*d*d*d) + 12.3046875*a*(b*b)*(d*d)*p + 7.2509765625*a*(p*p) - 0.21875*(b*b*b*b*b)*(d*d*d*d) - 4.921875*(b*b*b)*(d*d)*p - 11.6015625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU859(double a, double b, double p, double d, double s){
	return ((0.009765625*(a*a*a*a)*(d*d*d*d) - 0.15625*(a*a*a)*b*(d*d*d*d) + 0.546875*(a*a)*(b*b)*(d*d*d*d) + 0.537109375*(a*a)*(d*d)*p - 0.546875*a*(b*b*b)*(d*d*d*d) - 2.1484375*a*b*(d*d)*p + 0.13671875*(b*b*b*b)*(d*d*d*d) + 1.50390625*(b*b)*(d*d)*p + 1.04736328125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8510(double a, double b, double p, double d, double s){
	return (d*(0.009765625*(a*a*a)*(d*d) - 0.078125*(a*a)*b*(d*d) + 0.13671875*a*(b*b)*(d*d) + 0.1611328125*a*p - 0.0546875*(b*b*b)*(d*d) - 0.2578125*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8511(double a, double b, double p, double d, double s){
	return ((0.0048828125*(a*a)*(d*d) - 0.01953125*a*b*(d*d) + 0.013671875*(b*b)*(d*d) + 0.01904296875*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8512(double a, double b, double p, double d, double s){
	return (d*(0.001220703125*a - 0.001953125*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8513(double a, double b, double p, double d, double s){
	return (0.0001220703125/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU860(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 6.5625*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 24.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 252.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 630.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 315.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 7.5*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 315.0*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 1968.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 2756.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 442.96875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 120.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 2100.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 7350.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 4725.0*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 11.25*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 787.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 6890.625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 12403.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 2436.328125*(a*a)*(d*d)*(p*p*p*p*p*p) - 90.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 2205.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 9922.5*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 7796.25*a*b*(d*d)*(p*p*p*p*p*p) + 1.875*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 183.75*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 2067.1875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 4547.8125*(b*b)*(d*d)*(p*p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU861(double a, double b, double p, double d, double s){
	return (d*(-4.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 42.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 52.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 3.0*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 126.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 787.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1102.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 177.1875*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 90.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 1575.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 5512.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 3543.75*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 15.0*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1050.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 9187.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 16537.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 3248.4375*(a*a*a)*(d*d)*(p*p*p*p*p) - 225.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 5512.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 24806.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 19490.625*(a*a)*b*(d*d)*(p*p*p*p*p) + 11.25*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1102.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 12403.125*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 27286.875*a*(b*b)*(d*d)*(p*p*p*p*p) + 6334.453125*a*(p*p*p*p*p*p) - 52.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1653.75*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 9095.625*(b*b*b)*(d*d)*(p*p*p*p*p) - 8445.9375*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU862(double a, double b, double p, double d, double s){
	return ((7.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 52.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 78.75*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 12.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 252.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 945.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 630.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 3.75*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 315.0*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 2953.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 5512.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 1107.421875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 120.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 3150.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 14700.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 11812.5*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 11.25*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1181.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 13781.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 31007.8125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 7308.984375*(a*a)*(d*d)*(p*p*p*p*p) - 135.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 4410.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 24806.25*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 23388.75*a*b*(d*d)*(p*p*p*p*p) + 2.8125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 367.5*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 5167.96875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 13643.4375*(b*b)*(d*d)*(p*p*p*p*p) + 3695.09765625*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU863(double a, double b, double p, double d, double s){
	return (d*(-7.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 35.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 21.0 *(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 262.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 551.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 118.125*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 15.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 525.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 2756.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 2362.5*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 2.5*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 350.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4593.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 11025.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 2707.03125*(a*a*a)*(d*d)*(p*p*p*p) - 75.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 2756.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 16537.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 16242.1875*(a*a)*b*(d*d)*(p*p*p*p) + 3.75*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 551.25*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 8268.75*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 22739.0625*a*(b*b)*(d*d)*(p*p*p*p) + 6334.453125*a*(p*p*p*p*p) - 26.25*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1102.5*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 7579.6875*(b*b*b)*(d*d)*(p*p*p*p) - 8445.9375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU864(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 21.0 *(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 157.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 157.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 26.25*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 492.1875*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1378.125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 369.140625*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 10.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 525.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 3675.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 3937.5*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 0.9375*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 196.875*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 3445.3125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 10335.9375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 3045.41015625*(a*a)*(d*d)*(p*p*p*p) - 22.5*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1102.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 8268.75*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 9745.3125*a*b*(d*d)*(p*p*p*p) + 0.46875*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 91.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1722.65625*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 5684.765625*(b*b)*(d*d)*(p*p*p*p) + 1847.548828125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU865(double a, double b, double p, double d, double s){
	return (d*(-1.75*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 2.625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 13.125*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 55.125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 17.71875*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 26.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 275.625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 354.375*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 17.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 459.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1653.75*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 541.40625*(a*a*a)*(d*d)*(p*p*p) - 3.75*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 275.625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 2480.625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 3248.4375*(a*a)*b*(d*d)*(p*p*p) + 0.1875*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 55.125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 1240.3125*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 4547.8125*a*(b*b)*(d*d)*(p*p*p) + 1583.61328125*a*(p*p*p*p) - 2.625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 165.375*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 1515.9375*(b*b*b)*(d*d)*(p*p*p) - 2111.484375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU866(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.21875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 5.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 10.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 16.40625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 91.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 36.9140625*(a*a*a*a)*(d*d*d*d)*(p*p) - 17.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 245.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 393.75*(a*a*a)*b*(d*d*d*d)*(p*p) + 6.5625*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 229.6875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1033.59375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 406.0546875*(a*a)*(d*d)*(p*p*p) - 0.75*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 73.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 826.875*a*(b*b*b)*(d*d*d*d)*(p*p) - 1299.375*a*b*(d*d)*(p*p*p) + 0.015625*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 6.125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 172.265625*(b*b*b*b)*(d*d*d*d)*(p*p) + 757.96875*(b*b)*(d*d)*(p*p*p) + 307.9248046875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU867(double a, double b, double p, double d, double s){
	return (d*(-0.0625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.3125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.84375*(a*a*a*a*a)*(d*d*d*d)*p - 6.5625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 16.875*(a*a*a*a)*b*(d*d*d*d)*p + 10.9375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 78.75*(a*a*a)*(b*b)*(d*d*d*d)*p + 38.671875*(a*a*a)*(d*d)*(p*p) - 6.5625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 118.125*(a*a)*(b*b*b)*(d*d*d*d)*p - 232.03125*(a*a)*b*(d*d)*(p*p) + 1.3125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 59.0625*a*(b*b*b*b)*(d*d*d*d)*p + 324.84375*a*(b*b)*(d*d)*(p*p) + 150.8203125*a*(p*p*p) - 0.0625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 7.875*(b*b*b*b*b)*(d*d*d*d)*p - 108.28125*(b*b*b)*(d*d)*(p*p) - 201.09375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU868(double a, double b, double p, double d, double s){
	return ((0.00390625*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.1875*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.640625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.318359375*(a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 14.0625*(a*a*a)*b*(d*d*d*d)*p + 4.1015625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 36.9140625*(a*a)*(b*b)*(d*d*d*d)*p + 21.7529296875*(a*a)*(d*d)*(p*p) - 1.3125*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 29.53125*a*(b*b*b)*(d*d*d*d)*p - 69.609375*a*b*(d*d)*(p*p) + 0.109375*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 6.15234375*(b*b*b*b)*(d*d*d*d)*p + 40.60546875*(b*b)*(d*d)*(p*p) + 21.99462890625*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU869(double a, double b, double p, double d, double s){
	return (d*(0.01171875*(a*a*a*a*a)*(d*d*d*d) - 0.234375*(a*a*a*a)*b*(d*d*d*d) + 1.09375*(a*a*a)*(b*b)*(d*d*d*d) + 1.07421875*(a*a*a)*(d*d)*p - 1.640625*(a*a)*(b*b*b)*(d*d*d*d) - 6.4453125*(a*a)*b*(d*d)*p + 0.8203125*a*(b*b*b*b)*(d*d*d*d) + 9.0234375*a*(b*b)*(d*d)*p + 6.2841796875*a*(p*p) - 0.109375*(b*b*b*b*b)*(d*d*d*d) - 3.0078125*(b*b*b)*(d*d)*p - 8.37890625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8610(double a, double b, double p, double d, double s){
	return ((0.0146484375*(a*a*a*a)*(d*d*d*d) - 0.15625*(a*a*a)*b*(d*d*d*d) + 0.41015625*(a*a)*(b*b)*(d*d*d*d) + 0.4833984375*(a*a)*(d*d)*p - 0.328125*a*(b*b*b)*(d*d*d*d) - 1.546875*a*b*(d*d)*p + 0.068359375*(b*b*b*b)*(d*d*d*d) + 0.90234375*(b*b)*(d*d)*p + 0.733154296875*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8611(double a, double b, double p, double d, double s){
	return (d*(0.009765625*(a*a*a)*(d*d) - 0.05859375*(a*a)*b*(d*d) + 0.08203125*a*(b*b)*(d*d) + 0.1142578125*a*p - 0.02734375*(b*b*b)*(d*d) - 0.15234375*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8612(double a, double b, double p, double d, double s){
	return ((0.003662109375*(a*a)*(d*d) - 0.01171875*a*b*(d*d) + 0.0068359375*(b*b)*(d*d) + 0.0111083984375*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8613(double a, double b, double p, double d, double s){
	return (d*(0.000732421875*a - 0.0009765625*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8614(double a, double b, double p, double d, double s){
	return (6.103515625e-5/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU870(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 6.5625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 28.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 294.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 735.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 367.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 10.5*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 441.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 2756.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 3858.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 620.15625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 210.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 3675.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 12862.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 8268.75*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 26.25*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 1837.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 16078.125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 28940.625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 5684.765625*(a*a*a)*(d*d)*(p*p*p*p*p*p) - 315.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 7717.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 34728.75*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 27286.875*(a*a)*b*(d*d)*(p*p*p*p*p*p) + 13.125*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 1286.25*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 14470.3125*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 31834.6875*a*(b*b)*(d*d)*(p*p*p*p*p*p) + 7390.1953125*a*(p*p*p*p*p*p*p) - 52.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 1653.75*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 9095.625*(b*b*b)*(d*d)*(p*p*p*p*p*p) - 8445.9375*b*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU871(double a, double b, double p, double d, double s){
	return ((-4.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) - 42.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 52.5*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 3.5*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 147.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 918.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 1286.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 206.71875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 126.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 2205.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 7717.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 4961.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 26.25*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 1837.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 16078.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 28940.625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 5684.765625*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 525.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 12862.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 57881.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 45478.125*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 39.375*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 3858.75*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 43410.9375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 95504.0625*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 22170.5859375*(a*a)*(d*d)*(p*p*p*p*p*p) - 367.5*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 11576.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 63669.375*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 59121.5625*a*b*(d*d)*(p*p*p*p*p*p) + 6.5625*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 826.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 11369.53125*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 29560.78125*(b*b)*(d*d)*(p*p*p*p*p*p) + 7918.06640625*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU872(double a, double b, double p, double d, double s){
	return (d*(7.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 52.5*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 78.75*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 14.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 294.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 1102.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 735.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 5.25*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 441.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 4134.375*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 7717.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 1550.390625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 210.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 5512.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 25725.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 20671.875*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 26.25*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 2756.25*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 32156.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 72351.5625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 17054.296875*(a*a*a)*(d*d)*(p*p*p*p*p) - 472.5*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 15435.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 86821.875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 81860.625*(a*a)*b*(d*d)*(p*p*p*p*p) + 19.6875*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 2572.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 36175.78125*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 95504.0625*a*(b*b)*(d*d)*(p*p*p*p*p) + 25865.68359375*a*(p*p*p*p*p*p) - 105.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 4134.375*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 27286.875*(b*b*b)*(d*d)*(p*p*p*p*p) - 29560.78125*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU873(double a, double b, double p, double d, double s){
	return ((-7.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 35.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 24.5*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 306.25*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 643.125*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 137.8125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 21.0 *(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 735.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 3858.75*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 3307.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 4.375*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 612.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 8039.0625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 19293.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 4737.3046875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 175.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 6431.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 38587.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 37898.4375*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 13.125*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1929.375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 28940.625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 79586.71875*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 22170.5859375*(a*a)*(d*d)*(p*p*p*p*p) - 183.75*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 7717.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 53057.8125*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 59121.5625*a*b*(d*d)*(p*p*p*p*p) + 3.28125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 551.25*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 9474.609375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 29560.78125*(b*b)*(d*d)*(p*p*p*p*p) + 9237.744140625*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU874(double a, double b, double p, double d, double s){
	return (d*(4.375*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 24.5*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 183.75*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 183.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 36.75*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 689.0625*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1929.375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 516.796875*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 17.5*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 918.75*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 6431.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 6890.625*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 2.1875*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 459.375*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 8039.0625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 24117.1875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 7105.95703125*(a*a*a)*(d*d)*(p*p*p*p) - 78.75*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 3858.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 28940.625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 34108.59375*(a*a)*b*(d*d)*(p*p*p*p) + 3.28125*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 643.125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 12058.59375*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 39793.359375*a*(b*b)*(d*d)*(p*p*p*p) + 12932.841796875*a*(p*p*p*p*p) - 26.25*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1378.125*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 11369.53125*(b*b*b)*(d*d)*(p*p*p*p) - 14780.390625*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU875(double a, double b, double p, double d, double s){
	return ((-1.75*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 2.625*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 15.3125*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 64.3125*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 20.671875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 36.75*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 385.875*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 496.125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 30.625*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 803.90625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2894.0625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 947.4609375*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 8.75*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 643.125*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 5788.125*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 7579.6875*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 0.65625*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 192.9375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4341.09375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 15917.34375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 5542.646484375*(a*a)*(d*d)*(p*p*p*p) - 18.375*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1157.625*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 10611.5625*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 14780.390625*a*b*(d*d)*(p*p*p*p) + 0.328125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 82.6875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1894.921875*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 7390.1953125*(b*b)*(d*d)*(p*p*p*p) + 2771.3232421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU876(double a, double b, double p, double d, double s){
	return (d*(0.4375*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.21875*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 6.125*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 12.25*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 22.96875*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 128.625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 51.6796875*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 30.625*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 428.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 689.0625*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 15.3125*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 535.9375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 2411.71875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 947.4609375*(a*a*a)*(d*d)*(p*p*p) - 2.625*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 257.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 2894.0625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 4547.8125*(a*a)*b*(d*d)*(p*p*p) + 0.109375*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 42.875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 1205.859375*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 5305.78125*a*(b*b)*(d*d)*(p*p*p) + 2155.4736328125*a*(p*p*p*p) - 1.75*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 137.8125*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 1515.9375*(b*b*b)*(d*d)*(p*p*p) - 2463.3984375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU877(double a, double b, double p, double d, double s){
	return ((-0.0625*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 1.53125*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.984375*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 9.1875*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 23.625*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 19.140625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 137.8125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 67.67578125*(a*a*a*a)*(d*d*d*d)*(p*p) - 15.3125*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 275.625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 541.40625*(a*a*a)*b*(d*d*d*d)*(p*p) + 4.59375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 206.71875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1136.953125*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 527.87109375*(a*a)*(d*d)*(p*p*p) - 0.4375*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 55.125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 757.96875*a*(b*b*b)*(d*d*d*d)*(p*p) - 1407.65625*a*b*(d*d)*(p*p*p) + 0.0078125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 3.9375*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 135.3515625*(b*b*b*b)*(d*d*d*d)*(p*p) + 703.828125*(b*b)*(d*d)*(p*p*p) + 329.91943359375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU878(double a, double b, double p, double d, double s){
	return (d*(0.00390625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.21875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.296875*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.845703125*(a*a*a*a*a)*(d*d*d*d)*p - 7.65625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 24.609375*(a*a*a*a)*b*(d*d*d*d)*p + 9.5703125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 86.1328125*(a*a*a)*(b*b)*(d*d*d*d)*p + 50.7568359375*(a*a*a)*(d*d)*(p*p) - 4.59375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 103.359375*(a*a)*(b*b*b)*(d*d*d*d)*p - 243.6328125*(a*a)*b*(d*d)*(p*p) + 0.765625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 43.06640625*a*(b*b*b*b)*(d*d*d*d)*p + 284.23828125*a*(b*b)*(d*d)*(p*p) + 153.96240234375*a*(p*p*p) - 0.03125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 4.921875*(b*b*b*b*b)*(d*d*d*d)*p - 81.2109375*(b*b*b)*(d*d)*(p*p) - 175.95703125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU879(double a, double b, double p, double d, double s){
	return ((0.013671875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.328125*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.9140625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.8798828125*(a*a*a*a)*(d*d*d*d)*p - 3.828125*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 15.0390625*(a*a*a)*b*(d*d*d*d)*p + 2.87109375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 31.58203125*(a*a)*(b*b)*(d*d*d*d)*p + 21.99462890625*(a*a)*(d*d)*(p*p) - 0.765625*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 21.0546875*a*(b*b*b)*(d*d*d*d)*p - 58.65234375*a*b*(d*d)*(p*p) + 0.0546875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 3.759765625*(b*b*b*b)*(d*d*d*d)*p + 29.326171875*(b*b)*(d*d)*(p*p) + 18.328857421875*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8710(double a, double b, double p, double d, double s){
	return (d*(0.0205078125*(a*a*a*a*a)*(d*d*d*d) - 0.2734375*(a*a*a*a)*b*(d*d*d*d) + 0.95703125*(a*a*a)*(b*b)*(d*d*d*d) + 1.1279296875*(a*a*a)*(d*d)*p - 1.1484375*(a*a)*(b*b*b)*(d*d*d*d) - 5.4140625*(a*a)*b*(d*d)*p + 0.478515625*a*(b*b*b*b)*(d*d*d*d) + 6.31640625*a*(b*b)*(d*d)*p + 5.132080078125*a*(p*p) - 0.0546875*(b*b*b*b*b)*(d*d*d*d) - 1.8046875*(b*b*b)*(d*d)*p - 5.865234375*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8711(double a, double b, double p, double d, double s){
	return ((0.01708984375*(a*a*a*a)*(d*d*d*d) - 0.13671875*(a*a*a)*b*(d*d*d*d) + 0.287109375*(a*a)*(b*b)*(d*d*d*d) + 0.39990234375*(a*a)*(d*d)*p - 0.19140625*a*(b*b*b)*(d*d*d*d) - 1.06640625*a*b*(d*d)*p + 0.0341796875*(b*b*b*b)*(d*d*d*d) + 0.533203125*(b*b)*(d*d)*p + 0.4998779296875*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8712(double a, double b, double p, double d, double s){
	return (d*(0.008544921875*(a*a*a)*(d*d) - 0.041015625*(a*a)*b*(d*d) + 0.0478515625*a*(b*b)*(d*d) + 0.0777587890625*a*p - 0.013671875*(b*b*b)*(d*d) - 0.0888671875*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8713(double a, double b, double p, double d, double s){
	return ((0.0025634765625*(a*a)*(d*d) - 0.0068359375*a*b*(d*d) + 0.00341796875*(b*b)*(d*d) + 0.00640869140625*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8714(double a, double b, double p, double d, double s){
	return (d*(0.00042724609375*a - 0.00048828125*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8715(double a, double b, double p, double d, double s){
	return (3.0517578125e-5/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU880(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) + 6.5625*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p*p*p) - 32.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*p - 336.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*(p*p) - 840.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) - 420.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p*p) + 14.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*p + 588.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*(p*p) + 3675.0*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) + 5145.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) + 826.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p*p) - 336.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*(p*p) - 5880.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) - 20580.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) - 13230.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p*p) + 52.5*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*(p*p) + 3675.0*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) + 32156.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) + 57881.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p*p) + 11369.53125*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p*p) - 840.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) - 20580.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) - 92610.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p*p) - 72765.0*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p*p) + 52.5*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) + 5145.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) + 57881.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p*p) + 127338.75*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p*p) + 29560.78125*(a*a)*(d*d)*(p*p*p*p*p*p*p) - 420.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) - 13230.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p*p) - 72765.0*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p*p) - 67567.5*a*b*(d*d)*(p*p*p*p*p*p*p) + 6.5625*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) + 826.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p*p) + 11369.53125*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p*p) + 29560.78125*(b*b)*(d*d)*(p*p*p*p*p*p*p) + 7918.06640625*(p*p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU881(double a, double b, double p, double d, double s){
	return (d*(-4.0*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) - 42.0*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 52.5*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 4.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 168.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 1050.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 1470.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 236.25*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 168.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 2940.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 10290.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 6615.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 42.0*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 2940.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 25725.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 46305.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 9095.625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 1050.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 25725.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 115762.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 90956.25*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 105.0*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 10290.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 115762.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 254677.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 59121.5625*(a*a*a)*(d*d)*(p*p*p*p*p*p) - 1470.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 46305.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 254677.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 236486.25*(a*a)*b*(d*d)*(p*p*p*p*p*p) + 52.5*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 6615.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 90956.25*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 236486.25*a*(b*b)*(d*d)*(p*p*p*p*p*p) + 63344.53125*a*(p*p*p*p*p*p*p) - 236.25*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 9095.625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 59121.5625*(b*b*b)*(d*d)*(p*p*p*p*p*p) - 63344.53125*b*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU882(double a, double b, double p, double d, double s){
	return ((7.0*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 52.5*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 78.75*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p*p) - 16.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) - 336.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 1260.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 840.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 7.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 588.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 5512.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 10290.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 2067.1875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 336.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 8820.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 41160.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 33075.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 52.5*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 5512.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 64312.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 144703.125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 34108.59375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 1260.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 41160.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 231525.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 218295.0*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 78.75*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 10290.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 144703.125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 382016.25*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 103462.734375*(a*a)*(d*d)*(p*p*p*p*p*p) - 840.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 33075.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 218295.0*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 236486.25*a*b*(d*d)*(p*p*p*p*p*p) + 13.125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 2067.1875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 34108.59375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 103462.734375*(b*b)*(d*d)*(p*p*p*p*p*p) + 31672.265625*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU883(double a, double b, double p, double d, double s){
	return (d*(-7.0*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 35.0*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 28.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 350.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 735.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 157.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 28.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 980.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 5145.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 4410.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 7.0*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 980.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 12862.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 30870.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 7579.6875*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 350.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 12862.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 77175.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 75796.875*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 35.0*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 5145.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 77175.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 212231.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 59121.5625*(a*a*a)*(d*d)*(p*p*p*p*p) - 735.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 30870.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 212231.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 236486.25*(a*a)*b*(d*d)*(p*p*p*p*p) + 26.25*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 4410.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 75796.875*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 236486.25*a*(b*b)*(d*d)*(p*p*p*p*p) + 73901.953125*a*(p*p*p*p*p*p) - 157.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 7579.6875*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 59121.5625*(b*b*b)*(d*d)*(p*p*p*p*p) - 73901.953125*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU884(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p) - 28.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 210.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 210.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 49.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 918.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 2572.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 689.0625*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 28.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 1470.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 10290.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 11025.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 4.375*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 918.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 16078.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 48234.375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 14211.9140625*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 210.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 10290.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 77175.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 90956.25*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 13.125*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 2572.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 48234.375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 159173.4375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 51731.3671875*(a*a)*(d*d)*(p*p*p*p*p) - 210.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 11025.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 90956.25*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 118243.125*a*b*(d*d)*(p*p*p*p*p) + 3.28125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 689.0625*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 14211.9140625*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 51731.3671875*(b*b)*(d*d)*(p*p*p*p*p) + 18475.48828125*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU885(double a, double b, double p, double d, double s){
	return (d*(-1.75*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 2.625*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 17.5*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 73.5*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 23.625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 49.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 514.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 661.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 49.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 1286.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4630.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1515.9375*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 17.5*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 1286.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 11576.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 15159.375*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 1.75*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 514.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 11576.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 42446.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 14780.390625*(a*a*a)*(d*d)*(p*p*p*p) - 73.5*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 4630.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 42446.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 59121.5625*(a*a)*b*(d*d)*(p*p*p*p) + 2.625*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 661.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 15159.375*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 59121.5625*a*(b*b)*(d*d)*(p*p*p*p) + 22170.5859375*a*(p*p*p*p*p) - 23.625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1515.9375*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 14780.390625*(b*b*b)*(d*d)*(p*p*p*p) - 22170.5859375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU886(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.21875*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 7.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 14.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 30.625*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 171.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 68.90625*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 49.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 686.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 1102.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 30.625*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 1071.875*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4823.4375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1894.921875*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 7.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 686.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 7717.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 12127.5*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 0.4375*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 171.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4823.4375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 21223.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 8621.89453125*(a*a)*(d*d)*(p*p*p*p) - 14.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1102.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 12127.5*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 19707.1875*a*b*(d*d)*(p*p*p*p) + 0.21875*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 68.90625*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1894.921875*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 8621.89453125*(b*b)*(d*d)*(p*p*p*p) + 3695.09765625*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU887(double a, double b, double p, double d, double s){
	return (d*(-0.0625*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 1.75*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 1.125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 12.25*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 31.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 30.625*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 220.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 108.28125*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 30.625*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 551.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 1082.8125*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 12.25*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 551.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 3031.875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 1407.65625*(a*a*a)*(d*d)*(p*p*p) - 1.75*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 220.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 3031.875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 5630.625*(a*a)*b*(d*d)*(p*p*p) + 0.0625*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 31.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 1082.8125*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 5630.625*a*(b*b)*(d*d)*(p*p*p) + 2639.35546875*a*(p*p*p*p) - 1.125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 108.28125*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 1407.65625*(b*b*b)*(d*d)*(p*p*p) - 2639.35546875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU888(double a, double b, double p, double d, double s){
	return ((0.00390625*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 0.25*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 3.0625*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 2.4609375*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 12.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 39.375*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 19.140625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 172.265625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 101.513671875*(a*a*a*a)*(d*d*d*d)*(p*p) - 12.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 275.625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 649.6875*(a*a*a)*b*(d*d*d*d)*(p*p) + 3.0625*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 172.265625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1136.953125*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 615.849609375*(a*a)*(d*d)*(p*p*p) - 0.25*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 39.375*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 649.6875*a*(b*b*b)*(d*d*d*d)*(p*p) - 1407.65625*a*b*(d*d)*(p*p*p) + 0.00390625*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 2.4609375*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 101.513671875*(b*b*b*b)*(d*d*d*d)*(p*p) + 615.849609375*(b*b)*(d*d)*(p*p*p) + 329.91943359375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU889(double a, double b, double p, double d, double s){
	return (d*(0.015625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.4375*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.0625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.0078125*(a*a*a*a*a)*(d*d*d*d)*p - 7.65625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 30.078125*(a*a*a*a)*b*(d*d*d*d)*p + 7.65625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 84.21875*(a*a*a)*(b*b)*(d*d*d*d)*p + 58.65234375*(a*a*a)*(d*d)*(p*p) - 3.0625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 84.21875*(a*a)*(b*b*b)*(d*d*d*d)*p - 234.609375*(a*a)*b*(d*d)*(p*p) + 0.4375*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 30.078125*a*(b*b*b*b)*(d*d*d*d)*p + 234.609375*a*(b*b)*(d*d)*(p*p) + 146.630859375*a*(p*p*p) - 0.015625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 3.0078125*(b*b*b*b*b)*(d*d*d*d)*p - 58.65234375*(b*b*b)*(d*d)*(p*p) - 146.630859375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8810(double a, double b, double p, double d, double s){
	return ((0.02734375*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.4375*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.9140625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 2.255859375*(a*a*a*a)*(d*d*d*d)*p - 3.0625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 14.4375*(a*a*a)*b*(d*d*d*d)*p + 1.9140625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 25.265625*(a*a)*(b*b)*(d*d*d*d)*p + 20.5283203125*(a*a)*(d*d)*(p*p) - 0.4375*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 14.4375*a*(b*b*b)*(d*d*d*d)*p - 46.921875*a*b*(d*d)*(p*p) + 0.02734375*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 2.255859375*(b*b*b*b)*(d*d*d*d)*p + 20.5283203125*(b*b)*(d*d)*(p*p) + 14.6630859375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8811(double a, double b, double p, double d, double s){
	return (d*(0.02734375*(a*a*a*a*a)*(d*d*d*d) - 0.2734375*(a*a*a*a)*b*(d*d*d*d) + 0.765625*(a*a*a)*(b*b)*(d*d*d*d) + 1.06640625*(a*a*a)*(d*d)*p - 0.765625*(a*a)*(b*b*b)*(d*d*d*d) - 4.265625*(a*a)*b*(d*d)*p + 0.2734375*a*(b*b*b*b)*(d*d*d*d) + 4.265625*a*(b*b)*(d*d)*p + 3.9990234375*a*(p*p) - 0.02734375*(b*b*b*b*b)*(d*d*d*d) - 1.06640625*(b*b*b)*(d*d)*p - 3.9990234375*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8812(double a, double b, double p, double d, double s){
	return ((0.01708984375*(a*a*a*a)*(d*d*d*d) - 0.109375*(a*a*a)*b*(d*d*d*d) + 0.19140625*(a*a)*(b*b)*(d*d*d*d) + 0.31103515625*(a*a)*(d*d)*p - 0.109375*a*(b*b*b)*(d*d*d*d) - 0.7109375*a*b*(d*d)*p + 0.01708984375*(b*b*b*b)*(d*d*d*d) + 0.31103515625*(b*b)*(d*d)*p + 0.333251953125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8813(double a, double b, double p, double d, double s){
	return (d*(0.0068359375*(a*a*a)*(d*d) - 0.02734375*(a*a)*b*(d*d) + 0.02734375*a*(b*b)*(d*d) + 0.05126953125*a*p - 0.0068359375*(b*b*b)*(d*d) - 0.05126953125*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8814(double a, double b, double p, double d, double s){
	return ((0.001708984375*(a*a)*(d*d) - 0.00390625*a*b*(d*d) + 0.001708984375*(b*b)*(d*d) + 0.003662109375*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8815(double a, double b, double p, double d, double s){
	return (0.000244140625*d*(a - b)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline double MD_Et_GPU8816(double a, double b, double p, double d, double s){
	return (1.52587890625e-5/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

inline real_t callFunction(int index, real_t a, real_t b, real_t c, real_t d, real_t e) {
    switch(index) {
    case 0: return MD_Et_GPU000(a, b, c, d, e);
    case 1: return MD_Et_GPU010(a, b, c, d, e);
    case 2: return MD_Et_GPU011(a, b, c, d, e);
    case 3: return MD_Et_GPU020(a, b, c, d, e);
    case 4: return MD_Et_GPU021(a, b, c, d, e);
    case 5: return MD_Et_GPU022(a, b, c, d, e);
    case 6: return MD_Et_GPU030(a, b, c, d, e);
    case 7: return MD_Et_GPU031(a, b, c, d, e);
    case 8: return MD_Et_GPU032(a, b, c, d, e);
    case 9: return MD_Et_GPU033(a, b, c, d, e);
    case 10: return MD_Et_GPU040(a, b, c, d, e);
    case 11: return MD_Et_GPU041(a, b, c, d, e);
    case 12: return MD_Et_GPU042(a, b, c, d, e);
    case 13: return MD_Et_GPU043(a, b, c, d, e);
    case 14: return MD_Et_GPU044(a, b, c, d, e);
    case 15: return MD_Et_GPU050(a, b, c, d, e);
    case 16: return MD_Et_GPU051(a, b, c, d, e);
    case 17: return MD_Et_GPU052(a, b, c, d, e);
    case 18: return MD_Et_GPU053(a, b, c, d, e);
    case 19: return MD_Et_GPU054(a, b, c, d, e);
    case 20: return MD_Et_GPU055(a, b, c, d, e);
    case 21: return MD_Et_GPU060(a, b, c, d, e);
    case 22: return MD_Et_GPU061(a, b, c, d, e);
    case 23: return MD_Et_GPU062(a, b, c, d, e);
    case 24: return MD_Et_GPU063(a, b, c, d, e);
    case 25: return MD_Et_GPU064(a, b, c, d, e);
    case 26: return MD_Et_GPU065(a, b, c, d, e);
    case 27: return MD_Et_GPU066(a, b, c, d, e);
    case 28: return MD_Et_GPU070(a, b, c, d, e);
    case 29: return MD_Et_GPU071(a, b, c, d, e);
    case 30: return MD_Et_GPU072(a, b, c, d, e);
    case 31: return MD_Et_GPU073(a, b, c, d, e);
    case 32: return MD_Et_GPU074(a, b, c, d, e);
    case 33: return MD_Et_GPU075(a, b, c, d, e);
    case 34: return MD_Et_GPU076(a, b, c, d, e);
    case 35: return MD_Et_GPU077(a, b, c, d, e);
    case 36: return MD_Et_GPU080(a, b, c, d, e);
    case 37: return MD_Et_GPU081(a, b, c, d, e);
    case 38: return MD_Et_GPU082(a, b, c, d, e);
    case 39: return MD_Et_GPU083(a, b, c, d, e);
    case 40: return MD_Et_GPU084(a, b, c, d, e);
    case 41: return MD_Et_GPU085(a, b, c, d, e);
    case 42: return MD_Et_GPU086(a, b, c, d, e);
    case 43: return MD_Et_GPU087(a, b, c, d, e);
    case 44: return MD_Et_GPU088(a, b, c, d, e);
    case 45: return MD_Et_GPU100(a, b, c, d, e);
    case 46: return MD_Et_GPU101(a, b, c, d, e);
    case 47: return MD_Et_GPU110(a, b, c, d, e);
    case 48: return MD_Et_GPU111(a, b, c, d, e);
    case 49: return MD_Et_GPU112(a, b, c, d, e);
    case 50: return MD_Et_GPU120(a, b, c, d, e);
    case 51: return MD_Et_GPU121(a, b, c, d, e);
    case 52: return MD_Et_GPU122(a, b, c, d, e);
    case 53: return MD_Et_GPU123(a, b, c, d, e);
    case 54: return MD_Et_GPU130(a, b, c, d, e);
    case 55: return MD_Et_GPU131(a, b, c, d, e);
    case 56: return MD_Et_GPU132(a, b, c, d, e);
    case 57: return MD_Et_GPU133(a, b, c, d, e);
    case 58: return MD_Et_GPU134(a, b, c, d, e);
    case 59: return MD_Et_GPU140(a, b, c, d, e);
    case 60: return MD_Et_GPU141(a, b, c, d, e);
    case 61: return MD_Et_GPU142(a, b, c, d, e);
    case 62: return MD_Et_GPU143(a, b, c, d, e);
    case 63: return MD_Et_GPU144(a, b, c, d, e);
    case 64: return MD_Et_GPU145(a, b, c, d, e);
    case 65: return MD_Et_GPU150(a, b, c, d, e);
    case 66: return MD_Et_GPU151(a, b, c, d, e);
    case 67: return MD_Et_GPU152(a, b, c, d, e);
    case 68: return MD_Et_GPU153(a, b, c, d, e);
    case 69: return MD_Et_GPU154(a, b, c, d, e);
    case 70: return MD_Et_GPU155(a, b, c, d, e);
    case 71: return MD_Et_GPU156(a, b, c, d, e);
    case 72: return MD_Et_GPU160(a, b, c, d, e);
    case 73: return MD_Et_GPU161(a, b, c, d, e);
    case 74: return MD_Et_GPU162(a, b, c, d, e);
    case 75: return MD_Et_GPU163(a, b, c, d, e);
    case 76: return MD_Et_GPU164(a, b, c, d, e);
    case 77: return MD_Et_GPU165(a, b, c, d, e);
    case 78: return MD_Et_GPU166(a, b, c, d, e);
    case 79: return MD_Et_GPU167(a, b, c, d, e);
    case 80: return MD_Et_GPU170(a, b, c, d, e);
    case 81: return MD_Et_GPU171(a, b, c, d, e);
    case 82: return MD_Et_GPU172(a, b, c, d, e);
    case 83: return MD_Et_GPU173(a, b, c, d, e);
    case 84: return MD_Et_GPU174(a, b, c, d, e);
    case 85: return MD_Et_GPU175(a, b, c, d, e);
    case 86: return MD_Et_GPU176(a, b, c, d, e);
    case 87: return MD_Et_GPU177(a, b, c, d, e);
    case 88: return MD_Et_GPU178(a, b, c, d, e);
    case 89: return MD_Et_GPU180(a, b, c, d, e);
    case 90: return MD_Et_GPU181(a, b, c, d, e);
    case 91: return MD_Et_GPU182(a, b, c, d, e);
    case 92: return MD_Et_GPU183(a, b, c, d, e);
    case 93: return MD_Et_GPU184(a, b, c, d, e);
    case 94: return MD_Et_GPU185(a, b, c, d, e);
    case 95: return MD_Et_GPU186(a, b, c, d, e);
    case 96: return MD_Et_GPU187(a, b, c, d, e);
    case 97: return MD_Et_GPU188(a, b, c, d, e);
    case 98: return MD_Et_GPU189(a, b, c, d, e);
    case 99: return MD_Et_GPU200(a, b, c, d, e);
    case 100: return MD_Et_GPU201(a, b, c, d, e);
    case 101: return MD_Et_GPU202(a, b, c, d, e);
    case 102: return MD_Et_GPU210(a, b, c, d, e);
    case 103: return MD_Et_GPU211(a, b, c, d, e);
    case 104: return MD_Et_GPU212(a, b, c, d, e);
    case 105: return MD_Et_GPU213(a, b, c, d, e);
    case 106: return MD_Et_GPU220(a, b, c, d, e);
    case 107: return MD_Et_GPU221(a, b, c, d, e);
    case 108: return MD_Et_GPU222(a, b, c, d, e);
    case 109: return MD_Et_GPU223(a, b, c, d, e);
    case 110: return MD_Et_GPU224(a, b, c, d, e);
    case 111: return MD_Et_GPU230(a, b, c, d, e);
    case 112: return MD_Et_GPU231(a, b, c, d, e);
    case 113: return MD_Et_GPU232(a, b, c, d, e);
    case 114: return MD_Et_GPU233(a, b, c, d, e);
    case 115: return MD_Et_GPU234(a, b, c, d, e);
    case 116: return MD_Et_GPU235(a, b, c, d, e);
    case 117: return MD_Et_GPU240(a, b, c, d, e);
    case 118: return MD_Et_GPU241(a, b, c, d, e);
    case 119: return MD_Et_GPU242(a, b, c, d, e);
    case 120: return MD_Et_GPU243(a, b, c, d, e);
    case 121: return MD_Et_GPU244(a, b, c, d, e);
    case 122: return MD_Et_GPU245(a, b, c, d, e);
    case 123: return MD_Et_GPU246(a, b, c, d, e);
    case 124: return MD_Et_GPU250(a, b, c, d, e);
    case 125: return MD_Et_GPU251(a, b, c, d, e);
    case 126: return MD_Et_GPU252(a, b, c, d, e);
    case 127: return MD_Et_GPU253(a, b, c, d, e);
    case 128: return MD_Et_GPU254(a, b, c, d, e);
    case 129: return MD_Et_GPU255(a, b, c, d, e);
    case 130: return MD_Et_GPU256(a, b, c, d, e);
    case 131: return MD_Et_GPU257(a, b, c, d, e);
    case 132: return MD_Et_GPU260(a, b, c, d, e);
    case 133: return MD_Et_GPU261(a, b, c, d, e);
    case 134: return MD_Et_GPU262(a, b, c, d, e);
    case 135: return MD_Et_GPU263(a, b, c, d, e);
    case 136: return MD_Et_GPU264(a, b, c, d, e);
    case 137: return MD_Et_GPU265(a, b, c, d, e);
    case 138: return MD_Et_GPU266(a, b, c, d, e);
    case 139: return MD_Et_GPU267(a, b, c, d, e);
    case 140: return MD_Et_GPU268(a, b, c, d, e);
    case 141: return MD_Et_GPU270(a, b, c, d, e);
    case 142: return MD_Et_GPU271(a, b, c, d, e);
    case 143: return MD_Et_GPU272(a, b, c, d, e);
    case 144: return MD_Et_GPU273(a, b, c, d, e);
    case 145: return MD_Et_GPU274(a, b, c, d, e);
    case 146: return MD_Et_GPU275(a, b, c, d, e);
    case 147: return MD_Et_GPU276(a, b, c, d, e);
    case 148: return MD_Et_GPU277(a, b, c, d, e);
    case 149: return MD_Et_GPU278(a, b, c, d, e);
    case 150: return MD_Et_GPU279(a, b, c, d, e);
    case 151: return MD_Et_GPU280(a, b, c, d, e);
    case 152: return MD_Et_GPU281(a, b, c, d, e);
    case 153: return MD_Et_GPU282(a, b, c, d, e);
    case 154: return MD_Et_GPU283(a, b, c, d, e);
    case 155: return MD_Et_GPU284(a, b, c, d, e);
    case 156: return MD_Et_GPU285(a, b, c, d, e);
    case 157: return MD_Et_GPU286(a, b, c, d, e);
    case 158: return MD_Et_GPU287(a, b, c, d, e);
    case 159: return MD_Et_GPU288(a, b, c, d, e);
    case 160: return MD_Et_GPU289(a, b, c, d, e);
    case 161: return MD_Et_GPU2810(a, b, c, d, e);
    case 162: return MD_Et_GPU300(a, b, c, d, e);
    case 163: return MD_Et_GPU301(a, b, c, d, e);
    case 164: return MD_Et_GPU302(a, b, c, d, e);
    case 165: return MD_Et_GPU303(a, b, c, d, e);
    case 166: return MD_Et_GPU310(a, b, c, d, e);
    case 167: return MD_Et_GPU311(a, b, c, d, e);
    case 168: return MD_Et_GPU312(a, b, c, d, e);
    case 169: return MD_Et_GPU313(a, b, c, d, e);
    case 170: return MD_Et_GPU314(a, b, c, d, e);
    case 171: return MD_Et_GPU320(a, b, c, d, e);
    case 172: return MD_Et_GPU321(a, b, c, d, e);
    case 173: return MD_Et_GPU322(a, b, c, d, e);
    case 174: return MD_Et_GPU323(a, b, c, d, e);
    case 175: return MD_Et_GPU324(a, b, c, d, e);
    case 176: return MD_Et_GPU325(a, b, c, d, e);
    case 177: return MD_Et_GPU330(a, b, c, d, e);
    case 178: return MD_Et_GPU331(a, b, c, d, e);
    case 179: return MD_Et_GPU332(a, b, c, d, e);
    case 180: return MD_Et_GPU333(a, b, c, d, e);
    case 181: return MD_Et_GPU334(a, b, c, d, e);
    case 182: return MD_Et_GPU335(a, b, c, d, e);
    case 183: return MD_Et_GPU336(a, b, c, d, e);
    case 184: return MD_Et_GPU340(a, b, c, d, e);
    case 185: return MD_Et_GPU341(a, b, c, d, e);
    case 186: return MD_Et_GPU342(a, b, c, d, e);
    case 187: return MD_Et_GPU343(a, b, c, d, e);
    case 188: return MD_Et_GPU344(a, b, c, d, e);
    case 189: return MD_Et_GPU345(a, b, c, d, e);
    case 190: return MD_Et_GPU346(a, b, c, d, e);
    case 191: return MD_Et_GPU347(a, b, c, d, e);
    case 192: return MD_Et_GPU350(a, b, c, d, e);
    case 193: return MD_Et_GPU351(a, b, c, d, e);
    case 194: return MD_Et_GPU352(a, b, c, d, e);
    case 195: return MD_Et_GPU353(a, b, c, d, e);
    case 196: return MD_Et_GPU354(a, b, c, d, e);
    case 197: return MD_Et_GPU355(a, b, c, d, e);
    case 198: return MD_Et_GPU356(a, b, c, d, e);
    case 199: return MD_Et_GPU357(a, b, c, d, e);
    case 200: return MD_Et_GPU358(a, b, c, d, e);
    case 201: return MD_Et_GPU360(a, b, c, d, e);
    case 202: return MD_Et_GPU361(a, b, c, d, e);
    case 203: return MD_Et_GPU362(a, b, c, d, e);
    case 204: return MD_Et_GPU363(a, b, c, d, e);
    case 205: return MD_Et_GPU364(a, b, c, d, e);
    case 206: return MD_Et_GPU365(a, b, c, d, e);
    case 207: return MD_Et_GPU366(a, b, c, d, e);
    case 208: return MD_Et_GPU367(a, b, c, d, e);
    case 209: return MD_Et_GPU368(a, b, c, d, e);
    case 210: return MD_Et_GPU369(a, b, c, d, e);
    case 211: return MD_Et_GPU370(a, b, c, d, e);
    case 212: return MD_Et_GPU371(a, b, c, d, e);
    case 213: return MD_Et_GPU372(a, b, c, d, e);
    case 214: return MD_Et_GPU373(a, b, c, d, e);
    case 215: return MD_Et_GPU374(a, b, c, d, e);
    case 216: return MD_Et_GPU375(a, b, c, d, e);
    case 217: return MD_Et_GPU376(a, b, c, d, e);
    case 218: return MD_Et_GPU377(a, b, c, d, e);
    case 219: return MD_Et_GPU378(a, b, c, d, e);
    case 220: return MD_Et_GPU379(a, b, c, d, e);
    case 221: return MD_Et_GPU3710(a, b, c, d, e);
    case 222: return MD_Et_GPU380(a, b, c, d, e);
    case 223: return MD_Et_GPU381(a, b, c, d, e);
    case 224: return MD_Et_GPU382(a, b, c, d, e);
    case 225: return MD_Et_GPU383(a, b, c, d, e);
    case 226: return MD_Et_GPU384(a, b, c, d, e);
    case 227: return MD_Et_GPU385(a, b, c, d, e);
    case 228: return MD_Et_GPU386(a, b, c, d, e);
    case 229: return MD_Et_GPU387(a, b, c, d, e);
    case 230: return MD_Et_GPU388(a, b, c, d, e);
    case 231: return MD_Et_GPU389(a, b, c, d, e);
    case 232: return MD_Et_GPU3810(a, b, c, d, e);
    case 233: return MD_Et_GPU3811(a, b, c, d, e);
    case 234: return MD_Et_GPU400(a, b, c, d, e);
    case 235: return MD_Et_GPU401(a, b, c, d, e);
    case 236: return MD_Et_GPU402(a, b, c, d, e);
    case 237: return MD_Et_GPU403(a, b, c, d, e);
    case 238: return MD_Et_GPU404(a, b, c, d, e);
    case 239: return MD_Et_GPU410(a, b, c, d, e);
    case 240: return MD_Et_GPU411(a, b, c, d, e);
    case 241: return MD_Et_GPU412(a, b, c, d, e);
    case 242: return MD_Et_GPU413(a, b, c, d, e);
    case 243: return MD_Et_GPU414(a, b, c, d, e);
    case 244: return MD_Et_GPU415(a, b, c, d, e);
    case 245: return MD_Et_GPU420(a, b, c, d, e);
    case 246: return MD_Et_GPU421(a, b, c, d, e);
    case 247: return MD_Et_GPU422(a, b, c, d, e);
    case 248: return MD_Et_GPU423(a, b, c, d, e);
    case 249: return MD_Et_GPU424(a, b, c, d, e);
    case 250: return MD_Et_GPU425(a, b, c, d, e);
    case 251: return MD_Et_GPU426(a, b, c, d, e);
    case 252: return MD_Et_GPU430(a, b, c, d, e);
    case 253: return MD_Et_GPU431(a, b, c, d, e);
    case 254: return MD_Et_GPU432(a, b, c, d, e);
    case 255: return MD_Et_GPU433(a, b, c, d, e);
    case 256: return MD_Et_GPU434(a, b, c, d, e);
    case 257: return MD_Et_GPU435(a, b, c, d, e);
    case 258: return MD_Et_GPU436(a, b, c, d, e);
    case 259: return MD_Et_GPU437(a, b, c, d, e);
    case 260: return MD_Et_GPU440(a, b, c, d, e);
    case 261: return MD_Et_GPU441(a, b, c, d, e);
    case 262: return MD_Et_GPU442(a, b, c, d, e);
    case 263: return MD_Et_GPU443(a, b, c, d, e);
    case 264: return MD_Et_GPU444(a, b, c, d, e);
    case 265: return MD_Et_GPU445(a, b, c, d, e);
    case 266: return MD_Et_GPU446(a, b, c, d, e);
    case 267: return MD_Et_GPU447(a, b, c, d, e);
    case 268: return MD_Et_GPU448(a, b, c, d, e);
    case 269: return MD_Et_GPU450(a, b, c, d, e);
    case 270: return MD_Et_GPU451(a, b, c, d, e);
    case 271: return MD_Et_GPU452(a, b, c, d, e);
    case 272: return MD_Et_GPU453(a, b, c, d, e);
    case 273: return MD_Et_GPU454(a, b, c, d, e);
    case 274: return MD_Et_GPU455(a, b, c, d, e);
    case 275: return MD_Et_GPU456(a, b, c, d, e);
    case 276: return MD_Et_GPU457(a, b, c, d, e);
    case 277: return MD_Et_GPU458(a, b, c, d, e);
    case 278: return MD_Et_GPU459(a, b, c, d, e);
    case 279: return MD_Et_GPU460(a, b, c, d, e);
    case 280: return MD_Et_GPU461(a, b, c, d, e);
    case 281: return MD_Et_GPU462(a, b, c, d, e);
    case 282: return MD_Et_GPU463(a, b, c, d, e);
    case 283: return MD_Et_GPU464(a, b, c, d, e);
    case 284: return MD_Et_GPU465(a, b, c, d, e);
    case 285: return MD_Et_GPU466(a, b, c, d, e);
    case 286: return MD_Et_GPU467(a, b, c, d, e);
    case 287: return MD_Et_GPU468(a, b, c, d, e);
    case 288: return MD_Et_GPU469(a, b, c, d, e);
    case 289: return MD_Et_GPU4610(a, b, c, d, e);
    case 290: return MD_Et_GPU470(a, b, c, d, e);
    case 291: return MD_Et_GPU471(a, b, c, d, e);
    case 292: return MD_Et_GPU472(a, b, c, d, e);
    case 293: return MD_Et_GPU473(a, b, c, d, e);
    case 294: return MD_Et_GPU474(a, b, c, d, e);
    case 295: return MD_Et_GPU475(a, b, c, d, e);
    case 296: return MD_Et_GPU476(a, b, c, d, e);
    case 297: return MD_Et_GPU477(a, b, c, d, e);
    case 298: return MD_Et_GPU478(a, b, c, d, e);
    case 299: return MD_Et_GPU479(a, b, c, d, e);
    case 300: return MD_Et_GPU4710(a, b, c, d, e);
    case 301: return MD_Et_GPU4711(a, b, c, d, e);
    case 302: return MD_Et_GPU480(a, b, c, d, e);
    case 303: return MD_Et_GPU481(a, b, c, d, e);
    case 304: return MD_Et_GPU482(a, b, c, d, e);
    case 305: return MD_Et_GPU483(a, b, c, d, e);
    case 306: return MD_Et_GPU484(a, b, c, d, e);
    case 307: return MD_Et_GPU485(a, b, c, d, e);
    case 308: return MD_Et_GPU486(a, b, c, d, e);
    case 309: return MD_Et_GPU487(a, b, c, d, e);
    case 310: return MD_Et_GPU488(a, b, c, d, e);
    case 311: return MD_Et_GPU489(a, b, c, d, e);
    case 312: return MD_Et_GPU4810(a, b, c, d, e);
    case 313: return MD_Et_GPU4811(a, b, c, d, e);
    case 314: return MD_Et_GPU4812(a, b, c, d, e);
    case 315: return MD_Et_GPU500(a, b, c, d, e);
    case 316: return MD_Et_GPU501(a, b, c, d, e);
    case 317: return MD_Et_GPU502(a, b, c, d, e);
    case 318: return MD_Et_GPU503(a, b, c, d, e);
    case 319: return MD_Et_GPU504(a, b, c, d, e);
    case 320: return MD_Et_GPU505(a, b, c, d, e);
    case 321: return MD_Et_GPU510(a, b, c, d, e);
    case 322: return MD_Et_GPU511(a, b, c, d, e);
    case 323: return MD_Et_GPU512(a, b, c, d, e);
    case 324: return MD_Et_GPU513(a, b, c, d, e);
    case 325: return MD_Et_GPU514(a, b, c, d, e);
    case 326: return MD_Et_GPU515(a, b, c, d, e);
    case 327: return MD_Et_GPU516(a, b, c, d, e);
    case 328: return MD_Et_GPU520(a, b, c, d, e);
    case 329: return MD_Et_GPU521(a, b, c, d, e);
    case 330: return MD_Et_GPU522(a, b, c, d, e);
    case 331: return MD_Et_GPU523(a, b, c, d, e);
    case 332: return MD_Et_GPU524(a, b, c, d, e);
    case 333: return MD_Et_GPU525(a, b, c, d, e);
    case 334: return MD_Et_GPU526(a, b, c, d, e);
    case 335: return MD_Et_GPU527(a, b, c, d, e);
    case 336: return MD_Et_GPU530(a, b, c, d, e);
    case 337: return MD_Et_GPU531(a, b, c, d, e);
    case 338: return MD_Et_GPU532(a, b, c, d, e);
    case 339: return MD_Et_GPU533(a, b, c, d, e);
    case 340: return MD_Et_GPU534(a, b, c, d, e);
    case 341: return MD_Et_GPU535(a, b, c, d, e);
    case 342: return MD_Et_GPU536(a, b, c, d, e);
    case 343: return MD_Et_GPU537(a, b, c, d, e);
    case 344: return MD_Et_GPU538(a, b, c, d, e);
    case 345: return MD_Et_GPU540(a, b, c, d, e);
    case 346: return MD_Et_GPU541(a, b, c, d, e);
    case 347: return MD_Et_GPU542(a, b, c, d, e);
    case 348: return MD_Et_GPU543(a, b, c, d, e);
    case 349: return MD_Et_GPU544(a, b, c, d, e);
    case 350: return MD_Et_GPU545(a, b, c, d, e);
    case 351: return MD_Et_GPU546(a, b, c, d, e);
    case 352: return MD_Et_GPU547(a, b, c, d, e);
    case 353: return MD_Et_GPU548(a, b, c, d, e);
    case 354: return MD_Et_GPU549(a, b, c, d, e);
    case 355: return MD_Et_GPU550(a, b, c, d, e);
    case 356: return MD_Et_GPU551(a, b, c, d, e);
    case 357: return MD_Et_GPU552(a, b, c, d, e);
    case 358: return MD_Et_GPU553(a, b, c, d, e);
    case 359: return MD_Et_GPU554(a, b, c, d, e);
    case 360: return MD_Et_GPU555(a, b, c, d, e);
    case 361: return MD_Et_GPU556(a, b, c, d, e);
    case 362: return MD_Et_GPU557(a, b, c, d, e);
    case 363: return MD_Et_GPU558(a, b, c, d, e);
    case 364: return MD_Et_GPU559(a, b, c, d, e);
    case 365: return MD_Et_GPU5510(a, b, c, d, e);
    case 366: return MD_Et_GPU560(a, b, c, d, e);
    case 367: return MD_Et_GPU561(a, b, c, d, e);
    case 368: return MD_Et_GPU562(a, b, c, d, e);
    case 369: return MD_Et_GPU563(a, b, c, d, e);
    case 370: return MD_Et_GPU564(a, b, c, d, e);
    case 371: return MD_Et_GPU565(a, b, c, d, e);
    case 372: return MD_Et_GPU566(a, b, c, d, e);
    case 373: return MD_Et_GPU567(a, b, c, d, e);
    case 374: return MD_Et_GPU568(a, b, c, d, e);
    case 375: return MD_Et_GPU569(a, b, c, d, e);
    case 376: return MD_Et_GPU5610(a, b, c, d, e);
    case 377: return MD_Et_GPU5611(a, b, c, d, e);
    case 378: return MD_Et_GPU570(a, b, c, d, e);
    case 379: return MD_Et_GPU571(a, b, c, d, e);
    case 380: return MD_Et_GPU572(a, b, c, d, e);
    case 381: return MD_Et_GPU573(a, b, c, d, e);
    case 382: return MD_Et_GPU574(a, b, c, d, e);
    case 383: return MD_Et_GPU575(a, b, c, d, e);
    case 384: return MD_Et_GPU576(a, b, c, d, e);
    case 385: return MD_Et_GPU577(a, b, c, d, e);
    case 386: return MD_Et_GPU578(a, b, c, d, e);
    case 387: return MD_Et_GPU579(a, b, c, d, e);
    case 388: return MD_Et_GPU5710(a, b, c, d, e);
    case 389: return MD_Et_GPU5711(a, b, c, d, e);
    case 390: return MD_Et_GPU5712(a, b, c, d, e);
    case 391: return MD_Et_GPU580(a, b, c, d, e);
    case 392: return MD_Et_GPU581(a, b, c, d, e);
    case 393: return MD_Et_GPU582(a, b, c, d, e);
    case 394: return MD_Et_GPU583(a, b, c, d, e);
    case 395: return MD_Et_GPU584(a, b, c, d, e);
    case 396: return MD_Et_GPU585(a, b, c, d, e);
    case 397: return MD_Et_GPU586(a, b, c, d, e);
    case 398: return MD_Et_GPU587(a, b, c, d, e);
    case 399: return MD_Et_GPU588(a, b, c, d, e);
    case 400: return MD_Et_GPU589(a, b, c, d, e);
    case 401: return MD_Et_GPU5810(a, b, c, d, e);
    case 402: return MD_Et_GPU5811(a, b, c, d, e);
    case 403: return MD_Et_GPU5812(a, b, c, d, e);
    case 404: return MD_Et_GPU5813(a, b, c, d, e);
    case 405: return MD_Et_GPU600(a, b, c, d, e);
    case 406: return MD_Et_GPU601(a, b, c, d, e);
    case 407: return MD_Et_GPU602(a, b, c, d, e);
    case 408: return MD_Et_GPU603(a, b, c, d, e);
    case 409: return MD_Et_GPU604(a, b, c, d, e);
    case 410: return MD_Et_GPU605(a, b, c, d, e);
    case 411: return MD_Et_GPU606(a, b, c, d, e);
    case 412: return MD_Et_GPU610(a, b, c, d, e);
    case 413: return MD_Et_GPU611(a, b, c, d, e);
    case 414: return MD_Et_GPU612(a, b, c, d, e);
    case 415: return MD_Et_GPU613(a, b, c, d, e);
    case 416: return MD_Et_GPU614(a, b, c, d, e);
    case 417: return MD_Et_GPU615(a, b, c, d, e);
    case 418: return MD_Et_GPU616(a, b, c, d, e);
    case 419: return MD_Et_GPU617(a, b, c, d, e);
    case 420: return MD_Et_GPU620(a, b, c, d, e);
    case 421: return MD_Et_GPU621(a, b, c, d, e);
    case 422: return MD_Et_GPU622(a, b, c, d, e);
    case 423: return MD_Et_GPU623(a, b, c, d, e);
    case 424: return MD_Et_GPU624(a, b, c, d, e);
    case 425: return MD_Et_GPU625(a, b, c, d, e);
    case 426: return MD_Et_GPU626(a, b, c, d, e);
    case 427: return MD_Et_GPU627(a, b, c, d, e);
    case 428: return MD_Et_GPU628(a, b, c, d, e);
    case 429: return MD_Et_GPU630(a, b, c, d, e);
    case 430: return MD_Et_GPU631(a, b, c, d, e);
    case 431: return MD_Et_GPU632(a, b, c, d, e);
    case 432: return MD_Et_GPU633(a, b, c, d, e);
    case 433: return MD_Et_GPU634(a, b, c, d, e);
    case 434: return MD_Et_GPU635(a, b, c, d, e);
    case 435: return MD_Et_GPU636(a, b, c, d, e);
    case 436: return MD_Et_GPU637(a, b, c, d, e);
    case 437: return MD_Et_GPU638(a, b, c, d, e);
    case 438: return MD_Et_GPU639(a, b, c, d, e);
    case 439: return MD_Et_GPU640(a, b, c, d, e);
    case 440: return MD_Et_GPU641(a, b, c, d, e);
    case 441: return MD_Et_GPU642(a, b, c, d, e);
    case 442: return MD_Et_GPU643(a, b, c, d, e);
    case 443: return MD_Et_GPU644(a, b, c, d, e);
    case 444: return MD_Et_GPU645(a, b, c, d, e);
    case 445: return MD_Et_GPU646(a, b, c, d, e);
    case 446: return MD_Et_GPU647(a, b, c, d, e);
    case 447: return MD_Et_GPU648(a, b, c, d, e);
    case 448: return MD_Et_GPU649(a, b, c, d, e);
    case 449: return MD_Et_GPU6410(a, b, c, d, e);
    case 450: return MD_Et_GPU650(a, b, c, d, e);
    case 451: return MD_Et_GPU651(a, b, c, d, e);
    case 452: return MD_Et_GPU652(a, b, c, d, e);
    case 453: return MD_Et_GPU653(a, b, c, d, e);
    case 454: return MD_Et_GPU654(a, b, c, d, e);
    case 455: return MD_Et_GPU655(a, b, c, d, e);
    case 456: return MD_Et_GPU656(a, b, c, d, e);
    case 457: return MD_Et_GPU657(a, b, c, d, e);
    case 458: return MD_Et_GPU658(a, b, c, d, e);
    case 459: return MD_Et_GPU659(a, b, c, d, e);
    case 460: return MD_Et_GPU6510(a, b, c, d, e);
    case 461: return MD_Et_GPU6511(a, b, c, d, e);
    case 462: return MD_Et_GPU660(a, b, c, d, e);
    case 463: return MD_Et_GPU661(a, b, c, d, e);
    case 464: return MD_Et_GPU662(a, b, c, d, e);
    case 465: return MD_Et_GPU663(a, b, c, d, e);
    case 466: return MD_Et_GPU664(a, b, c, d, e);
    case 467: return MD_Et_GPU665(a, b, c, d, e);
    case 468: return MD_Et_GPU666(a, b, c, d, e);
    case 469: return MD_Et_GPU667(a, b, c, d, e);
    case 470: return MD_Et_GPU668(a, b, c, d, e);
    case 471: return MD_Et_GPU669(a, b, c, d, e);
    case 472: return MD_Et_GPU6610(a, b, c, d, e);
    case 473: return MD_Et_GPU6611(a, b, c, d, e);
    case 474: return MD_Et_GPU6612(a, b, c, d, e);
    case 475: return MD_Et_GPU670(a, b, c, d, e);
    case 476: return MD_Et_GPU671(a, b, c, d, e);
    case 477: return MD_Et_GPU672(a, b, c, d, e);
    case 478: return MD_Et_GPU673(a, b, c, d, e);
    case 479: return MD_Et_GPU674(a, b, c, d, e);
    case 480: return MD_Et_GPU675(a, b, c, d, e);
    case 481: return MD_Et_GPU676(a, b, c, d, e);
    case 482: return MD_Et_GPU677(a, b, c, d, e);
    case 483: return MD_Et_GPU678(a, b, c, d, e);
    case 484: return MD_Et_GPU679(a, b, c, d, e);
    case 485: return MD_Et_GPU6710(a, b, c, d, e);
    case 486: return MD_Et_GPU6711(a, b, c, d, e);
    case 487: return MD_Et_GPU6712(a, b, c, d, e);
    case 488: return MD_Et_GPU6713(a, b, c, d, e);
    case 489: return MD_Et_GPU680(a, b, c, d, e);
    case 490: return MD_Et_GPU681(a, b, c, d, e);
    case 491: return MD_Et_GPU682(a, b, c, d, e);
    case 492: return MD_Et_GPU683(a, b, c, d, e);
    case 493: return MD_Et_GPU684(a, b, c, d, e);
    case 494: return MD_Et_GPU685(a, b, c, d, e);
    case 495: return MD_Et_GPU686(a, b, c, d, e);
    case 496: return MD_Et_GPU687(a, b, c, d, e);
    case 497: return MD_Et_GPU688(a, b, c, d, e);
    case 498: return MD_Et_GPU689(a, b, c, d, e);
    case 499: return MD_Et_GPU6810(a, b, c, d, e);
    case 500: return MD_Et_GPU6811(a, b, c, d, e);
    case 501: return MD_Et_GPU6812(a, b, c, d, e);
    case 502: return MD_Et_GPU6813(a, b, c, d, e);
    case 503: return MD_Et_GPU6814(a, b, c, d, e);
    case 504: return MD_Et_GPU700(a, b, c, d, e);
    case 505: return MD_Et_GPU701(a, b, c, d, e);
    case 506: return MD_Et_GPU702(a, b, c, d, e);
    case 507: return MD_Et_GPU703(a, b, c, d, e);
    case 508: return MD_Et_GPU704(a, b, c, d, e);
    case 509: return MD_Et_GPU705(a, b, c, d, e);
    case 510: return MD_Et_GPU706(a, b, c, d, e);
    case 511: return MD_Et_GPU707(a, b, c, d, e);
    case 512: return MD_Et_GPU710(a, b, c, d, e);
    case 513: return MD_Et_GPU711(a, b, c, d, e);
    case 514: return MD_Et_GPU712(a, b, c, d, e);
    case 515: return MD_Et_GPU713(a, b, c, d, e);
    case 516: return MD_Et_GPU714(a, b, c, d, e);
    case 517: return MD_Et_GPU715(a, b, c, d, e);
    case 518: return MD_Et_GPU716(a, b, c, d, e);
    case 519: return MD_Et_GPU717(a, b, c, d, e);
    case 520: return MD_Et_GPU718(a, b, c, d, e);
    case 521: return MD_Et_GPU720(a, b, c, d, e);
    case 522: return MD_Et_GPU721(a, b, c, d, e);
    case 523: return MD_Et_GPU722(a, b, c, d, e);
    case 524: return MD_Et_GPU723(a, b, c, d, e);
    case 525: return MD_Et_GPU724(a, b, c, d, e);
    case 526: return MD_Et_GPU725(a, b, c, d, e);
    case 527: return MD_Et_GPU726(a, b, c, d, e);
    case 528: return MD_Et_GPU727(a, b, c, d, e);
    case 529: return MD_Et_GPU728(a, b, c, d, e);
    case 530: return MD_Et_GPU729(a, b, c, d, e);
    case 531: return MD_Et_GPU730(a, b, c, d, e);
    case 532: return MD_Et_GPU731(a, b, c, d, e);
    case 533: return MD_Et_GPU732(a, b, c, d, e);
    case 534: return MD_Et_GPU733(a, b, c, d, e);
    case 535: return MD_Et_GPU734(a, b, c, d, e);
    case 536: return MD_Et_GPU735(a, b, c, d, e);
    case 537: return MD_Et_GPU736(a, b, c, d, e);
    case 538: return MD_Et_GPU737(a, b, c, d, e);
    case 539: return MD_Et_GPU738(a, b, c, d, e);
    case 540: return MD_Et_GPU739(a, b, c, d, e);
    case 541: return MD_Et_GPU7310(a, b, c, d, e);
    case 542: return MD_Et_GPU740(a, b, c, d, e);
    case 543: return MD_Et_GPU741(a, b, c, d, e);
    case 544: return MD_Et_GPU742(a, b, c, d, e);
    case 545: return MD_Et_GPU743(a, b, c, d, e);
    case 546: return MD_Et_GPU744(a, b, c, d, e);
    case 547: return MD_Et_GPU745(a, b, c, d, e);
    case 548: return MD_Et_GPU746(a, b, c, d, e);
    case 549: return MD_Et_GPU747(a, b, c, d, e);
    case 550: return MD_Et_GPU748(a, b, c, d, e);
    case 551: return MD_Et_GPU749(a, b, c, d, e);
    case 552: return MD_Et_GPU7410(a, b, c, d, e);
    case 553: return MD_Et_GPU7411(a, b, c, d, e);
    case 554: return MD_Et_GPU750(a, b, c, d, e);
    case 555: return MD_Et_GPU751(a, b, c, d, e);
    case 556: return MD_Et_GPU752(a, b, c, d, e);
    case 557: return MD_Et_GPU753(a, b, c, d, e);
    case 558: return MD_Et_GPU754(a, b, c, d, e);
    case 559: return MD_Et_GPU755(a, b, c, d, e);
    case 560: return MD_Et_GPU756(a, b, c, d, e);
    case 561: return MD_Et_GPU757(a, b, c, d, e);
    case 562: return MD_Et_GPU758(a, b, c, d, e);
    case 563: return MD_Et_GPU759(a, b, c, d, e);
    case 564: return MD_Et_GPU7510(a, b, c, d, e);
    case 565: return MD_Et_GPU7511(a, b, c, d, e);
    case 566: return MD_Et_GPU7512(a, b, c, d, e);
    case 567: return MD_Et_GPU760(a, b, c, d, e);
    case 568: return MD_Et_GPU761(a, b, c, d, e);
    case 569: return MD_Et_GPU762(a, b, c, d, e);
    case 570: return MD_Et_GPU763(a, b, c, d, e);
    case 571: return MD_Et_GPU764(a, b, c, d, e);
    case 572: return MD_Et_GPU765(a, b, c, d, e);
    case 573: return MD_Et_GPU766(a, b, c, d, e);
    case 574: return MD_Et_GPU767(a, b, c, d, e);
    case 575: return MD_Et_GPU768(a, b, c, d, e);
    case 576: return MD_Et_GPU769(a, b, c, d, e);
    case 577: return MD_Et_GPU7610(a, b, c, d, e);
    case 578: return MD_Et_GPU7611(a, b, c, d, e);
    case 579: return MD_Et_GPU7612(a, b, c, d, e);
    case 580: return MD_Et_GPU7613(a, b, c, d, e);
    case 581: return MD_Et_GPU770(a, b, c, d, e);
    case 582: return MD_Et_GPU771(a, b, c, d, e);
    case 583: return MD_Et_GPU772(a, b, c, d, e);
    case 584: return MD_Et_GPU773(a, b, c, d, e);
    case 585: return MD_Et_GPU774(a, b, c, d, e);
    case 586: return MD_Et_GPU775(a, b, c, d, e);
    case 587: return MD_Et_GPU776(a, b, c, d, e);
    case 588: return MD_Et_GPU777(a, b, c, d, e);
    case 589: return MD_Et_GPU778(a, b, c, d, e);
    case 590: return MD_Et_GPU779(a, b, c, d, e);
    case 591: return MD_Et_GPU7710(a, b, c, d, e);
    case 592: return MD_Et_GPU7711(a, b, c, d, e);
    case 593: return MD_Et_GPU7712(a, b, c, d, e);
    case 594: return MD_Et_GPU7713(a, b, c, d, e);
    case 595: return MD_Et_GPU7714(a, b, c, d, e);
    case 596: return MD_Et_GPU780(a, b, c, d, e);
    case 597: return MD_Et_GPU781(a, b, c, d, e);
    case 598: return MD_Et_GPU782(a, b, c, d, e);
    case 599: return MD_Et_GPU783(a, b, c, d, e);
    case 600: return MD_Et_GPU784(a, b, c, d, e);
    case 601: return MD_Et_GPU785(a, b, c, d, e);
    case 602: return MD_Et_GPU786(a, b, c, d, e);
    case 603: return MD_Et_GPU787(a, b, c, d, e);
    case 604: return MD_Et_GPU788(a, b, c, d, e);
    case 605: return MD_Et_GPU789(a, b, c, d, e);
    case 606: return MD_Et_GPU7810(a, b, c, d, e);
    case 607: return MD_Et_GPU7811(a, b, c, d, e);
    case 608: return MD_Et_GPU7812(a, b, c, d, e);
    case 609: return MD_Et_GPU7813(a, b, c, d, e);
    case 610: return MD_Et_GPU7814(a, b, c, d, e);
    case 611: return MD_Et_GPU7815(a, b, c, d, e);
    case 612: return MD_Et_GPU800(a, b, c, d, e);
    case 613: return MD_Et_GPU801(a, b, c, d, e);
    case 614: return MD_Et_GPU802(a, b, c, d, e);
    case 615: return MD_Et_GPU803(a, b, c, d, e);
    case 616: return MD_Et_GPU804(a, b, c, d, e);
    case 617: return MD_Et_GPU805(a, b, c, d, e);
    case 618: return MD_Et_GPU806(a, b, c, d, e);
    case 619: return MD_Et_GPU807(a, b, c, d, e);
    case 620: return MD_Et_GPU808(a, b, c, d, e);
    case 621: return MD_Et_GPU810(a, b, c, d, e);
    case 622: return MD_Et_GPU811(a, b, c, d, e);
    case 623: return MD_Et_GPU812(a, b, c, d, e);
    case 624: return MD_Et_GPU813(a, b, c, d, e);
    case 625: return MD_Et_GPU814(a, b, c, d, e);
    case 626: return MD_Et_GPU815(a, b, c, d, e);
    case 627: return MD_Et_GPU816(a, b, c, d, e);
    case 628: return MD_Et_GPU817(a, b, c, d, e);
    case 629: return MD_Et_GPU818(a, b, c, d, e);
    case 630: return MD_Et_GPU819(a, b, c, d, e);
    case 631: return MD_Et_GPU820(a, b, c, d, e);
    case 632: return MD_Et_GPU821(a, b, c, d, e);
    case 633: return MD_Et_GPU822(a, b, c, d, e);
    case 634: return MD_Et_GPU823(a, b, c, d, e);
    case 635: return MD_Et_GPU824(a, b, c, d, e);
    case 636: return MD_Et_GPU825(a, b, c, d, e);
    case 637: return MD_Et_GPU826(a, b, c, d, e);
    case 638: return MD_Et_GPU827(a, b, c, d, e);
    case 639: return MD_Et_GPU828(a, b, c, d, e);
    case 640: return MD_Et_GPU829(a, b, c, d, e);
    case 641: return MD_Et_GPU8210(a, b, c, d, e);
    case 642: return MD_Et_GPU830(a, b, c, d, e);
    case 643: return MD_Et_GPU831(a, b, c, d, e);
    case 644: return MD_Et_GPU832(a, b, c, d, e);
    case 645: return MD_Et_GPU833(a, b, c, d, e);
    case 646: return MD_Et_GPU834(a, b, c, d, e);
    case 647: return MD_Et_GPU835(a, b, c, d, e);
    case 648: return MD_Et_GPU836(a, b, c, d, e);
    case 649: return MD_Et_GPU837(a, b, c, d, e);
    case 650: return MD_Et_GPU838(a, b, c, d, e);
    case 651: return MD_Et_GPU839(a, b, c, d, e);
    case 652: return MD_Et_GPU8310(a, b, c, d, e);
    case 653: return MD_Et_GPU8311(a, b, c, d, e);
    case 654: return MD_Et_GPU840(a, b, c, d, e);
    case 655: return MD_Et_GPU841(a, b, c, d, e);
    case 656: return MD_Et_GPU842(a, b, c, d, e);
    case 657: return MD_Et_GPU843(a, b, c, d, e);
    case 658: return MD_Et_GPU844(a, b, c, d, e);
    case 659: return MD_Et_GPU845(a, b, c, d, e);
    case 660: return MD_Et_GPU846(a, b, c, d, e);
    case 661: return MD_Et_GPU847(a, b, c, d, e);
    case 662: return MD_Et_GPU848(a, b, c, d, e);
    case 663: return MD_Et_GPU849(a, b, c, d, e);
    case 664: return MD_Et_GPU8410(a, b, c, d, e);
    case 665: return MD_Et_GPU8411(a, b, c, d, e);
    case 666: return MD_Et_GPU8412(a, b, c, d, e);
    case 667: return MD_Et_GPU850(a, b, c, d, e);
    case 668: return MD_Et_GPU851(a, b, c, d, e);
    case 669: return MD_Et_GPU852(a, b, c, d, e);
    case 670: return MD_Et_GPU853(a, b, c, d, e);
    case 671: return MD_Et_GPU854(a, b, c, d, e);
    case 672: return MD_Et_GPU855(a, b, c, d, e);
    case 673: return MD_Et_GPU856(a, b, c, d, e);
    case 674: return MD_Et_GPU857(a, b, c, d, e);
    case 675: return MD_Et_GPU858(a, b, c, d, e);
    case 676: return MD_Et_GPU859(a, b, c, d, e);
    case 677: return MD_Et_GPU8510(a, b, c, d, e);
    case 678: return MD_Et_GPU8511(a, b, c, d, e);
    case 679: return MD_Et_GPU8512(a, b, c, d, e);
    case 680: return MD_Et_GPU8513(a, b, c, d, e);
    case 681: return MD_Et_GPU860(a, b, c, d, e);
    case 682: return MD_Et_GPU861(a, b, c, d, e);
    case 683: return MD_Et_GPU862(a, b, c, d, e);
    case 684: return MD_Et_GPU863(a, b, c, d, e);
    case 685: return MD_Et_GPU864(a, b, c, d, e);
    case 686: return MD_Et_GPU865(a, b, c, d, e);
    case 687: return MD_Et_GPU866(a, b, c, d, e);
    case 688: return MD_Et_GPU867(a, b, c, d, e);
    case 689: return MD_Et_GPU868(a, b, c, d, e);
    case 690: return MD_Et_GPU869(a, b, c, d, e);
    case 691: return MD_Et_GPU8610(a, b, c, d, e);
    case 692: return MD_Et_GPU8611(a, b, c, d, e);
    case 693: return MD_Et_GPU8612(a, b, c, d, e);
    case 694: return MD_Et_GPU8613(a, b, c, d, e);
    case 695: return MD_Et_GPU8614(a, b, c, d, e);
    case 696: return MD_Et_GPU870(a, b, c, d, e);
    case 697: return MD_Et_GPU871(a, b, c, d, e);
    case 698: return MD_Et_GPU872(a, b, c, d, e);
    case 699: return MD_Et_GPU873(a, b, c, d, e);
    case 700: return MD_Et_GPU874(a, b, c, d, e);
    case 701: return MD_Et_GPU875(a, b, c, d, e);
    case 702: return MD_Et_GPU876(a, b, c, d, e);
    case 703: return MD_Et_GPU877(a, b, c, d, e);
    case 704: return MD_Et_GPU878(a, b, c, d, e);
    case 705: return MD_Et_GPU879(a, b, c, d, e);
    case 706: return MD_Et_GPU8710(a, b, c, d, e);
    case 707: return MD_Et_GPU8711(a, b, c, d, e);
    case 708: return MD_Et_GPU8712(a, b, c, d, e);
    case 709: return MD_Et_GPU8713(a, b, c, d, e);
    case 710: return MD_Et_GPU8714(a, b, c, d, e);
    case 711: return MD_Et_GPU8715(a, b, c, d, e);
    case 712: return MD_Et_GPU880(a, b, c, d, e);
    case 713: return MD_Et_GPU881(a, b, c, d, e);
    case 714: return MD_Et_GPU882(a, b, c, d, e);
    case 715: return MD_Et_GPU883(a, b, c, d, e);
    case 716: return MD_Et_GPU884(a, b, c, d, e);
    case 717: return MD_Et_GPU885(a, b, c, d, e);
    case 718: return MD_Et_GPU886(a, b, c, d, e);
    case 719: return MD_Et_GPU887(a, b, c, d, e);
    case 720: return MD_Et_GPU888(a, b, c, d, e);
    case 721: return MD_Et_GPU889(a, b, c, d, e);
    case 722: return MD_Et_GPU8810(a, b, c, d, e);
    case 723: return MD_Et_GPU8811(a, b, c, d, e);
    case 724: return MD_Et_GPU8812(a, b, c, d, e);
    case 725: return MD_Et_GPU8813(a, b, c, d, e);
    case 726: return MD_Et_GPU8814(a, b, c, d, e);
    case 727: return MD_Et_GPU8815(a, b, c, d, e);
    case 728: return MD_Et_GPU8816(a, b, c, d, e);
    default:
        sycl::ext::oneapi::experimental::printf("Invalid index\n");
        return -1.0;
    }
}

// MD法におけるEに関して，(i,l,t)の対応するデバイス関数を選択
inline real_t MD_Et_NonRecursion(int i, int l, int t, real_t alpha, real_t beta, real_t dist){
//    return MD_EtArray[4 * i * (10 + i) + (i + l) * (i + l + 1) / 2 + t](
    return callFunction(4 * i * (10 + i) + (i + l) * (i + l + 1) / 2 + t,
        alpha, beta, alpha + beta, dist,
        sycl::exp(-alpha * beta / (alpha + beta) * dist * dist));
}


// 1電子の計算で使用
inline real_t Et_GPU(int i, int l, int t, real_t alpha, real_t beta, real_t dist){
//                     real_t (*MD_EtArray[])(real_t, real_t, real_t, real_t, real_t) **MD_EtArray){
	if( i<0 || l<0 || t<0 || i+l<t){ // 範囲外の処理
        return 0.0;
    }else{
		return callFunction(4*i*(10+i) + (i+l)*(i+l+1)/2 + t,alpha, beta, alpha+beta, dist, 1.0);
	}
}

}
