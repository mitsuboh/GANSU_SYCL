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
//#include <dpct/dpct.hpp>
//#include <dpct/dpl_utils.hpp>
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

inline double MD_Et_grad000(double a, double b, double p, double d, double s, double g){
	return g;
}

inline double MD_Et_grad010(double a, double b, double p, double d, double s, double g){
	return a*(d*g + s)/p;
}

inline double MD_Et_grad011(double a, double b, double p, double d, double s, double g){
	return 0.5*g/p;
}

inline double MD_Et_grad020(double a, double b, double p, double d, double s, double g){
	return (d*(a*a)*(d*g + 2*s) + 0.5*g*p)/(p*p);
}

inline double MD_Et_grad021(double a, double b, double p, double d, double s, double g){
	return a*(d*g + s)/(p*p);
}

inline double MD_Et_grad022(double a, double b, double p, double d, double s, double g){
	return 0.25*g/(p*p);
}

inline double MD_Et_grad030(double a, double b, double p, double d, double s, double g){
	return a*((d*d*d)*(a*a)*g + 3.0*(d*d)*(a*a)*s + 1.5*d*g*p + 1.5*p*s)/(p*p*p);
}

inline double MD_Et_grad031(double a, double b, double p, double d, double s, double g){
	return (1.5*d*(a*a)*(d*g + 2*s) + 0.75*g*p)/(p*p*p);
}

inline double MD_Et_grad032(double a, double b, double p, double d, double s, double g){
	return 0.75*a*(d*g + s)/(p*p*p);
}

inline double MD_Et_grad033(double a, double b, double p, double d, double s, double g){
	return 0.125*g/(p*p*p);
}

inline double MD_Et_grad040(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d)*(a*a*a*a)*g + 4.0*(d*d*d)*(a*a*a*a)*s + 3.0*(d*d)*(a*a)*g*p + 6.0*d*(a*a)*p*s + 0.75*g*(p*p))/(p*p*p*p);
}

inline double MD_Et_grad041(double a, double b, double p, double d, double s, double g){
	return a*(2.0*(d*d*d)*(a*a)*g + 6.0*(d*d)*(a*a)*s + 3.0*d*g*p + 3.0*p*s)/(p*p*p*p);
}

inline double MD_Et_grad042(double a, double b, double p, double d, double s, double g){
	return (1.5*d*(a*a)*(d*g + 2*s) + 0.75*g*p)/(p*p*p*p);
}

inline double MD_Et_grad043(double a, double b, double p, double d, double s, double g){
	return 0.5*a*(d*g + s)/(p*p*p*p);
}

inline double MD_Et_grad044(double a, double b, double p, double d, double s, double g){
	return 0.0625*g/(p*p*p*p);
}

inline double MD_Et_grad050(double a, double b, double p, double d, double s, double g){
	return a*((d*d*d*d*d)*(a*a*a*a)*g + 5.0*(d*d*d*d)*(a*a*a*a)*s + 5.0*(d*d*d)*(a*a)*g*p + 15.0*(d*d)*(a*a)*p*s + 3.75*d*g*(p*p) + 3.75*(p*p)*s)/(p*p*p*p*p);
}

inline double MD_Et_grad051(double a, double b, double p, double d, double s, double g){
	return (2.5*(d*d*d*d)*(a*a*a*a)*g + 10.0*(d*d*d)*(a*a*a*a)*s + 7.5*(d*d)*(a*a)*g*p + 15.0*d*(a*a)*p*s + 1.875*g*(p*p))/(p*p*p*p*p);
}

inline double MD_Et_grad052(double a, double b, double p, double d, double s, double g){
	return a*(2.5*(d*d*d)*(a*a)*g + 7.5*(d*d)*(a*a)*s + 3.75*d*g*p + 3.75*p*s)/(p*p*p*p*p);
}

inline double MD_Et_grad053(double a, double b, double p, double d, double s, double g){
	return (1.25*d*(a*a)*(d*g + 2*s) + 0.625*g*p)/(p*p*p*p*p);
}

inline double MD_Et_grad054(double a, double b, double p, double d, double s, double g){
	return 0.3125*a*(d*g + s)/(p*p*p*p*p);
}

inline double MD_Et_grad055(double a, double b, double p, double d, double s, double g){
	return 0.03125*g/(p*p*p*p*p);
}

inline double MD_Et_grad060(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d)*(a*a*a*a*a*a)*g + 6.0*(d*d*d*d*d)*(a*a*a*a*a*a)*s + 7.5*(d*d*d*d)*(a*a*a*a)*g*p + 30.0*(d*d*d)*(a*a*a*a)*p*s + 11.25*(d*d)*(a*a)*g*(p*p) + 22.5*d*(a*a)*(p*p)*s + 1.875*g*(p*p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad061(double a, double b, double p, double d, double s, double g){
	return a*(3.0*(d*d*d*d*d)*(a*a*a*a)*g + 15.0*(d*d*d*d)*(a*a*a*a)*s + 15.0*(d*d*d)*(a*a)*g*p + 45.0*(d*d)*(a*a)*p*s + 11.25*d*g*(p*p) + 11.25*(p*p)*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad062(double a, double b, double p, double d, double s, double g){
	return (3.75*(d*d*d*d)*(a*a*a*a)*g + 15.0*(d*d*d)*(a*a*a*a)*s + 11.25*(d*d)*(a*a)*g*p + 22.5*d*(a*a)*p*s + 2.8125*g*(p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad063(double a, double b, double p, double d, double s, double g){
	return a*(2.5*(d*d*d)*(a*a)*g + 7.5*(d*d)*(a*a)*s + 3.75*d*g*p + 3.75*p*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad064(double a, double b, double p, double d, double s, double g){
	return (0.9375*d*(a*a)*(d*g + 2*s) + 0.46875*g*p)/(p*p*p*p*p*p);
}

inline double MD_Et_grad065(double a, double b, double p, double d, double s, double g){
	return 0.1875*a*(d*g + s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad066(double a, double b, double p, double d, double s, double g){
	return 0.015625*g/(p*p*p*p*p*p);
}

inline double MD_Et_grad070(double a, double b, double p, double d, double s, double g){
	return a*((d*d*d*d*d*d*d)*(a*a*a*a*a*a)*g + 7.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*s + 10.5*(d*d*d*d*d)*(a*a*a*a)*g*p + 52.5*(d*d*d*d)*(a*a*a*a)*p*s + 26.25*(d*d*d)*(a*a)*g*(p*p) + 78.75*(d*d)*(a*a)*(p*p)*s + 13.125*d*g*(p*p*p) + 13.125*(p*p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad071(double a, double b, double p, double d, double s, double g){
	return (3.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g + 21.0 *(d*d*d*d*d)*(a*a*a*a*a*a)*s + 26.25*(d*d*d*d)*(a*a*a*a)*g*p + 105.0*(d*d*d)*(a*a*a*a)*p*s + 39.375*(d*d)*(a*a)*g*(p*p) + 78.75*d*(a*a)*(p*p)*s + 6.5625*g*(p*p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad072(double a, double b, double p, double d, double s, double g){
	return a*(5.25*(d*d*d*d*d)*(a*a*a*a)*g + 26.25*(d*d*d*d)*(a*a*a*a)*s + 26.25*(d*d*d)*(a*a)*g*p + 78.75*(d*d)*(a*a)*p*s + 19.6875*d*g*(p*p) + 19.6875*(p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad073(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d)*(a*a*a*a)*g + 17.5*(d*d*d)*(a*a*a*a)*s + 13.125*(d*d)*(a*a)*g*p + 26.25*d*(a*a)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad074(double a, double b, double p, double d, double s, double g){
	return a*(2.1875*(d*d*d)*(a*a)*g + 6.5625*(d*d)*(a*a)*s + 3.28125*d*g*p + 3.28125*p*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad075(double a, double b, double p, double d, double s, double g){
	return (0.65625*d*(a*a)*(d*g + 2*s) + 0.328125*g*p)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad076(double a, double b, double p, double d, double s, double g){
	return 0.109375*a*(d*g + s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad077(double a, double b, double p, double d, double s, double g){
	return 0.0078125*g/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad080(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g + 8.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*s + 14.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p + 84.0*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s + 52.5*(d*d*d*d)*(a*a*a*a)*g*(p*p) + 210.0*(d*d*d)*(a*a*a*a)*(p*p)*s + 52.5*(d*d)*(a*a)*g*(p*p*p) + 105.0*d*(a*a)*(p*p*p)*s + 6.5625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad081(double a, double b, double p, double d, double s, double g){
	return a*(4.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*g + 28.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*s + 42.0*(d*d*d*d*d)*(a*a*a*a)*g*p + 210.0*(d*d*d*d)*(a*a*a*a)*p*s + 105.0*(d*d*d)*(a*a)*g*(p*p) + 315.0*(d*d)*(a*a)*(p*p)*s + 52.5*d*g*(p*p*p) + 52.5*(p*p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad082(double a, double b, double p, double d, double s, double g){
	return (7.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g + 42.0*(d*d*d*d*d)*(a*a*a*a*a*a)*s + 52.5*(d*d*d*d)*(a*a*a*a)*g*p + 210.0*(d*d*d)*(a*a*a*a)*p*s + 78.75*(d*d)*(a*a)*g*(p*p) + 157.5*d*(a*a)*(p*p)*s + 13.125*g*(p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad083(double a, double b, double p, double d, double s, double g){
	return a*(7.0*(d*d*d*d*d)*(a*a*a*a)*g + 35.0*(d*d*d*d)*(a*a*a*a)*s + 35.0*(d*d*d)*(a*a)*g*p + 105.0*(d*d)*(a*a)*p*s + 26.25*d*g*(p*p) + 26.25*(p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad084(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d)*(a*a*a*a)*g + 17.5*(d*d*d)*(a*a*a*a)*s + 13.125*(d*d)*(a*a)*g*p + 26.25*d*(a*a)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad085(double a, double b, double p, double d, double s, double g){
	return a*(1.75*(d*d*d)*(a*a)*g + 5.25*(d*d)*(a*a)*s + 2.625*d*g*p + 2.625*p*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad086(double a, double b, double p, double d, double s, double g){
	return (0.4375*d*(a*a)*(d*g + 2*s) + 0.21875*g*p)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad087(double a, double b, double p, double d, double s, double g){
	return 0.0625*a*(d*g + s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad088(double a, double b, double p, double d, double s, double g){
	return 0.00390625*g/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad100(double a, double b, double p, double d, double s, double g){
	return b*(-d*g - s)/p;
}

inline double MD_Et_grad101(double a, double b, double p, double d, double s, double g){
	return 0.5*g/p;
}

inline double MD_Et_grad110(double a, double b, double p, double d, double s, double g){
	return (-d*a*b*(d*g + 2*s) + 0.5*g*p)/(p*p);
}

inline double MD_Et_grad111(double a, double b, double p, double d, double s, double g){
	return 0.5*(d*a*g - d*b*g + a*s - b*s)/(p*p);
}

inline double MD_Et_grad112(double a, double b, double p, double d, double s, double g){
	return 0.25*g/(p*p);
}

inline double MD_Et_grad120(double a, double b, double p, double d, double s, double g){
	return (a*p*(d*g + s) - b*(d*(d*(a*a)*(d*g + 2*s) + 0.5*g*p) + s*((d*d)*(a*a) + 0.5*p)))/(p*p*p);
}

inline double MD_Et_grad121(double a, double b, double p, double d, double s, double g){
	return (0.5*d*(a*a)*(d*g + 2*s) - d*a*b*(d*g + 2*s) + 0.75*g*p)/(p*p*p);
}

inline double MD_Et_grad122(double a, double b, double p, double d, double s, double g){
	return (0.5*d*a*g - 0.25*d*b*g + 0.5*a*s - 0.25*b*s)/(p*p*p);
}

inline double MD_Et_grad123(double a, double b, double p, double d, double s, double g){
	return 0.125*g/(p*p*p);
}

inline double MD_Et_grad130(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d)*(a*a*a)*b*g - 4.0*(d*d*d)*(a*a*a)*b*s + 1.5*(d*d)*(a*a)*g*p - 1.5*(d*d)*a*b*g*p + 3.0*d*(a*a)*p*s - 3.0*d*a*b*p*s + 0.75*g*(p*p))/(p*p*p*p);
}

inline double MD_Et_grad131(double a, double b, double p, double d, double s, double g){
	return (0.5*(d*d*d)*(a*a*a)*g - 1.5*(d*d*d)*(a*a)*b*g + 1.5*(d*d)*(a*a*a)*s - 4.5*(d*d)*(a*a)*b*s + 2.25*d*a*g*p - 0.75*d*b*g*p + 2.25*a*p*s - 0.75*b*p*s)/(p*p*p*p);
}

inline double MD_Et_grad132(double a, double b, double p, double d, double s, double g){
	return 0.75*(d*(a*a)*(d*g + 2*s) - d*a*b*(d*g + 2*s) + g*p)/(p*p*p*p);
}

inline double MD_Et_grad133(double a, double b, double p, double d, double s, double g){
	return (0.375*d*a*g - 0.125*d*b*g + 0.375*a*s - 0.125*b*s)/(p*p*p*p);
}

inline double MD_Et_grad134(double a, double b, double p, double d, double s, double g){
	return 0.0625*g/(p*p*p*p);
}

inline double MD_Et_grad140(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d)*(a*a*a*a)*b*g - 5.0*(d*d*d*d)*(a*a*a*a)*b*s + 2.0*(d*d*d)*(a*a*a)*g*p - 3.0*(d*d*d)*(a*a)*b*g*p + 6.0*(d*d)*(a*a*a)*p*s - 9.0*(d*d)*(a*a)*b*p*s + 3.0*d*a*g*(p*p) - 0.75*d*b*g*(p*p) + 3.0*a*(p*p)*s - 0.75*b*(p*p)*s)/(p*p*p*p*p);
}

inline double MD_Et_grad141(double a, double b, double p, double d, double s, double g){
	return (0.5*(d*d*d*d)*(a*a*a*a)*g - 2.0*(d*d*d*d)*(a*a*a)*b*g + 2.0*(d*d*d)*(a*a*a*a)*s - 8.0*(d*d*d)*(a*a*a)*b*s + 4.5*(d*d)*(a*a)*g*p - 3.0*(d*d)*a*b*g*p + 9.0*d*(a*a)*p*s - 6.0*d*a*b*p*s + 1.875*g*(p*p))/(p*p*p*p*p);
}

inline double MD_Et_grad142(double a, double b, double p, double d, double s, double g){
	return ((d*d*d)*(a*a*a)*g - 1.5*(d*d*d)*(a*a)*b*g + 3.0*(d*d)*(a*a*a)*s - 4.5*(d*d)*(a*a)*b*s + 3.0*d*a*g*p - 0.75*d*b*g*p + 3.0*a*p*s - 0.75*b*p*s)/(p*p*p*p*p);
}

inline double MD_Et_grad143(double a, double b, double p, double d, double s, double g){
	return (0.75*d*(a*a)*(d*g + 2*s) - 0.5*d*a*b*(d*g + 2*s) + 0.625*g*p)/(p*p*p*p*p);
}

inline double MD_Et_grad144(double a, double b, double p, double d, double s, double g){
	return (0.25*d*a*g - 0.0625*d*b*g + 0.25*a*s - 0.0625*b*s)/(p*p*p*p*p);
}

inline double MD_Et_grad145(double a, double b, double p, double d, double s, double g){
	return 0.03125*g/(p*p*p*p*p);
}

inline double MD_Et_grad150(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d)*(a*a*a*a*a)*b*g - 6.0*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 2.5*(d*d*d*d)*(a*a*a*a)*g*p - 5.0*(d*d*d*d)*(a*a*a)*b*g*p + 10.0*(d*d*d)*(a*a*a*a)*p*s - 20.0*(d*d*d)*(a*a*a)*b*p*s + 7.5*(d*d)*(a*a)*g*(p*p) - 3.75*(d*d)*a*b*g*(p*p) + 15.0*d*(a*a)*(p*p)*s - 7.5*d*a*b*(p*p)*s + 1.875*g*(p*p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad151(double a, double b, double p, double d, double s, double g){
	return (0.5*(d*d*d*d*d)*(a*a*a*a*a)*g - 2.5*(d*d*d*d*d)*(a*a*a*a)*b*g + 2.5*(d*d*d*d)*(a*a*a*a*a)*s - 12.5*(d*d*d*d)*(a*a*a*a)*b*s + 7.5*(d*d*d)*(a*a*a)*g*p - 7.5*(d*d*d)*(a*a)*b*g*p + 22.5*(d*d)*(a*a*a)*p*s - 22.5*(d*d)*(a*a)*b*p*s + 9.375*d*a*g*(p*p) - 1.875*d*b*g*(p*p) + 9.375*a*(p*p)*s - 1.875*b*(p*p)*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad152(double a, double b, double p, double d, double s, double g){
	return (1.25*(d*d*d*d)*(a*a*a*a)*g - 2.5*(d*d*d*d)*(a*a*a)*b*g + 5.0*(d*d*d)*(a*a*a*a)*s - 10.0*(d*d*d)*(a*a*a)*b*s + 7.5*(d*d)*(a*a)*g*p - 3.75*(d*d)*a*b*g*p + 15.0*d*(a*a)*p*s - 7.5*d*a*b*p*s + 2.8125*g*(p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad153(double a, double b, double p, double d, double s, double g){
	return (1.25*(d*d*d)*(a*a*a)*g - 1.25*(d*d*d)*(a*a)*b*g + 3.75*(d*d)*(a*a*a)*s - 3.75*(d*d)*(a*a)*b*s + 3.125*d*a*g*p - 0.625*d*b*g*p + 3.125*a*p*s - 0.625*b*p*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad154(double a, double b, double p, double d, double s, double g){
	return (0.625*d*(a*a)*(d*g + 2*s) - 0.3125*d*a*b*(d*g + 2*s) + 0.46875*g*p)/(p*p*p*p*p*p);
}

inline double MD_Et_grad155(double a, double b, double p, double d, double s, double g){
	return (0.15625*d*a*g - 0.03125*d*b*g + 0.15625*a*s - 0.03125*b*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad156(double a, double b, double p, double d, double s, double g){
	return 0.015625*g/(p*p*p*p*p*p);
}

inline double MD_Et_grad160(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g - 7.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 3.0*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 7.5*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 15.0*(d*d*d*d)*(a*a*a*a*a)*p*s - 37.5*(d*d*d*d)*(a*a*a*a)*b*p*s + 15.0*(d*d*d)*(a*a*a)*g*(p*p) - 11.25*(d*d*d)*(a*a)*b*g*(p*p) + 45.0*(d*d)*(a*a*a)*(p*p)*s - 33.75*(d*d)*(a*a)*b*(p*p)*s + 11.25*d*a*g*(p*p*p) - 1.875*d*b*g*(p*p*p) + 11.25*a*(p*p*p)*s - 1.875*b*(p*p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad161(double a, double b, double p, double d, double s, double g){
	return (0.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 3.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 3.0*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 18.0*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 11.25*(d*d*d*d)*(a*a*a*a)*g*p - 15.0*(d*d*d*d)*(a*a*a)*b*g*p + 45.0*(d*d*d)*(a*a*a*a)*p*s - 60.0*(d*d*d)*(a*a*a)*b*p*s + 28.125*(d*d)*(a*a)*g*(p*p) - 11.25*(d*d)*a*b*g*(p*p) + 56.25*d*(a*a)*(p*p)*s - 22.5*d*a*b*(p*p)*s + 6.5625*g*(p*p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad162(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d)*(a*a*a*a*a)*g - 3.75*(d*d*d*d*d)*(a*a*a*a)*b*g + 7.5*(d*d*d*d)*(a*a*a*a*a)*s - 18.75*(d*d*d*d)*(a*a*a*a)*b*s + 15.0*(d*d*d)*(a*a*a)*g*p - 11.25*(d*d*d)*(a*a)*b*g*p + 45.0*(d*d)*(a*a*a)*p*s - 33.75*(d*d)*(a*a)*b*p*s + 16.875*d*a*g*(p*p) - 2.8125*d*b*g*(p*p) + 16.875*a*(p*p)*s - 2.8125*b*(p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad163(double a, double b, double p, double d, double s, double g){
	return (1.875*(d*d*d*d)*(a*a*a*a)*g - 2.5*(d*d*d*d)*(a*a*a)*b*g + 7.5*(d*d*d)*(a*a*a*a)*s - 10.0*(d*d*d)*(a*a*a)*b*s + 9.375*(d*d)*(a*a)*g*p - 3.75*(d*d)*a*b*g*p + 18.75*d*(a*a)*p*s - 7.5*d*a*b*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad164(double a, double b, double p, double d, double s, double g){
	return (1.25*(d*d*d)*(a*a*a)*g - 0.9375*(d*d*d)*(a*a)*b*g + 3.75*(d*d)*(a*a*a)*s - 2.8125*(d*d)*(a*a)*b*s + 2.8125*d*a*g*p - 0.46875*d*b*g*p + 2.8125*a*p*s - 0.46875*b*p*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad165(double a, double b, double p, double d, double s, double g){
	return (0.46875*d*(a*a)*(d*g + 2*s) - 0.1875*d*a*b*(d*g + 2*s) + 0.328125*g*p)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad166(double a, double b, double p, double d, double s, double g){
	return (0.09375*d*a*g - 0.015625*d*b*g + 0.09375*a*s - 0.015625*b*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad167(double a, double b, double p, double d, double s, double g){
	return 0.0078125*g/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad170(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g - 8.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 3.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 10.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 21.0 *(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 63.0*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 26.25*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 26.25*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 105.0*(d*d*d)*(a*a*a*a)*(p*p)*s - 105.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 39.375*(d*d)*(a*a)*g*(p*p*p) - 13.125*(d*d)*a*b*g*(p*p*p) + 78.75*d*(a*a)*(p*p*p)*s - 26.25*d*a*b*(p*p*p)*s + 6.5625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad171(double a, double b, double p, double d, double s, double g){
	return (0.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 3.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 3.5*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 24.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 15.75*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 26.25*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 78.75*(d*d*d*d)*(a*a*a*a*a)*p*s - 131.25*(d*d*d*d)*(a*a*a*a)*b*p*s + 65.625*(d*d*d)*(a*a*a)*g*(p*p) - 39.375*(d*d*d)*(a*a)*b*g*(p*p) + 196.875*(d*d)*(a*a*a)*(p*p)*s - 118.125*(d*d)*(a*a)*b*(p*p)*s + 45.9375*d*a*g*(p*p*p) - 6.5625*d*b*g*(p*p*p) + 45.9375*a*(p*p*p)*s - 6.5625*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad172(double a, double b, double p, double d, double s, double g){
	return (1.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 5.25*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 10.5*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 31.5*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 26.25*(d*d*d*d)*(a*a*a*a)*g*p - 26.25*(d*d*d*d)*(a*a*a)*b*g*p + 105.0*(d*d*d)*(a*a*a*a)*p*s - 105.0*(d*d*d)*(a*a*a)*b*p*s + 59.0625*(d*d)*(a*a)*g*(p*p) - 19.6875*(d*d)*a*b*g*(p*p) + 118.125*d*(a*a)*(p*p)*s - 39.375*d*a*b*(p*p)*s + 13.125*g*(p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad173(double a, double b, double p, double d, double s, double g){
	return (2.625*(d*d*d*d*d)*(a*a*a*a*a)*g - 4.375*(d*d*d*d*d)*(a*a*a*a)*b*g + 13.125*(d*d*d*d)*(a*a*a*a*a)*s - 21.875*(d*d*d*d)*(a*a*a*a)*b*s + 21.875*(d*d*d)*(a*a*a)*g*p - 13.125*(d*d*d)*(a*a)*b*g*p + 65.625*(d*d)*(a*a*a)*p*s - 39.375*(d*d)*(a*a)*b*p*s + 22.96875*d*a*g*(p*p) - 3.28125*d*b*g*(p*p) + 22.96875*a*(p*p)*s - 3.28125*b*(p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad174(double a, double b, double p, double d, double s, double g){
	return (2.1875*(d*d*d*d)*(a*a*a*a)*g - 2.1875*(d*d*d*d)*(a*a*a)*b*g + 8.75*(d*d*d)*(a*a*a*a)*s - 8.75*(d*d*d)*(a*a*a)*b*s + 9.84375*(d*d)*(a*a)*g*p - 3.28125*(d*d)*a*b*g*p + 19.6875*d*(a*a)*p*s - 6.5625*d*a*b*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad175(double a, double b, double p, double d, double s, double g){
	return (1.09375*(d*d*d)*(a*a*a)*g - 0.65625*(d*d*d)*(a*a)*b*g + 3.28125*(d*d)*(a*a*a)*s - 1.96875*(d*d)*(a*a)*b*s + 2.296875*d*a*g*p - 0.328125*d*b*g*p + 2.296875*a*p*s - 0.328125*b*p*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad176(double a, double b, double p, double d, double s, double g){
	return (0.328125*d*(a*a)*(d*g + 2*s) - 0.109375*d*a*b*(d*g + 2*s) + 0.21875*g*p)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad177(double a, double b, double p, double d, double s, double g){
	return (0.0546875*d*a*g - 0.0078125*d*b*g + 0.0546875*a*s - 0.0078125*b*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad178(double a, double b, double p, double d, double s, double g){
	return 0.00390625*g/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad180(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g - 9.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*s + 4.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 14.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 28.0*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 98.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 42.0*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 52.5*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 210.0*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 262.5*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 105.0*(d*d*d)*(a*a*a)*g*(p*p*p) - 52.5*(d*d*d)*(a*a)*b*g*(p*p*p) + 315.0*(d*d)*(a*a*a)*(p*p*p)*s - 157.5*(d*d)*(a*a)*b*(p*p*p)*s + 52.5*d*a*g*(p*p*p*p) - 6.5625*d*b*g*(p*p*p*p) + 52.5*a*(p*p*p*p)*s - 6.5625*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad181(double a, double b, double p, double d, double s, double g){
	return (0.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g - 4.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 4.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*s - 32.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 21.0 *(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 42.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 126.0*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 252.0*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 131.25*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 105.0*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 525.0*(d*d*d)*(a*a*a*a)*(p*p)*s - 420.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 183.75*(d*d)*(a*a)*g*(p*p*p) - 52.5*(d*d)*a*b*g*(p*p*p) + 367.5*d*(a*a)*(p*p*p)*s - 105.0*d*a*b*(p*p*p)*s + 29.53125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad182(double a, double b, double p, double d, double s, double g){
	return (2.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 7.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 14.0*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 49.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 42.0*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 52.5*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 210.0*(d*d*d*d)*(a*a*a*a*a)*p*s - 262.5*(d*d*d*d)*(a*a*a*a)*b*p*s + 157.5*(d*d*d)*(a*a*a)*g*(p*p) - 78.75*(d*d*d)*(a*a)*b*g*(p*p) + 472.5*(d*d)*(a*a*a)*(p*p)*s - 236.25*(d*d)*(a*a)*b*(p*p)*s + 105.0*d*a*g*(p*p*p) - 13.125*d*b*g*(p*p*p) + 105.0*a*(p*p*p)*s - 13.125*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad183(double a, double b, double p, double d, double s, double g){
	return (3.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 7.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 21.0 *(d*d*d*d*d)*(a*a*a*a*a*a)*s - 42.0*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 43.75*(d*d*d*d)*(a*a*a*a)*g*p - 35.0*(d*d*d*d)*(a*a*a)*b*g*p + 175.0*(d*d*d)*(a*a*a*a)*p*s - 140.0*(d*d*d)*(a*a*a)*b*p*s + 91.875*(d*d)*(a*a)*g*(p*p) - 26.25*(d*d)*a*b*g*(p*p) + 183.75*d*(a*a)*(p*p)*s - 52.5*d*a*b*(p*p)*s + 19.6875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad184(double a, double b, double p, double d, double s, double g){
	return (3.5*(d*d*d*d*d)*(a*a*a*a*a)*g - 4.375*(d*d*d*d*d)*(a*a*a*a)*b*g + 17.5*(d*d*d*d)*(a*a*a*a*a)*s - 21.875*(d*d*d*d)*(a*a*a*a)*b*s + 26.25*(d*d*d)*(a*a*a)*g*p - 13.125*(d*d*d)*(a*a)*b*g*p + 78.75*(d*d)*(a*a*a)*p*s - 39.375*(d*d)*(a*a)*b*p*s + 26.25*d*a*g*(p*p) - 3.28125*d*b*g*(p*p) + 26.25*a*(p*p)*s - 3.28125*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad185(double a, double b, double p, double d, double s, double g){
	return (2.1875*(d*d*d*d)*(a*a*a*a)*g - 1.75*(d*d*d*d)*(a*a*a)*b*g + 8.75*(d*d*d)*(a*a*a*a)*s - 7.0*(d*d*d)*(a*a*a)*b*s + 9.1875*(d*d)*(a*a)*g*p - 2.625*(d*d)*a*b*g*p + 18.375*d*(a*a)*p*s - 5.25*d*a*b*p*s + 2.953125*g*(p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad186(double a, double b, double p, double d, double s, double g){
	return (0.875*(d*d*d)*(a*a*a)*g - 0.4375*(d*d*d)*(a*a)*b*g + 2.625*(d*d)*(a*a*a)*s - 1.3125*(d*d)*(a*a)*b*s + 1.75*d*a*g*p - 0.21875*d*b*g*p + 1.75*a*p*s - 0.21875*b*p*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad187(double a, double b, double p, double d, double s, double g){
	return (0.21875*d*(a*a)*(d*g + 2*s) - 0.0625*d*a*b*(d*g + 2*s) + 0.140625*g*p)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad188(double a, double b, double p, double d, double s, double g){
	return (0.03125*d*a*g - 0.00390625*d*b*g + 0.03125*a*s - 0.00390625*b*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad189(double a, double b, double p, double d, double s, double g){
	return 0.001953125*g/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad200(double a, double b, double p, double d, double s, double g){
	return (d*(b*b)*(d*g + 2*s) + 0.5*g*p)/(p*p);
}

inline double MD_Et_grad201(double a, double b, double p, double d, double s, double g){
	return b*(-d*g - s)/(p*p);
}

inline double MD_Et_grad202(double a, double b, double p, double d, double s, double g){
	return 0.25*g/(p*p);
}

inline double MD_Et_grad210(double a, double b, double p, double d, double s, double g){
	return (b*(d*(d*a*b*(d*g + 2*s) - 0.5*g*p) + s*((d*d)*a*b - 0.5*p)) + 0.5*p*(d*a*g - d*b*g + a*s - b*s))/(p*p*p);
}

inline double MD_Et_grad211(double a, double b, double p, double d, double s, double g){
	return (-(d*d)*a*b*g + 0.5*(d*d)*(b*b)*g - 2.0*d*a*b*s + d*(b*b)*s + 0.75*g*p)/(p*p*p);
}

inline double MD_Et_grad212(double a, double b, double p, double d, double s, double g){
	return (0.25*d*a*g - 0.5*d*b*g + 0.25*a*s - 0.5*b*s)/(p*p*p);
}

inline double MD_Et_grad213(double a, double b, double p, double d, double s, double g){
	return 0.125*g/(p*p*p);
}

inline double MD_Et_grad220(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d)*(a*a)*(b*b)*g + 4.0*(d*d*d)*(a*a)*(b*b)*s + 0.5*(d*d)*(a*a)*g*p - 2.0*(d*d)*a*b*g*p + 0.5*(d*d)*(b*b)*g*p + d*(a*a)*p*s - 4.0*d*a*b*p*s + d*(b*b)*p*s + 0.75*g*(p*p))/(p*p*p*p);
}

inline double MD_Et_grad221(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d)*(a*a)*b*g + (d*d*d)*a*(b*b)*g - 3.0*(d*d)*(a*a)*b*s + 3.0*(d*d)*a*(b*b)*s + 1.5*d*a*g*p - 1.5*d*b*g*p + 1.5*a*p*s - 1.5*b*p*s)/(p*p*p*p);
}

inline double MD_Et_grad222(double a, double b, double p, double d, double s, double g){
	return (0.25*(d*d)*(a*a)*g - (d*d)*a*b*g + 0.25*(d*d)*(b*b)*g + 0.5*d*(a*a)*s - 2.0*d*a*b*s + 0.5*d*(b*b)*s + 0.75*g*p)/(p*p*p*p);
}

inline double MD_Et_grad223(double a, double b, double p, double d, double s, double g){
	return 0.25*(d*a*g - d*b*g + a*s - b*s)/(p*p*p*p);
}

inline double MD_Et_grad224(double a, double b, double p, double d, double s, double g){
	return 0.0625*g/(p*p*p*p);
}

inline double MD_Et_grad230(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d)*(a*a*a)*(b*b)*g + 5.0*(d*d*d*d)*(a*a*a)*(b*b)*s + 0.5*(d*d*d)*(a*a*a)*g*p - 3.0*(d*d*d)*(a*a)*b*g*p + 1.5*(d*d*d)*a*(b*b)*g*p + 1.5*(d*d)*(a*a*a)*p*s - 9.0*(d*d)*(a*a)*b*p*s + 4.5*(d*d)*a*(b*b)*p*s + 2.25*d*a*g*(p*p) - 1.5*d*b*g*(p*p) + 2.25*a*(p*p)*s - 1.5*b*(p*p)*s)/(p*p*p*p*p);
}

inline double MD_Et_grad231(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d)*(a*a*a)*b*g + 1.5*(d*d*d*d)*(a*a)*(b*b)*g - 4.0*(d*d*d)*(a*a*a)*b*s + 6.0*(d*d*d)*(a*a)*(b*b)*s + 2.25*(d*d)*(a*a)*g*p - 4.5*(d*d)*a*b*g*p + 0.75*(d*d)*(b*b)*g*p + 4.5*d*(a*a)*p*s - 9.0*d*a*b*p*s + 1.5*d*(b*b)*p*s + 1.875*g*(p*p))/(p*p*p*p*p);
}

inline double MD_Et_grad232(double a, double b, double p, double d, double s, double g){
	return (0.25*(d*d*d)*(a*a*a)*g - 1.5*(d*d*d)*(a*a)*b*g + 0.75*(d*d*d)*a*(b*b)*g + 0.75*(d*d)*(a*a*a)*s - 4.5*(d*d)*(a*a)*b*s + 2.25*(d*d)*a*(b*b)*s + 2.25*d*a*g*p - 1.5*d*b*g*p + 2.25*a*p*s - 1.5*b*p*s)/(p*p*p*p*p);
}

inline double MD_Et_grad233(double a, double b, double p, double d, double s, double g){
	return (0.375*(d*d)*(a*a)*g - 0.75*(d*d)*a*b*g + 0.125*(d*d)*(b*b)*g + 0.75*d*(a*a)*s - 1.5*d*a*b*s + 0.25*d*(b*b)*s + 0.625*g*p)/(p*p*p*p*p);
}

inline double MD_Et_grad234(double a, double b, double p, double d, double s, double g){
	return (0.1875*d*a*g - 0.125*d*b*g + 0.1875*a*s - 0.125*b*s)/(p*p*p*p*p);
}

inline double MD_Et_grad235(double a, double b, double p, double d, double s, double g){
	return 0.03125*g/(p*p*p*p*p);
}

inline double MD_Et_grad240(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g + 6.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s + 0.5*(d*d*d*d)*(a*a*a*a)*g*p - 4.0*(d*d*d*d)*(a*a*a)*b*g*p + 3.0*(d*d*d*d)*(a*a)*(b*b)*g*p + 2.0*(d*d*d)*(a*a*a*a)*p*s - 16.0*(d*d*d)*(a*a*a)*b*p*s + 12.0*(d*d*d)*(a*a)*(b*b)*p*s + 4.5*(d*d)*(a*a)*g*(p*p) - 6.0*(d*d)*a*b*g*(p*p) + 0.75*(d*d)*(b*b)*g*(p*p) + 9.0*d*(a*a)*(p*p)*s - 12.0*d*a*b*(p*p)*s + 1.5*d*(b*b)*(p*p)*s + 1.875*g*(p*p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad241(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d)*(a*a*a*a)*b*g + 2.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 5.0*(d*d*d*d)*(a*a*a*a)*b*s + 10.0*(d*d*d*d)*(a*a*a)*(b*b)*s + 3.0*(d*d*d)*(a*a*a)*g*p - 9.0*(d*d*d)*(a*a)*b*g*p + 3.0*(d*d*d)*a*(b*b)*g*p + 9.0*(d*d)*(a*a*a)*p*s - 27.0*(d*d)*(a*a)*b*p*s + 9.0*(d*d)*a*(b*b)*p*s + 7.5*d*a*g*(p*p) - 3.75*d*b*g*(p*p) + 7.5*a*(p*p)*s - 3.75*b*(p*p)*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad242(double a, double b, double p, double d, double s, double g){
	return (0.25*(d*d*d*d)*(a*a*a*a)*g - 2.0*(d*d*d*d)*(a*a*a)*b*g + 1.5*(d*d*d*d)*(a*a)*(b*b)*g + (d*d*d)*(a*a*a*a)*s - 8.0*(d*d*d)*(a*a*a)*b*s + 6.0*(d*d*d)*(a*a)*(b*b)*s + 4.5*(d*d)*(a*a)*g*p - 6.0*(d*d)*a*b*g*p + 0.75*(d*d)*(b*b)*g*p + 9.0*d*(a*a)*p*s - 12.0*d*a*b*p*s + 1.5*d*(b*b)*p*s + 2.8125*g*(p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad243(double a, double b, double p, double d, double s, double g){
	return (0.5*(d*d*d)*(a*a*a)*g - 1.5*(d*d*d)*(a*a)*b*g + 0.5*(d*d*d)*a*(b*b)*g + 1.5*(d*d)*(a*a*a)*s - 4.5*(d*d)*(a*a)*b*s + 1.5*(d*d)*a*(b*b)*s + 2.5*d*a*g*p - 1.25*d*b*g*p + 2.5*a*p*s - 1.25*b*p*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad244(double a, double b, double p, double d, double s, double g){
	return (0.375*(d*d)*(a*a)*g - 0.5*(d*d)*a*b*g + 0.0625*(d*d)*(b*b)*g + 0.75*d*(a*a)*s - d*a*b*s + 0.125*d*(b*b)*s + 0.46875*g*p)/(p*p*p*p*p*p);
}

inline double MD_Et_grad245(double a, double b, double p, double d, double s, double g){
	return (0.125*d*a*g - 0.0625*d*b*g + 0.125*a*s - 0.0625*b*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad246(double a, double b, double p, double d, double s, double g){
	return 0.015625*g/(p*p*p*p*p*p);
}

inline double MD_Et_grad250(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g + 7.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s + 0.5*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 5.0*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 5.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p + 2.5*(d*d*d*d)*(a*a*a*a*a)*p*s - 25.0*(d*d*d*d)*(a*a*a*a)*b*p*s + 25.0*(d*d*d*d)*(a*a*a)*(b*b)*p*s + 7.5*(d*d*d)*(a*a*a)*g*(p*p) - 15.0*(d*d*d)*(a*a)*b*g*(p*p) + 3.75*(d*d*d)*a*(b*b)*g*(p*p) + 22.5*(d*d)*(a*a*a)*(p*p)*s - 45.0*(d*d)*(a*a)*b*(p*p)*s + 11.25*(d*d)*a*(b*b)*(p*p)*s + 9.375*d*a*g*(p*p*p) - 3.75*d*b*g*(p*p*p) + 9.375*a*(p*p*p)*s - 3.75*b*(p*p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad251(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 2.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 6.0*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 15.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s + 3.75*(d*d*d*d)*(a*a*a*a)*g*p - 15.0*(d*d*d*d)*(a*a*a)*b*g*p + 7.5*(d*d*d*d)*(a*a)*(b*b)*g*p + 15.0*(d*d*d)*(a*a*a*a)*p*s - 60.0*(d*d*d)*(a*a*a)*b*p*s + 30.0*(d*d*d)*(a*a)*(b*b)*p*s + 18.75*(d*d)*(a*a)*g*(p*p) - 18.75*(d*d)*a*b*g*(p*p) + 1.875*(d*d)*(b*b)*g*(p*p) + 37.5*d*(a*a)*(p*p)*s - 37.5*d*a*b*(p*p)*s + 3.75*d*(b*b)*(p*p)*s + 6.5625*g*(p*p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad252(double a, double b, double p, double d, double s, double g){
	return (0.25*(d*d*d*d*d)*(a*a*a*a*a)*g - 2.5*(d*d*d*d*d)*(a*a*a*a)*b*g + 2.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g + 1.25*(d*d*d*d)*(a*a*a*a*a)*s - 12.5*(d*d*d*d)*(a*a*a*a)*b*s + 12.5*(d*d*d*d)*(a*a*a)*(b*b)*s + 7.5*(d*d*d)*(a*a*a)*g*p - 15.0*(d*d*d)*(a*a)*b*g*p + 3.75*(d*d*d)*a*(b*b)*g*p + 22.5*(d*d)*(a*a*a)*p*s - 45.0*(d*d)*(a*a)*b*p*s + 11.25*(d*d)*a*(b*b)*p*s + 14.0625*d*a*g*(p*p) - 5.625*d*b*g*(p*p) + 14.0625*a*(p*p)*s - 5.625*b*(p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad253(double a, double b, double p, double d, double s, double g){
	return (0.625*(d*d*d*d)*(a*a*a*a)*g - 2.5*(d*d*d*d)*(a*a*a)*b*g + 1.25*(d*d*d*d)*(a*a)*(b*b)*g + 2.5*(d*d*d)*(a*a*a*a)*s - 10.0*(d*d*d)*(a*a*a)*b*s + 5.0*(d*d*d)*(a*a)*(b*b)*s + 6.25*(d*d)*(a*a)*g*p - 6.25*(d*d)*a*b*g*p + 0.625*(d*d)*(b*b)*g*p + 12.5*d*(a*a)*p*s - 12.5*d*a*b*p*s + 1.25*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad254(double a, double b, double p, double d, double s, double g){
	return (0.625*(d*d*d)*(a*a*a)*g - 1.25*(d*d*d)*(a*a)*b*g + 0.3125*(d*d*d)*a*(b*b)*g + 1.875*(d*d)*(a*a*a)*s - 3.75*(d*d)*(a*a)*b*s + 0.9375*(d*d)*a*(b*b)*s + 2.34375*d*a*g*p - 0.9375*d*b*g*p + 2.34375*a*p*s - 0.9375*b*p*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad255(double a, double b, double p, double d, double s, double g){
	return (0.3125*(d*d)*(a*a)*g - 0.3125*(d*d)*a*b*g + 0.03125*(d*d)*(b*b)*g + 0.625*d*(a*a)*s - 0.625*d*a*b*s + 0.0625*d*(b*b)*s + 0.328125*g*p)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad256(double a, double b, double p, double d, double s, double g){
	return (0.078125*d*a*g - 0.03125*d*b*g + 0.078125*a*s - 0.03125*b*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad257(double a, double b, double p, double d, double s, double g){
	return 0.0078125*g/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad260(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g + 8.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s + 0.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 6.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 7.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p + 3.0*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 36.0*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 45.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s + 11.25*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 30.0*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 11.25*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) + 45.0*(d*d*d)*(a*a*a*a)*(p*p)*s - 120.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 45.0*(d*d*d)*(a*a)*(b*b)*(p*p)*s + 28.125*(d*d)*(a*a)*g*(p*p*p) - 22.5*(d*d)*a*b*g*(p*p*p) + 1.875*(d*d)*(b*b)*g*(p*p*p) + 56.25*d*(a*a)*(p*p*p)*s - 45.0*d*a*b*(p*p*p)*s + 3.75*d*(b*b)*(p*p*p)*s + 6.5625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad261(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 3.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 7.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 21.0 *(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s + 4.5*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 22.5*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 15.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p + 22.5*(d*d*d*d)*(a*a*a*a*a)*p*s - 112.5*(d*d*d*d)*(a*a*a*a)*b*p*s + 75.0*(d*d*d*d)*(a*a*a)*(b*b)*p*s + 37.5*(d*d*d)*(a*a*a)*g*(p*p) - 56.25*(d*d*d)*(a*a)*b*g*(p*p) + 11.25*(d*d*d)*a*(b*b)*g*(p*p) + 112.5*(d*d)*(a*a*a)*(p*p)*s - 168.75*(d*d)*(a*a)*b*(p*p)*s + 33.75*(d*d)*a*(b*b)*(p*p)*s + 39.375*d*a*g*(p*p*p) - 13.125*d*b*g*(p*p*p) + 39.375*a*(p*p*p)*s - 13.125*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad262(double a, double b, double p, double d, double s, double g){
	return (0.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 3.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 3.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g + 1.5*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 18.0*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 22.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s + 11.25*(d*d*d*d)*(a*a*a*a)*g*p - 30.0*(d*d*d*d)*(a*a*a)*b*g*p + 11.25*(d*d*d*d)*(a*a)*(b*b)*g*p + 45.0*(d*d*d)*(a*a*a*a)*p*s - 120.0*(d*d*d)*(a*a*a)*b*p*s + 45.0*(d*d*d)*(a*a)*(b*b)*p*s + 42.1875*(d*d)*(a*a)*g*(p*p) - 33.75*(d*d)*a*b*g*(p*p) + 2.8125*(d*d)*(b*b)*g*(p*p) + 84.375*d*(a*a)*(p*p)*s - 67.5*d*a*b*(p*p)*s + 5.625*d*(b*b)*(p*p)*s + 13.125*g*(p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad263(double a, double b, double p, double d, double s, double g){
	return (0.75*(d*d*d*d*d)*(a*a*a*a*a)*g - 3.75*(d*d*d*d*d)*(a*a*a*a)*b*g + 2.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g + 3.75*(d*d*d*d)*(a*a*a*a*a)*s - 18.75*(d*d*d*d)*(a*a*a*a)*b*s + 12.5*(d*d*d*d)*(a*a*a)*(b*b)*s + 12.5*(d*d*d)*(a*a*a)*g*p - 18.75*(d*d*d)*(a*a)*b*g*p + 3.75*(d*d*d)*a*(b*b)*g*p + 37.5*(d*d)*(a*a*a)*p*s - 56.25*(d*d)*(a*a)*b*p*s + 11.25*(d*d)*a*(b*b)*p*s + 19.6875*d*a*g*(p*p) - 6.5625*d*b*g*(p*p) + 19.6875*a*(p*p)*s - 6.5625*b*(p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad264(double a, double b, double p, double d, double s, double g){
	return (0.9375*(d*d*d*d)*(a*a*a*a)*g - 2.5*(d*d*d*d)*(a*a*a)*b*g + 0.9375*(d*d*d*d)*(a*a)*(b*b)*g + 3.75*(d*d*d)*(a*a*a*a)*s - 10.0*(d*d*d)*(a*a*a)*b*s + 3.75*(d*d*d)*(a*a)*(b*b)*s + 7.03125*(d*d)*(a*a)*g*p - 5.625*(d*d)*a*b*g*p + 0.46875*(d*d)*(b*b)*g*p + 14.0625*d*(a*a)*p*s - 11.25*d*a*b*p*s + 0.9375*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad265(double a, double b, double p, double d, double s, double g){
	return (0.625*(d*d*d)*(a*a*a)*g - 0.9375*(d*d*d)*(a*a)*b*g + 0.1875*(d*d*d)*a*(b*b)*g + 1.875*(d*d)*(a*a*a)*s - 2.8125*(d*d)*(a*a)*b*s + 0.5625*(d*d)*a*(b*b)*s + 1.96875*d*a*g*p - 0.65625*d*b*g*p + 1.96875*a*p*s - 0.65625*b*p*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad266(double a, double b, double p, double d, double s, double g){
	return (0.234375*(d*d)*(a*a)*g - 0.1875*(d*d)*a*b*g + 0.015625*(d*d)*(b*b)*g + 0.46875*d*(a*a)*s - 0.375*d*a*b*s + 0.03125*d*(b*b)*s + 0.21875*g*p)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad267(double a, double b, double p, double d, double s, double g){
	return (0.046875*d*a*g - 0.015625*d*b*g + 0.046875*a*s - 0.015625*b*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad268(double a, double b, double p, double d, double s, double g){
	return 0.00390625*g/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad270(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g + 9.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s + 0.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 7.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 10.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p + 3.5*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 49.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 73.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s + 15.75*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 52.5*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 26.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) + 78.75*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 262.5*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 131.25*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s + 65.625*(d*d*d)*(a*a*a)*g*(p*p*p) - 78.75*(d*d*d)*(a*a)*b*g*(p*p*p) + 13.125*(d*d*d)*a*(b*b)*g*(p*p*p) + 196.875*(d*d)*(a*a*a)*(p*p*p)*s - 236.25*(d*d)*(a*a)*b*(p*p*p)*s + 39.375*(d*d)*a*(b*b)*(p*p*p)*s + 45.9375*d*a*g*(p*p*p*p) - 13.125*d*b*g*(p*p*p*p) + 45.9375*a*(p*p*p*p)*s - 13.125*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad271(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 3.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 8.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 28.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s + 5.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 31.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 26.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p + 31.5*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 189.0*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 157.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s + 65.625*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 131.25*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 39.375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) + 262.5*(d*d*d)*(a*a*a*a)*(p*p)*s - 525.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 157.5*(d*d*d)*(a*a)*(b*b)*(p*p)*s + 137.8125*(d*d)*(a*a)*g*(p*p*p) - 91.875*(d*d)*a*b*g*(p*p*p) + 6.5625*(d*d)*(b*b)*g*(p*p*p) + 275.625*d*(a*a)*(p*p*p)*s - 183.75*d*a*b*(p*p*p)*s + 13.125*d*(b*b)*(p*p*p)*s + 29.53125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad272(double a, double b, double p, double d, double s, double g){
	return (0.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 3.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 5.25*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g + 1.75*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 24.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 36.75*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s + 15.75*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 52.5*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 26.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p + 78.75*(d*d*d*d)*(a*a*a*a*a)*p*s - 262.5*(d*d*d*d)*(a*a*a*a)*b*p*s + 131.25*(d*d*d*d)*(a*a*a)*(b*b)*p*s + 98.4375*(d*d*d)*(a*a*a)*g*(p*p) - 118.125*(d*d*d)*(a*a)*b*g*(p*p) + 19.6875*(d*d*d)*a*(b*b)*g*(p*p) + 295.3125*(d*d)*(a*a*a)*(p*p)*s - 354.375*(d*d)*(a*a)*b*(p*p)*s + 59.0625*(d*d)*a*(b*b)*(p*p)*s + 91.875*d*a*g*(p*p*p) - 26.25*d*b*g*(p*p*p) + 91.875*a*(p*p*p)*s - 26.25*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad273(double a, double b, double p, double d, double s, double g){
	return (0.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 5.25*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 4.375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g + 5.25*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 31.5*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 26.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s + 21.875*(d*d*d*d)*(a*a*a*a)*g*p - 43.75*(d*d*d*d)*(a*a*a)*b*g*p + 13.125*(d*d*d*d)*(a*a)*(b*b)*g*p + 87.5*(d*d*d)*(a*a*a*a)*p*s - 175.0*(d*d*d)*(a*a*a)*b*p*s + 52.5*(d*d*d)*(a*a)*(b*b)*p*s + 68.90625*(d*d)*(a*a)*g*(p*p) - 45.9375*(d*d)*a*b*g*(p*p) + 3.28125*(d*d)*(b*b)*g*(p*p) + 137.8125*d*(a*a)*(p*p)*s - 91.875*d*a*b*(p*p)*s + 6.5625*d*(b*b)*(p*p)*s + 19.6875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad274(double a, double b, double p, double d, double s, double g){
	return (1.3125*(d*d*d*d*d)*(a*a*a*a*a)*g - 4.375*(d*d*d*d*d)*(a*a*a*a)*b*g + 2.1875*(d*d*d*d*d)*(a*a*a)*(b*b)*g + 6.5625*(d*d*d*d)*(a*a*a*a*a)*s - 21.875*(d*d*d*d)*(a*a*a*a)*b*s + 10.9375*(d*d*d*d)*(a*a*a)*(b*b)*s + 16.40625*(d*d*d)*(a*a*a)*g*p - 19.6875*(d*d*d)*(a*a)*b*g*p + 3.28125*(d*d*d)*a*(b*b)*g*p + 49.21875*(d*d)*(a*a*a)*p*s - 59.0625*(d*d)*(a*a)*b*p*s + 9.84375*(d*d)*a*(b*b)*p*s + 22.96875*d*a*g*(p*p) - 6.5625*d*b*g*(p*p) + 22.96875*a*(p*p)*s - 6.5625*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad275(double a, double b, double p, double d, double s, double g){
	return (1.09375*(d*d*d*d)*(a*a*a*a)*g - 2.1875*(d*d*d*d)*(a*a*a)*b*g + 0.65625*(d*d*d*d)*(a*a)*(b*b)*g + 4.375*(d*d*d)*(a*a*a*a)*s - 8.75*(d*d*d)*(a*a*a)*b*s + 2.625*(d*d*d)*(a*a)*(b*b)*s + 6.890625*(d*d)*(a*a)*g*p - 4.59375*(d*d)*a*b*g*p + 0.328125*(d*d)*(b*b)*g*p + 13.78125*d*(a*a)*p*s - 9.1875*d*a*b*p*s + 0.65625*d*(b*b)*p*s + 2.953125*g*(p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad276(double a, double b, double p, double d, double s, double g){
	return (0.546875*(d*d*d)*(a*a*a)*g - 0.65625*(d*d*d)*(a*a)*b*g + 0.109375*(d*d*d)*a*(b*b)*g + 1.640625*(d*d)*(a*a*a)*s - 1.96875*(d*d)*(a*a)*b*s + 0.328125*(d*d)*a*(b*b)*s + 1.53125*d*a*g*p - 0.4375*d*b*g*p + 1.53125*a*p*s - 0.4375*b*p*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad277(double a, double b, double p, double d, double s, double g){
	return (0.1640625*(d*d)*(a*a)*g - 0.109375*(d*d)*a*b*g + 0.0078125*(d*d)*(b*b)*g + 0.328125*d*(a*a)*s - 0.21875*d*a*b*s + 0.015625*d*(b*b)*s + 0.140625*g*p)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad278(double a, double b, double p, double d, double s, double g){
	return (0.02734375*d*a*g - 0.0078125*d*b*g + 0.02734375*a*s - 0.0078125*b*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad279(double a, double b, double p, double d, double s, double g){
	return 0.001953125*g/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad280(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g + 10.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*s + 0.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*p - 8.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 14.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p + 4.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*p*s - 64.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 112.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s + 21.0 *(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 84.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 52.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) + 126.0*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 504.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 315.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s + 131.25*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 210.0*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 52.5*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) + 525.0*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 840.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 210.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s + 183.75*(d*d)*(a*a)*g*(p*p*p*p) - 105.0*(d*d)*a*b*g*(p*p*p*p) + 6.5625*(d*d)*(b*b)*g*(p*p*p*p) + 367.5*d*(a*a)*(p*p*p*p)*s - 210.0*d*a*b*(p*p*p*p)*s + 13.125*d*(b*b)*(p*p*p*p)*s + 29.53125*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad281(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g + 4.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 9.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*s + 36.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s + 6.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 42.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 42.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p + 42.0*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 294.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 294.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s + 105.0*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 262.5*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 105.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) + 525.0*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 1312.5*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 525.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s + 367.5*(d*d*d)*(a*a*a)*g*(p*p*p) - 367.5*(d*d*d)*(a*a)*b*g*(p*p*p) + 52.5*(d*d*d)*a*(b*b)*g*(p*p*p) + 1102.5*(d*d)*(a*a*a)*(p*p*p)*s - 1102.5*(d*d)*(a*a)*b*(p*p*p)*s + 157.5*(d*d)*a*(b*b)*(p*p*p)*s + 236.25*d*a*g*(p*p*p*p) - 59.0625*d*b*g*(p*p*p*p) + 236.25*a*(p*p*p*p)*s - 59.0625*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad282(double a, double b, double p, double d, double s, double g){
	return (0.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g - 4.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 7.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g + 2.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*s - 32.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 56.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s + 21.0 *(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 84.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 52.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p + 126.0*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 504.0*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 315.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s + 196.875*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 315.0*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 78.75*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) + 787.5*(d*d*d)*(a*a*a*a)*(p*p)*s - 1260.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 315.0*(d*d*d)*(a*a)*(b*b)*(p*p)*s + 367.5*(d*d)*(a*a)*g*(p*p*p) - 210.0*(d*d)*a*b*g*(p*p*p) + 13.125*(d*d)*(b*b)*g*(p*p*p) + 735.0*d*(a*a)*(p*p*p)*s - 420.0*d*a*b*(p*p*p)*s + 26.25*d*(b*b)*(p*p*p)*s + 73.828125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad283(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 7.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 7.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g + 7.0*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 49.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 49.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s + 35.0*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 87.5*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 35.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p + 175.0*(d*d*d*d)*(a*a*a*a*a)*p*s - 437.5*(d*d*d*d)*(a*a*a*a)*b*p*s + 175.0*(d*d*d*d)*(a*a*a)*(b*b)*p*s + 183.75*(d*d*d)*(a*a*a)*g*(p*p) - 183.75*(d*d*d)*(a*a)*b*g*(p*p) + 26.25*(d*d*d)*a*(b*b)*g*(p*p) + 551.25*(d*d)*(a*a*a)*(p*p)*s - 551.25*(d*d)*(a*a)*b*(p*p)*s + 78.75*(d*d)*a*(b*b)*(p*p)*s + 157.5*d*a*g*(p*p*p) - 39.375*d*b*g*(p*p*p) + 157.5*a*(p*p*p)*s - 39.375*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad284(double a, double b, double p, double d, double s, double g){
	return (1.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 7.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 4.375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g + 10.5*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 42.0*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 26.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s + 32.8125*(d*d*d*d)*(a*a*a*a)*g*p - 52.5*(d*d*d*d)*(a*a*a)*b*g*p + 13.125*(d*d*d*d)*(a*a)*(b*b)*g*p + 131.25*(d*d*d)*(a*a*a*a)*p*s - 210.0*(d*d*d)*(a*a*a)*b*p*s + 52.5*(d*d*d)*(a*a)*(b*b)*p*s + 91.875*(d*d)*(a*a)*g*(p*p) - 52.5*(d*d)*a*b*g*(p*p) + 3.28125*(d*d)*(b*b)*g*(p*p) + 183.75*d*(a*a)*(p*p)*s - 105.0*d*a*b*(p*p)*s + 6.5625*d*(b*b)*(p*p)*s + 24.609375*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad285(double a, double b, double p, double d, double s, double g){
	return (1.75*(d*d*d*d*d)*(a*a*a*a*a)*g - 4.375*(d*d*d*d*d)*(a*a*a*a)*b*g + 1.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g + 8.75*(d*d*d*d)*(a*a*a*a*a)*s - 21.875*(d*d*d*d)*(a*a*a*a)*b*s + 8.75*(d*d*d*d)*(a*a*a)*(b*b)*s + 18.375*(d*d*d)*(a*a*a)*g*p - 18.375*(d*d*d)*(a*a)*b*g*p + 2.625*(d*d*d)*a*(b*b)*g*p + 55.125*(d*d)*(a*a*a)*p*s - 55.125*(d*d)*(a*a)*b*p*s + 7.875*(d*d)*a*(b*b)*p*s + 23.625*d*a*g*(p*p) - 5.90625*d*b*g*(p*p) + 23.625*a*(p*p)*s - 5.90625*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad286(double a, double b, double p, double d, double s, double g){
	return (1.09375*(d*d*d*d)*(a*a*a*a)*g - 1.75*(d*d*d*d)*(a*a*a)*b*g + 0.4375*(d*d*d*d)*(a*a)*(b*b)*g + 4.375*(d*d*d)*(a*a*a*a)*s - 7.0*(d*d*d)*(a*a*a)*b*s + 1.75*(d*d*d)*(a*a)*(b*b)*s + 6.125*(d*d)*(a*a)*g*p - 3.5*(d*d)*a*b*g*p + 0.21875*(d*d)*(b*b)*g*p + 12.25*d*(a*a)*p*s - 7.0*d*a*b*p*s + 0.4375*d*(b*b)*p*s + 2.4609375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad287(double a, double b, double p, double d, double s, double g){
	return (0.4375*(d*d*d)*(a*a*a)*g - 0.4375*(d*d*d)*(a*a)*b*g + 0.0625*(d*d*d)*a*(b*b)*g + 1.3125*(d*d)*(a*a*a)*s - 1.3125*(d*d)*(a*a)*b*s + 0.1875*(d*d)*a*(b*b)*s + 1.125*d*a*g*p - 0.28125*d*b*g*p + 1.125*a*p*s - 0.28125*b*p*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad288(double a, double b, double p, double d, double s, double g){
	return (0.109375*(d*d)*(a*a)*g - 0.0625*(d*d)*a*b*g + 0.00390625*(d*d)*(b*b)*g + 0.21875*d*(a*a)*s - 0.125*d*a*b*s + 0.0078125*d*(b*b)*s + 0.087890625*g*p)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad289(double a, double b, double p, double d, double s, double g){
	return (0.015625*d*a*g - 0.00390625*d*b*g + 0.015625*a*s - 0.00390625*b*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad2810(double a, double b, double p, double d, double s, double g){
	return 0.0009765625*g/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad300(double a, double b, double p, double d, double s, double g){
	return b*(-(d*d*d)*(b*b)*g - 3.0*(d*d)*(b*b)*s - 1.5*d*g*p - 1.5*p*s)/(p*p*p);
}

inline double MD_Et_grad301(double a, double b, double p, double d, double s, double g){
	return (1.5*d*(b*b)*(d*g + 2*s) + 0.75*g*p)/(p*p*p);
}

inline double MD_Et_grad302(double a, double b, double p, double d, double s, double g){
	return 0.75*b*(-d*g - s)/(p*p*p);
}

inline double MD_Et_grad303(double a, double b, double p, double d, double s, double g){
	return 0.125*g/(p*p*p);
}

inline double MD_Et_grad310(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d)*a*(b*b*b)*g - 4.0*(d*d*d)*a*(b*b*b)*s - 1.5*(d*d)*a*b*g*p + 1.5*(d*d)*(b*b)*g*p - 3.0*d*a*b*p*s + 3.0*d*(b*b)*p*s + 0.75*g*(p*p))/(p*p*p*p);
}

inline double MD_Et_grad311(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d)*a*(b*b)*g - 0.5*(d*d*d)*(b*b*b)*g + 4.5*(d*d)*a*(b*b)*s - 1.5*(d*d)*(b*b*b)*s + 0.75*d*a*g*p - 2.25*d*b*g*p + 0.75*a*p*s - 2.25*b*p*s)/(p*p*p*p);
}

inline double MD_Et_grad312(double a, double b, double p, double d, double s, double g){
	return (-0.75*(d*d)*a*b*g + 0.75*(d*d)*(b*b)*g - 1.5*d*a*b*s + 1.5*d*(b*b)*s + 0.75*g*p)/(p*p*p*p);
}

inline double MD_Et_grad313(double a, double b, double p, double d, double s, double g){
	return (0.125*d*a*g - 0.375*d*b*g + 0.125*a*s - 0.375*b*s)/(p*p*p*p);
}

inline double MD_Et_grad314(double a, double b, double p, double d, double s, double g){
	return 0.0625*g/(p*p*p*p);
}

inline double MD_Et_grad320(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d)*(a*a)*(b*b*b)*g - 5.0*(d*d*d*d)*(a*a)*(b*b*b)*s - 1.5*(d*d*d)*(a*a)*b*g*p + 3.0*(d*d*d)*a*(b*b)*g*p - 0.5*(d*d*d)*(b*b*b)*g*p - 4.5*(d*d)*(a*a)*b*p*s + 9.0*(d*d)*a*(b*b)*p*s - 1.5*(d*d)*(b*b*b)*p*s + 1.5*d*a*g*(p*p) - 2.25*d*b*g*(p*p) + 1.5*a*(p*p)*s - 2.25*b*(p*p)*s)/(p*p*p*p*p);
}

inline double MD_Et_grad321(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d)*(a*a)*(b*b)*g - (d*d*d*d)*a*(b*b*b)*g + 6.0*(d*d*d)*(a*a)*(b*b)*s - 4.0*(d*d*d)*a*(b*b*b)*s + 0.75*(d*d)*(a*a)*g*p - 4.5*(d*d)*a*b*g*p + 2.25*(d*d)*(b*b)*g*p + 1.5*d*(a*a)*p*s - 9.0*d*a*b*p*s + 4.5*d*(b*b)*p*s + 1.875*g*(p*p))/(p*p*p*p*p);
}

inline double MD_Et_grad322(double a, double b, double p, double d, double s, double g){
	return (-0.75*(d*d*d)*(a*a)*b*g + 1.5*(d*d*d)*a*(b*b)*g - 0.25*(d*d*d)*(b*b*b)*g - 2.25*(d*d)*(a*a)*b*s + 4.5*(d*d)*a*(b*b)*s - 0.75*(d*d)*(b*b*b)*s + 1.5*d*a*g*p - 2.25*d*b*g*p + 1.5*a*p*s - 2.25*b*p*s)/(p*p*p*p*p);
}

inline double MD_Et_grad323(double a, double b, double p, double d, double s, double g){
	return (0.125*(d*d)*(a*a)*g - 0.75*(d*d)*a*b*g + 0.375*(d*d)*(b*b)*g + 0.25*d*(a*a)*s - 1.5*d*a*b*s + 0.75*d*(b*b)*s + 0.625*g*p)/(p*p*p*p*p);
}

inline double MD_Et_grad324(double a, double b, double p, double d, double s, double g){
	return (0.125*d*a*g - 0.1875*d*b*g + 0.125*a*s - 0.1875*b*s)/(p*p*p*p*p);
}

inline double MD_Et_grad325(double a, double b, double p, double d, double s, double g){
	return 0.03125*g/(p*p*p*p*p);
}

inline double MD_Et_grad330(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g - 6.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s - 1.5*(d*d*d*d)*(a*a*a)*b*g*p + 4.5*(d*d*d*d)*(a*a)*(b*b)*g*p - 1.5*(d*d*d*d)*a*(b*b*b)*g*p - 6.0*(d*d*d)*(a*a*a)*b*p*s + 18.0*(d*d*d)*(a*a)*(b*b)*p*s - 6.0*(d*d*d)*a*(b*b*b)*p*s + 2.25*(d*d)*(a*a)*g*(p*p) - 6.75*(d*d)*a*b*g*(p*p) + 2.25*(d*d)*(b*b)*g*(p*p) + 4.5*d*(a*a)*(p*p)*s - 13.5*d*a*b*(p*p)*s + 4.5*d*(b*b)*(p*p)*s + 1.875*g*(p*p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad331(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 1.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 7.5*(d*d*d*d)*(a*a*a)*(b*b)*s - 7.5*(d*d*d*d)*(a*a)*(b*b*b)*s + 0.75*(d*d*d)*(a*a*a)*g*p - 6.75*(d*d*d)*(a*a)*b*g*p + 6.75*(d*d*d)*a*(b*b)*g*p - 0.75*(d*d*d)*(b*b*b)*g*p + 2.25*(d*d)*(a*a*a)*p*s - 20.25*(d*d)*(a*a)*b*p*s + 20.25*(d*d)*a*(b*b)*p*s - 2.25*(d*d)*(b*b*b)*p*s + 5.625*d*a*g*(p*p) - 5.625*d*b*g*(p*p) + 5.625*a*(p*p)*s - 5.625*b*(p*p)*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad332(double a, double b, double p, double d, double s, double g){
	return (-0.75*(d*d*d*d)*(a*a*a)*b*g + 2.25*(d*d*d*d)*(a*a)*(b*b)*g - 0.75*(d*d*d*d)*a*(b*b*b)*g - 3.0*(d*d*d)*(a*a*a)*b*s + 9.0*(d*d*d)*(a*a)*(b*b)*s - 3.0*(d*d*d)*a*(b*b*b)*s + 2.25*(d*d)*(a*a)*g*p - 6.75*(d*d)*a*b*g*p + 2.25*(d*d)*(b*b)*g*p + 4.5*d*(a*a)*p*s - 13.5*d*a*b*p*s + 4.5*d*(b*b)*p*s + 2.8125*g*(p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad333(double a, double b, double p, double d, double s, double g){
	return (0.125*(d*d*d)*(a*a*a)*g - 1.125*(d*d*d)*(a*a)*b*g + 1.125*(d*d*d)*a*(b*b)*g - 0.125*(d*d*d)*(b*b*b)*g + 0.375*(d*d)*(a*a*a)*s - 3.375*(d*d)*(a*a)*b*s + 3.375*(d*d)*a*(b*b)*s - 0.375*(d*d)*(b*b*b)*s + 1.875*d*a*g*p - 1.875*d*b*g*p + 1.875*a*p*s - 1.875*b*p*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad334(double a, double b, double p, double d, double s, double g){
	return (0.1875*(d*d)*(a*a)*g - 0.5625*(d*d)*a*b*g + 0.1875*(d*d)*(b*b)*g + 0.375*d*(a*a)*s - 1.125*d*a*b*s + 0.375*d*(b*b)*s + 0.46875*g*p)/(p*p*p*p*p*p);
}

inline double MD_Et_grad335(double a, double b, double p, double d, double s, double g){
	return 0.09375*(d*a*g - d*b*g + a*s - b*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad336(double a, double b, double p, double d, double s, double g){
	return 0.015625*g/(p*p*p*p*p*p);
}

inline double MD_Et_grad340(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g - 7.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s - 1.5*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 6.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 3.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p - 7.5*(d*d*d*d)*(a*a*a*a)*b*p*s + 30.0*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 15.0*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 3.0*(d*d*d)*(a*a*a)*g*(p*p) - 13.5*(d*d*d)*(a*a)*b*g*(p*p) + 9.0*(d*d*d)*a*(b*b)*g*(p*p) - 0.75*(d*d*d)*(b*b*b)*g*(p*p) + 9.0*(d*d)*(a*a*a)*(p*p)*s - 40.5*(d*d)*(a*a)*b*(p*p)*s + 27.0*(d*d)*a*(b*b)*(p*p)*s - 2.25*(d*d)*(b*b*b)*(p*p)*s + 7.5*d*a*g*(p*p*p) - 5.625*d*b*g*(p*p*p) + 7.5*a*(p*p*p)*s - 5.625*b*(p*p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad341(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 2.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 9.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 12.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 0.75*(d*d*d*d)*(a*a*a*a)*g*p - 9.0*(d*d*d*d)*(a*a*a)*b*g*p + 13.5*(d*d*d*d)*(a*a)*(b*b)*g*p - 3.0*(d*d*d*d)*a*(b*b*b)*g*p + 3.0*(d*d*d)*(a*a*a*a)*p*s - 36.0*(d*d*d)*(a*a*a)*b*p*s + 54.0*(d*d*d)*(a*a)*(b*b)*p*s - 12.0*(d*d*d)*a*(b*b*b)*p*s + 11.25*(d*d)*(a*a)*g*(p*p) - 22.5*(d*d)*a*b*g*(p*p) + 5.625*(d*d)*(b*b)*g*(p*p) + 22.5*d*(a*a)*(p*p)*s - 45.0*d*a*b*(p*p)*s + 11.25*d*(b*b)*(p*p)*s + 6.5625*g*(p*p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad342(double a, double b, double p, double d, double s, double g){
	return (-0.75*(d*d*d*d*d)*(a*a*a*a)*b*g + 3.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 1.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g - 3.75*(d*d*d*d)*(a*a*a*a)*b*s + 15.0*(d*d*d*d)*(a*a*a)*(b*b)*s - 7.5*(d*d*d*d)*(a*a)*(b*b*b)*s + 3.0*(d*d*d)*(a*a*a)*g*p - 13.5*(d*d*d)*(a*a)*b*g*p + 9.0*(d*d*d)*a*(b*b)*g*p - 0.75*(d*d*d)*(b*b*b)*g*p + 9.0*(d*d)*(a*a*a)*p*s - 40.5*(d*d)*(a*a)*b*p*s + 27.0*(d*d)*a*(b*b)*p*s - 2.25*(d*d)*(b*b*b)*p*s + 11.25*d*a*g*(p*p) - 8.4375*d*b*g*(p*p) + 11.25*a*(p*p)*s - 8.4375*b*(p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad343(double a, double b, double p, double d, double s, double g){
	return (0.125*(d*d*d*d)*(a*a*a*a)*g - 1.5*(d*d*d*d)*(a*a*a)*b*g + 2.25*(d*d*d*d)*(a*a)*(b*b)*g - 0.5*(d*d*d*d)*a*(b*b*b)*g + 0.5*(d*d*d)*(a*a*a*a)*s - 6.0*(d*d*d)*(a*a*a)*b*s + 9.0*(d*d*d)*(a*a)*(b*b)*s - 2.0*(d*d*d)*a*(b*b*b)*s + 3.75*(d*d)*(a*a)*g*p - 7.5*(d*d)*a*b*g*p + 1.875*(d*d)*(b*b)*g*p + 7.5*d*(a*a)*p*s - 15.0*d*a*b*p*s + 3.75*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad344(double a, double b, double p, double d, double s, double g){
	return (0.25*(d*d*d)*(a*a*a)*g - 1.125*(d*d*d)*(a*a)*b*g + 0.75*(d*d*d)*a*(b*b)*g - 0.0625*(d*d*d)*(b*b*b)*g + 0.75*(d*d)*(a*a*a)*s - 3.375*(d*d)*(a*a)*b*s + 2.25*(d*d)*a*(b*b)*s - 0.1875*(d*d)*(b*b*b)*s + 1.875*d*a*g*p - 1.40625*d*b*g*p + 1.875*a*p*s - 1.40625*b*p*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad345(double a, double b, double p, double d, double s, double g){
	return (0.1875*(d*d)*(a*a)*g - 0.375*(d*d)*a*b*g + 0.09375*(d*d)*(b*b)*g + 0.375*d*(a*a)*s - 0.75*d*a*b*s + 0.1875*d*(b*b)*s + 0.328125*g*p)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad346(double a, double b, double p, double d, double s, double g){
	return (0.0625*d*a*g - 0.046875*d*b*g + 0.0625*a*s - 0.046875*b*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad347(double a, double b, double p, double d, double s, double g){
	return 0.0078125*g/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad350(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g - 8.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s - 1.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 7.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 5.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p - 9.0*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 45.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 30.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 3.75*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 22.5*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 22.5*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 3.75*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 15.0*(d*d*d)*(a*a*a*a)*(p*p)*s - 90.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 90.0*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 15.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 18.75*(d*d)*(a*a)*g*(p*p*p) - 28.125*(d*d)*a*b*g*(p*p*p) + 5.625*(d*d)*(b*b)*g*(p*p*p) + 37.5*d*(a*a)*(p*p*p)*s - 56.25*d*a*b*(p*p*p)*s + 11.25*d*(b*b)*(p*p*p)*s + 6.5625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad351(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 2.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 10.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 17.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 0.75*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 11.25*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 22.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 7.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 3.75*(d*d*d*d)*(a*a*a*a*a)*p*s - 56.25*(d*d*d*d)*(a*a*a*a)*b*p*s + 112.5*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 37.5*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 18.75*(d*d*d)*(a*a*a)*g*(p*p) - 56.25*(d*d*d)*(a*a)*b*g*(p*p) + 28.125*(d*d*d)*a*(b*b)*g*(p*p) - 1.875*(d*d*d)*(b*b*b)*g*(p*p) + 56.25*(d*d)*(a*a*a)*(p*p)*s - 168.75*(d*d)*(a*a)*b*(p*p)*s + 84.375*(d*d)*a*(b*b)*(p*p)*s - 5.625*(d*d)*(b*b*b)*(p*p)*s + 32.8125*d*a*g*(p*p*p) - 19.6875*d*b*g*(p*p*p) + 32.8125*a*(p*p*p)*s - 19.6875*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad352(double a, double b, double p, double d, double s, double g){
	return (-0.75*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 3.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 2.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g - 4.5*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 22.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 15.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 3.75*(d*d*d*d)*(a*a*a*a)*g*p - 22.5*(d*d*d*d)*(a*a*a)*b*g*p + 22.5*(d*d*d*d)*(a*a)*(b*b)*g*p - 3.75*(d*d*d*d)*a*(b*b*b)*g*p + 15.0*(d*d*d)*(a*a*a*a)*p*s - 90.0*(d*d*d)*(a*a*a)*b*p*s + 90.0*(d*d*d)*(a*a)*(b*b)*p*s - 15.0*(d*d*d)*a*(b*b*b)*p*s + 28.125*(d*d)*(a*a)*g*(p*p) - 42.1875*(d*d)*a*b*g*(p*p) + 8.4375*(d*d)*(b*b)*g*(p*p) + 56.25*d*(a*a)*(p*p)*s - 84.375*d*a*b*(p*p)*s + 16.875*d*(b*b)*(p*p)*s + 13.125*g*(p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad353(double a, double b, double p, double d, double s, double g){
	return (0.125*(d*d*d*d*d)*(a*a*a*a*a)*g - 1.875*(d*d*d*d*d)*(a*a*a*a)*b*g + 3.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 1.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.625*(d*d*d*d)*(a*a*a*a*a)*s - 9.375*(d*d*d*d)*(a*a*a*a)*b*s + 18.75*(d*d*d*d)*(a*a*a)*(b*b)*s - 6.25*(d*d*d*d)*(a*a)*(b*b*b)*s + 6.25*(d*d*d)*(a*a*a)*g*p - 18.75*(d*d*d)*(a*a)*b*g*p + 9.375*(d*d*d)*a*(b*b)*g*p - 0.625*(d*d*d)*(b*b*b)*g*p + 18.75*(d*d)*(a*a*a)*p*s - 56.25*(d*d)*(a*a)*b*p*s + 28.125*(d*d)*a*(b*b)*p*s - 1.875*(d*d)*(b*b*b)*p*s + 16.40625*d*a*g*(p*p) - 9.84375*d*b*g*(p*p) + 16.40625*a*(p*p)*s - 9.84375*b*(p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad354(double a, double b, double p, double d, double s, double g){
	return (0.3125*(d*d*d*d)*(a*a*a*a)*g - 1.875*(d*d*d*d)*(a*a*a)*b*g + 1.875*(d*d*d*d)*(a*a)*(b*b)*g - 0.3125*(d*d*d*d)*a*(b*b*b)*g + 1.25*(d*d*d)*(a*a*a*a)*s - 7.5*(d*d*d)*(a*a*a)*b*s + 7.5*(d*d*d)*(a*a)*(b*b)*s - 1.25*(d*d*d)*a*(b*b*b)*s + 4.6875*(d*d)*(a*a)*g*p - 7.03125*(d*d)*a*b*g*p + 1.40625*(d*d)*(b*b)*g*p + 9.375*d*(a*a)*p*s - 14.0625*d*a*b*p*s + 2.8125*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad355(double a, double b, double p, double d, double s, double g){
	return (0.3125*(d*d*d)*(a*a*a)*g - 0.9375*(d*d*d)*(a*a)*b*g + 0.46875*(d*d*d)*a*(b*b)*g - 0.03125*(d*d*d)*(b*b*b)*g + 0.9375*(d*d)*(a*a*a)*s - 2.8125*(d*d)*(a*a)*b*s + 1.40625*(d*d)*a*(b*b)*s - 0.09375*(d*d)*(b*b*b)*s + 1.640625*d*a*g*p - 0.984375*d*b*g*p + 1.640625*a*p*s - 0.984375*b*p*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad356(double a, double b, double p, double d, double s, double g){
	return (0.15625*(d*d)*(a*a)*g - 0.234375*(d*d)*a*b*g + 0.046875*(d*d)*(b*b)*g + 0.3125*d*(a*a)*s - 0.46875*d*a*b*s + 0.09375*d*(b*b)*s + 0.21875*g*p)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad357(double a, double b, double p, double d, double s, double g){
	return (0.0390625*d*a*g - 0.0234375*d*b*g + 0.0390625*a*s - 0.0234375*b*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad358(double a, double b, double p, double d, double s, double g){
	return 0.00390625*g/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad360(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g - 9.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s - 1.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 9.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 7.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p - 10.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 63.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 52.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 4.5*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 33.75*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 45.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 11.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 22.5*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 168.75*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 225.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 56.25*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 37.5*(d*d*d)*(a*a*a)*g*(p*p*p) - 84.375*(d*d*d)*(a*a)*b*g*(p*p*p) + 33.75*(d*d*d)*a*(b*b)*g*(p*p*p) - 1.875*(d*d*d)*(b*b*b)*g*(p*p*p) + 112.5*(d*d)*(a*a*a)*(p*p*p)*s - 253.125*(d*d)*(a*a)*b*(p*p*p)*s + 101.25*(d*d)*a*(b*b)*(p*p*p)*s - 5.625*(d*d)*(b*b*b)*(p*p*p)*s + 39.375*d*a*g*(p*p*p*p) - 19.6875*d*b*g*(p*p*p*p) + 39.375*a*(p*p*p*p)*s - 19.6875*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad361(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 3.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 12.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 24.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 0.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 13.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 33.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 15.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 4.5*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 81.0 *(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 202.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 90.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 28.125*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 112.5*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 84.375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 11.25*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 112.5*(d*d*d)*(a*a*a*a)*(p*p)*s - 450.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 337.5*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 45.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 98.4375*(d*d)*(a*a)*g*(p*p*p) - 118.125*(d*d)*a*b*g*(p*p*p) + 19.6875*(d*d)*(b*b)*g*(p*p*p) + 196.875*d*(a*a)*(p*p*p)*s - 236.25*d*a*b*(p*p*p)*s + 39.375*d*(b*b)*(p*p*p)*s + 29.53125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad362(double a, double b, double p, double d, double s, double g){
	return (-0.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 4.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 3.75*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g - 5.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 31.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 26.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 4.5*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 33.75*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 45.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 11.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 22.5*(d*d*d*d)*(a*a*a*a*a)*p*s - 168.75*(d*d*d*d)*(a*a*a*a)*b*p*s + 225.0*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 56.25*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 56.25*(d*d*d)*(a*a*a)*g*(p*p) - 126.5625*(d*d*d)*(a*a)*b*g*(p*p) + 50.625*(d*d*d)*a*(b*b)*g*(p*p) - 2.8125*(d*d*d)*(b*b*b)*g*(p*p) + 168.75*(d*d)*(a*a*a)*(p*p)*s - 379.6875*(d*d)*(a*a)*b*(p*p)*s + 151.875*(d*d)*a*(b*b)*(p*p)*s - 8.4375*(d*d)*(b*b*b)*(p*p)*s + 78.75*d*a*g*(p*p*p) - 39.375*d*b*g*(p*p*p) + 78.75*a*(p*p*p)*s - 39.375*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad363(double a, double b, double p, double d, double s, double g){
	return (0.125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 2.25*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 5.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 2.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 0.75*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 13.5*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 33.75*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 15.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 9.375*(d*d*d*d)*(a*a*a*a)*g*p - 37.5*(d*d*d*d)*(a*a*a)*b*g*p + 28.125*(d*d*d*d)*(a*a)*(b*b)*g*p - 3.75*(d*d*d*d)*a*(b*b*b)*g*p + 37.5*(d*d*d)*(a*a*a*a)*p*s - 150.0*(d*d*d)*(a*a*a)*b*p*s + 112.5*(d*d*d)*(a*a)*(b*b)*p*s - 15.0*(d*d*d)*a*(b*b*b)*p*s + 49.21875*(d*d)*(a*a)*g*(p*p) - 59.0625*(d*d)*a*b*g*(p*p) + 9.84375*(d*d)*(b*b)*g*(p*p) + 98.4375*d*(a*a)*(p*p)*s - 118.125*d*a*b*(p*p)*s + 19.6875*d*(b*b)*(p*p)*s + 19.6875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad364(double a, double b, double p, double d, double s, double g){
	return (0.375*(d*d*d*d*d)*(a*a*a*a*a)*g - 2.8125*(d*d*d*d*d)*(a*a*a*a)*b*g + 3.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 0.9375*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 1.875*(d*d*d*d)*(a*a*a*a*a)*s - 14.0625*(d*d*d*d)*(a*a*a*a)*b*s + 18.75*(d*d*d*d)*(a*a*a)*(b*b)*s - 4.6875*(d*d*d*d)*(a*a)*(b*b*b)*s + 9.375*(d*d*d)*(a*a*a)*g*p - 21.09375*(d*d*d)*(a*a)*b*g*p + 8.4375*(d*d*d)*a*(b*b)*g*p - 0.46875*(d*d*d)*(b*b*b)*g*p + 28.125*(d*d)*(a*a*a)*p*s - 63.28125*(d*d)*(a*a)*b*p*s + 25.3125*(d*d)*a*(b*b)*p*s - 1.40625*(d*d)*(b*b*b)*p*s + 19.6875*d*a*g*(p*p) - 9.84375*d*b*g*(p*p) + 19.6875*a*(p*p)*s - 9.84375*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad365(double a, double b, double p, double d, double s, double g){
	return (0.46875*(d*d*d*d)*(a*a*a*a)*g - 1.875*(d*d*d*d)*(a*a*a)*b*g + 1.40625*(d*d*d*d)*(a*a)*(b*b)*g - 0.1875*(d*d*d*d)*a*(b*b*b)*g + 1.875*(d*d*d)*(a*a*a*a)*s - 7.5*(d*d*d)*(a*a*a)*b*s + 5.625*(d*d*d)*(a*a)*(b*b)*s - 0.75*(d*d*d)*a*(b*b*b)*s + 4.921875*(d*d)*(a*a)*g*p - 5.90625*(d*d)*a*b*g*p + 0.984375*(d*d)*(b*b)*g*p + 9.84375*d*(a*a)*p*s - 11.8125*d*a*b*p*s + 1.96875*d*(b*b)*p*s + 2.953125*g*(p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad366(double a, double b, double p, double d, double s, double g){
	return (0.3125*(d*d*d)*(a*a*a)*g - 0.703125*(d*d*d)*(a*a)*b*g + 0.28125*(d*d*d)*a*(b*b)*g - 0.015625*(d*d*d)*(b*b*b)*g + 0.9375*(d*d)*(a*a*a)*s - 2.109375*(d*d)*(a*a)*b*s + 0.84375*(d*d)*a*(b*b)*s - 0.046875*(d*d)*(b*b*b)*s + 1.3125*d*a*g*p - 0.65625*d*b*g*p + 1.3125*a*p*s - 0.65625*b*p*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad367(double a, double b, double p, double d, double s, double g){
	return (0.1171875*(d*d)*(a*a)*g - 0.140625*(d*d)*a*b*g + 0.0234375*(d*d)*(b*b)*g + 0.234375*d*(a*a)*s - 0.28125*d*a*b*s + 0.046875*d*(b*b)*s + 0.140625*g*p)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad368(double a, double b, double p, double d, double s, double g){
	return (0.0234375*d*a*g - 0.01171875*d*b*g + 0.0234375*a*s - 0.01171875*b*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad369(double a, double b, double p, double d, double s, double g){
	return 0.001953125*g/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad370(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g - 10.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*s - 1.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 10.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 10.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p - 12.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 84.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 84.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 5.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 47.25*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 78.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 26.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 31.5*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 283.5*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 472.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 157.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 65.625*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 196.875*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 118.125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 13.125*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 262.5*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 787.5*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 472.5*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 52.5*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 137.8125*(d*d)*(a*a)*g*(p*p*p*p) - 137.8125*(d*d)*a*b*g*(p*p*p*p) + 19.6875*(d*d)*(b*b)*g*(p*p*p*p) + 275.625*d*(a*a)*(p*p*p*p)*s - 275.625*d*a*b*(p*p*p*p)*s + 39.375*d*(b*b)*(p*p*p*p)*s + 29.53125*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad371(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 3.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 13.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s - 31.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 0.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 15.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 47.25*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 26.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 5.25*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 110.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 330.75*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 183.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 39.375*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 196.875*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 196.875*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 39.375*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 196.875*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 984.375*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 984.375*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 196.875*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 229.6875*(d*d*d)*(a*a*a)*g*(p*p*p) - 413.4375*(d*d*d)*(a*a)*b*g*(p*p*p) + 137.8125*(d*d*d)*a*(b*b)*g*(p*p*p) - 6.5625*(d*d*d)*(b*b*b)*g*(p*p*p) + 689.0625*(d*d)*(a*a*a)*(p*p*p)*s - 1240.3125*(d*d)*(a*a)*b*(p*p*p)*s + 413.4375*(d*d)*a*(b*b)*(p*p*p)*s - 19.6875*(d*d)*(b*b*b)*(p*p*p)*s + 206.71875*d*a*g*(p*p*p*p) - 88.59375*d*b*g*(p*p*p*p) + 206.71875*a*(p*p*p*p)*s - 88.59375*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad372(double a, double b, double p, double d, double s, double g){
	return (-0.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 5.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 5.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g - 6.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 42.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 42.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 5.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 47.25*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 78.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 26.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 31.5*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 283.5*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 472.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 157.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 98.4375*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 295.3125*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 177.1875*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 19.6875*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 393.75*(d*d*d)*(a*a*a*a)*(p*p)*s - 1181.25*(d*d*d)*(a*a*a)*b*(p*p)*s + 708.75*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 78.75*(d*d*d)*a*(b*b*b)*(p*p)*s + 275.625*(d*d)*(a*a)*g*(p*p*p) - 275.625*(d*d)*a*b*g*(p*p*p) + 39.375*(d*d)*(b*b)*g*(p*p*p) + 551.25*d*(a*a)*(p*p*p)*s - 551.25*d*a*b*(p*p*p)*s + 78.75*d*(b*b)*(p*p*p)*s + 73.828125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad373(double a, double b, double p, double d, double s, double g){
	return (0.125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 2.625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 7.875*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 4.375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 0.875*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 18.375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 55.125*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 30.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 13.125*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 65.625*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 65.625*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 13.125*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 65.625*(d*d*d*d)*(a*a*a*a*a)*p*s - 328.125*(d*d*d*d)*(a*a*a*a)*b*p*s + 328.125*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 65.625*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 114.84375*(d*d*d)*(a*a*a)*g*(p*p) - 206.71875*(d*d*d)*(a*a)*b*g*(p*p) + 68.90625*(d*d*d)*a*(b*b)*g*(p*p) - 3.28125*(d*d*d)*(b*b*b)*g*(p*p) + 344.53125*(d*d)*(a*a*a)*(p*p)*s - 620.15625*(d*d)*(a*a)*b*(p*p)*s + 206.71875*(d*d)*a*(b*b)*(p*p)*s - 9.84375*(d*d)*(b*b*b)*(p*p)*s + 137.8125*d*a*g*(p*p*p) - 59.0625*d*b*g*(p*p*p) + 137.8125*a*(p*p*p)*s - 59.0625*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad374(double a, double b, double p, double d, double s, double g){
	return (0.4375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 3.9375*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 6.5625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 2.1875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 2.625*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 23.625*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 39.375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 13.125*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 16.40625*(d*d*d*d)*(a*a*a*a)*g*p - 49.21875*(d*d*d*d)*(a*a*a)*b*g*p + 29.53125*(d*d*d*d)*(a*a)*(b*b)*g*p - 3.28125*(d*d*d*d)*a*(b*b*b)*g*p + 65.625*(d*d*d)*(a*a*a*a)*p*s - 196.875*(d*d*d)*(a*a*a)*b*p*s + 118.125*(d*d*d)*(a*a)*(b*b)*p*s - 13.125*(d*d*d)*a*(b*b*b)*p*s + 68.90625*(d*d)*(a*a)*g*(p*p) - 68.90625*(d*d)*a*b*g*(p*p) + 9.84375*(d*d)*(b*b)*g*(p*p) + 137.8125*d*(a*a)*(p*p)*s - 137.8125*d*a*b*(p*p)*s + 19.6875*d*(b*b)*(p*p)*s + 24.609375*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad375(double a, double b, double p, double d, double s, double g){
	return (0.65625*(d*d*d*d*d)*(a*a*a*a*a)*g - 3.28125*(d*d*d*d*d)*(a*a*a*a)*b*g + 3.28125*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 0.65625*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 3.28125*(d*d*d*d)*(a*a*a*a*a)*s - 16.40625*(d*d*d*d)*(a*a*a*a)*b*s + 16.40625*(d*d*d*d)*(a*a*a)*(b*b)*s - 3.28125*(d*d*d*d)*(a*a)*(b*b*b)*s + 11.484375*(d*d*d)*(a*a*a)*g*p - 20.671875*(d*d*d)*(a*a)*b*g*p + 6.890625*(d*d*d)*a*(b*b)*g*p - 0.328125*(d*d*d)*(b*b*b)*g*p + 34.453125*(d*d)*(a*a*a)*p*s - 62.015625*(d*d)*(a*a)*b*p*s + 20.671875*(d*d)*a*(b*b)*p*s - 0.984375*(d*d)*(b*b*b)*p*s + 20.671875*d*a*g*(p*p) - 8.859375*d*b*g*(p*p) + 20.671875*a*(p*p)*s - 8.859375*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad376(double a, double b, double p, double d, double s, double g){
	return (0.546875*(d*d*d*d)*(a*a*a*a)*g - 1.640625*(d*d*d*d)*(a*a*a)*b*g + 0.984375*(d*d*d*d)*(a*a)*(b*b)*g - 0.109375*(d*d*d*d)*a*(b*b*b)*g + 2.1875*(d*d*d)*(a*a*a*a)*s - 6.5625*(d*d*d)*(a*a*a)*b*s + 3.9375*(d*d*d)*(a*a)*(b*b)*s - 0.4375*(d*d*d)*a*(b*b*b)*s + 4.59375*(d*d)*(a*a)*g*p - 4.59375*(d*d)*a*b*g*p + 0.65625*(d*d)*(b*b)*g*p + 9.1875*d*(a*a)*p*s - 9.1875*d*a*b*p*s + 1.3125*d*(b*b)*p*s + 2.4609375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad377(double a, double b, double p, double d, double s, double g){
	return (0.2734375*(d*d*d)*(a*a*a)*g - 0.4921875*(d*d*d)*(a*a)*b*g + 0.1640625*(d*d*d)*a*(b*b)*g - 0.0078125*(d*d*d)*(b*b*b)*g + 0.8203125*(d*d)*(a*a*a)*s - 1.4765625*(d*d)*(a*a)*b*s + 0.4921875*(d*d)*a*(b*b)*s - 0.0234375*(d*d)*(b*b*b)*s + 0.984375*d*a*g*p - 0.421875*d*b*g*p + 0.984375*a*p*s - 0.421875*b*p*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad378(double a, double b, double p, double d, double s, double g){
	return (0.08203125*(d*d)*(a*a)*g - 0.08203125*(d*d)*a*b*g + 0.01171875*(d*d)*(b*b)*g + 0.1640625*d*(a*a)*s - 0.1640625*d*a*b*s + 0.0234375*d*(b*b)*s + 0.087890625*g*p)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad379(double a, double b, double p, double d, double s, double g){
	return (0.013671875*d*a*g - 0.005859375*d*b*g + 0.013671875*a*s - 0.005859375*b*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad3710(double a, double b, double p, double d, double s, double g){
	return 0.0009765625*g/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad380(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*g - 11.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*s - 1.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g*p + 12.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*p - 14.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p - 13.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*p*s + 108.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*p*s - 126.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 6.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p) - 63.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 126.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 52.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 42.0*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p)*s - 441.0 *(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 882.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 367.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 105.0*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 393.75*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 315.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 52.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 525.0*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 1968.75*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 1575.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 262.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 367.5*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 551.25*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 157.5*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 6.5625*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 1102.5*(d*d)*(a*a*a)*(p*p*p*p)*s - 1653.75*(d*d)*(a*a)*b*(p*p*p*p)*s + 472.5*(d*d)*a*(b*b)*(p*p*p*p)*s - 19.6875*(d*d)*(b*b*b)*(p*p*p*p)*s + 236.25*d*a*g*(p*p*p*p*p) - 88.59375*d*b*g*(p*p*p*p*p) + 236.25*a*(p*p*p*p*p)*s - 88.59375*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad381(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g - 4.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g + 15.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*s - 40.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*s + 0.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*p - 18.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 63.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 42.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 6.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*p*s - 144.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 504.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 336.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 52.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 315.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 393.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 105.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 315.0*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 1890.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 2362.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 630.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 459.375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 1102.5*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 551.25*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 52.5*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 1837.5*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 4410.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 2205.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 210.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 826.875*(d*d)*(a*a)*g*(p*p*p*p) - 708.75*(d*d)*a*b*g*(p*p*p*p) + 88.59375*(d*d)*(b*b)*g*(p*p*p*p) + 1653.75*d*(a*a)*(p*p*p*p)*s - 1417.5*d*a*b*(p*p*p*p)*s + 177.1875*d*(b*b)*(p*p*p*p)*s + 162.421875*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad382(double a, double b, double p, double d, double s, double g){
	return (-0.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g + 6.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 7.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g - 6.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*s + 54.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s - 63.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 6.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 63.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 126.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 52.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 42.0*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 441.0 *(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 882.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 367.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 157.5*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 590.625*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 472.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 78.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 787.5*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 2953.125*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 2362.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 393.75*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 735.0*(d*d*d)*(a*a*a)*g*(p*p*p) - 1102.5*(d*d*d)*(a*a)*b*g*(p*p*p) + 315.0*(d*d*d)*a*(b*b)*g*(p*p*p) - 13.125*(d*d*d)*(b*b*b)*g*(p*p*p) + 2205.0*(d*d)*(a*a*a)*(p*p*p)*s - 3307.5*(d*d)*(a*a)*b*(p*p*p)*s + 945.0*(d*d)*a*(b*b)*(p*p*p)*s - 39.375*(d*d)*(b*b*b)*(p*p*p)*s + 590.625*d*a*g*(p*p*p*p) - 221.484375*d*b*g*(p*p*p*p) + 590.625*a*(p*p*p*p)*s - 221.484375*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad383(double a, double b, double p, double d, double s, double g){
	return (0.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g - 3.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 10.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 7.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + (d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*s - 24.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 84.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 56.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 17.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 105.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 131.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 35.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 105.0*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 630.0*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 787.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 210.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 229.6875*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 551.25*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 275.625*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 26.25*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 918.75*(d*d*d)*(a*a*a*a)*(p*p)*s - 2205.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 1102.5*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 105.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 551.25*(d*d)*(a*a)*g*(p*p*p) - 472.5*(d*d)*a*b*g*(p*p*p) + 59.0625*(d*d)*(b*b)*g*(p*p*p) + 1102.5*d*(a*a)*(p*p*p)*s - 945.0*d*a*b*(p*p*p)*s + 118.125*d*(b*b)*(p*p*p)*s + 135.3515625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad384(double a, double b, double p, double d, double s, double g){
	return (0.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 5.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 10.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 4.375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 3.5*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 36.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 73.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 30.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 26.25*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 98.4375*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 78.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 13.125*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 131.25*(d*d*d*d)*(a*a*a*a*a)*p*s - 492.1875*(d*d*d*d)*(a*a*a*a)*b*p*s + 393.75*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 65.625*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 183.75*(d*d*d)*(a*a*a)*g*(p*p) - 275.625*(d*d*d)*(a*a)*b*g*(p*p) + 78.75*(d*d*d)*a*(b*b)*g*(p*p) - 3.28125*(d*d*d)*(b*b*b)*g*(p*p) + 551.25*(d*d)*(a*a*a)*(p*p)*s - 826.875*(d*d)*(a*a)*b*(p*p)*s + 236.25*(d*d)*a*(b*b)*(p*p)*s - 9.84375*(d*d)*(b*b*b)*(p*p)*s + 196.875*d*a*g*(p*p*p) - 73.828125*d*b*g*(p*p*p) + 196.875*a*(p*p*p)*s - 73.828125*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad385(double a, double b, double p, double d, double s, double g){
	return (0.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 5.25*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 6.5625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 1.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 5.25*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 31.5*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 39.375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 10.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 22.96875*(d*d*d*d)*(a*a*a*a)*g*p - 55.125*(d*d*d*d)*(a*a*a)*b*g*p + 27.5625*(d*d*d*d)*(a*a)*(b*b)*g*p - 2.625*(d*d*d*d)*a*(b*b*b)*g*p + 91.875*(d*d*d)*(a*a*a*a)*p*s - 220.5*(d*d*d)*(a*a*a)*b*p*s + 110.25*(d*d*d)*(a*a)*(b*b)*p*s - 10.5*(d*d*d)*a*(b*b*b)*p*s + 82.6875*(d*d)*(a*a)*g*(p*p) - 70.875*(d*d)*a*b*g*(p*p) + 8.859375*(d*d)*(b*b)*g*(p*p) + 165.375*d*(a*a)*(p*p)*s - 141.75*d*a*b*(p*p)*s + 17.71875*d*(b*b)*(p*p)*s + 27.0703125*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad386(double a, double b, double p, double d, double s, double g){
	return (0.875*(d*d*d*d*d)*(a*a*a*a*a)*g - 3.28125*(d*d*d*d*d)*(a*a*a*a)*b*g + 2.625*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 0.4375*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 4.375*(d*d*d*d)*(a*a*a*a*a)*s - 16.40625*(d*d*d*d)*(a*a*a*a)*b*s + 13.125*(d*d*d*d)*(a*a*a)*(b*b)*s - 2.1875*(d*d*d*d)*(a*a)*(b*b*b)*s + 12.25*(d*d*d)*(a*a*a)*g*p - 18.375*(d*d*d)*(a*a)*b*g*p + 5.25*(d*d*d)*a*(b*b)*g*p - 0.21875*(d*d*d)*(b*b*b)*g*p + 36.75*(d*d)*(a*a*a)*p*s - 55.125*(d*d)*(a*a)*b*p*s + 15.75*(d*d)*a*(b*b)*p*s - 0.65625*(d*d)*(b*b*b)*p*s + 19.6875*d*a*g*(p*p) - 7.3828125*d*b*g*(p*p) + 19.6875*a*(p*p)*s - 7.3828125*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad387(double a, double b, double p, double d, double s, double g){
	return (0.546875*(d*d*d*d)*(a*a*a*a)*g - 1.3125*(d*d*d*d)*(a*a*a)*b*g + 0.65625*(d*d*d*d)*(a*a)*(b*b)*g - 0.0625*(d*d*d*d)*a*(b*b*b)*g + 2.1875*(d*d*d)*(a*a*a*a)*s - 5.25*(d*d*d)*(a*a*a)*b*s + 2.625*(d*d*d)*(a*a)*(b*b)*s - 0.25*(d*d*d)*a*(b*b*b)*s + 3.9375*(d*d)*(a*a)*g*p - 3.375*(d*d)*a*b*g*p + 0.421875*(d*d)*(b*b)*g*p + 7.875*d*(a*a)*p*s - 6.75*d*a*b*p*s + 0.84375*d*(b*b)*p*s + 1.93359375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad388(double a, double b, double p, double d, double s, double g){
	return (0.21875*(d*d*d)*(a*a*a)*g - 0.328125*(d*d*d)*(a*a)*b*g + 0.09375*(d*d*d)*a*(b*b)*g - 0.00390625*(d*d*d)*(b*b*b)*g + 0.65625*(d*d)*(a*a*a)*s - 0.984375*(d*d)*(a*a)*b*s + 0.28125*(d*d)*a*(b*b)*s - 0.01171875*(d*d)*(b*b*b)*s + 0.703125*d*a*g*p - 0.263671875*d*b*g*p + 0.703125*a*p*s - 0.263671875*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad389(double a, double b, double p, double d, double s, double g){
	return (0.0546875*(d*d)*(a*a)*g - 0.046875*(d*d)*a*b*g + 0.005859375*(d*d)*(b*b)*g + 0.109375*d*(a*a)*s - 0.09375*d*a*b*s + 0.01171875*d*(b*b)*s + 0.0537109375*g*p)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad3810(double a, double b, double p, double d, double s, double g){
	return (0.0078125*d*a*g - 0.0029296875*d*b*g + 0.0078125*a*s - 0.0029296875*b*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad3811(double a, double b, double p, double d, double s, double g){
	return 0.00048828125*g/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad400(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d)*(b*b*b*b)*g + 4.0*(d*d*d)*(b*b*b*b)*s + 3.0*(d*d)*(b*b)*g*p + 6.0*d*(b*b)*p*s + 0.75*g*(p*p))/(p*p*p*p);
}

inline double MD_Et_grad401(double a, double b, double p, double d, double s, double g){
	return b*(-2.0*(d*d*d)*(b*b)*g - 6.0*(d*d)*(b*b)*s - 3.0*d*g*p - 3.0*p*s)/(p*p*p*p);
}

inline double MD_Et_grad402(double a, double b, double p, double d, double s, double g){
	return (1.5*d*(b*b)*(d*g + 2*s) + 0.75*g*p)/(p*p*p*p);
}

inline double MD_Et_grad403(double a, double b, double p, double d, double s, double g){
	return 0.5*b*(-d*g - s)/(p*p*p*p);
}

inline double MD_Et_grad404(double a, double b, double p, double d, double s, double g){
	return 0.0625*g/(p*p*p*p);
}

inline double MD_Et_grad410(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d)*a*(b*b*b*b)*g + 5.0*(d*d*d*d)*a*(b*b*b*b)*s + 3.0*(d*d*d)*a*(b*b)*g*p - 2.0*(d*d*d)*(b*b*b)*g*p + 9.0*(d*d)*a*(b*b)*p*s - 6.0*(d*d)*(b*b*b)*p*s + 0.75*d*a*g*(p*p) - 3.0*d*b*g*(p*p) + 0.75*a*(p*p)*s - 3.0*b*(p*p)*s)/(p*p*p*p*p);
}

inline double MD_Et_grad411(double a, double b, double p, double d, double s, double g){
	return (-2.0*(d*d*d*d)*a*(b*b*b)*g + 0.5*(d*d*d*d)*(b*b*b*b)*g - 8.0*(d*d*d)*a*(b*b*b)*s + 2.0*(d*d*d)*(b*b*b*b)*s - 3.0*(d*d)*a*b*g*p + 4.5*(d*d)*(b*b)*g*p - 6.0*d*a*b*p*s + 9.0*d*(b*b)*p*s + 1.875*g*(p*p))/(p*p*p*p*p);
}

inline double MD_Et_grad412(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d)*a*(b*b)*g - (d*d*d)*(b*b*b)*g + 4.5*(d*d)*a*(b*b)*s - 3.0*(d*d)*(b*b*b)*s + 0.75*d*a*g*p - 3.0*d*b*g*p + 0.75*a*p*s - 3.0*b*p*s)/(p*p*p*p*p);
}

inline double MD_Et_grad413(double a, double b, double p, double d, double s, double g){
	return (-0.5*(d*d)*a*b*g + 0.75*(d*d)*(b*b)*g - d*a*b*s + 1.5*d*(b*b)*s + 0.625*g*p)/(p*p*p*p*p);
}

inline double MD_Et_grad414(double a, double b, double p, double d, double s, double g){
	return (0.0625*d*a*g - 0.25*d*b*g + 0.0625*a*s - 0.25*b*s)/(p*p*p*p*p);
}

inline double MD_Et_grad415(double a, double b, double p, double d, double s, double g){
	return 0.03125*g/(p*p*p*p*p);
}

inline double MD_Et_grad420(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g + 6.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s + 3.0*(d*d*d*d)*(a*a)*(b*b)*g*p - 4.0*(d*d*d*d)*a*(b*b*b)*g*p + 0.5*(d*d*d*d)*(b*b*b*b)*g*p + 12.0*(d*d*d)*(a*a)*(b*b)*p*s - 16.0*(d*d*d)*a*(b*b*b)*p*s + 2.0*(d*d*d)*(b*b*b*b)*p*s + 0.75*(d*d)*(a*a)*g*(p*p) - 6.0*(d*d)*a*b*g*(p*p) + 4.5*(d*d)*(b*b)*g*(p*p) + 1.5*d*(a*a)*(p*p)*s - 12.0*d*a*b*(p*p)*s + 9.0*d*(b*b)*(p*p)*s + 1.875*g*(p*p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad421(double a, double b, double p, double d, double s, double g){
	return (-2.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g + (d*d*d*d*d)*a*(b*b*b*b)*g - 10.0*(d*d*d*d)*(a*a)*(b*b*b)*s + 5.0*(d*d*d*d)*a*(b*b*b*b)*s - 3.0*(d*d*d)*(a*a)*b*g*p + 9.0*(d*d*d)*a*(b*b)*g*p - 3.0*(d*d*d)*(b*b*b)*g*p - 9.0*(d*d)*(a*a)*b*p*s + 27.0*(d*d)*a*(b*b)*p*s - 9.0*(d*d)*(b*b*b)*p*s + 3.75*d*a*g*(p*p) - 7.5*d*b*g*(p*p) + 3.75*a*(p*p)*s - 7.5*b*(p*p)*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad422(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d)*(a*a)*(b*b)*g - 2.0*(d*d*d*d)*a*(b*b*b)*g + 0.25*(d*d*d*d)*(b*b*b*b)*g + 6.0*(d*d*d)*(a*a)*(b*b)*s - 8.0*(d*d*d)*a*(b*b*b)*s + (d*d*d)*(b*b*b*b)*s + 0.75*(d*d)*(a*a)*g*p - 6.0*(d*d)*a*b*g*p + 4.5*(d*d)*(b*b)*g*p + 1.5*d*(a*a)*p*s - 12.0*d*a*b*p*s + 9.0*d*(b*b)*p*s + 2.8125*g*(p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad423(double a, double b, double p, double d, double s, double g){
	return (-0.5*(d*d*d)*(a*a)*b*g + 1.5*(d*d*d)*a*(b*b)*g - 0.5*(d*d*d)*(b*b*b)*g - 1.5*(d*d)*(a*a)*b*s + 4.5*(d*d)*a*(b*b)*s - 1.5*(d*d)*(b*b*b)*s + 1.25*d*a*g*p - 2.5*d*b*g*p + 1.25*a*p*s - 2.5*b*p*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad424(double a, double b, double p, double d, double s, double g){
	return (0.0625*(d*d)*(a*a)*g - 0.5*(d*d)*a*b*g + 0.375*(d*d)*(b*b)*g + 0.125*d*(a*a)*s - d*a*b*s + 0.75*d*(b*b)*s + 0.46875*g*p)/(p*p*p*p*p*p);
}

inline double MD_Et_grad425(double a, double b, double p, double d, double s, double g){
	return (0.0625*d*a*g - 0.125*d*b*g + 0.0625*a*s - 0.125*b*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad426(double a, double b, double p, double d, double s, double g){
	return 0.015625*g/(p*p*p*p*p*p);
}

inline double MD_Et_grad430(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g + 7.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s + 3.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 6.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 1.5*(d*d*d*d*d)*a*(b*b*b*b)*g*p + 15.0*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 30.0*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 7.5*(d*d*d*d)*a*(b*b*b*b)*p*s + 0.75*(d*d*d)*(a*a*a)*g*(p*p) - 9.0*(d*d*d)*(a*a)*b*g*(p*p) + 13.5*(d*d*d)*a*(b*b)*g*(p*p) - 3.0*(d*d*d)*(b*b*b)*g*(p*p) + 2.25*(d*d)*(a*a*a)*(p*p)*s - 27.0*(d*d)*(a*a)*b*(p*p)*s + 40.5*(d*d)*a*(b*b)*(p*p)*s - 9.0*(d*d)*(b*b*b)*(p*p)*s + 5.625*d*a*g*(p*p*p) - 7.5*d*b*g*(p*p*p) + 5.625*a*(p*p*p)*s - 7.5*b*(p*p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad431(double a, double b, double p, double d, double s, double g){
	return (-2.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 1.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 12.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 9.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 3.0*(d*d*d*d)*(a*a*a)*b*g*p + 13.5*(d*d*d*d)*(a*a)*(b*b)*g*p - 9.0*(d*d*d*d)*a*(b*b*b)*g*p + 0.75*(d*d*d*d)*(b*b*b*b)*g*p - 12.0*(d*d*d)*(a*a*a)*b*p*s + 54.0*(d*d*d)*(a*a)*(b*b)*p*s - 36.0*(d*d*d)*a*(b*b*b)*p*s + 3.0*(d*d*d)*(b*b*b*b)*p*s + 5.625*(d*d)*(a*a)*g*(p*p) - 22.5*(d*d)*a*b*g*(p*p) + 11.25*(d*d)*(b*b)*g*(p*p) + 11.25*d*(a*a)*(p*p)*s - 45.0*d*a*b*(p*p)*s + 22.5*d*(b*b)*(p*p)*s + 6.5625*g*(p*p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad432(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 3.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.75*(d*d*d*d*d)*a*(b*b*b*b)*g + 7.5*(d*d*d*d)*(a*a*a)*(b*b)*s - 15.0*(d*d*d*d)*(a*a)*(b*b*b)*s + 3.75*(d*d*d*d)*a*(b*b*b*b)*s + 0.75*(d*d*d)*(a*a*a)*g*p - 9.0*(d*d*d)*(a*a)*b*g*p + 13.5*(d*d*d)*a*(b*b)*g*p - 3.0*(d*d*d)*(b*b*b)*g*p + 2.25*(d*d)*(a*a*a)*p*s - 27.0*(d*d)*(a*a)*b*p*s + 40.5*(d*d)*a*(b*b)*p*s - 9.0*(d*d)*(b*b*b)*p*s + 8.4375*d*a*g*(p*p) - 11.25*d*b*g*(p*p) + 8.4375*a*(p*p)*s - 11.25*b*(p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad433(double a, double b, double p, double d, double s, double g){
	return (-0.5*(d*d*d*d)*(a*a*a)*b*g + 2.25*(d*d*d*d)*(a*a)*(b*b)*g - 1.5*(d*d*d*d)*a*(b*b*b)*g + 0.125*(d*d*d*d)*(b*b*b*b)*g - 2.0*(d*d*d)*(a*a*a)*b*s + 9.0*(d*d*d)*(a*a)*(b*b)*s - 6.0*(d*d*d)*a*(b*b*b)*s + 0.5*(d*d*d)*(b*b*b*b)*s + 1.875*(d*d)*(a*a)*g*p - 7.5*(d*d)*a*b*g*p + 3.75*(d*d)*(b*b)*g*p + 3.75*d*(a*a)*p*s - 15.0*d*a*b*p*s + 7.5*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad434(double a, double b, double p, double d, double s, double g){
	return (0.0625*(d*d*d)*(a*a*a)*g - 0.75*(d*d*d)*(a*a)*b*g + 1.125*(d*d*d)*a*(b*b)*g - 0.25*(d*d*d)*(b*b*b)*g + 0.1875*(d*d)*(a*a*a)*s - 2.25*(d*d)*(a*a)*b*s + 3.375*(d*d)*a*(b*b)*s - 0.75*(d*d)*(b*b*b)*s + 1.40625*d*a*g*p - 1.875*d*b*g*p + 1.40625*a*p*s - 1.875*b*p*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad435(double a, double b, double p, double d, double s, double g){
	return (0.09375*(d*d)*(a*a)*g - 0.375*(d*d)*a*b*g + 0.1875*(d*d)*(b*b)*g + 0.1875*d*(a*a)*s - 0.75*d*a*b*s + 0.375*d*(b*b)*s + 0.328125*g*p)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad436(double a, double b, double p, double d, double s, double g){
	return (0.046875*d*a*g - 0.0625*d*b*g + 0.046875*a*s - 0.0625*b*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad437(double a, double b, double p, double d, double s, double g){
	return 0.0078125*g/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad440(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g + 8.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s + 3.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 8.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 3.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p + 18.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 48.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 18.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s + 0.75*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 12.0*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 27.0*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 12.0*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 0.75*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 3.0*(d*d*d)*(a*a*a*a)*(p*p)*s - 48.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 108.0*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 48.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 3.0*(d*d*d)*(b*b*b*b)*(p*p)*s + 11.25*(d*d)*(a*a)*g*(p*p*p) - 30.0*(d*d)*a*b*g*(p*p*p) + 11.25*(d*d)*(b*b)*g*(p*p*p) + 22.5*d*(a*a)*(p*p*p)*s - 60.0*d*a*b*(p*p*p)*s + 22.5*d*(b*b)*(p*p*p)*s + 6.5625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad441(double a, double b, double p, double d, double s, double g){
	return (-2.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 2.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 14.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 14.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 3.0*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 18.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 18.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 3.0*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 15.0*(d*d*d*d)*(a*a*a*a)*b*p*s + 90.0*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 90.0*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 15.0*(d*d*d*d)*a*(b*b*b*b)*p*s + 7.5*(d*d*d)*(a*a*a)*g*(p*p) - 45.0*(d*d*d)*(a*a)*b*g*(p*p) + 45.0*(d*d*d)*a*(b*b)*g*(p*p) - 7.5*(d*d*d)*(b*b*b)*g*(p*p) + 22.5*(d*d)*(a*a*a)*(p*p)*s - 135.0*(d*d)*(a*a)*b*(p*p)*s + 135.0*(d*d)*a*(b*b)*(p*p)*s - 22.5*(d*d)*(b*b*b)*(p*p)*s + 26.25*d*a*g*(p*p*p) - 26.25*d*b*g*(p*p*p) + 26.25*a*(p*p*p)*s - 26.25*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad442(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 4.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 1.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g + 9.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 24.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 9.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s + 0.75*(d*d*d*d)*(a*a*a*a)*g*p - 12.0*(d*d*d*d)*(a*a*a)*b*g*p + 27.0*(d*d*d*d)*(a*a)*(b*b)*g*p - 12.0*(d*d*d*d)*a*(b*b*b)*g*p + 0.75*(d*d*d*d)*(b*b*b*b)*g*p + 3.0*(d*d*d)*(a*a*a*a)*p*s - 48.0*(d*d*d)*(a*a*a)*b*p*s + 108.0*(d*d*d)*(a*a)*(b*b)*p*s - 48.0*(d*d*d)*a*(b*b*b)*p*s + 3.0*(d*d*d)*(b*b*b*b)*p*s + 16.875*(d*d)*(a*a)*g*(p*p) - 45.0*(d*d)*a*b*g*(p*p) + 16.875*(d*d)*(b*b)*g*(p*p) + 33.75*d*(a*a)*(p*p)*s - 90.0*d*a*b*(p*p)*s + 33.75*d*(b*b)*(p*p)*s + 13.125*g*(p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad443(double a, double b, double p, double d, double s, double g){
	return (-0.5*(d*d*d*d*d)*(a*a*a*a)*b*g + 3.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 3.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.5*(d*d*d*d*d)*a*(b*b*b*b)*g - 2.5*(d*d*d*d)*(a*a*a*a)*b*s + 15.0*(d*d*d*d)*(a*a*a)*(b*b)*s - 15.0*(d*d*d*d)*(a*a)*(b*b*b)*s + 2.5*(d*d*d*d)*a*(b*b*b*b)*s + 2.5*(d*d*d)*(a*a*a)*g*p - 15.0*(d*d*d)*(a*a)*b*g*p + 15.0*(d*d*d)*a*(b*b)*g*p - 2.5*(d*d*d)*(b*b*b)*g*p + 7.5*(d*d)*(a*a*a)*p*s - 45.0*(d*d)*(a*a)*b*p*s + 45.0*(d*d)*a*(b*b)*p*s - 7.5*(d*d)*(b*b*b)*p*s + 13.125*d*a*g*(p*p) - 13.125*d*b*g*(p*p) + 13.125*a*(p*p)*s - 13.125*b*(p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad444(double a, double b, double p, double d, double s, double g){
	return (0.0625*(d*d*d*d)*(a*a*a*a)*g - (d*d*d*d)*(a*a*a)*b*g + 2.25*(d*d*d*d)*(a*a)*(b*b)*g - (d*d*d*d)*a*(b*b*b)*g + 0.0625*(d*d*d*d)*(b*b*b*b)*g + 0.25*(d*d*d)*(a*a*a*a)*s - 4.0*(d*d*d)*(a*a*a)*b*s + 9.0*(d*d*d)*(a*a)*(b*b)*s - 4.0*(d*d*d)*a*(b*b*b)*s + 0.25*(d*d*d)*(b*b*b*b)*s + 2.8125*(d*d)*(a*a)*g*p - 7.5*(d*d)*a*b*g*p + 2.8125*(d*d)*(b*b)*g*p + 5.625*d*(a*a)*p*s - 15.0*d*a*b*p*s + 5.625*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad445(double a, double b, double p, double d, double s, double g){
	return (0.125*(d*d*d)*(a*a*a)*g - 0.75*(d*d*d)*(a*a)*b*g + 0.75*(d*d*d)*a*(b*b)*g - 0.125*(d*d*d)*(b*b*b)*g + 0.375*(d*d)*(a*a*a)*s - 2.25*(d*d)*(a*a)*b*s + 2.25*(d*d)*a*(b*b)*s - 0.375*(d*d)*(b*b*b)*s + 1.3125*d*a*g*p - 1.3125*d*b*g*p + 1.3125*a*p*s - 1.3125*b*p*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad446(double a, double b, double p, double d, double s, double g){
	return (0.09375*(d*d)*(a*a)*g - 0.25*(d*d)*a*b*g + 0.09375*(d*d)*(b*b)*g + 0.1875*d*(a*a)*s - 0.5*d*a*b*s + 0.1875*d*(b*b)*s + 0.21875*g*p)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad447(double a, double b, double p, double d, double s, double g){
	return 0.03125*(d*a*g - d*b*g + a*s - b*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad448(double a, double b, double p, double d, double s, double g){
	return 0.00390625*g/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad450(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g + 9.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s + 3.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 10.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 5.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p + 21.0 *(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 70.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 35.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s + 0.75*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 15.0*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 45.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 30.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 3.75*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) + 3.75*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 75.0*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 225.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 150.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 18.75*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s + 18.75*(d*d*d)*(a*a*a)*g*(p*p*p) - 75.0*(d*d*d)*(a*a)*b*g*(p*p*p) + 56.25*(d*d*d)*a*(b*b)*g*(p*p*p) - 7.5*(d*d*d)*(b*b*b)*g*(p*p*p) + 56.25*(d*d)*(a*a*a)*(p*p*p)*s - 225.0*(d*d)*(a*a)*b*(p*p*p)*s + 168.75*(d*d)*a*(b*b)*(p*p*p)*s - 22.5*(d*d)*(b*b*b)*(p*p*p)*s + 32.8125*d*a*g*(p*p*p*p) - 26.25*d*b*g*(p*p*p*p) + 32.8125*a*(p*p*p*p)*s - 26.25*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad451(double a, double b, double p, double d, double s, double g){
	return (-2.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 2.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 16.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 20.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 3.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 22.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 30.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 7.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 18.0*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 135.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 180.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 45.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s + 9.375*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 75.0*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 112.5*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 37.5*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 1.875*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 37.5*(d*d*d)*(a*a*a*a)*(p*p)*s - 300.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 450.0*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 150.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 7.5*(d*d*d)*(b*b*b*b)*(p*p)*s + 65.625*(d*d)*(a*a)*g*(p*p*p) - 131.25*(d*d)*a*b*g*(p*p*p) + 39.375*(d*d)*(b*b)*g*(p*p*p) + 131.25*d*(a*a)*(p*p*p)*s - 262.5*d*a*b*(p*p*p)*s + 78.75*d*(b*b)*(p*p*p)*s + 29.53125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad452(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 5.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 2.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g + 10.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 35.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 17.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s + 0.75*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 15.0*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 45.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 30.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 3.75*(d*d*d*d*d)*a*(b*b*b*b)*g*p + 3.75*(d*d*d*d)*(a*a*a*a*a)*p*s - 75.0*(d*d*d*d)*(a*a*a*a)*b*p*s + 225.0*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 150.0*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 18.75*(d*d*d*d)*a*(b*b*b*b)*p*s + 28.125*(d*d*d)*(a*a*a)*g*(p*p) - 112.5*(d*d*d)*(a*a)*b*g*(p*p) + 84.375*(d*d*d)*a*(b*b)*g*(p*p) - 11.25*(d*d*d)*(b*b*b)*g*(p*p) + 84.375*(d*d)*(a*a*a)*(p*p)*s - 337.5*(d*d)*(a*a)*b*(p*p)*s + 253.125*(d*d)*a*(b*b)*(p*p)*s - 33.75*(d*d)*(b*b*b)*(p*p)*s + 65.625*d*a*g*(p*p*p) - 52.5*d*b*g*(p*p*p) + 65.625*a*(p*p*p)*s - 52.5*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad453(double a, double b, double p, double d, double s, double g){
	return (-0.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 3.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 5.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 1.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 3.0*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 22.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 30.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 7.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s + 3.125*(d*d*d*d)*(a*a*a*a)*g*p - 25.0*(d*d*d*d)*(a*a*a)*b*g*p + 37.5*(d*d*d*d)*(a*a)*(b*b)*g*p - 12.5*(d*d*d*d)*a*(b*b*b)*g*p + 0.625*(d*d*d*d)*(b*b*b*b)*g*p + 12.5*(d*d*d)*(a*a*a*a)*p*s - 100.0*(d*d*d)*(a*a*a)*b*p*s + 150.0*(d*d*d)*(a*a)*(b*b)*p*s - 50.0*(d*d*d)*a*(b*b*b)*p*s + 2.5*(d*d*d)*(b*b*b*b)*p*s + 32.8125*(d*d)*(a*a)*g*(p*p) - 65.625*(d*d)*a*b*g*(p*p) + 19.6875*(d*d)*(b*b)*g*(p*p) + 65.625*d*(a*a)*(p*p)*s - 131.25*d*a*b*(p*p)*s + 39.375*d*(b*b)*(p*p)*s + 19.6875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad454(double a, double b, double p, double d, double s, double g){
	return (0.0625*(d*d*d*d*d)*(a*a*a*a*a)*g - 1.25*(d*d*d*d*d)*(a*a*a*a)*b*g + 3.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 2.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.3125*(d*d*d*d*d)*a*(b*b*b*b)*g + 0.3125*(d*d*d*d)*(a*a*a*a*a)*s - 6.25*(d*d*d*d)*(a*a*a*a)*b*s + 18.75*(d*d*d*d)*(a*a*a)*(b*b)*s - 12.5*(d*d*d*d)*(a*a)*(b*b*b)*s + 1.5625*(d*d*d*d)*a*(b*b*b*b)*s + 4.6875*(d*d*d)*(a*a*a)*g*p - 18.75*(d*d*d)*(a*a)*b*g*p + 14.0625*(d*d*d)*a*(b*b)*g*p - 1.875*(d*d*d)*(b*b*b)*g*p + 14.0625*(d*d)*(a*a*a)*p*s - 56.25*(d*d)*(a*a)*b*p*s + 42.1875*(d*d)*a*(b*b)*p*s - 5.625*(d*d)*(b*b*b)*p*s + 16.40625*d*a*g*(p*p) - 13.125*d*b*g*(p*p) + 16.40625*a*(p*p)*s - 13.125*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad455(double a, double b, double p, double d, double s, double g){
	return (0.15625*(d*d*d*d)*(a*a*a*a)*g - 1.25*(d*d*d*d)*(a*a*a)*b*g + 1.875*(d*d*d*d)*(a*a)*(b*b)*g - 0.625*(d*d*d*d)*a*(b*b*b)*g + 0.03125*(d*d*d*d)*(b*b*b*b)*g + 0.625*(d*d*d)*(a*a*a*a)*s - 5.0*(d*d*d)*(a*a*a)*b*s + 7.5*(d*d*d)*(a*a)*(b*b)*s - 2.5*(d*d*d)*a*(b*b*b)*s + 0.125*(d*d*d)*(b*b*b*b)*s + 3.28125*(d*d)*(a*a)*g*p - 6.5625*(d*d)*a*b*g*p + 1.96875*(d*d)*(b*b)*g*p + 6.5625*d*(a*a)*p*s - 13.125*d*a*b*p*s + 3.9375*d*(b*b)*p*s + 2.953125*g*(p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad456(double a, double b, double p, double d, double s, double g){
	return (0.15625*(d*d*d)*(a*a*a)*g - 0.625*(d*d*d)*(a*a)*b*g + 0.46875*(d*d*d)*a*(b*b)*g - 0.0625*(d*d*d)*(b*b*b)*g + 0.46875*(d*d)*(a*a*a)*s - 1.875*(d*d)*(a*a)*b*s + 1.40625*(d*d)*a*(b*b)*s - 0.1875*(d*d)*(b*b*b)*s + 1.09375*d*a*g*p - 0.875*d*b*g*p + 1.09375*a*p*s - 0.875*b*p*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad457(double a, double b, double p, double d, double s, double g){
	return (0.078125*(d*d)*(a*a)*g - 0.15625*(d*d)*a*b*g + 0.046875*(d*d)*(b*b)*g + 0.15625*d*(a*a)*s - 0.3125*d*a*b*s + 0.09375*d*(b*b)*s + 0.140625*g*p)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad458(double a, double b, double p, double d, double s, double g){
	return (0.01953125*d*a*g - 0.015625*d*b*g + 0.01953125*a*s - 0.015625*b*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad459(double a, double b, double p, double d, double s, double g){
	return 0.001953125*g/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad460(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g + 10.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s + 3.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 12.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 7.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p + 24.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 96.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 60.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s + 0.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 18.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 67.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 60.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 11.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) + 4.5*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 108.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 405.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 360.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 67.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s + 28.125*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 150.0*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 168.75*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 45.0*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 1.875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 112.5*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 600.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 675.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 180.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 7.5*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 98.4375*(d*d)*(a*a)*g*(p*p*p*p) - 157.5*(d*d)*a*b*g*(p*p*p*p) + 39.375*(d*d)*(b*b)*g*(p*p*p*p) + 196.875*d*(a*a)*(p*p*p*p)*s - 315.0*d*a*b*(p*p*p*p)*s + 78.75*d*(b*b)*(p*p*p*p)*s + 29.53125*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad461(double a, double b, double p, double d, double s, double g){
	return (-2.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 3.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 18.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 27.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 3.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 27.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 45.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 15.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 21.0 *(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 189.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 315.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 105.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s + 11.25*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 112.5*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 225.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 112.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 11.25*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) + 56.25*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 562.5*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 1125.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 562.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 56.25*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s + 131.25*(d*d*d)*(a*a*a)*g*(p*p*p) - 393.75*(d*d*d)*(a*a)*b*g*(p*p*p) + 236.25*(d*d*d)*a*(b*b)*g*(p*p*p) - 26.25*(d*d*d)*(b*b*b)*g*(p*p*p) + 393.75*(d*d)*(a*a*a)*(p*p*p)*s - 1181.25*(d*d)*(a*a)*b*(p*p*p)*s + 708.75*(d*d)*a*(b*b)*(p*p*p)*s - 78.75*(d*d)*(b*b*b)*(p*p*p)*s + 177.1875*d*a*g*(p*p*p*p) - 118.125*d*b*g*(p*p*p*p) + 177.1875*a*(p*p*p*p)*s - 118.125*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad462(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 6.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 3.75*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g + 12.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 48.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 30.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s + 0.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 18.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 67.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 60.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 11.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p + 4.5*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 108.0*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 405.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 360.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 67.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s + 42.1875*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 225.0*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 253.125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 67.5*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 2.8125*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 168.75*(d*d*d)*(a*a*a*a)*(p*p)*s - 900.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 1012.5*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 270.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 11.25*(d*d*d)*(b*b*b*b)*(p*p)*s + 196.875*(d*d)*(a*a)*g*(p*p*p) - 315.0*(d*d)*a*b*g*(p*p*p) + 78.75*(d*d)*(b*b)*g*(p*p*p) + 393.75*d*(a*a)*(p*p*p)*s - 630.0*d*a*b*(p*p*p)*s + 157.5*d*(b*b)*(p*p*p)*s + 73.828125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad463(double a, double b, double p, double d, double s, double g){
	return (-0.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 4.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 7.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 2.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 3.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 31.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 52.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 17.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s + 3.75*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 37.5*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 75.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 37.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 3.75*(d*d*d*d*d)*a*(b*b*b*b)*g*p + 18.75*(d*d*d*d)*(a*a*a*a*a)*p*s - 187.5*(d*d*d*d)*(a*a*a*a)*b*p*s + 375.0*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 187.5*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 18.75*(d*d*d*d)*a*(b*b*b*b)*p*s + 65.625*(d*d*d)*(a*a*a)*g*(p*p) - 196.875*(d*d*d)*(a*a)*b*g*(p*p) + 118.125*(d*d*d)*a*(b*b)*g*(p*p) - 13.125*(d*d*d)*(b*b*b)*g*(p*p) + 196.875*(d*d)*(a*a*a)*(p*p)*s - 590.625*(d*d)*(a*a)*b*(p*p)*s + 354.375*(d*d)*a*(b*b)*(p*p)*s - 39.375*(d*d)*(b*b*b)*(p*p)*s + 118.125*d*a*g*(p*p*p) - 78.75*d*b*g*(p*p*p) + 118.125*a*(p*p*p)*s - 78.75*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad464(double a, double b, double p, double d, double s, double g){
	return (0.0625*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 1.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 5.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 5.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 0.9375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g + 0.375*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 9.0*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 33.75*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 30.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 5.625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s + 7.03125*(d*d*d*d)*(a*a*a*a)*g*p - 37.5*(d*d*d*d)*(a*a*a)*b*g*p + 42.1875*(d*d*d*d)*(a*a)*(b*b)*g*p - 11.25*(d*d*d*d)*a*(b*b*b)*g*p + 0.46875*(d*d*d*d)*(b*b*b*b)*g*p + 28.125*(d*d*d)*(a*a*a*a)*p*s - 150.0*(d*d*d)*(a*a*a)*b*p*s + 168.75*(d*d*d)*(a*a)*(b*b)*p*s - 45.0*(d*d*d)*a*(b*b*b)*p*s + 1.875*(d*d*d)*(b*b*b*b)*p*s + 49.21875*(d*d)*(a*a)*g*(p*p) - 78.75*(d*d)*a*b*g*(p*p) + 19.6875*(d*d)*(b*b)*g*(p*p) + 98.4375*d*(a*a)*(p*p)*s - 157.5*d*a*b*(p*p)*s + 39.375*d*(b*b)*(p*p)*s + 24.609375*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad465(double a, double b, double p, double d, double s, double g){
	return (0.1875*(d*d*d*d*d)*(a*a*a*a*a)*g - 1.875*(d*d*d*d*d)*(a*a*a*a)*b*g + 3.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 1.875*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.1875*(d*d*d*d*d)*a*(b*b*b*b)*g + 0.9375*(d*d*d*d)*(a*a*a*a*a)*s - 9.375*(d*d*d*d)*(a*a*a*a)*b*s + 18.75*(d*d*d*d)*(a*a*a)*(b*b)*s - 9.375*(d*d*d*d)*(a*a)*(b*b*b)*s + 0.9375*(d*d*d*d)*a*(b*b*b*b)*s + 6.5625*(d*d*d)*(a*a*a)*g*p - 19.6875*(d*d*d)*(a*a)*b*g*p + 11.8125*(d*d*d)*a*(b*b)*g*p - 1.3125*(d*d*d)*(b*b*b)*g*p + 19.6875*(d*d)*(a*a*a)*p*s - 59.0625*(d*d)*(a*a)*b*p*s + 35.4375*(d*d)*a*(b*b)*p*s - 3.9375*(d*d)*(b*b*b)*p*s + 17.71875*d*a*g*(p*p) - 11.8125*d*b*g*(p*p) + 17.71875*a*(p*p)*s - 11.8125*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad466(double a, double b, double p, double d, double s, double g){
	return (0.234375*(d*d*d*d)*(a*a*a*a)*g - 1.25*(d*d*d*d)*(a*a*a)*b*g + 1.40625*(d*d*d*d)*(a*a)*(b*b)*g - 0.375*(d*d*d*d)*a*(b*b*b)*g + 0.015625*(d*d*d*d)*(b*b*b*b)*g + 0.9375*(d*d*d)*(a*a*a*a)*s - 5.0*(d*d*d)*(a*a*a)*b*s + 5.625*(d*d*d)*(a*a)*(b*b)*s - 1.5*(d*d*d)*a*(b*b*b)*s + 0.0625*(d*d*d)*(b*b*b*b)*s + 3.28125*(d*d)*(a*a)*g*p - 5.25*(d*d)*a*b*g*p + 1.3125*(d*d)*(b*b)*g*p + 6.5625*d*(a*a)*p*s - 10.5*d*a*b*p*s + 2.625*d*(b*b)*p*s + 2.4609375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad467(double a, double b, double p, double d, double s, double g){
	return (0.15625*(d*d*d)*(a*a*a)*g - 0.46875*(d*d*d)*(a*a)*b*g + 0.28125*(d*d*d)*a*(b*b)*g - 0.03125*(d*d*d)*(b*b*b)*g + 0.46875*(d*d)*(a*a*a)*s - 1.40625*(d*d)*(a*a)*b*s + 0.84375*(d*d)*a*(b*b)*s - 0.09375*(d*d)*(b*b*b)*s + 0.84375*d*a*g*p - 0.5625*d*b*g*p + 0.84375*a*p*s - 0.5625*b*p*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad468(double a, double b, double p, double d, double s, double g){
	return (0.05859375*(d*d)*(a*a)*g - 0.09375*(d*d)*a*b*g + 0.0234375*(d*d)*(b*b)*g + 0.1171875*d*(a*a)*s - 0.1875*d*a*b*s + 0.046875*d*(b*b)*s + 0.087890625*g*p)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad469(double a, double b, double p, double d, double s, double g){
	return (0.01171875*d*a*g - 0.0078125*d*b*g + 0.01171875*a*s - 0.0078125*b*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad4610(double a, double b, double p, double d, double s, double g){
	return 0.0009765625*g/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad470(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g + 11.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*s + 3.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*p - 14.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 10.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p + 27.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*p*s - 126.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 94.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s + 0.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p) - 21.0 *(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 94.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 105.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 26.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) + 5.25*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p)*s - 147.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 661.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 735.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 183.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s + 39.375*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 262.5*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 393.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 157.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 13.125*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) + 196.875*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 1312.5*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 1968.75*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 787.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 65.625*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s + 229.6875*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 551.25*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 275.625*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 26.25*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 689.0625*(d*d)*(a*a*a)*(p*p*p*p)*s - 1653.75*(d*d)*(a*a)*b*(p*p*p*p)*s + 826.875*(d*d)*a*(b*b)*(p*p*p*p)*s - 78.75*(d*d)*(b*b*b)*(p*p*p*p)*s + 206.71875*d*a*g*(p*p*p*p*p) - 118.125*d*b*g*(p*p*p*p*p) + 206.71875*a*(p*p*p*p*p)*s - 118.125*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad471(double a, double b, double p, double d, double s, double g){
	return (-2.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g + 3.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 20.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*s + 35.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 3.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 31.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 63.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 26.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 24.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 252.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 504.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 210.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s + 13.125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 157.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 393.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 262.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 39.375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) + 78.75*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 945.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 2362.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 1575.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 236.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s + 229.6875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 918.75*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 826.875*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 183.75*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 6.5625*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 918.75*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 3675.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 3307.5*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 735.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 26.25*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 620.15625*(d*d)*(a*a)*g*(p*p*p*p) - 826.875*(d*d)*a*b*g*(p*p*p*p) + 177.1875*(d*d)*(b*b)*g*(p*p*p*p) + 1240.3125*d*(a*a)*(p*p*p*p)*s - 1653.75*d*a*b*(p*p*p*p)*s + 354.375*d*(b*b)*(p*p*p*p)*s + 162.421875*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad472(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 7.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 5.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g + 13.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s - 63.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 47.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s + 0.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 21.0 *(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 94.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 105.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 26.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p + 5.25*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 147.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 661.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 735.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 183.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s + 59.0625*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 393.75*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 590.625*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 236.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 19.6875*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) + 295.3125*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 1968.75*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 2953.125*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 1181.25*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 98.4375*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s + 459.375*(d*d*d)*(a*a*a)*g*(p*p*p) - 1102.5*(d*d*d)*(a*a)*b*g*(p*p*p) + 551.25*(d*d*d)*a*(b*b)*g*(p*p*p) - 52.5*(d*d*d)*(b*b*b)*g*(p*p*p) + 1378.125*(d*d)*(a*a*a)*(p*p*p)*s - 3307.5*(d*d)*(a*a)*b*(p*p*p)*s + 1653.75*(d*d)*a*(b*b)*(p*p*p)*s - 157.5*(d*d)*(b*b*b)*(p*p*p)*s + 516.796875*d*a*g*(p*p*p*p) - 295.3125*d*b*g*(p*p*p*p) + 516.796875*a*(p*p*p*p)*s - 295.3125*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad473(double a, double b, double p, double d, double s, double g){
	return (-0.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 5.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 10.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 4.375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 4.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 42.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 84.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 35.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s + 4.375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 52.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 131.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 87.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 13.125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p + 26.25*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 315.0*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 787.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 525.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 78.75*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s + 114.84375*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 459.375*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 413.4375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 91.875*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 3.28125*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 459.375*(d*d*d)*(a*a*a*a)*(p*p)*s - 1837.5*(d*d*d)*(a*a*a)*b*(p*p)*s + 1653.75*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 367.5*(d*d*d)*a*(b*b*b)*(p*p)*s + 13.125*(d*d*d)*(b*b*b*b)*(p*p)*s + 413.4375*(d*d)*(a*a)*g*(p*p*p) - 551.25*(d*d)*a*b*g*(p*p*p) + 118.125*(d*d)*(b*b)*g*(p*p*p) + 826.875*d*(a*a)*(p*p*p)*s - 1102.5*d*a*b*(p*p*p)*s + 236.25*d*(b*b)*(p*p*p)*s + 135.3515625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad474(double a, double b, double p, double d, double s, double g){
	return (0.0625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 1.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 7.875*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 8.75*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 2.1875*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g + 0.4375*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 12.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 55.125*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 61.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 15.3125*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s + 9.84375*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 65.625*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 98.4375*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 39.375*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 3.28125*(d*d*d*d*d)*a*(b*b*b*b)*g*p + 49.21875*(d*d*d*d)*(a*a*a*a*a)*p*s - 328.125*(d*d*d*d)*(a*a*a*a)*b*p*s + 492.1875*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 196.875*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 16.40625*(d*d*d*d)*a*(b*b*b*b)*p*s + 114.84375*(d*d*d)*(a*a*a)*g*(p*p) - 275.625*(d*d*d)*(a*a)*b*g*(p*p) + 137.8125*(d*d*d)*a*(b*b)*g*(p*p) - 13.125*(d*d*d)*(b*b*b)*g*(p*p) + 344.53125*(d*d)*(a*a*a)*(p*p)*s - 826.875*(d*d)*(a*a)*b*(p*p)*s + 413.4375*(d*d)*a*(b*b)*(p*p)*s - 39.375*(d*d)*(b*b*b)*(p*p)*s + 172.265625*d*a*g*(p*p*p) - 98.4375*d*b*g*(p*p*p) + 172.265625*a*(p*p*p)*s - 98.4375*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad475(double a, double b, double p, double d, double s, double g){
	return (0.21875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 2.625*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 6.5625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 4.375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 0.65625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g + 1.3125*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 15.75*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 39.375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 26.25*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 3.9375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s + 11.484375*(d*d*d*d)*(a*a*a*a)*g*p - 45.9375*(d*d*d*d)*(a*a*a)*b*g*p + 41.34375*(d*d*d*d)*(a*a)*(b*b)*g*p - 9.1875*(d*d*d*d)*a*(b*b*b)*g*p + 0.328125*(d*d*d*d)*(b*b*b*b)*g*p + 45.9375*(d*d*d)*(a*a*a*a)*p*s - 183.75*(d*d*d)*(a*a*a)*b*p*s + 165.375*(d*d*d)*(a*a)*(b*b)*p*s - 36.75*(d*d*d)*a*(b*b*b)*p*s + 1.3125*(d*d*d)*(b*b*b*b)*p*s + 62.015625*(d*d)*(a*a)*g*(p*p) - 82.6875*(d*d)*a*b*g*(p*p) + 17.71875*(d*d)*(b*b)*g*(p*p) + 124.03125*d*(a*a)*(p*p)*s - 165.375*d*a*b*(p*p)*s + 35.4375*d*(b*b)*(p*p)*s + 27.0703125*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad476(double a, double b, double p, double d, double s, double g){
	return (0.328125*(d*d*d*d*d)*(a*a*a*a*a)*g - 2.1875*(d*d*d*d*d)*(a*a*a*a)*b*g + 3.28125*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 1.3125*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.109375*(d*d*d*d*d)*a*(b*b*b*b)*g + 1.640625*(d*d*d*d)*(a*a*a*a*a)*s - 10.9375*(d*d*d*d)*(a*a*a*a)*b*s + 16.40625*(d*d*d*d)*(a*a*a)*(b*b)*s - 6.5625*(d*d*d*d)*(a*a)*(b*b*b)*s + 0.546875*(d*d*d*d)*a*(b*b*b*b)*s + 7.65625*(d*d*d)*(a*a*a)*g*p - 18.375*(d*d*d)*(a*a)*b*g*p + 9.1875*(d*d*d)*a*(b*b)*g*p - 0.875*(d*d*d)*(b*b*b)*g*p + 22.96875*(d*d)*(a*a*a)*p*s - 55.125*(d*d)*(a*a)*b*p*s + 27.5625*(d*d)*a*(b*b)*p*s - 2.625*(d*d)*(b*b*b)*p*s + 17.2265625*d*a*g*(p*p) - 9.84375*d*b*g*(p*p) + 17.2265625*a*(p*p)*s - 9.84375*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad477(double a, double b, double p, double d, double s, double g){
	return (0.2734375*(d*d*d*d)*(a*a*a*a)*g - 1.09375*(d*d*d*d)*(a*a*a)*b*g + 0.984375*(d*d*d*d)*(a*a)*(b*b)*g - 0.21875*(d*d*d*d)*a*(b*b*b)*g + 0.0078125*(d*d*d*d)*(b*b*b*b)*g + 1.09375*(d*d*d)*(a*a*a*a)*s - 4.375*(d*d*d)*(a*a*a)*b*s + 3.9375*(d*d*d)*(a*a)*(b*b)*s - 0.875*(d*d*d)*a*(b*b*b)*s + 0.03125*(d*d*d)*(b*b*b*b)*s + 2.953125*(d*d)*(a*a)*g*p - 3.9375*(d*d)*a*b*g*p + 0.84375*(d*d)*(b*b)*g*p + 5.90625*d*(a*a)*p*s - 7.875*d*a*b*p*s + 1.6875*d*(b*b)*p*s + 1.93359375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad478(double a, double b, double p, double d, double s, double g){
	return (0.13671875*(d*d*d)*(a*a*a)*g - 0.328125*(d*d*d)*(a*a)*b*g + 0.1640625*(d*d*d)*a*(b*b)*g - 0.015625*(d*d*d)*(b*b*b)*g + 0.41015625*(d*d)*(a*a*a)*s - 0.984375*(d*d)*(a*a)*b*s + 0.4921875*(d*d)*a*(b*b)*s - 0.046875*(d*d)*(b*b*b)*s + 0.615234375*d*a*g*p - 0.3515625*d*b*g*p + 0.615234375*a*p*s - 0.3515625*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad479(double a, double b, double p, double d, double s, double g){
	return (0.041015625*(d*d)*(a*a)*g - 0.0546875*(d*d)*a*b*g + 0.01171875*(d*d)*(b*b)*g + 0.08203125*d*(a*a)*s - 0.109375*d*a*b*s + 0.0234375*d*(b*b)*s + 0.0537109375*g*p)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad4710(double a, double b, double p, double d, double s, double g){
	return (0.0068359375*d*a*g - 0.00390625*d*b*g + 0.0068359375*a*s - 0.00390625*b*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad4711(double a, double b, double p, double d, double s, double g){
	return 0.00048828125*g/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad480(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*g + 12.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*s + 3.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g*p - 16.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*p + 14.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*p + 30.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*p*s - 160.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*p*s + 140.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*p*s + 0.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*(p*p) - 24.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p) + 126.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p) - 168.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 52.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) + 6.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(p*p)*s - 192.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p)*s + 1008.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p)*s - 1344.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 420.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s + 52.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p) - 420.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 787.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 420.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 52.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) + 315.0*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p)*s - 2520.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 4725.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 2520.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 315.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s + 459.375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 1470.0*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 1102.5*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 210.0*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 6.5625*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 1837.5*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 5880.0*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 4410.0*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 840.0*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 26.25*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 826.875*(d*d)*(a*a)*g*(p*p*p*p*p) - 945.0*(d*d)*a*b*g*(p*p*p*p*p) + 177.1875*(d*d)*(b*b)*g*(p*p*p*p*p) + 1653.75*d*(a*a)*(p*p*p*p*p)*s - 1890.0*d*a*b*(p*p*p*p*p)*s + 354.375*d*(b*b)*(p*p*p*p*p)*s + 162.421875*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad481(double a, double b, double p, double d, double s, double g){
	return (-2.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*g + 4.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g - 22.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*s + 44.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*s - 3.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g*p + 36.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*p - 84.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 42.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 27.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*p*s + 324.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*p*s - 756.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 378.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s + 15.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p) - 210.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 630.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 525.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 105.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) + 105.0*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p)*s - 1470.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 4410.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 3675.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 735.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s + 367.5*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 1837.5*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 2205.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 735.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 52.5*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) + 1837.5*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 9187.5*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 11025.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 3675.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 262.5*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s + 1653.75*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 3307.5*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 1417.5*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 118.125*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 4961.25*(d*d)*(a*a*a)*(p*p*p*p)*s - 9922.5*(d*d)*(a*a)*b*(p*p*p*p)*s + 4252.5*(d*d)*a*(b*b)*(p*p*p*p)*s - 354.375*(d*d)*(b*b*b)*(p*p*p*p)*s + 1299.375*d*a*g*(p*p*p*p*p) - 649.6875*d*b*g*(p*p*p*p*p) + 1299.375*a*(p*p*p*p*p)*s - 649.6875*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad482(double a, double b, double p, double d, double s, double g){
	return (1.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g - 8.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g + 7.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g + 15.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*s - 80.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*s + 70.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s + 0.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*p - 24.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 126.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 168.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 52.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p + 6.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*p*s - 192.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 1008.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 1344.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 420.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s + 78.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 630.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 1181.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 630.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 78.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) + 472.5*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 3780.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 7087.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 3780.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 472.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s + 918.75*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 2940.0*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 2205.0*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 420.0*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 13.125*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 3675.0*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 11760.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 8820.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 1680.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 52.5*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 2067.1875*(d*d)*(a*a)*g*(p*p*p*p) - 2362.5*(d*d)*a*b*g*(p*p*p*p) + 442.96875*(d*d)*(b*b)*g*(p*p*p*p) + 4134.375*d*(a*a)*(p*p*p*p)*s - 4725.0*d*a*b*(p*p*p*p)*s + 885.9375*d*(b*b)*(p*p*p*p)*s + 487.265625*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad483(double a, double b, double p, double d, double s, double g){
	return (-0.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g + 6.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 14.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 7.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 4.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*s + 54.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s - 126.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 63.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s + 5.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 70.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 210.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 175.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 35.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p + 35.0*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 490.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 1470.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 1225.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 245.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s + 183.75*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 918.75*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 1102.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 367.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 26.25*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) + 918.75*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 4593.75*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 5512.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 1837.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 131.25*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s + 1102.5*(d*d*d)*(a*a*a)*g*(p*p*p) - 2205.0*(d*d*d)*(a*a)*b*g*(p*p*p) + 945.0*(d*d*d)*a*(b*b)*g*(p*p*p) - 78.75*(d*d*d)*(b*b*b)*g*(p*p*p) + 3307.5*(d*d)*(a*a*a)*(p*p*p)*s - 6615.0*(d*d)*(a*a)*b*(p*p*p)*s + 2835.0*(d*d)*a*(b*b)*(p*p*p)*s - 236.25*(d*d)*(b*b*b)*(p*p*p)*s + 1082.8125*d*a*g*(p*p*p*p) - 541.40625*d*b*g*(p*p*p*p) + 1082.8125*a*(p*p*p*p)*s - 541.40625*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad484(double a, double b, double p, double d, double s, double g){
	return (0.0625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g - 2.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 10.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 14.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 4.375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g + 0.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*s - 16.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 84.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 112.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 35.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s + 13.125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 105.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 196.875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 105.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 13.125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p + 78.75*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 630.0*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 1181.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 630.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 78.75*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s + 229.6875*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 735.0*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 551.25*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 105.0*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 3.28125*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 918.75*(d*d*d)*(a*a*a*a)*(p*p)*s - 2940.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 2205.0*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 420.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 13.125*(d*d*d)*(b*b*b*b)*(p*p)*s + 689.0625*(d*d)*(a*a)*g*(p*p*p) - 787.5*(d*d)*a*b*g*(p*p*p) + 147.65625*(d*d)*(b*b)*g*(p*p*p) + 1378.125*d*(a*a)*(p*p*p)*s - 1575.0*d*a*b*(p*p*p)*s + 295.3125*d*(b*b)*(p*p*p)*s + 203.02734375*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad485(double a, double b, double p, double d, double s, double g){
	return (0.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 3.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 10.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 8.75*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 1.75*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g + 1.75*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 24.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 73.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 61.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 12.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s + 18.375*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 91.875*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 110.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 36.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 2.625*(d*d*d*d*d)*a*(b*b*b*b)*g*p + 91.875*(d*d*d*d)*(a*a*a*a*a)*p*s - 459.375*(d*d*d*d)*(a*a*a*a)*b*p*s + 551.25*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 183.75*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 13.125*(d*d*d*d)*a*(b*b*b*b)*p*s + 165.375*(d*d*d)*(a*a*a)*g*(p*p) - 330.75*(d*d*d)*(a*a)*b*g*(p*p) + 141.75*(d*d*d)*a*(b*b)*g*(p*p) - 11.8125*(d*d*d)*(b*b*b)*g*(p*p) + 496.125*(d*d)*(a*a*a)*(p*p)*s - 992.25*(d*d)*(a*a)*b*(p*p)*s + 425.25*(d*d)*a*(b*b)*(p*p)*s - 35.4375*(d*d)*(b*b*b)*(p*p)*s + 216.5625*d*a*g*(p*p*p) - 108.28125*d*b*g*(p*p*p) + 216.5625*a*(p*p*p)*s - 108.28125*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad486(double a, double b, double p, double d, double s, double g){
	return (0.4375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 3.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 6.5625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 3.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 0.4375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g + 2.625*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 21.0 *(d*d*d*d*d)*(a*a*a*a*a)*b*s + 39.375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 21.0 *(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 2.625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s + 15.3125*(d*d*d*d)*(a*a*a*a)*g*p - 49.0*(d*d*d*d)*(a*a*a)*b*g*p + 36.75*(d*d*d*d)*(a*a)*(b*b)*g*p - 7.0*(d*d*d*d)*a*(b*b*b)*g*p + 0.21875*(d*d*d*d)*(b*b*b*b)*g*p + 61.25*(d*d*d)*(a*a*a*a)*p*s - 196.0*(d*d*d)*(a*a*a)*b*p*s + 147.0*(d*d*d)*(a*a)*(b*b)*p*s - 28.0*(d*d*d)*a*(b*b*b)*p*s + 0.875*(d*d*d)*(b*b*b*b)*p*s + 68.90625*(d*d)*(a*a)*g*(p*p) - 78.75*(d*d)*a*b*g*(p*p) + 14.765625*(d*d)*(b*b)*g*(p*p) + 137.8125*d*(a*a)*(p*p)*s - 157.5*d*a*b*(p*p)*s + 29.53125*d*(b*b)*(p*p)*s + 27.0703125*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad487(double a, double b, double p, double d, double s, double g){
	return (0.4375*(d*d*d*d*d)*(a*a*a*a*a)*g - 2.1875*(d*d*d*d*d)*(a*a*a*a)*b*g + 2.625*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 0.875*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.0625*(d*d*d*d*d)*a*(b*b*b*b)*g + 2.1875*(d*d*d*d)*(a*a*a*a*a)*s - 10.9375*(d*d*d*d)*(a*a*a*a)*b*s + 13.125*(d*d*d*d)*(a*a*a)*(b*b)*s - 4.375*(d*d*d*d)*(a*a)*(b*b*b)*s + 0.3125*(d*d*d*d)*a*(b*b*b*b)*s + 7.875*(d*d*d)*(a*a*a)*g*p - 15.75*(d*d*d)*(a*a)*b*g*p + 6.75*(d*d*d)*a*(b*b)*g*p - 0.5625*(d*d*d)*(b*b*b)*g*p + 23.625*(d*d)*(a*a*a)*p*s - 47.25*(d*d)*(a*a)*b*p*s + 20.25*(d*d)*a*(b*b)*p*s - 1.6875*(d*d)*(b*b*b)*p*s + 15.46875*d*a*g*(p*p) - 7.734375*d*b*g*(p*p) + 15.46875*a*(p*p)*s - 7.734375*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad488(double a, double b, double p, double d, double s, double g){
	return (0.2734375*(d*d*d*d)*(a*a*a*a)*g - 0.875*(d*d*d*d)*(a*a*a)*b*g + 0.65625*(d*d*d*d)*(a*a)*(b*b)*g - 0.125*(d*d*d*d)*a*(b*b*b)*g + 0.00390625*(d*d*d*d)*(b*b*b*b)*g + 1.09375*(d*d*d)*(a*a*a*a)*s - 3.5*(d*d*d)*(a*a*a)*b*s + 2.625*(d*d*d)*(a*a)*(b*b)*s - 0.5*(d*d*d)*a*(b*b*b)*s + 0.015625*(d*d*d)*(b*b*b*b)*s + 2.4609375*(d*d)*(a*a)*g*p - 2.8125*(d*d)*a*b*g*p + 0.52734375*(d*d)*(b*b)*g*p + 4.921875*d*(a*a)*p*s - 5.625*d*a*b*p*s + 1.0546875*d*(b*b)*p*s + 1.4501953125*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad489(double a, double b, double p, double d, double s, double g){
	return (0.109375*(d*d*d)*(a*a*a)*g - 0.21875*(d*d*d)*(a*a)*b*g + 0.09375*(d*d*d)*a*(b*b)*g - 0.0078125*(d*d*d)*(b*b*b)*g + 0.328125*(d*d)*(a*a*a)*s - 0.65625*(d*d)*(a*a)*b*s + 0.28125*(d*d)*a*(b*b)*s - 0.0234375*(d*d)*(b*b*b)*s + 0.4296875*d*a*g*p - 0.21484375*d*b*g*p + 0.4296875*a*p*s - 0.21484375*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad4810(double a, double b, double p, double d, double s, double g){
	return (0.02734375*(d*d)*(a*a)*g - 0.03125*(d*d)*a*b*g + 0.005859375*(d*d)*(b*b)*g + 0.0546875*d*(a*a)*s - 0.0625*d*a*b*s + 0.01171875*d*(b*b)*s + 0.0322265625*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad4811(double a, double b, double p, double d, double s, double g){
	return (0.00390625*d*a*g - 0.001953125*d*b*g + 0.00390625*a*s - 0.001953125*b*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad4812(double a, double b, double p, double d, double s, double g){
	return 0.000244140625*g/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad500(double a, double b, double p, double d, double s, double g){
	return b*(-(d*d*d*d*d)*(b*b*b*b)*g - 5.0*(d*d*d*d)*(b*b*b*b)*s - 5.0*(d*d*d)*(b*b)*g*p - 15.0*(d*d)*(b*b)*p*s - 3.75*d*g*(p*p) - 3.75*(p*p)*s)/(p*p*p*p*p);
}

inline double MD_Et_grad501(double a, double b, double p, double d, double s, double g){
	return (2.5*(d*d*d*d)*(b*b*b*b)*g + 10.0*(d*d*d)*(b*b*b*b)*s + 7.5*(d*d)*(b*b)*g*p + 15.0*d*(b*b)*p*s + 1.875*g*(p*p))/(p*p*p*p*p);
}

inline double MD_Et_grad502(double a, double b, double p, double d, double s, double g){
	return b*(-2.5*(d*d*d)*(b*b)*g - 7.5*(d*d)*(b*b)*s - 3.75*d*g*p - 3.75*p*s)/(p*p*p*p*p);
}

inline double MD_Et_grad503(double a, double b, double p, double d, double s, double g){
	return (1.25*d*(b*b)*(d*g + 2*s) + 0.625*g*p)/(p*p*p*p*p);
}

inline double MD_Et_grad504(double a, double b, double p, double d, double s, double g){
	return 0.3125*b*(-d*g - s)/(p*p*p*p*p);
}

inline double MD_Et_grad505(double a, double b, double p, double d, double s, double g){
	return 0.03125*g/(p*p*p*p*p);
}

inline double MD_Et_grad510(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d)*a*(b*b*b*b*b)*g - 6.0*(d*d*d*d*d)*a*(b*b*b*b*b)*s - 5.0*(d*d*d*d)*a*(b*b*b)*g*p + 2.5*(d*d*d*d)*(b*b*b*b)*g*p - 20.0*(d*d*d)*a*(b*b*b)*p*s + 10.0*(d*d*d)*(b*b*b*b)*p*s - 3.75*(d*d)*a*b*g*(p*p) + 7.5*(d*d)*(b*b)*g*(p*p) - 7.5*d*a*b*(p*p)*s + 15.0*d*(b*b)*(p*p)*s + 1.875*g*(p*p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad511(double a, double b, double p, double d, double s, double g){
	return (2.5*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.5*(d*d*d*d*d)*(b*b*b*b*b)*g + 12.5*(d*d*d*d)*a*(b*b*b*b)*s - 2.5*(d*d*d*d)*(b*b*b*b*b)*s + 7.5*(d*d*d)*a*(b*b)*g*p - 7.5*(d*d*d)*(b*b*b)*g*p + 22.5*(d*d)*a*(b*b)*p*s - 22.5*(d*d)*(b*b*b)*p*s + 1.875*d*a*g*(p*p) - 9.375*d*b*g*(p*p) + 1.875*a*(p*p)*s - 9.375*b*(p*p)*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad512(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d)*a*(b*b*b)*g + 1.25*(d*d*d*d)*(b*b*b*b)*g - 10.0*(d*d*d)*a*(b*b*b)*s + 5.0*(d*d*d)*(b*b*b*b)*s - 3.75*(d*d)*a*b*g*p + 7.5*(d*d)*(b*b)*g*p - 7.5*d*a*b*p*s + 15.0*d*(b*b)*p*s + 2.8125*g*(p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad513(double a, double b, double p, double d, double s, double g){
	return (1.25*(d*d*d)*a*(b*b)*g - 1.25*(d*d*d)*(b*b*b)*g + 3.75*(d*d)*a*(b*b)*s - 3.75*(d*d)*(b*b*b)*s + 0.625*d*a*g*p - 3.125*d*b*g*p + 0.625*a*p*s - 3.125*b*p*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad514(double a, double b, double p, double d, double s, double g){
	return (-0.3125*(d*d)*a*b*g + 0.625*(d*d)*(b*b)*g - 0.625*d*a*b*s + 1.25*d*(b*b)*s + 0.46875*g*p)/(p*p*p*p*p*p);
}

inline double MD_Et_grad515(double a, double b, double p, double d, double s, double g){
	return (0.03125*d*a*g - 0.15625*d*b*g + 0.03125*a*s - 0.15625*b*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad516(double a, double b, double p, double d, double s, double g){
	return 0.015625*g/(p*p*p*p*p*p);
}

inline double MD_Et_grad520(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g - 7.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s - 5.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 5.0*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 0.5*(d*d*d*d*d)*(b*b*b*b*b)*g*p - 25.0*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 25.0*(d*d*d*d)*a*(b*b*b*b)*p*s - 2.5*(d*d*d*d)*(b*b*b*b*b)*p*s - 3.75*(d*d*d)*(a*a)*b*g*(p*p) + 15.0*(d*d*d)*a*(b*b)*g*(p*p) - 7.5*(d*d*d)*(b*b*b)*g*(p*p) - 11.25*(d*d)*(a*a)*b*(p*p)*s + 45.0*(d*d)*a*(b*b)*(p*p)*s - 22.5*(d*d)*(b*b*b)*(p*p)*s + 3.75*d*a*g*(p*p*p) - 9.375*d*b*g*(p*p*p) + 3.75*a*(p*p*p)*s - 9.375*b*(p*p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad521(double a, double b, double p, double d, double s, double g){
	return (2.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - (d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 15.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 6.0*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 7.5*(d*d*d*d)*(a*a)*(b*b)*g*p - 15.0*(d*d*d*d)*a*(b*b*b)*g*p + 3.75*(d*d*d*d)*(b*b*b*b)*g*p + 30.0*(d*d*d)*(a*a)*(b*b)*p*s - 60.0*(d*d*d)*a*(b*b*b)*p*s + 15.0*(d*d*d)*(b*b*b*b)*p*s + 1.875*(d*d)*(a*a)*g*(p*p) - 18.75*(d*d)*a*b*g*(p*p) + 18.75*(d*d)*(b*b)*g*(p*p) + 3.75*d*(a*a)*(p*p)*s - 37.5*d*a*b*(p*p)*s + 37.5*d*(b*b)*(p*p)*s + 6.5625*g*(p*p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad522(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 2.5*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.25*(d*d*d*d*d)*(b*b*b*b*b)*g - 12.5*(d*d*d*d)*(a*a)*(b*b*b)*s + 12.5*(d*d*d*d)*a*(b*b*b*b)*s - 1.25*(d*d*d*d)*(b*b*b*b*b)*s - 3.75*(d*d*d)*(a*a)*b*g*p + 15.0*(d*d*d)*a*(b*b)*g*p - 7.5*(d*d*d)*(b*b*b)*g*p - 11.25*(d*d)*(a*a)*b*p*s + 45.0*(d*d)*a*(b*b)*p*s - 22.5*(d*d)*(b*b*b)*p*s + 5.625*d*a*g*(p*p) - 14.0625*d*b*g*(p*p) + 5.625*a*(p*p)*s - 14.0625*b*(p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad523(double a, double b, double p, double d, double s, double g){
	return (1.25*(d*d*d*d)*(a*a)*(b*b)*g - 2.5*(d*d*d*d)*a*(b*b*b)*g + 0.625*(d*d*d*d)*(b*b*b*b)*g + 5.0*(d*d*d)*(a*a)*(b*b)*s - 10.0*(d*d*d)*a*(b*b*b)*s + 2.5*(d*d*d)*(b*b*b*b)*s + 0.625*(d*d)*(a*a)*g*p - 6.25*(d*d)*a*b*g*p + 6.25*(d*d)*(b*b)*g*p + 1.25*d*(a*a)*p*s - 12.5*d*a*b*p*s + 12.5*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad524(double a, double b, double p, double d, double s, double g){
	return (-0.3125*(d*d*d)*(a*a)*b*g + 1.25*(d*d*d)*a*(b*b)*g - 0.625*(d*d*d)*(b*b*b)*g - 0.9375*(d*d)*(a*a)*b*s + 3.75*(d*d)*a*(b*b)*s - 1.875*(d*d)*(b*b*b)*s + 0.9375*d*a*g*p - 2.34375*d*b*g*p + 0.9375*a*p*s - 2.34375*b*p*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad525(double a, double b, double p, double d, double s, double g){
	return (0.03125*(d*d)*(a*a)*g - 0.3125*(d*d)*a*b*g + 0.3125*(d*d)*(b*b)*g + 0.0625*d*(a*a)*s - 0.625*d*a*b*s + 0.625*d*(b*b)*s + 0.328125*g*p)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad526(double a, double b, double p, double d, double s, double g){
	return (0.03125*d*a*g - 0.078125*d*b*g + 0.03125*a*s - 0.078125*b*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad527(double a, double b, double p, double d, double s, double g){
	return 0.0078125*g/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad530(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g - 8.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s - 5.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 7.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 1.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p - 30.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 45.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 9.0*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s - 3.75*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 22.5*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 22.5*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 3.75*(d*d*d*d)*(b*b*b*b)*g*(p*p) - 15.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 90.0*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 90.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 15.0*(d*d*d)*(b*b*b*b)*(p*p)*s + 5.625*(d*d)*(a*a)*g*(p*p*p) - 28.125*(d*d)*a*b*g*(p*p*p) + 18.75*(d*d)*(b*b)*g*(p*p*p) + 11.25*d*(a*a)*(p*p*p)*s - 56.25*d*a*b*(p*p*p)*s + 37.5*d*(b*b)*(p*p*p)*s + 6.5625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad531(double a, double b, double p, double d, double s, double g){
	return (2.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 1.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 17.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 10.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 7.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 22.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 11.25*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 0.75*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 37.5*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 112.5*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 56.25*(d*d*d*d)*a*(b*b*b*b)*p*s - 3.75*(d*d*d*d)*(b*b*b*b*b)*p*s + 1.875*(d*d*d)*(a*a*a)*g*(p*p) - 28.125*(d*d*d)*(a*a)*b*g*(p*p) + 56.25*(d*d*d)*a*(b*b)*g*(p*p) - 18.75*(d*d*d)*(b*b*b)*g*(p*p) + 5.625*(d*d)*(a*a*a)*(p*p)*s - 84.375*(d*d)*(a*a)*b*(p*p)*s + 168.75*(d*d)*a*(b*b)*(p*p)*s - 56.25*(d*d)*(b*b*b)*(p*p)*s + 19.6875*d*a*g*(p*p*p) - 32.8125*d*b*g*(p*p*p) + 19.6875*a*(p*p*p)*s - 32.8125*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad532(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 3.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.75*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g - 15.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 22.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 4.5*(d*d*d*d*d)*a*(b*b*b*b*b)*s - 3.75*(d*d*d*d)*(a*a*a)*b*g*p + 22.5*(d*d*d*d)*(a*a)*(b*b)*g*p - 22.5*(d*d*d*d)*a*(b*b*b)*g*p + 3.75*(d*d*d*d)*(b*b*b*b)*g*p - 15.0*(d*d*d)*(a*a*a)*b*p*s + 90.0*(d*d*d)*(a*a)*(b*b)*p*s - 90.0*(d*d*d)*a*(b*b*b)*p*s + 15.0*(d*d*d)*(b*b*b*b)*p*s + 8.4375*(d*d)*(a*a)*g*(p*p) - 42.1875*(d*d)*a*b*g*(p*p) + 28.125*(d*d)*(b*b)*g*(p*p) + 16.875*d*(a*a)*(p*p)*s - 84.375*d*a*b*(p*p)*s + 56.25*d*(b*b)*(p*p)*s + 13.125*g*(p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad533(double a, double b, double p, double d, double s, double g){
	return (1.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 3.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 1.875*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.125*(d*d*d*d*d)*(b*b*b*b*b)*g + 6.25*(d*d*d*d)*(a*a*a)*(b*b)*s - 18.75*(d*d*d*d)*(a*a)*(b*b*b)*s + 9.375*(d*d*d*d)*a*(b*b*b*b)*s - 0.625*(d*d*d*d)*(b*b*b*b*b)*s + 0.625*(d*d*d)*(a*a*a)*g*p - 9.375*(d*d*d)*(a*a)*b*g*p + 18.75*(d*d*d)*a*(b*b)*g*p - 6.25*(d*d*d)*(b*b*b)*g*p + 1.875*(d*d)*(a*a*a)*p*s - 28.125*(d*d)*(a*a)*b*p*s + 56.25*(d*d)*a*(b*b)*p*s - 18.75*(d*d)*(b*b*b)*p*s + 9.84375*d*a*g*(p*p) - 16.40625*d*b*g*(p*p) + 9.84375*a*(p*p)*s - 16.40625*b*(p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad534(double a, double b, double p, double d, double s, double g){
	return (-0.3125*(d*d*d*d)*(a*a*a)*b*g + 1.875*(d*d*d*d)*(a*a)*(b*b)*g - 1.875*(d*d*d*d)*a*(b*b*b)*g + 0.3125*(d*d*d*d)*(b*b*b*b)*g - 1.25*(d*d*d)*(a*a*a)*b*s + 7.5*(d*d*d)*(a*a)*(b*b)*s - 7.5*(d*d*d)*a*(b*b*b)*s + 1.25*(d*d*d)*(b*b*b*b)*s + 1.40625*(d*d)*(a*a)*g*p - 7.03125*(d*d)*a*b*g*p + 4.6875*(d*d)*(b*b)*g*p + 2.8125*d*(a*a)*p*s - 14.0625*d*a*b*p*s + 9.375*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad535(double a, double b, double p, double d, double s, double g){
	return (0.03125*(d*d*d)*(a*a*a)*g - 0.46875*(d*d*d)*(a*a)*b*g + 0.9375*(d*d*d)*a*(b*b)*g - 0.3125*(d*d*d)*(b*b*b)*g + 0.09375*(d*d)*(a*a*a)*s - 1.40625*(d*d)*(a*a)*b*s + 2.8125*(d*d)*a*(b*b)*s - 0.9375*(d*d)*(b*b*b)*s + 0.984375*d*a*g*p - 1.640625*d*b*g*p + 0.984375*a*p*s - 1.640625*b*p*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad536(double a, double b, double p, double d, double s, double g){
	return (0.046875*(d*d)*(a*a)*g - 0.234375*(d*d)*a*b*g + 0.15625*(d*d)*(b*b)*g + 0.09375*d*(a*a)*s - 0.46875*d*a*b*s + 0.3125*d*(b*b)*s + 0.21875*g*p)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad537(double a, double b, double p, double d, double s, double g){
	return (0.0234375*d*a*g - 0.0390625*d*b*g + 0.0234375*a*s - 0.0390625*b*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad538(double a, double b, double p, double d, double s, double g){
	return 0.00390625*g/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad540(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g - 9.0*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s - 5.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 10.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 3.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p - 35.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 70.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 21.0 *(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s - 3.75*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 30.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 45.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 15.0*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 0.75*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) - 18.75*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 150.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 225.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 75.0*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 3.75*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 7.5*(d*d*d)*(a*a*a)*g*(p*p*p) - 56.25*(d*d*d)*(a*a)*b*g*(p*p*p) + 75.0*(d*d*d)*a*(b*b)*g*(p*p*p) - 18.75*(d*d*d)*(b*b*b)*g*(p*p*p) + 22.5*(d*d)*(a*a*a)*(p*p*p)*s - 168.75*(d*d)*(a*a)*b*(p*p*p)*s + 225.0*(d*d)*a*(b*b)*(p*p*p)*s - 56.25*(d*d)*(b*b*b)*(p*p*p)*s + 26.25*d*a*g*(p*p*p*p) - 32.8125*d*b*g*(p*p*p*p) + 26.25*a*(p*p*p*p)*s - 32.8125*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad541(double a, double b, double p, double d, double s, double g){
	return (2.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 2.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 20.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 16.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 30.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 22.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 3.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 45.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 180.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 135.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 18.0*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 1.875*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 37.5*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 112.5*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 75.0*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 9.375*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 7.5*(d*d*d)*(a*a*a*a)*(p*p)*s - 150.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 450.0*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 300.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 37.5*(d*d*d)*(b*b*b*b)*(p*p)*s + 39.375*(d*d)*(a*a)*g*(p*p*p) - 131.25*(d*d)*a*b*g*(p*p*p) + 65.625*(d*d)*(b*b)*g*(p*p*p) + 78.75*d*(a*a)*(p*p*p)*s - 262.5*d*a*b*(p*p*p)*s + 131.25*d*(b*b)*(p*p*p)*s + 29.53125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad542(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 5.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 1.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g - 17.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 35.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 10.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s - 3.75*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 30.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 45.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 15.0*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 0.75*(d*d*d*d*d)*(b*b*b*b*b)*g*p - 18.75*(d*d*d*d)*(a*a*a*a)*b*p*s + 150.0*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 225.0*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 75.0*(d*d*d*d)*a*(b*b*b*b)*p*s - 3.75*(d*d*d*d)*(b*b*b*b*b)*p*s + 11.25*(d*d*d)*(a*a*a)*g*(p*p) - 84.375*(d*d*d)*(a*a)*b*g*(p*p) + 112.5*(d*d*d)*a*(b*b)*g*(p*p) - 28.125*(d*d*d)*(b*b*b)*g*(p*p) + 33.75*(d*d)*(a*a*a)*(p*p)*s - 253.125*(d*d)*(a*a)*b*(p*p)*s + 337.5*(d*d)*a*(b*b)*(p*p)*s - 84.375*(d*d)*(b*b*b)*(p*p)*s + 52.5*d*a*g*(p*p*p) - 65.625*d*b*g*(p*p*p) + 52.5*a*(p*p*p)*s - 65.625*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad543(double a, double b, double p, double d, double s, double g){
	return (1.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 5.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 3.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 7.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 30.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 22.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 3.0*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.625*(d*d*d*d)*(a*a*a*a)*g*p - 12.5*(d*d*d*d)*(a*a*a)*b*g*p + 37.5*(d*d*d*d)*(a*a)*(b*b)*g*p - 25.0*(d*d*d*d)*a*(b*b*b)*g*p + 3.125*(d*d*d*d)*(b*b*b*b)*g*p + 2.5*(d*d*d)*(a*a*a*a)*p*s - 50.0*(d*d*d)*(a*a*a)*b*p*s + 150.0*(d*d*d)*(a*a)*(b*b)*p*s - 100.0*(d*d*d)*a*(b*b*b)*p*s + 12.5*(d*d*d)*(b*b*b*b)*p*s + 19.6875*(d*d)*(a*a)*g*(p*p) - 65.625*(d*d)*a*b*g*(p*p) + 32.8125*(d*d)*(b*b)*g*(p*p) + 39.375*d*(a*a)*(p*p)*s - 131.25*d*a*b*(p*p)*s + 65.625*d*(b*b)*(p*p)*s + 19.6875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad544(double a, double b, double p, double d, double s, double g){
	return (-0.3125*(d*d*d*d*d)*(a*a*a*a)*b*g + 2.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 3.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 1.25*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.0625*(d*d*d*d*d)*(b*b*b*b*b)*g - 1.5625*(d*d*d*d)*(a*a*a*a)*b*s + 12.5*(d*d*d*d)*(a*a*a)*(b*b)*s - 18.75*(d*d*d*d)*(a*a)*(b*b*b)*s + 6.25*(d*d*d*d)*a*(b*b*b*b)*s - 0.3125*(d*d*d*d)*(b*b*b*b*b)*s + 1.875*(d*d*d)*(a*a*a)*g*p - 14.0625*(d*d*d)*(a*a)*b*g*p + 18.75*(d*d*d)*a*(b*b)*g*p - 4.6875*(d*d*d)*(b*b*b)*g*p + 5.625*(d*d)*(a*a*a)*p*s - 42.1875*(d*d)*(a*a)*b*p*s + 56.25*(d*d)*a*(b*b)*p*s - 14.0625*(d*d)*(b*b*b)*p*s + 13.125*d*a*g*(p*p) - 16.40625*d*b*g*(p*p) + 13.125*a*(p*p)*s - 16.40625*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad545(double a, double b, double p, double d, double s, double g){
	return (0.03125*(d*d*d*d)*(a*a*a*a)*g - 0.625*(d*d*d*d)*(a*a*a)*b*g + 1.875*(d*d*d*d)*(a*a)*(b*b)*g - 1.25*(d*d*d*d)*a*(b*b*b)*g + 0.15625*(d*d*d*d)*(b*b*b*b)*g + 0.125*(d*d*d)*(a*a*a*a)*s - 2.5*(d*d*d)*(a*a*a)*b*s + 7.5*(d*d*d)*(a*a)*(b*b)*s - 5.0*(d*d*d)*a*(b*b*b)*s + 0.625*(d*d*d)*(b*b*b*b)*s + 1.96875*(d*d)*(a*a)*g*p - 6.5625*(d*d)*a*b*g*p + 3.28125*(d*d)*(b*b)*g*p + 3.9375*d*(a*a)*p*s - 13.125*d*a*b*p*s + 6.5625*d*(b*b)*p*s + 2.953125*g*(p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad546(double a, double b, double p, double d, double s, double g){
	return (0.0625*(d*d*d)*(a*a*a)*g - 0.46875*(d*d*d)*(a*a)*b*g + 0.625*(d*d*d)*a*(b*b)*g - 0.15625*(d*d*d)*(b*b*b)*g + 0.1875*(d*d)*(a*a*a)*s - 1.40625*(d*d)*(a*a)*b*s + 1.875*(d*d)*a*(b*b)*s - 0.46875*(d*d)*(b*b*b)*s + 0.875*d*a*g*p - 1.09375*d*b*g*p + 0.875*a*p*s - 1.09375*b*p*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad547(double a, double b, double p, double d, double s, double g){
	return (0.046875*(d*d)*(a*a)*g - 0.15625*(d*d)*a*b*g + 0.078125*(d*d)*(b*b)*g + 0.09375*d*(a*a)*s - 0.3125*d*a*b*s + 0.15625*d*(b*b)*s + 0.140625*g*p)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad548(double a, double b, double p, double d, double s, double g){
	return (0.015625*d*a*g - 0.01953125*d*b*g + 0.015625*a*s - 0.01953125*b*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad549(double a, double b, double p, double d, double s, double g){
	return 0.001953125*g/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad550(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g - 10.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s - 5.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 12.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 5.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p - 40.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 100.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 40.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s - 3.75*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 37.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 75.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 37.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 3.75*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) - 22.5*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 225.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 450.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 225.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 22.5*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 9.375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 93.75*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 187.5*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 93.75*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 9.375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 37.5*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 375.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 750.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 375.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 37.5*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 65.625*(d*d)*(a*a)*g*(p*p*p*p) - 164.0625*(d*d)*a*b*g*(p*p*p*p) + 65.625*(d*d)*(b*b)*g*(p*p*p*p) + 131.25*d*(a*a)*(p*p*p*p)*s - 328.125*d*a*b*(p*p*p*p)*s + 131.25*d*(b*b)*(p*p*p*p)*s + 29.53125*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad551(double a, double b, double p, double d, double s, double g){
	return (2.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 2.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 22.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 22.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 37.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 37.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 7.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 52.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 262.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 262.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 52.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 1.875*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 46.875*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 187.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 187.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 46.875*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 1.875*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 9.375*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 234.375*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 937.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 937.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 234.375*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 9.375*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 65.625*(d*d*d)*(a*a*a)*g*(p*p*p) - 328.125*(d*d*d)*(a*a)*b*g*(p*p*p) + 328.125*(d*d*d)*a*(b*b)*g*(p*p*p) - 65.625*(d*d*d)*(b*b*b)*g*(p*p*p) + 196.875*(d*d)*(a*a*a)*(p*p*p)*s - 984.375*(d*d)*(a*a)*b*(p*p*p)*s + 984.375*(d*d)*a*(b*b)*(p*p*p)*s - 196.875*(d*d)*(b*b*b)*(p*p*p)*s + 147.65625*d*a*g*(p*p*p*p) - 147.65625*d*b*g*(p*p*p*p) + 147.65625*a*(p*p*p*p)*s - 147.65625*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad552(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 6.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 2.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g - 20.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 50.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 20.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s - 3.75*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 37.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 75.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 37.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 3.75*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p - 22.5*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 225.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 450.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 225.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 22.5*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 14.0625*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 140.625*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 281.25*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 140.625*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 14.0625*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 56.25*(d*d*d)*(a*a*a*a)*(p*p)*s - 562.5*(d*d*d)*(a*a*a)*b*(p*p)*s + 1125.0*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 562.5*(d*d*d)*a*(b*b*b)*(p*p)*s + 56.25*(d*d*d)*(b*b*b*b)*(p*p)*s + 131.25*(d*d)*(a*a)*g*(p*p*p) - 328.125*(d*d)*a*b*g*(p*p*p) + 131.25*(d*d)*(b*b)*g*(p*p*p) + 262.5*d*(a*a)*(p*p*p)*s - 656.25*d*a*b*(p*p*p)*s + 262.5*d*(b*b)*(p*p*p)*s + 73.828125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad553(double a, double b, double p, double d, double s, double g){
	return (1.25*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 6.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 6.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 1.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 8.75*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 43.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 43.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 8.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 0.625*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 15.625*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 62.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 62.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 15.625*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 0.625*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 3.125*(d*d*d*d)*(a*a*a*a*a)*p*s - 78.125*(d*d*d*d)*(a*a*a*a)*b*p*s + 312.5*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 312.5*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 78.125*(d*d*d*d)*a*(b*b*b*b)*p*s - 3.125*(d*d*d*d)*(b*b*b*b*b)*p*s + 32.8125*(d*d*d)*(a*a*a)*g*(p*p) - 164.0625*(d*d*d)*(a*a)*b*g*(p*p) + 164.0625*(d*d*d)*a*(b*b)*g*(p*p) - 32.8125*(d*d*d)*(b*b*b)*g*(p*p) + 98.4375*(d*d)*(a*a*a)*(p*p)*s - 492.1875*(d*d)*(a*a)*b*(p*p)*s + 492.1875*(d*d)*a*(b*b)*(p*p)*s - 98.4375*(d*d)*(b*b*b)*(p*p)*s + 98.4375*d*a*g*(p*p*p) - 98.4375*d*b*g*(p*p*p) + 98.4375*a*(p*p*p)*s - 98.4375*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad554(double a, double b, double p, double d, double s, double g){
	return (-0.3125*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 3.125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 6.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 3.125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.3125*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g - 1.875*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 18.75*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 37.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 18.75*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 1.875*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 2.34375*(d*d*d*d)*(a*a*a*a)*g*p - 23.4375*(d*d*d*d)*(a*a*a)*b*g*p + 46.875*(d*d*d*d)*(a*a)*(b*b)*g*p - 23.4375*(d*d*d*d)*a*(b*b*b)*g*p + 2.34375*(d*d*d*d)*(b*b*b*b)*g*p + 9.375*(d*d*d)*(a*a*a*a)*p*s - 93.75*(d*d*d)*(a*a*a)*b*p*s + 187.5*(d*d*d)*(a*a)*(b*b)*p*s - 93.75*(d*d*d)*a*(b*b*b)*p*s + 9.375*(d*d*d)*(b*b*b*b)*p*s + 32.8125*(d*d)*(a*a)*g*(p*p) - 82.03125*(d*d)*a*b*g*(p*p) + 32.8125*(d*d)*(b*b)*g*(p*p) + 65.625*d*(a*a)*(p*p)*s - 164.0625*d*a*b*(p*p)*s + 65.625*d*(b*b)*(p*p)*s + 24.609375*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad555(double a, double b, double p, double d, double s, double g){
	return (0.03125*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.78125*(d*d*d*d*d)*(a*a*a*a)*b*g + 3.125*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 3.125*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.78125*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.03125*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.15625*(d*d*d*d)*(a*a*a*a*a)*s - 3.90625*(d*d*d*d)*(a*a*a*a)*b*s + 15.625*(d*d*d*d)*(a*a*a)*(b*b)*s - 15.625*(d*d*d*d)*(a*a)*(b*b*b)*s + 3.90625*(d*d*d*d)*a*(b*b*b*b)*s - 0.15625*(d*d*d*d)*(b*b*b*b*b)*s + 3.28125*(d*d*d)*(a*a*a)*g*p - 16.40625*(d*d*d)*(a*a)*b*g*p + 16.40625*(d*d*d)*a*(b*b)*g*p - 3.28125*(d*d*d)*(b*b*b)*g*p + 9.84375*(d*d)*(a*a*a)*p*s - 49.21875*(d*d)*(a*a)*b*p*s + 49.21875*(d*d)*a*(b*b)*p*s - 9.84375*(d*d)*(b*b*b)*p*s + 14.765625*d*a*g*(p*p) - 14.765625*d*b*g*(p*p) + 14.765625*a*(p*p)*s - 14.765625*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad556(double a, double b, double p, double d, double s, double g){
	return (0.078125*(d*d*d*d)*(a*a*a*a)*g - 0.78125*(d*d*d*d)*(a*a*a)*b*g + 1.5625*(d*d*d*d)*(a*a)*(b*b)*g - 0.78125*(d*d*d*d)*a*(b*b*b)*g + 0.078125*(d*d*d*d)*(b*b*b*b)*g + 0.3125*(d*d*d)*(a*a*a*a)*s - 3.125*(d*d*d)*(a*a*a)*b*s + 6.25*(d*d*d)*(a*a)*(b*b)*s - 3.125*(d*d*d)*a*(b*b*b)*s + 0.3125*(d*d*d)*(b*b*b*b)*s + 2.1875*(d*d)*(a*a)*g*p - 5.46875*(d*d)*a*b*g*p + 2.1875*(d*d)*(b*b)*g*p + 4.375*d*(a*a)*p*s - 10.9375*d*a*b*p*s + 4.375*d*(b*b)*p*s + 2.4609375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad557(double a, double b, double p, double d, double s, double g){
	return (0.078125*(d*d*d)*(a*a*a)*g - 0.390625*(d*d*d)*(a*a)*b*g + 0.390625*(d*d*d)*a*(b*b)*g - 0.078125*(d*d*d)*(b*b*b)*g + 0.234375*(d*d)*(a*a*a)*s - 1.171875*(d*d)*(a*a)*b*s + 1.171875*(d*d)*a*(b*b)*s - 0.234375*(d*d)*(b*b*b)*s + 0.703125*d*a*g*p - 0.703125*d*b*g*p + 0.703125*a*p*s - 0.703125*b*p*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad558(double a, double b, double p, double d, double s, double g){
	return (0.0390625*(d*d)*(a*a)*g - 0.09765625*(d*d)*a*b*g + 0.0390625*(d*d)*(b*b)*g + 0.078125*d*(a*a)*s - 0.1953125*d*a*b*s + 0.078125*d*(b*b)*s + 0.087890625*g*p)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad559(double a, double b, double p, double d, double s, double g){
	return 0.009765625*(d*a*g - d*b*g + a*s - b*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad5510(double a, double b, double p, double d, double s, double g){
	return 0.0009765625*g/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad560(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g - 11.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*s - 5.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 15.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 7.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p - 45.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 135.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 67.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s - 3.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 45.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 112.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 75.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 11.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) - 26.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 315.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 787.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 525.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 78.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 11.25*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 140.625*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 375.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 281.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 56.25*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 1.875*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 56.25*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 703.125*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 1875.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 1406.25*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 281.25*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 9.375*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 131.25*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 492.1875*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 393.75*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 65.625*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 393.75*(d*d)*(a*a*a)*(p*p*p*p)*s - 1476.5625*(d*d)*(a*a)*b*(p*p*p*p)*s + 1181.25*(d*d)*a*(b*b)*(p*p*p*p)*s - 196.875*(d*d)*(b*b*b)*(p*p*p*p)*s + 177.1875*d*a*g*(p*p*p*p*p) - 147.65625*d*b*g*(p*p*p*p*p) + 177.1875*a*(p*p*p*p*p)*s - 147.65625*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad561(double a, double b, double p, double d, double s, double g){
	return (2.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 3.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 25.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 30.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 45.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 56.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 15.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 60.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 360.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 450.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 120.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 1.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 56.25*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 281.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 375.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 140.625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 11.25*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 11.25*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 337.5*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 1687.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 2250.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 843.75*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 67.5*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 98.4375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 656.25*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 984.375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 393.75*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 32.8125*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 393.75*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 2625.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 3937.5*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 1575.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 131.25*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 442.96875*(d*d)*(a*a)*g*(p*p*p*p) - 885.9375*(d*d)*a*b*g*(p*p*p*p) + 295.3125*(d*d)*(b*b)*g*(p*p*p*p) + 885.9375*d*(a*a)*(p*p*p*p)*s - 1771.875*d*a*b*(p*p*p*p)*s + 590.625*d*(b*b)*(p*p*p*p)*s + 162.421875*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad562(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 7.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 3.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g - 22.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 67.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 33.75*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s - 3.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 45.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 112.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 75.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 11.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p - 26.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 315.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 787.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 525.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 78.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 16.875*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 210.9375*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 562.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 421.875*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 84.375*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 2.8125*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 84.375*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 1054.6875*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 2812.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 2109.375*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 421.875*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 14.0625*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 262.5*(d*d*d)*(a*a*a)*g*(p*p*p) - 984.375*(d*d*d)*(a*a)*b*g*(p*p*p) + 787.5*(d*d*d)*a*(b*b)*g*(p*p*p) - 131.25*(d*d*d)*(b*b*b)*g*(p*p*p) + 787.5*(d*d)*(a*a*a)*(p*p*p)*s - 2953.125*(d*d)*(a*a)*b*(p*p*p)*s + 2362.5*(d*d)*a*(b*b)*(p*p*p)*s - 393.75*(d*d)*(b*b*b)*(p*p*p)*s + 442.96875*d*a*g*(p*p*p*p) - 369.140625*d*b*g*(p*p*p*p) + 442.96875*a*(p*p*p*p)*s - 369.140625*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad563(double a, double b, double p, double d, double s, double g){
	return (1.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 7.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 9.375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 2.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 10.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 60.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 75.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 20.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 0.625*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 18.75*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 93.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 125.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 46.875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 3.75*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 3.75*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 112.5*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 562.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 750.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 281.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 22.5*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 49.21875*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 328.125*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 492.1875*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 196.875*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 16.40625*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 196.875*(d*d*d)*(a*a*a*a)*(p*p)*s - 1312.5*(d*d*d)*(a*a*a)*b*(p*p)*s + 1968.75*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 787.5*(d*d*d)*a*(b*b*b)*(p*p)*s + 65.625*(d*d*d)*(b*b*b*b)*(p*p)*s + 295.3125*(d*d)*(a*a)*g*(p*p*p) - 590.625*(d*d)*a*b*g*(p*p*p) + 196.875*(d*d)*(b*b)*g*(p*p*p) + 590.625*d*(a*a)*(p*p*p)*s - 1181.25*d*a*b*(p*p*p)*s + 393.75*d*(b*b)*(p*p*p)*s + 135.3515625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad564(double a, double b, double p, double d, double s, double g){
	return (-0.3125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 3.75*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 9.375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 6.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 0.9375*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g - 2.1875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 26.25*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 65.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 43.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 6.5625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 2.8125*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 35.15625*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 93.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 70.3125*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 14.0625*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 0.46875*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 14.0625*(d*d*d*d)*(a*a*a*a*a)*p*s - 175.78125*(d*d*d*d)*(a*a*a*a)*b*p*s + 468.75*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 351.5625*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 70.3125*(d*d*d*d)*a*(b*b*b*b)*p*s - 2.34375*(d*d*d*d)*(b*b*b*b*b)*p*s + 65.625*(d*d*d)*(a*a*a)*g*(p*p) - 246.09375*(d*d*d)*(a*a)*b*g*(p*p) + 196.875*(d*d*d)*a*(b*b)*g*(p*p) - 32.8125*(d*d*d)*(b*b*b)*g*(p*p) + 196.875*(d*d)*(a*a*a)*(p*p)*s - 738.28125*(d*d)*(a*a)*b*(p*p)*s + 590.625*(d*d)*a*(b*b)*(p*p)*s - 98.4375*(d*d)*(b*b*b)*(p*p)*s + 147.65625*d*a*g*(p*p*p) - 123.046875*d*b*g*(p*p*p) + 147.65625*a*(p*p*p)*s - 123.046875*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad565(double a, double b, double p, double d, double s, double g){
	return (0.03125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 0.9375*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 4.6875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 6.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 2.34375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.1875*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.1875*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 5.625*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 28.125*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 37.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 14.0625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 1.125*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 4.921875*(d*d*d*d)*(a*a*a*a)*g*p - 32.8125*(d*d*d*d)*(a*a*a)*b*g*p + 49.21875*(d*d*d*d)*(a*a)*(b*b)*g*p - 19.6875*(d*d*d*d)*a*(b*b*b)*g*p + 1.640625*(d*d*d*d)*(b*b*b*b)*g*p + 19.6875*(d*d*d)*(a*a*a*a)*p*s - 131.25*(d*d*d)*(a*a*a)*b*p*s + 196.875*(d*d*d)*(a*a)*(b*b)*p*s - 78.75*(d*d*d)*a*(b*b*b)*p*s + 6.5625*(d*d*d)*(b*b*b*b)*p*s + 44.296875*(d*d)*(a*a)*g*(p*p) - 88.59375*(d*d)*a*b*g*(p*p) + 29.53125*(d*d)*(b*b)*g*(p*p) + 88.59375*d*(a*a)*(p*p)*s - 177.1875*d*a*b*(p*p)*s + 59.0625*d*(b*b)*(p*p)*s + 27.0703125*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad566(double a, double b, double p, double d, double s, double g){
	return (0.09375*(d*d*d*d*d)*(a*a*a*a*a)*g - 1.171875*(d*d*d*d*d)*(a*a*a*a)*b*g + 3.125*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 2.34375*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.46875*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.015625*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.46875*(d*d*d*d)*(a*a*a*a*a)*s - 5.859375*(d*d*d*d)*(a*a*a*a)*b*s + 15.625*(d*d*d*d)*(a*a*a)*(b*b)*s - 11.71875*(d*d*d*d)*(a*a)*(b*b*b)*s + 2.34375*(d*d*d*d)*a*(b*b*b*b)*s - 0.078125*(d*d*d*d)*(b*b*b*b*b)*s + 4.375*(d*d*d)*(a*a*a)*g*p - 16.40625*(d*d*d)*(a*a)*b*g*p + 13.125*(d*d*d)*a*(b*b)*g*p - 2.1875*(d*d*d)*(b*b*b)*g*p + 13.125*(d*d)*(a*a*a)*p*s - 49.21875*(d*d)*(a*a)*b*p*s + 39.375*(d*d)*a*(b*b)*p*s - 6.5625*(d*d)*(b*b*b)*p*s + 14.765625*d*a*g*(p*p) - 12.3046875*d*b*g*(p*p) + 14.765625*a*(p*p)*s - 12.3046875*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad567(double a, double b, double p, double d, double s, double g){
	return (0.1171875*(d*d*d*d)*(a*a*a*a)*g - 0.78125*(d*d*d*d)*(a*a*a)*b*g + 1.171875*(d*d*d*d)*(a*a)*(b*b)*g - 0.46875*(d*d*d*d)*a*(b*b*b)*g + 0.0390625*(d*d*d*d)*(b*b*b*b)*g + 0.46875*(d*d*d)*(a*a*a*a)*s - 3.125*(d*d*d)*(a*a*a)*b*s + 4.6875*(d*d*d)*(a*a)*(b*b)*s - 1.875*(d*d*d)*a*(b*b*b)*s + 0.15625*(d*d*d)*(b*b*b*b)*s + 2.109375*(d*d)*(a*a)*g*p - 4.21875*(d*d)*a*b*g*p + 1.40625*(d*d)*(b*b)*g*p + 4.21875*d*(a*a)*p*s - 8.4375*d*a*b*p*s + 2.8125*d*(b*b)*p*s + 1.93359375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad568(double a, double b, double p, double d, double s, double g){
	return (0.078125*(d*d*d)*(a*a*a)*g - 0.29296875*(d*d*d)*(a*a)*b*g + 0.234375*(d*d*d)*a*(b*b)*g - 0.0390625*(d*d*d)*(b*b*b)*g + 0.234375*(d*d)*(a*a*a)*s - 0.87890625*(d*d)*(a*a)*b*s + 0.703125*(d*d)*a*(b*b)*s - 0.1171875*(d*d)*(b*b*b)*s + 0.52734375*d*a*g*p - 0.439453125*d*b*g*p + 0.52734375*a*p*s - 0.439453125*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad569(double a, double b, double p, double d, double s, double g){
	return (0.029296875*(d*d)*(a*a)*g - 0.05859375*(d*d)*a*b*g + 0.01953125*(d*d)*(b*b)*g + 0.05859375*d*(a*a)*s - 0.1171875*d*a*b*s + 0.0390625*d*(b*b)*s + 0.0537109375*g*p)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad5610(double a, double b, double p, double d, double s, double g){
	return (0.005859375*d*a*g - 0.0048828125*d*b*g + 0.005859375*a*s - 0.0048828125*b*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad5611(double a, double b, double p, double d, double s, double g){
	return 0.00048828125*g/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad570(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g - 12.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*s - 5.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*p + 17.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*p - 10.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p - 50.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*p*s + 175.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*p*s - 105.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s - 3.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p) + 52.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p) - 157.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 131.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 26.25*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) - 30.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p)*s + 420.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p)*s - 1260.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 1050.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 210.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 13.125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p) - 196.875*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 656.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 656.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 196.875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 13.125*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 78.75*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p)*s - 1181.25*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 3937.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 3937.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 1181.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 78.75*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 229.6875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 1148.4375*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 1378.125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 459.375*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 32.8125*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 918.75*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 4593.75*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 5512.5*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 1837.5*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 131.25*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 620.15625*(d*d)*(a*a)*g*(p*p*p*p*p) - 1033.59375*(d*d)*a*b*g*(p*p*p*p*p) + 295.3125*(d*d)*(b*b)*g*(p*p*p*p*p) + 1240.3125*d*(a*a)*(p*p*p*p*p)*s - 2067.1875*d*a*b*(p*p*p*p*p)*s + 590.625*d*(b*b)*(p*p*p*p*p)*s + 162.421875*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad571(double a, double b, double p, double d, double s, double g){
	return (2.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g - 3.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g + 27.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*s - 38.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*p - 52.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 78.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 26.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 67.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*p*s - 472.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 708.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 236.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 1.875*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p) - 65.625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 393.75*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 656.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 328.125*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 39.375*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 13.125*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p)*s - 459.375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 2756.25*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 4593.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 2296.875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 275.625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 137.8125*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 1148.4375*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 2296.875*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 1378.125*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 229.6875*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 6.5625*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 689.0625*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 5742.1875*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 11484.375*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 6890.625*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 1148.4375*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 32.8125*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 1033.59375*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 3100.78125*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 2067.1875*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 295.3125*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 3100.78125*(d*d)*(a*a*a)*(p*p*p*p)*s - 9302.34375*(d*d)*(a*a)*b*(p*p*p*p)*s + 6201.5625*(d*d)*a*(b*b)*(p*p*p*p)*s - 885.9375*(d*d)*(b*b*b)*(p*p*p*p)*s + 1136.953125*d*a*g*(p*p*p*p*p) - 812.109375*d*b*g*(p*p*p*p*p) + 1136.953125*a*(p*p*p*p*p)*s - 812.109375*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad572(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g + 8.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 5.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g - 25.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*s + 87.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 52.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s - 3.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 52.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 157.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 131.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 26.25*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p - 30.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 420.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 1260.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 1050.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 210.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 19.6875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 295.3125*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 984.375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 984.375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 295.3125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 19.6875*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 118.125*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 1771.875*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 5906.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 5906.25*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 1771.875*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 118.125*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 459.375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 2296.875*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 2756.25*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 918.75*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 65.625*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 1837.5*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 9187.5*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 11025.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 3675.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 262.5*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 1550.390625*(d*d)*(a*a)*g*(p*p*p*p) - 2583.984375*(d*d)*a*b*g*(p*p*p*p) + 738.28125*(d*d)*(b*b)*g*(p*p*p*p) + 3100.78125*d*(a*a)*(p*p*p*p)*s - 5167.96875*d*a*b*(p*p*p*p)*s + 1476.5625*d*(b*b)*(p*p*p*p)*s + 487.265625*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad573(double a, double b, double p, double d, double s, double g){
	return (1.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 8.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 13.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 4.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 11.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s - 78.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 118.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 39.375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 0.625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 21.875*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 131.25*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 218.75*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 109.375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 13.125*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 4.375*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 153.125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 918.75*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 1531.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 765.625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 91.875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 68.90625*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 574.21875*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 1148.4375*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 689.0625*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 114.84375*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 3.28125*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 344.53125*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 2871.09375*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 5742.1875*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 3445.3125*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 574.21875*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 16.40625*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 689.0625*(d*d*d)*(a*a*a)*g*(p*p*p) - 2067.1875*(d*d*d)*(a*a)*b*g*(p*p*p) + 1378.125*(d*d*d)*a*(b*b)*g*(p*p*p) - 196.875*(d*d*d)*(b*b*b)*g*(p*p*p) + 2067.1875*(d*d)*(a*a*a)*(p*p*p)*s - 6201.5625*(d*d)*(a*a)*b*(p*p*p)*s + 4134.375*(d*d)*a*(b*b)*(p*p*p)*s - 590.625*(d*d)*(b*b*b)*(p*p*p)*s + 947.4609375*d*a*g*(p*p*p*p) - 676.7578125*d*b*g*(p*p*p*p) + 947.4609375*a*(p*p*p*p)*s - 676.7578125*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad574(double a, double b, double p, double d, double s, double g){
	return (-0.3125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 4.375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 13.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 10.9375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 2.1875*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g - 2.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 35.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 105.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 87.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 17.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 3.28125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 49.21875*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 164.0625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 164.0625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 49.21875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 3.28125*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 19.6875*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 295.3125*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 984.375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 984.375*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 295.3125*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 19.6875*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 114.84375*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 574.21875*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 689.0625*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 229.6875*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 16.40625*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 459.375*(d*d*d)*(a*a*a*a)*(p*p)*s - 2296.875*(d*d*d)*(a*a*a)*b*(p*p)*s + 2756.25*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 918.75*(d*d*d)*a*(b*b*b)*(p*p)*s + 65.625*(d*d*d)*(b*b*b*b)*(p*p)*s + 516.796875*(d*d)*(a*a)*g*(p*p*p) - 861.328125*(d*d)*a*b*g*(p*p*p) + 246.09375*(d*d)*(b*b)*g*(p*p*p) + 1033.59375*d*(a*a)*(p*p*p)*s - 1722.65625*d*a*b*(p*p*p)*s + 492.1875*d*(b*b)*(p*p*p)*s + 203.02734375*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad575(double a, double b, double p, double d, double s, double g){
	return (0.03125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 1.09375*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 6.5625*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 10.9375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 5.46875*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 0.65625*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.21875*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 7.65625*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 45.9375*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 76.5625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 38.28125*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 4.59375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 6.890625*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 57.421875*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 114.84375*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 68.90625*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 11.484375*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 0.328125*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 34.453125*(d*d*d*d)*(a*a*a*a*a)*p*s - 287.109375*(d*d*d*d)*(a*a*a*a)*b*p*s + 574.21875*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 344.53125*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 57.421875*(d*d*d*d)*a*(b*b*b*b)*p*s - 1.640625*(d*d*d*d)*(b*b*b*b*b)*p*s + 103.359375*(d*d*d)*(a*a*a)*g*(p*p) - 310.078125*(d*d*d)*(a*a)*b*g*(p*p) + 206.71875*(d*d*d)*a*(b*b)*g*(p*p) - 29.53125*(d*d*d)*(b*b*b)*g*(p*p) + 310.078125*(d*d)*(a*a*a)*(p*p)*s - 930.234375*(d*d)*(a*a)*b*(p*p)*s + 620.15625*(d*d)*a*(b*b)*(p*p)*s - 88.59375*(d*d)*(b*b*b)*(p*p)*s + 189.4921875*d*a*g*(p*p*p) - 135.3515625*d*b*g*(p*p*p) + 189.4921875*a*(p*p*p)*s - 135.3515625*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad576(double a, double b, double p, double d, double s, double g){
	return (0.109375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 1.640625*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 5.46875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 5.46875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 1.640625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.109375*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.65625*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 9.84375*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 32.8125*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 32.8125*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 9.84375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 0.65625*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 7.65625*(d*d*d*d)*(a*a*a*a)*g*p - 38.28125*(d*d*d*d)*(a*a*a)*b*g*p + 45.9375*(d*d*d*d)*(a*a)*(b*b)*g*p - 15.3125*(d*d*d*d)*a*(b*b*b)*g*p + 1.09375*(d*d*d*d)*(b*b*b*b)*g*p + 30.625*(d*d*d)*(a*a*a*a)*p*s - 153.125*(d*d*d)*(a*a*a)*b*p*s + 183.75*(d*d*d)*(a*a)*(b*b)*p*s - 61.25*(d*d*d)*a*(b*b*b)*p*s + 4.375*(d*d*d)*(b*b*b*b)*p*s + 51.6796875*(d*d)*(a*a)*g*(p*p) - 86.1328125*(d*d)*a*b*g*(p*p) + 24.609375*(d*d)*(b*b)*g*(p*p) + 103.359375*d*(a*a)*(p*p)*s - 172.265625*d*a*b*(p*p)*s + 49.21875*d*(b*b)*(p*p)*s + 27.0703125*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad577(double a, double b, double p, double d, double s, double g){
	return (0.1640625*(d*d*d*d*d)*(a*a*a*a*a)*g - 1.3671875*(d*d*d*d*d)*(a*a*a*a)*b*g + 2.734375*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 1.640625*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.2734375*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.0078125*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.8203125*(d*d*d*d)*(a*a*a*a*a)*s - 6.8359375*(d*d*d*d)*(a*a*a*a)*b*s + 13.671875*(d*d*d*d)*(a*a*a)*(b*b)*s - 8.203125*(d*d*d*d)*(a*a)*(b*b*b)*s + 1.3671875*(d*d*d*d)*a*(b*b*b*b)*s - 0.0390625*(d*d*d*d)*(b*b*b*b*b)*s + 4.921875*(d*d*d)*(a*a*a)*g*p - 14.765625*(d*d*d)*(a*a)*b*g*p + 9.84375*(d*d*d)*a*(b*b)*g*p - 1.40625*(d*d*d)*(b*b*b)*g*p + 14.765625*(d*d)*(a*a*a)*p*s - 44.296875*(d*d)*(a*a)*b*p*s + 29.53125*(d*d)*a*(b*b)*p*s - 4.21875*(d*d)*(b*b*b)*p*s + 13.53515625*d*a*g*(p*p) - 9.66796875*d*b*g*(p*p) + 13.53515625*a*(p*p)*s - 9.66796875*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad578(double a, double b, double p, double d, double s, double g){
	return (0.13671875*(d*d*d*d)*(a*a*a*a)*g - 0.68359375*(d*d*d*d)*(a*a*a)*b*g + 0.8203125*(d*d*d*d)*(a*a)*(b*b)*g - 0.2734375*(d*d*d*d)*a*(b*b*b)*g + 0.01953125*(d*d*d*d)*(b*b*b*b)*g + 0.546875*(d*d*d)*(a*a*a*a)*s - 2.734375*(d*d*d)*(a*a*a)*b*s + 3.28125*(d*d*d)*(a*a)*(b*b)*s - 1.09375*(d*d*d)*a*(b*b*b)*s + 0.078125*(d*d*d)*(b*b*b*b)*s + 1.845703125*(d*d)*(a*a)*g*p - 3.076171875*(d*d)*a*b*g*p + 0.87890625*(d*d)*(b*b)*g*p + 3.69140625*d*(a*a)*p*s - 6.15234375*d*a*b*p*s + 1.7578125*d*(b*b)*p*s + 1.4501953125*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad579(double a, double b, double p, double d, double s, double g){
	return (0.068359375*(d*d*d)*(a*a*a)*g - 0.205078125*(d*d*d)*(a*a)*b*g + 0.13671875*(d*d*d)*a*(b*b)*g - 0.01953125*(d*d*d)*(b*b*b)*g + 0.205078125*(d*d)*(a*a*a)*s - 0.615234375*(d*d)*(a*a)*b*s + 0.41015625*(d*d)*a*(b*b)*s - 0.05859375*(d*d)*(b*b*b)*s + 0.3759765625*d*a*g*p - 0.2685546875*d*b*g*p + 0.3759765625*a*p*s - 0.2685546875*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad5710(double a, double b, double p, double d, double s, double g){
	return (0.0205078125*(d*d)*(a*a)*g - 0.0341796875*(d*d)*a*b*g + 0.009765625*(d*d)*(b*b)*g + 0.041015625*d*(a*a)*s - 0.068359375*d*a*b*s + 0.01953125*d*(b*b)*s + 0.0322265625*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad5711(double a, double b, double p, double d, double s, double g){
	return (0.00341796875*d*a*g - 0.00244140625*d*b*g + 0.00341796875*a*s - 0.00244140625*b*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad5712(double a, double b, double p, double d, double s, double g){
	return 0.000244140625*g/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad580(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*g - 13.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*s - 5.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*g*p + 20.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g*p - 14.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g*p - 55.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*p*s + 220.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*p*s - 154.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*p*s - 3.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g*(p*p) + 60.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*(p*p) - 210.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 210.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 52.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p) - 33.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*(p*p)*s + 540.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*(p*p)*s - 1890.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 1890.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 472.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 15.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p*p) - 262.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p*p) + 1050.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p) - 1312.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p) + 525.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p) - 52.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p) + 105.0*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p*p)*s - 1837.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p*p)*s + 7350.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p)*s - 9187.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p)*s + 3675.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p)*s - 367.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p)*s + 367.5*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p) - 2296.875*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p) + 3675.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p) - 1837.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p) + 262.5*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p) - 6.5625*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p) + 1837.5*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p)*s - 11484.375*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p)*s + 18375.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p)*s - 9187.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p)*s + 1312.5*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p)*s - 32.8125*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p)*s + 1653.75*(d*d*d)*(a*a*a)*g*(p*p*p*p*p) - 4134.375*(d*d*d)*(a*a)*b*g*(p*p*p*p*p) + 2362.5*(d*d*d)*a*(b*b)*g*(p*p*p*p*p) - 295.3125*(d*d*d)*(b*b*b)*g*(p*p*p*p*p) + 4961.25*(d*d)*(a*a*a)*(p*p*p*p*p)*s - 12403.125*(d*d)*(a*a)*b*(p*p*p*p*p)*s + 7087.5*(d*d)*a*(b*b)*(p*p*p*p*p)*s - 885.9375*(d*d)*(b*b*b)*(p*p*p*p*p)*s + 1299.375*d*a*g*(p*p*p*p*p*p) - 812.109375*d*b*g*(p*p*p*p*p*p) + 1299.375*a*(p*p*p*p*p*p)*s - 812.109375*b*(p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad581(double a, double b, double p, double d, double s, double g){
	return (2.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*g - 4.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g + 30.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*s - 48.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g*p - 60.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*p + 105.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*p - 42.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p + 75.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*p*s - 600.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*p*s + 1050.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*p*s - 420.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s + 1.875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*(p*p) - 75.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p) + 525.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p) - 1050.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 656.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 105.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 15.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(p*p)*s - 600.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p)*s + 4200.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p)*s - 8400.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 5250.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 840.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 183.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p) - 1837.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 4593.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 3675.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 918.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 52.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 1102.5*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p)*s - 11025.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 27562.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 22050.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 5512.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 315.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 2067.1875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 8268.75*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 8268.75*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 2362.5*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 147.65625*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 8268.75*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 33075.0*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 33075.0*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 9450.0*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 590.625*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 4547.8125*(d*d)*(a*a)*g*(p*p*p*p*p) - 6496.875*(d*d)*a*b*g*(p*p*p*p*p) + 1624.21875*(d*d)*(b*b)*g*(p*p*p*p*p) + 9095.625*d*(a*a)*(p*p*p*p*p)*s - 12993.75*d*a*b*(p*p*p*p*p)*s + 3248.4375*d*(b*b)*(p*p*p*p*p)*s + 1055.7421875*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad582(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*g + 10.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g - 7.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g - 27.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*s + 110.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*s - 77.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*s - 3.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g*p + 60.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*p - 210.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 210.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 52.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p - 33.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*p*s + 540.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*p*s - 1890.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 1890.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 472.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 22.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p) - 393.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 1575.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 1968.75*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 787.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 78.75*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 157.5*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p)*s - 2756.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 11025.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 13781.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 5512.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 551.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 735.0*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 4593.75*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 7350.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 3675.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 525.0*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 13.125*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 3675.0*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 22968.75*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 36750.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 18375.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 2625.0*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 65.625*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 4134.375*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 10335.9375*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 5906.25*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 738.28125*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 12403.125*(d*d)*(a*a*a)*(p*p*p*p)*s - 31007.8125*(d*d)*(a*a)*b*(p*p*p*p)*s + 17718.75*(d*d)*a*(b*b)*(p*p*p*p)*s - 2214.84375*(d*d)*(b*b*b)*(p*p*p*p)*s + 3898.125*d*a*g*(p*p*p*p*p) - 2436.328125*d*b*g*(p*p*p*p*p) + 3898.125*a*(p*p*p*p*p)*s - 2436.328125*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad583(double a, double b, double p, double d, double s, double g){
	return (1.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g - 10.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g + 17.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 7.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 12.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*s - 100.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*s + 175.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 70.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 0.625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*p - 25.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 175.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 350.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 218.75*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 35.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 5.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*p*s - 200.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 1400.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 2800.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 1750.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 280.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 91.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 918.75*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 2296.875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 1837.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 459.375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 26.25*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 551.25*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 5512.5*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 13781.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 11025.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 2756.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 157.5*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 1378.125*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 5512.5*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 5512.5*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 1575.0*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 98.4375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 5512.5*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 22050.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 22050.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 6300.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 393.75*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 3789.84375*(d*d)*(a*a)*g*(p*p*p*p) - 5414.0625*(d*d)*a*b*g*(p*p*p*p) + 1353.515625*(d*d)*(b*b)*g*(p*p*p*p) + 7579.6875*d*(a*a)*(p*p*p*p)*s - 10828.125*d*a*b*(p*p*p*p)*s + 2707.03125*d*(b*b)*(p*p*p*p)*s + 1055.7421875*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad584(double a, double b, double p, double d, double s, double g){
	return (-0.3125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g + 5.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 17.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 17.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 4.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g - 2.8125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*s + 45.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s - 157.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 157.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 39.375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 3.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 65.625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 262.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 328.125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 131.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 13.125*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 26.25*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 459.375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 1837.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 2296.875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 918.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 91.875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 183.75*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 1148.4375*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 1837.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 918.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 131.25*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 3.28125*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 918.75*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 5742.1875*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 9187.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 4593.75*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 656.25*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 16.40625*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 1378.125*(d*d*d)*(a*a*a)*g*(p*p*p) - 3445.3125*(d*d*d)*(a*a)*b*g*(p*p*p) + 1968.75*(d*d*d)*a*(b*b)*g*(p*p*p) - 246.09375*(d*d*d)*(b*b*b)*g*(p*p*p) + 4134.375*(d*d)*(a*a*a)*(p*p*p)*s - 10335.9375*(d*d)*(a*a)*b*(p*p*p)*s + 5906.25*(d*d)*a*(b*b)*(p*p*p)*s - 738.28125*(d*d)*(b*b*b)*(p*p*p)*s + 1624.21875*d*a*g*(p*p*p*p) - 1015.13671875*d*b*g*(p*p*p*p) + 1624.21875*a*(p*p*p*p)*s - 1015.13671875*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad585(double a, double b, double p, double d, double s, double g){
	return (0.03125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g - 1.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 8.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 17.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 10.9375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 1.75*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 0.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*s - 10.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 70.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 140.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 87.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 14.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 9.1875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 91.875*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 229.6875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 183.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 45.9375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 2.625*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 55.125*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 551.25*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 1378.125*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 1102.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 275.625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 15.75*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 206.71875*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 826.875*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 826.875*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 236.25*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 14.765625*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 826.875*(d*d*d)*(a*a*a*a)*(p*p)*s - 3307.5*(d*d*d)*(a*a*a)*b*(p*p)*s + 3307.5*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 945.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 59.0625*(d*d*d)*(b*b*b*b)*(p*p)*s + 757.96875*(d*d)*(a*a)*g*(p*p*p) - 1082.8125*(d*d)*a*b*g*(p*p*p) + 270.703125*(d*d)*(b*b)*g*(p*p*p) + 1515.9375*d*(a*a)*(p*p*p)*s - 2165.625*d*a*b*(p*p*p)*s + 541.40625*d*(b*b)*(p*p*p)*s + 263.935546875*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad586(double a, double b, double p, double d, double s, double g){
	return (0.125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 2.1875*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 8.75*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 10.9375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 4.375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 0.4375*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.875*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 15.3125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 61.25*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 76.5625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 30.625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 3.0625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 12.25*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 76.5625*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 122.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 61.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 8.75*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 0.21875*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 61.25*(d*d*d*d)*(a*a*a*a*a)*p*s - 382.8125*(d*d*d*d)*(a*a*a*a)*b*p*s + 612.5*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 306.25*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 43.75*(d*d*d*d)*a*(b*b*b*b)*p*s - 1.09375*(d*d*d*d)*(b*b*b*b*b)*p*s + 137.8125*(d*d*d)*(a*a*a)*g*(p*p) - 344.53125*(d*d*d)*(a*a)*b*g*(p*p) + 196.875*(d*d*d)*a*(b*b)*g*(p*p) - 24.609375*(d*d*d)*(b*b*b)*g*(p*p) + 413.4375*(d*d)*(a*a*a)*(p*p)*s - 1033.59375*(d*d)*(a*a)*b*(p*p)*s + 590.625*(d*d)*a*(b*b)*(p*p)*s - 73.828125*(d*d)*(b*b*b)*(p*p)*s + 216.5625*d*a*g*(p*p*p) - 135.3515625*d*b*g*(p*p*p) + 216.5625*a*(p*p*p)*s - 135.3515625*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad587(double a, double b, double p, double d, double s, double g){
	return (0.21875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 2.1875*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 5.46875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 4.375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 1.09375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.0625*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 1.3125*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 13.125*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 32.8125*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 26.25*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 6.5625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 0.375*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 9.84375*(d*d*d*d)*(a*a*a*a)*g*p - 39.375*(d*d*d*d)*(a*a*a)*b*g*p + 39.375*(d*d*d*d)*(a*a)*(b*b)*g*p - 11.25*(d*d*d*d)*a*(b*b*b)*g*p + 0.703125*(d*d*d*d)*(b*b*b*b)*g*p + 39.375*(d*d*d)*(a*a*a*a)*p*s - 157.5*(d*d*d)*(a*a*a)*b*p*s + 157.5*(d*d*d)*(a*a)*(b*b)*p*s - 45.0*(d*d*d)*a*(b*b*b)*p*s + 2.8125*(d*d*d)*(b*b*b*b)*p*s + 54.140625*(d*d)*(a*a)*g*(p*p) - 77.34375*(d*d)*a*b*g*(p*p) + 19.3359375*(d*d)*(b*b)*g*(p*p) + 108.28125*d*(a*a)*(p*p)*s - 154.6875*d*a*b*(p*p)*s + 38.671875*d*(b*b)*(p*p)*s + 25.13671875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad588(double a, double b, double p, double d, double s, double g){
	return (0.21875*(d*d*d*d*d)*(a*a*a*a*a)*g - 1.3671875*(d*d*d*d*d)*(a*a*a*a)*b*g + 2.1875*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 1.09375*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.15625*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.00390625*(d*d*d*d*d)*(b*b*b*b*b)*g + 1.09375*(d*d*d*d)*(a*a*a*a*a)*s - 6.8359375*(d*d*d*d)*(a*a*a*a)*b*s + 10.9375*(d*d*d*d)*(a*a*a)*(b*b)*s - 5.46875*(d*d*d*d)*(a*a)*(b*b*b)*s + 0.78125*(d*d*d*d)*a*(b*b*b*b)*s - 0.01953125*(d*d*d*d)*(b*b*b*b*b)*s + 4.921875*(d*d*d)*(a*a*a)*g*p - 12.3046875*(d*d*d)*(a*a)*b*g*p + 7.03125*(d*d*d)*a*(b*b)*g*p - 0.87890625*(d*d*d)*(b*b*b)*g*p + 14.765625*(d*d)*(a*a*a)*p*s - 36.9140625*(d*d)*(a*a)*b*p*s + 21.09375*(d*d)*a*(b*b)*p*s - 2.63671875*(d*d)*(b*b*b)*p*s + 11.6015625*d*a*g*(p*p) - 7.2509765625*d*b*g*(p*p) + 11.6015625*a*(p*p)*s - 7.2509765625*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad589(double a, double b, double p, double d, double s, double g){
	return (0.13671875*(d*d*d*d)*(a*a*a*a)*g - 0.546875*(d*d*d*d)*(a*a*a)*b*g + 0.546875*(d*d*d*d)*(a*a)*(b*b)*g - 0.15625*(d*d*d*d)*a*(b*b*b)*g + 0.009765625*(d*d*d*d)*(b*b*b*b)*g + 0.546875*(d*d*d)*(a*a*a*a)*s - 2.1875*(d*d*d)*(a*a*a)*b*s + 2.1875*(d*d*d)*(a*a)*(b*b)*s - 0.625*(d*d*d)*a*(b*b*b)*s + 0.0390625*(d*d*d)*(b*b*b*b)*s + 1.50390625*(d*d)*(a*a)*g*p - 2.1484375*(d*d)*a*b*g*p + 0.537109375*(d*d)*(b*b)*g*p + 3.0078125*d*(a*a)*p*s - 4.296875*d*a*b*p*s + 1.07421875*d*(b*b)*p*s + 1.04736328125*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad5810(double a, double b, double p, double d, double s, double g){
	return (0.0546875*(d*d*d)*(a*a*a)*g - 0.13671875*(d*d*d)*(a*a)*b*g + 0.078125*(d*d*d)*a*(b*b)*g - 0.009765625*(d*d*d)*(b*b*b)*g + 0.1640625*(d*d)*(a*a*a)*s - 0.41015625*(d*d)*(a*a)*b*s + 0.234375*(d*d)*a*(b*b)*s - 0.029296875*(d*d)*(b*b*b)*s + 0.2578125*d*a*g*p - 0.1611328125*d*b*g*p + 0.2578125*a*p*s - 0.1611328125*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad5811(double a, double b, double p, double d, double s, double g){
	return (0.013671875*(d*d)*(a*a)*g - 0.01953125*(d*d)*a*b*g + 0.0048828125*(d*d)*(b*b)*g + 0.02734375*d*(a*a)*s - 0.0390625*d*a*b*s + 0.009765625*d*(b*b)*s + 0.01904296875*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad5812(double a, double b, double p, double d, double s, double g){
	return (0.001953125*d*a*g - 0.001220703125*d*b*g + 0.001953125*a*s - 0.001220703125*b*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad5813(double a, double b, double p, double d, double s, double g){
	return 0.0001220703125*g/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad600(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 6.0*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 7.5*(d*d*d*d)*(b*b*b*b)*g*p + 30.0*(d*d*d)*(b*b*b*b)*p*s + 11.25*(d*d)*(b*b)*g*(p*p) + 22.5*d*(b*b)*(p*p)*s + 1.875*g*(p*p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad601(double a, double b, double p, double d, double s, double g){
	return b*(-3.0*(d*d*d*d*d)*(b*b*b*b)*g - 15.0*(d*d*d*d)*(b*b*b*b)*s - 15.0*(d*d*d)*(b*b)*g*p - 45.0*(d*d)*(b*b)*p*s - 11.25*d*g*(p*p) - 11.25*(p*p)*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad602(double a, double b, double p, double d, double s, double g){
	return (3.75*(d*d*d*d)*(b*b*b*b)*g + 15.0*(d*d*d)*(b*b*b*b)*s + 11.25*(d*d)*(b*b)*g*p + 22.5*d*(b*b)*p*s + 2.8125*g*(p*p))/(p*p*p*p*p*p);
}

inline double MD_Et_grad603(double a, double b, double p, double d, double s, double g){
	return b*(-2.5*(d*d*d)*(b*b)*g - 7.5*(d*d)*(b*b)*s - 3.75*d*g*p - 3.75*p*s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad604(double a, double b, double p, double d, double s, double g){
	return (0.9375*d*(b*b)*(d*g + 2*s) + 0.46875*g*p)/(p*p*p*p*p*p);
}

inline double MD_Et_grad605(double a, double b, double p, double d, double s, double g){
	return 0.1875*b*(-d*g - s)/(p*p*p*p*p*p);
}

inline double MD_Et_grad606(double a, double b, double p, double d, double s, double g){
	return 0.015625*g/(p*p*p*p*p*p);
}

inline double MD_Et_grad610(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g + 7.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s + 7.5*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 3.0*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 37.5*(d*d*d*d)*a*(b*b*b*b)*p*s - 15.0*(d*d*d*d)*(b*b*b*b*b)*p*s + 11.25*(d*d*d)*a*(b*b)*g*(p*p) - 15.0*(d*d*d)*(b*b*b)*g*(p*p) + 33.75*(d*d)*a*(b*b)*(p*p)*s - 45.0*(d*d)*(b*b*b)*(p*p)*s + 1.875*d*a*g*(p*p*p) - 11.25*d*b*g*(p*p*p) + 1.875*a*(p*p*p)*s - 11.25*b*(p*p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad611(double a, double b, double p, double d, double s, double g){
	return (-3.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.5*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g - 18.0*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 3.0*(d*d*d*d*d)*(b*b*b*b*b*b)*s - 15.0*(d*d*d*d)*a*(b*b*b)*g*p + 11.25*(d*d*d*d)*(b*b*b*b)*g*p - 60.0*(d*d*d)*a*(b*b*b)*p*s + 45.0*(d*d*d)*(b*b*b*b)*p*s - 11.25*(d*d)*a*b*g*(p*p) + 28.125*(d*d)*(b*b)*g*(p*p) - 22.5*d*a*b*(p*p)*s + 56.25*d*(b*b)*(p*p)*s + 6.5625*g*(p*p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad612(double a, double b, double p, double d, double s, double g){
	return (3.75*(d*d*d*d*d)*a*(b*b*b*b)*g - 1.5*(d*d*d*d*d)*(b*b*b*b*b)*g + 18.75*(d*d*d*d)*a*(b*b*b*b)*s - 7.5*(d*d*d*d)*(b*b*b*b*b)*s + 11.25*(d*d*d)*a*(b*b)*g*p - 15.0*(d*d*d)*(b*b*b)*g*p + 33.75*(d*d)*a*(b*b)*p*s - 45.0*(d*d)*(b*b*b)*p*s + 2.8125*d*a*g*(p*p) - 16.875*d*b*g*(p*p) + 2.8125*a*(p*p)*s - 16.875*b*(p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad613(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d)*a*(b*b*b)*g + 1.875*(d*d*d*d)*(b*b*b*b)*g - 10.0*(d*d*d)*a*(b*b*b)*s + 7.5*(d*d*d)*(b*b*b*b)*s - 3.75*(d*d)*a*b*g*p + 9.375*(d*d)*(b*b)*g*p - 7.5*d*a*b*p*s + 18.75*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad614(double a, double b, double p, double d, double s, double g){
	return (0.9375*(d*d*d)*a*(b*b)*g - 1.25*(d*d*d)*(b*b*b)*g + 2.8125*(d*d)*a*(b*b)*s - 3.75*(d*d)*(b*b*b)*s + 0.46875*d*a*g*p - 2.8125*d*b*g*p + 0.46875*a*p*s - 2.8125*b*p*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad615(double a, double b, double p, double d, double s, double g){
	return (-0.1875*(d*d)*a*b*g + 0.46875*(d*d)*(b*b)*g - 0.375*d*a*b*s + 0.9375*d*(b*b)*s + 0.328125*g*p)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad616(double a, double b, double p, double d, double s, double g){
	return (0.015625*d*a*g - 0.09375*d*b*g + 0.015625*a*s - 0.09375*b*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad617(double a, double b, double p, double d, double s, double g){
	return 0.0078125*g/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad620(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g + 8.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 6.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 0.5*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 45.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 36.0*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 3.0*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 11.25*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 30.0*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 11.25*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 45.0*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 120.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 45.0*(d*d*d)*(b*b*b*b)*(p*p)*s + 1.875*(d*d)*(a*a)*g*(p*p*p) - 22.5*(d*d)*a*b*g*(p*p*p) + 28.125*(d*d)*(b*b)*g*(p*p*p) + 3.75*d*(a*a)*(p*p*p)*s - 45.0*d*a*b*(p*p*p)*s + 56.25*d*(b*b)*(p*p*p)*s + 6.5625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad621(double a, double b, double p, double d, double s, double g){
	return (-3.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + (d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 21.0 *(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 7.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 15.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 22.5*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 4.5*(d*d*d*d*d)*(b*b*b*b*b)*g*p - 75.0*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 112.5*(d*d*d*d)*a*(b*b*b*b)*p*s - 22.5*(d*d*d*d)*(b*b*b*b*b)*p*s - 11.25*(d*d*d)*(a*a)*b*g*(p*p) + 56.25*(d*d*d)*a*(b*b)*g*(p*p) - 37.5*(d*d*d)*(b*b*b)*g*(p*p) - 33.75*(d*d)*(a*a)*b*(p*p)*s + 168.75*(d*d)*a*(b*b)*(p*p)*s - 112.5*(d*d)*(b*b*b)*(p*p)*s + 13.125*d*a*g*(p*p*p) - 39.375*d*b*g*(p*p*p) + 13.125*a*(p*p*p)*s - 39.375*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad622(double a, double b, double p, double d, double s, double g){
	return (3.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 3.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.25*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 22.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 18.0*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 1.5*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 11.25*(d*d*d*d)*(a*a)*(b*b)*g*p - 30.0*(d*d*d*d)*a*(b*b*b)*g*p + 11.25*(d*d*d*d)*(b*b*b*b)*g*p + 45.0*(d*d*d)*(a*a)*(b*b)*p*s - 120.0*(d*d*d)*a*(b*b*b)*p*s + 45.0*(d*d*d)*(b*b*b*b)*p*s + 2.8125*(d*d)*(a*a)*g*(p*p) - 33.75*(d*d)*a*b*g*(p*p) + 42.1875*(d*d)*(b*b)*g*(p*p) + 5.625*d*(a*a)*(p*p)*s - 67.5*d*a*b*(p*p)*s + 84.375*d*(b*b)*(p*p)*s + 13.125*g*(p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad623(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 3.75*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.75*(d*d*d*d*d)*(b*b*b*b*b)*g - 12.5*(d*d*d*d)*(a*a)*(b*b*b)*s + 18.75*(d*d*d*d)*a*(b*b*b*b)*s - 3.75*(d*d*d*d)*(b*b*b*b*b)*s - 3.75*(d*d*d)*(a*a)*b*g*p + 18.75*(d*d*d)*a*(b*b)*g*p - 12.5*(d*d*d)*(b*b*b)*g*p - 11.25*(d*d)*(a*a)*b*p*s + 56.25*(d*d)*a*(b*b)*p*s - 37.5*(d*d)*(b*b*b)*p*s + 6.5625*d*a*g*(p*p) - 19.6875*d*b*g*(p*p) + 6.5625*a*(p*p)*s - 19.6875*b*(p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad624(double a, double b, double p, double d, double s, double g){
	return (0.9375*(d*d*d*d)*(a*a)*(b*b)*g - 2.5*(d*d*d*d)*a*(b*b*b)*g + 0.9375*(d*d*d*d)*(b*b*b*b)*g + 3.75*(d*d*d)*(a*a)*(b*b)*s - 10.0*(d*d*d)*a*(b*b*b)*s + 3.75*(d*d*d)*(b*b*b*b)*s + 0.46875*(d*d)*(a*a)*g*p - 5.625*(d*d)*a*b*g*p + 7.03125*(d*d)*(b*b)*g*p + 0.9375*d*(a*a)*p*s - 11.25*d*a*b*p*s + 14.0625*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad625(double a, double b, double p, double d, double s, double g){
	return (-0.1875*(d*d*d)*(a*a)*b*g + 0.9375*(d*d*d)*a*(b*b)*g - 0.625*(d*d*d)*(b*b*b)*g - 0.5625*(d*d)*(a*a)*b*s + 2.8125*(d*d)*a*(b*b)*s - 1.875*(d*d)*(b*b*b)*s + 0.65625*d*a*g*p - 1.96875*d*b*g*p + 0.65625*a*p*s - 1.96875*b*p*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad626(double a, double b, double p, double d, double s, double g){
	return (0.015625*(d*d)*(a*a)*g - 0.1875*(d*d)*a*b*g + 0.234375*(d*d)*(b*b)*g + 0.03125*d*(a*a)*s - 0.375*d*a*b*s + 0.46875*d*(b*b)*s + 0.21875*g*p)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad627(double a, double b, double p, double d, double s, double g){
	return (0.015625*d*a*g - 0.046875*d*b*g + 0.015625*a*s - 0.046875*b*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad628(double a, double b, double p, double d, double s, double g){
	return 0.00390625*g/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad630(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g + 9.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 9.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 1.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p + 52.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 63.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 10.5*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s + 11.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 45.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 33.75*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 4.5*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 56.25*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 225.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 168.75*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 22.5*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 1.875*(d*d*d)*(a*a*a)*g*(p*p*p) - 33.75*(d*d*d)*(a*a)*b*g*(p*p*p) + 84.375*(d*d*d)*a*(b*b)*g*(p*p*p) - 37.5*(d*d*d)*(b*b*b)*g*(p*p*p) + 5.625*(d*d)*(a*a*a)*(p*p*p)*s - 101.25*(d*d)*(a*a)*b*(p*p*p)*s + 253.125*(d*d)*a*(b*b)*(p*p*p)*s - 112.5*(d*d)*(b*b*b)*(p*p*p)*s + 19.6875*d*a*g*(p*p*p*p) - 39.375*d*b*g*(p*p*p*p) + 19.6875*a*(p*p*p*p)*s - 39.375*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad631(double a, double b, double p, double d, double s, double g){
	return (-3.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 1.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 24.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 12.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 15.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 33.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 13.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 0.75*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p - 90.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 202.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 81.0 *(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 4.5*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s - 11.25*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 84.375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 112.5*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 28.125*(d*d*d*d)*(b*b*b*b)*g*(p*p) - 45.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 337.5*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 450.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 112.5*(d*d*d)*(b*b*b*b)*(p*p)*s + 19.6875*(d*d)*(a*a)*g*(p*p*p) - 118.125*(d*d)*a*b*g*(p*p*p) + 98.4375*(d*d)*(b*b)*g*(p*p*p) + 39.375*d*(a*a)*(p*p*p)*s - 236.25*d*a*b*(p*p*p)*s + 196.875*d*(b*b)*(p*p*p)*s + 29.53125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad632(double a, double b, double p, double d, double s, double g){
	return (3.75*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 4.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.75*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g + 26.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 31.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 5.25*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s + 11.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 45.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 33.75*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 4.5*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 56.25*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 225.0*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 168.75*(d*d*d*d)*a*(b*b*b*b)*p*s - 22.5*(d*d*d*d)*(b*b*b*b*b)*p*s + 2.8125*(d*d*d)*(a*a*a)*g*(p*p) - 50.625*(d*d*d)*(a*a)*b*g*(p*p) + 126.5625*(d*d*d)*a*(b*b)*g*(p*p) - 56.25*(d*d*d)*(b*b*b)*g*(p*p) + 8.4375*(d*d)*(a*a*a)*(p*p)*s - 151.875*(d*d)*(a*a)*b*(p*p)*s + 379.6875*(d*d)*a*(b*b)*(p*p)*s - 168.75*(d*d)*(b*b*b)*(p*p)*s + 39.375*d*a*g*(p*p*p) - 78.75*d*b*g*(p*p*p) + 39.375*a*(p*p*p)*s - 78.75*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad633(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 5.625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 2.25*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g - 15.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 33.75*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 13.5*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.75*(d*d*d*d*d)*(b*b*b*b*b*b)*s - 3.75*(d*d*d*d)*(a*a*a)*b*g*p + 28.125*(d*d*d*d)*(a*a)*(b*b)*g*p - 37.5*(d*d*d*d)*a*(b*b*b)*g*p + 9.375*(d*d*d*d)*(b*b*b*b)*g*p - 15.0*(d*d*d)*(a*a*a)*b*p*s + 112.5*(d*d*d)*(a*a)*(b*b)*p*s - 150.0*(d*d*d)*a*(b*b*b)*p*s + 37.5*(d*d*d)*(b*b*b*b)*p*s + 9.84375*(d*d)*(a*a)*g*(p*p) - 59.0625*(d*d)*a*b*g*(p*p) + 49.21875*(d*d)*(b*b)*g*(p*p) + 19.6875*d*(a*a)*(p*p)*s - 118.125*d*a*b*(p*p)*s + 98.4375*d*(b*b)*(p*p)*s + 19.6875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad634(double a, double b, double p, double d, double s, double g){
	return (0.9375*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 3.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 2.8125*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.375*(d*d*d*d*d)*(b*b*b*b*b)*g + 4.6875*(d*d*d*d)*(a*a*a)*(b*b)*s - 18.75*(d*d*d*d)*(a*a)*(b*b*b)*s + 14.0625*(d*d*d*d)*a*(b*b*b*b)*s - 1.875*(d*d*d*d)*(b*b*b*b*b)*s + 0.46875*(d*d*d)*(a*a*a)*g*p - 8.4375*(d*d*d)*(a*a)*b*g*p + 21.09375*(d*d*d)*a*(b*b)*g*p - 9.375*(d*d*d)*(b*b*b)*g*p + 1.40625*(d*d)*(a*a*a)*p*s - 25.3125*(d*d)*(a*a)*b*p*s + 63.28125*(d*d)*a*(b*b)*p*s - 28.125*(d*d)*(b*b*b)*p*s + 9.84375*d*a*g*(p*p) - 19.6875*d*b*g*(p*p) + 9.84375*a*(p*p)*s - 19.6875*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad635(double a, double b, double p, double d, double s, double g){
	return (-0.1875*(d*d*d*d)*(a*a*a)*b*g + 1.40625*(d*d*d*d)*(a*a)*(b*b)*g - 1.875*(d*d*d*d)*a*(b*b*b)*g + 0.46875*(d*d*d*d)*(b*b*b*b)*g - 0.75*(d*d*d)*(a*a*a)*b*s + 5.625*(d*d*d)*(a*a)*(b*b)*s - 7.5*(d*d*d)*a*(b*b*b)*s + 1.875*(d*d*d)*(b*b*b*b)*s + 0.984375*(d*d)*(a*a)*g*p - 5.90625*(d*d)*a*b*g*p + 4.921875*(d*d)*(b*b)*g*p + 1.96875*d*(a*a)*p*s - 11.8125*d*a*b*p*s + 9.84375*d*(b*b)*p*s + 2.953125*g*(p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad636(double a, double b, double p, double d, double s, double g){
	return (0.015625*(d*d*d)*(a*a*a)*g - 0.28125*(d*d*d)*(a*a)*b*g + 0.703125*(d*d*d)*a*(b*b)*g - 0.3125*(d*d*d)*(b*b*b)*g + 0.046875*(d*d)*(a*a*a)*s - 0.84375*(d*d)*(a*a)*b*s + 2.109375*(d*d)*a*(b*b)*s - 0.9375*(d*d)*(b*b*b)*s + 0.65625*d*a*g*p - 1.3125*d*b*g*p + 0.65625*a*p*s - 1.3125*b*p*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad637(double a, double b, double p, double d, double s, double g){
	return (0.0234375*(d*d)*(a*a)*g - 0.140625*(d*d)*a*b*g + 0.1171875*(d*d)*(b*b)*g + 0.046875*d*(a*a)*s - 0.28125*d*a*b*s + 0.234375*d*(b*b)*s + 0.140625*g*p)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad638(double a, double b, double p, double d, double s, double g){
	return (0.01171875*d*a*g - 0.0234375*d*b*g + 0.01171875*a*s - 0.0234375*b*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad639(double a, double b, double p, double d, double s, double g){
	return 0.001953125*g/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad640(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g + 10.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 12.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 3.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p + 60.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 96.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 24.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s + 11.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 60.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 67.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 18.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 0.75*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 67.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 360.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 405.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 108.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 4.5*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 1.875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 45.0*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 168.75*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 150.0*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 28.125*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 7.5*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 180.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 675.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 600.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 112.5*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 39.375*(d*d)*(a*a)*g*(p*p*p*p) - 157.5*(d*d)*a*b*g*(p*p*p*p) + 98.4375*(d*d)*(b*b)*g*(p*p*p*p) + 78.75*d*(a*a)*(p*p*p*p)*s - 315.0*d*a*b*(p*p*p*p)*s + 196.875*d*(b*b)*(p*p*p*p)*s + 29.53125*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad641(double a, double b, double p, double d, double s, double g){
	return (-3.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 2.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 27.0*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 18.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 15.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 45.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 27.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 3.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 105.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 315.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 189.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 21.0 *(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 11.25*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 112.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 225.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 112.5*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 11.25*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) - 56.25*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 562.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 1125.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 562.5*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 56.25*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 26.25*(d*d*d)*(a*a*a)*g*(p*p*p) - 236.25*(d*d*d)*(a*a)*b*g*(p*p*p) + 393.75*(d*d*d)*a*(b*b)*g*(p*p*p) - 131.25*(d*d*d)*(b*b*b)*g*(p*p*p) + 78.75*(d*d)*(a*a*a)*(p*p*p)*s - 708.75*(d*d)*(a*a)*b*(p*p*p)*s + 1181.25*(d*d)*a*(b*b)*(p*p*p)*s - 393.75*(d*d)*(b*b*b)*(p*p*p)*s + 118.125*d*a*g*(p*p*p*p) - 177.1875*d*b*g*(p*p*p*p) + 118.125*a*(p*p*p*p)*s - 177.1875*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad642(double a, double b, double p, double d, double s, double g){
	return (3.75*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 6.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 1.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g + 30.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 48.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 12.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s + 11.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 60.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 67.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 18.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 0.75*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 67.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 360.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 405.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 108.0*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 4.5*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 2.8125*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 67.5*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 253.125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 225.0*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 42.1875*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 11.25*(d*d*d)*(a*a*a*a)*(p*p)*s - 270.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 1012.5*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 900.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 168.75*(d*d*d)*(b*b*b*b)*(p*p)*s + 78.75*(d*d)*(a*a)*g*(p*p*p) - 315.0*(d*d)*a*b*g*(p*p*p) + 196.875*(d*d)*(b*b)*g*(p*p*p) + 157.5*d*(a*a)*(p*p*p)*s - 630.0*d*a*b*(p*p*p)*s + 393.75*d*(b*b)*(p*p*p)*s + 73.828125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad643(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 7.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 4.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 17.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 52.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 31.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 3.5*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 3.75*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 37.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 75.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 37.5*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 3.75*(d*d*d*d*d)*(b*b*b*b*b)*g*p - 18.75*(d*d*d*d)*(a*a*a*a)*b*p*s + 187.5*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 375.0*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 187.5*(d*d*d*d)*a*(b*b*b*b)*p*s - 18.75*(d*d*d*d)*(b*b*b*b*b)*p*s + 13.125*(d*d*d)*(a*a*a)*g*(p*p) - 118.125*(d*d*d)*(a*a)*b*g*(p*p) + 196.875*(d*d*d)*a*(b*b)*g*(p*p) - 65.625*(d*d*d)*(b*b*b)*g*(p*p) + 39.375*(d*d)*(a*a*a)*(p*p)*s - 354.375*(d*d)*(a*a)*b*(p*p)*s + 590.625*(d*d)*a*(b*b)*(p*p)*s - 196.875*(d*d)*(b*b*b)*(p*p)*s + 78.75*d*a*g*(p*p*p) - 118.125*d*b*g*(p*p*p) + 78.75*a*(p*p*p)*s - 118.125*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad644(double a, double b, double p, double d, double s, double g){
	return (0.9375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 5.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 5.625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 1.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.0625*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 5.625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 30.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 33.75*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 9.0*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.375*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 0.46875*(d*d*d*d)*(a*a*a*a)*g*p - 11.25*(d*d*d*d)*(a*a*a)*b*g*p + 42.1875*(d*d*d*d)*(a*a)*(b*b)*g*p - 37.5*(d*d*d*d)*a*(b*b*b)*g*p + 7.03125*(d*d*d*d)*(b*b*b*b)*g*p + 1.875*(d*d*d)*(a*a*a*a)*p*s - 45.0*(d*d*d)*(a*a*a)*b*p*s + 168.75*(d*d*d)*(a*a)*(b*b)*p*s - 150.0*(d*d*d)*a*(b*b*b)*p*s + 28.125*(d*d*d)*(b*b*b*b)*p*s + 19.6875*(d*d)*(a*a)*g*(p*p) - 78.75*(d*d)*a*b*g*(p*p) + 49.21875*(d*d)*(b*b)*g*(p*p) + 39.375*d*(a*a)*(p*p)*s - 157.5*d*a*b*(p*p)*s + 98.4375*d*(b*b)*(p*p)*s + 24.609375*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad645(double a, double b, double p, double d, double s, double g){
	return (-0.1875*(d*d*d*d*d)*(a*a*a*a)*b*g + 1.875*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 3.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 1.875*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.1875*(d*d*d*d*d)*(b*b*b*b*b)*g - 0.9375*(d*d*d*d)*(a*a*a*a)*b*s + 9.375*(d*d*d*d)*(a*a*a)*(b*b)*s - 18.75*(d*d*d*d)*(a*a)*(b*b*b)*s + 9.375*(d*d*d*d)*a*(b*b*b*b)*s - 0.9375*(d*d*d*d)*(b*b*b*b*b)*s + 1.3125*(d*d*d)*(a*a*a)*g*p - 11.8125*(d*d*d)*(a*a)*b*g*p + 19.6875*(d*d*d)*a*(b*b)*g*p - 6.5625*(d*d*d)*(b*b*b)*g*p + 3.9375*(d*d)*(a*a*a)*p*s - 35.4375*(d*d)*(a*a)*b*p*s + 59.0625*(d*d)*a*(b*b)*p*s - 19.6875*(d*d)*(b*b*b)*p*s + 11.8125*d*a*g*(p*p) - 17.71875*d*b*g*(p*p) + 11.8125*a*(p*p)*s - 17.71875*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad646(double a, double b, double p, double d, double s, double g){
	return (0.015625*(d*d*d*d)*(a*a*a*a)*g - 0.375*(d*d*d*d)*(a*a*a)*b*g + 1.40625*(d*d*d*d)*(a*a)*(b*b)*g - 1.25*(d*d*d*d)*a*(b*b*b)*g + 0.234375*(d*d*d*d)*(b*b*b*b)*g + 0.0625*(d*d*d)*(a*a*a*a)*s - 1.5*(d*d*d)*(a*a*a)*b*s + 5.625*(d*d*d)*(a*a)*(b*b)*s - 5.0*(d*d*d)*a*(b*b*b)*s + 0.9375*(d*d*d)*(b*b*b*b)*s + 1.3125*(d*d)*(a*a)*g*p - 5.25*(d*d)*a*b*g*p + 3.28125*(d*d)*(b*b)*g*p + 2.625*d*(a*a)*p*s - 10.5*d*a*b*p*s + 6.5625*d*(b*b)*p*s + 2.4609375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad647(double a, double b, double p, double d, double s, double g){
	return (0.03125*(d*d*d)*(a*a*a)*g - 0.28125*(d*d*d)*(a*a)*b*g + 0.46875*(d*d*d)*a*(b*b)*g - 0.15625*(d*d*d)*(b*b*b)*g + 0.09375*(d*d)*(a*a*a)*s - 0.84375*(d*d)*(a*a)*b*s + 1.40625*(d*d)*a*(b*b)*s - 0.46875*(d*d)*(b*b*b)*s + 0.5625*d*a*g*p - 0.84375*d*b*g*p + 0.5625*a*p*s - 0.84375*b*p*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad648(double a, double b, double p, double d, double s, double g){
	return (0.0234375*(d*d)*(a*a)*g - 0.09375*(d*d)*a*b*g + 0.05859375*(d*d)*(b*b)*g + 0.046875*d*(a*a)*s - 0.1875*d*a*b*s + 0.1171875*d*(b*b)*s + 0.087890625*g*p)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad649(double a, double b, double p, double d, double s, double g){
	return (0.0078125*d*a*g - 0.01171875*d*b*g + 0.0078125*a*s - 0.01171875*b*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6410(double a, double b, double p, double d, double s, double g){
	return 0.0009765625*g/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad650(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g + 11.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 15.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 5.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p + 67.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 135.0*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 45.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s + 11.25*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 75.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 112.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 45.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 3.75*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) + 78.75*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 525.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 787.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 315.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 26.25*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s + 1.875*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 56.25*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 281.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 375.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 140.625*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 11.25*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 9.375*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 281.25*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 1406.25*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 1875.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 703.125*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 56.25*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 65.625*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 393.75*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 492.1875*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 131.25*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 196.875*(d*d)*(a*a*a)*(p*p*p*p)*s - 1181.25*(d*d)*(a*a)*b*(p*p*p*p)*s + 1476.5625*(d*d)*a*(b*b)*(p*p*p*p)*s - 393.75*(d*d)*(b*b*b)*(p*p*p*p)*s + 147.65625*d*a*g*(p*p*p*p*p) - 177.1875*d*b*g*(p*p*p*p*p) + 147.65625*a*(p*p*p*p*p)*s - 177.1875*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad651(double a, double b, double p, double d, double s, double g){
	return (-3.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 2.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g - 30.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 25.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s - 15.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 56.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 45.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 7.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 120.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 450.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 360.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 60.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 11.25*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 140.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 375.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 281.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 56.25*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 1.875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) - 67.5*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 843.75*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 2250.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 1687.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 337.5*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 11.25*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 32.8125*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 393.75*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 984.375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 656.25*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 98.4375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 131.25*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 1575.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 3937.5*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 2625.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 393.75*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 295.3125*(d*d)*(a*a)*g*(p*p*p*p) - 885.9375*(d*d)*a*b*g*(p*p*p*p) + 442.96875*(d*d)*(b*b)*g*(p*p*p*p) + 590.625*d*(a*a)*(p*p*p*p)*s - 1771.875*d*a*b*(p*p*p*p)*s + 885.9375*d*(b*b)*(p*p*p*p)*s + 162.421875*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad652(double a, double b, double p, double d, double s, double g){
	return (3.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 7.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 2.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g + 33.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 67.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 22.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s + 11.25*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 75.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 112.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 45.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 3.75*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p + 78.75*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 525.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 787.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 315.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 26.25*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s + 2.8125*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 84.375*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 421.875*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 562.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 210.9375*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 16.875*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 14.0625*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 421.875*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 2109.375*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 2812.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 1054.6875*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 84.375*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 131.25*(d*d*d)*(a*a*a)*g*(p*p*p) - 787.5*(d*d*d)*(a*a)*b*g*(p*p*p) + 984.375*(d*d*d)*a*(b*b)*g*(p*p*p) - 262.5*(d*d*d)*(b*b*b)*g*(p*p*p) + 393.75*(d*d)*(a*a*a)*(p*p*p)*s - 2362.5*(d*d)*(a*a)*b*(p*p*p)*s + 2953.125*(d*d)*a*(b*b)*(p*p*p)*s - 787.5*(d*d)*(b*b*b)*(p*p*p)*s + 369.140625*d*a*g*(p*p*p*p) - 442.96875*d*b*g*(p*p*p*p) + 369.140625*a*(p*p*p*p)*s - 442.96875*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad653(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 9.375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 7.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 1.25*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 20.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 75.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 60.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 10.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 3.75*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 46.875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 125.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 93.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 18.75*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 0.625*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p - 22.5*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 281.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 750.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 562.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 112.5*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 3.75*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 16.40625*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 196.875*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 492.1875*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 328.125*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 49.21875*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 65.625*(d*d*d)*(a*a*a*a)*(p*p)*s - 787.5*(d*d*d)*(a*a*a)*b*(p*p)*s + 1968.75*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 1312.5*(d*d*d)*a*(b*b*b)*(p*p)*s + 196.875*(d*d*d)*(b*b*b*b)*(p*p)*s + 196.875*(d*d)*(a*a)*g*(p*p*p) - 590.625*(d*d)*a*b*g*(p*p*p) + 295.3125*(d*d)*(b*b)*g*(p*p*p) + 393.75*d*(a*a)*(p*p*p)*s - 1181.25*d*a*b*(p*p*p)*s + 590.625*d*(b*b)*(p*p*p)*s + 135.3515625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad654(double a, double b, double p, double d, double s, double g){
	return (0.9375*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 6.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 9.375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 3.75*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.3125*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g + 6.5625*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 43.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 65.625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 26.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 2.1875*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s + 0.46875*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 14.0625*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 70.3125*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 93.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 35.15625*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 2.8125*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 2.34375*(d*d*d*d)*(a*a*a*a*a)*p*s - 70.3125*(d*d*d*d)*(a*a*a*a)*b*p*s + 351.5625*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 468.75*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 175.78125*(d*d*d*d)*a*(b*b*b*b)*p*s - 14.0625*(d*d*d*d)*(b*b*b*b*b)*p*s + 32.8125*(d*d*d)*(a*a*a)*g*(p*p) - 196.875*(d*d*d)*(a*a)*b*g*(p*p) + 246.09375*(d*d*d)*a*(b*b)*g*(p*p) - 65.625*(d*d*d)*(b*b*b)*g*(p*p) + 98.4375*(d*d)*(a*a*a)*(p*p)*s - 590.625*(d*d)*(a*a)*b*(p*p)*s + 738.28125*(d*d)*a*(b*b)*(p*p)*s - 196.875*(d*d)*(b*b*b)*(p*p)*s + 123.046875*d*a*g*(p*p*p) - 147.65625*d*b*g*(p*p*p) + 123.046875*a*(p*p*p)*s - 147.65625*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad655(double a, double b, double p, double d, double s, double g){
	return (-0.1875*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 2.34375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 6.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 4.6875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.9375*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.03125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g - 1.125*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 14.0625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 37.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 28.125*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 5.625*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.1875*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 1.640625*(d*d*d*d)*(a*a*a*a)*g*p - 19.6875*(d*d*d*d)*(a*a*a)*b*g*p + 49.21875*(d*d*d*d)*(a*a)*(b*b)*g*p - 32.8125*(d*d*d*d)*a*(b*b*b)*g*p + 4.921875*(d*d*d*d)*(b*b*b*b)*g*p + 6.5625*(d*d*d)*(a*a*a*a)*p*s - 78.75*(d*d*d)*(a*a*a)*b*p*s + 196.875*(d*d*d)*(a*a)*(b*b)*p*s - 131.25*(d*d*d)*a*(b*b*b)*p*s + 19.6875*(d*d*d)*(b*b*b*b)*p*s + 29.53125*(d*d)*(a*a)*g*(p*p) - 88.59375*(d*d)*a*b*g*(p*p) + 44.296875*(d*d)*(b*b)*g*(p*p) + 59.0625*d*(a*a)*(p*p)*s - 177.1875*d*a*b*(p*p)*s + 88.59375*d*(b*b)*(p*p)*s + 27.0703125*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad656(double a, double b, double p, double d, double s, double g){
	return (0.015625*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.46875*(d*d*d*d*d)*(a*a*a*a)*b*g + 2.34375*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 3.125*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 1.171875*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.09375*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.078125*(d*d*d*d)*(a*a*a*a*a)*s - 2.34375*(d*d*d*d)*(a*a*a*a)*b*s + 11.71875*(d*d*d*d)*(a*a*a)*(b*b)*s - 15.625*(d*d*d*d)*(a*a)*(b*b*b)*s + 5.859375*(d*d*d*d)*a*(b*b*b*b)*s - 0.46875*(d*d*d*d)*(b*b*b*b*b)*s + 2.1875*(d*d*d)*(a*a*a)*g*p - 13.125*(d*d*d)*(a*a)*b*g*p + 16.40625*(d*d*d)*a*(b*b)*g*p - 4.375*(d*d*d)*(b*b*b)*g*p + 6.5625*(d*d)*(a*a*a)*p*s - 39.375*(d*d)*(a*a)*b*p*s + 49.21875*(d*d)*a*(b*b)*p*s - 13.125*(d*d)*(b*b*b)*p*s + 12.3046875*d*a*g*(p*p) - 14.765625*d*b*g*(p*p) + 12.3046875*a*(p*p)*s - 14.765625*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad657(double a, double b, double p, double d, double s, double g){
	return (0.0390625*(d*d*d*d)*(a*a*a*a)*g - 0.46875*(d*d*d*d)*(a*a*a)*b*g + 1.171875*(d*d*d*d)*(a*a)*(b*b)*g - 0.78125*(d*d*d*d)*a*(b*b*b)*g + 0.1171875*(d*d*d*d)*(b*b*b*b)*g + 0.15625*(d*d*d)*(a*a*a*a)*s - 1.875*(d*d*d)*(a*a*a)*b*s + 4.6875*(d*d*d)*(a*a)*(b*b)*s - 3.125*(d*d*d)*a*(b*b*b)*s + 0.46875*(d*d*d)*(b*b*b*b)*s + 1.40625*(d*d)*(a*a)*g*p - 4.21875*(d*d)*a*b*g*p + 2.109375*(d*d)*(b*b)*g*p + 2.8125*d*(a*a)*p*s - 8.4375*d*a*b*p*s + 4.21875*d*(b*b)*p*s + 1.93359375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad658(double a, double b, double p, double d, double s, double g){
	return (0.0390625*(d*d*d)*(a*a*a)*g - 0.234375*(d*d*d)*(a*a)*b*g + 0.29296875*(d*d*d)*a*(b*b)*g - 0.078125*(d*d*d)*(b*b*b)*g + 0.1171875*(d*d)*(a*a*a)*s - 0.703125*(d*d)*(a*a)*b*s + 0.87890625*(d*d)*a*(b*b)*s - 0.234375*(d*d)*(b*b*b)*s + 0.439453125*d*a*g*p - 0.52734375*d*b*g*p + 0.439453125*a*p*s - 0.52734375*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad659(double a, double b, double p, double d, double s, double g){
	return (0.01953125*(d*d)*(a*a)*g - 0.05859375*(d*d)*a*b*g + 0.029296875*(d*d)*(b*b)*g + 0.0390625*d*(a*a)*s - 0.1171875*d*a*b*s + 0.05859375*d*(b*b)*s + 0.0537109375*g*p)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6510(double a, double b, double p, double d, double s, double g){
	return (0.0048828125*d*a*g - 0.005859375*d*b*g + 0.0048828125*a*s - 0.005859375*b*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6511(double a, double b, double p, double d, double s, double g){
	return 0.00048828125*g/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad660(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g + 12.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*p - 18.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p + 7.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*p + 75.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*p*s - 180.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s + 75.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*p*s + 11.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p) - 90.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 168.75*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 90.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 11.25*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p) + 90.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p)*s - 720.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 1350.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 720.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 90.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p)*s + 1.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p) - 67.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 421.875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 750.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 421.875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 67.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 1.875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p) + 11.25*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p)*s - 405.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 2531.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 4500.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 2531.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 405.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 11.25*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p)*s + 98.4375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 787.5*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 1476.5625*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 787.5*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 98.4375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 393.75*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 3150.0*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 5906.25*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 3150.0*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 393.75*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 442.96875*(d*d)*(a*a)*g*(p*p*p*p*p) - 1063.125*(d*d)*a*b*g*(p*p*p*p*p) + 442.96875*(d*d)*(b*b)*g*(p*p*p*p*p) + 885.9375*d*(a*a)*(p*p*p*p*p)*s - 2126.25*d*a*b*(p*p*p*p*p)*s + 885.9375*d*(b*b)*(p*p*p*p*p)*s + 162.421875*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad661(double a, double b, double p, double d, double s, double g){
	return (-3.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g + 3.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g - 33.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*s + 33.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*s - 15.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 67.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 67.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 15.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 135.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 607.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 607.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 135.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s - 11.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 168.75*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 562.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 562.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 168.75*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 11.25*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) - 78.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 1181.25*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 3937.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 3937.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 1181.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 78.75*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s + 39.375*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 590.625*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 1968.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 1968.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 590.625*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 39.375*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 196.875*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 2953.125*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 9843.75*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 9843.75*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 2953.125*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 196.875*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 590.625*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 2657.8125*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 2657.8125*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 590.625*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 1771.875*(d*d)*(a*a*a)*(p*p*p*p)*s - 7973.4375*(d*d)*(a*a)*b*(p*p*p*p)*s + 7973.4375*(d*d)*a*(b*b)*(p*p*p*p)*s - 1771.875*(d*d)*(b*b*b)*(p*p*p*p)*s + 974.53125*d*a*g*(p*p*p*p*p) - 974.53125*d*b*g*(p*p*p*p*p) + 974.53125*a*(p*p*p*p*p)*s - 974.53125*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad662(double a, double b, double p, double d, double s, double g){
	return (3.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 9.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 3.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g + 37.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 90.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 37.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s + 11.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 90.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 168.75*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 90.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 11.25*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p + 90.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 720.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 1350.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 720.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 90.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s + 2.8125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 101.25*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 632.8125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 1125.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 632.8125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 101.25*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 2.8125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 16.875*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 607.5*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 3796.875*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 6750.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 3796.875*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 607.5*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 16.875*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 196.875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 1575.0*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 2953.125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 1575.0*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 196.875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 787.5*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 6300.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 11812.5*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 6300.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 787.5*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 1107.421875*(d*d)*(a*a)*g*(p*p*p*p) - 2657.8125*(d*d)*a*b*g*(p*p*p*p) + 1107.421875*(d*d)*(b*b)*g*(p*p*p*p) + 2214.84375*d*(a*a)*(p*p*p*p)*s - 5315.625*d*a*b*(p*p*p*p)*s + 2214.84375*d*(b*b)*(p*p*p*p)*s + 487.265625*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad663(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 11.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 11.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 2.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 22.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 101.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 101.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 22.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 3.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 56.25*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 187.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 187.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 56.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 3.75*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 26.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 393.75*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 1312.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 1312.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 393.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 26.25*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s + 19.6875*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 295.3125*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 984.375*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 984.375*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 295.3125*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 19.6875*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 98.4375*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 1476.5625*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 4921.875*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 4921.875*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 1476.5625*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 98.4375*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 393.75*(d*d*d)*(a*a*a)*g*(p*p*p) - 1771.875*(d*d*d)*(a*a)*b*g*(p*p*p) + 1771.875*(d*d*d)*a*(b*b)*g*(p*p*p) - 393.75*(d*d*d)*(b*b*b)*g*(p*p*p) + 1181.25*(d*d)*(a*a*a)*(p*p*p)*s - 5315.625*(d*d)*(a*a)*b*(p*p*p)*s + 5315.625*(d*d)*a*(b*b)*(p*p*p)*s - 1181.25*(d*d)*(b*b*b)*(p*p*p)*s + 812.109375*d*a*g*(p*p*p*p) - 812.109375*d*b*g*(p*p*p*p) + 812.109375*a*(p*p*p*p)*s - 812.109375*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad664(double a, double b, double p, double d, double s, double g){
	return (0.9375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 7.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 14.0625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 7.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 0.9375*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g + 7.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 60.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 112.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 60.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s + 0.46875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 16.875*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 105.46875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 187.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 105.46875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 16.875*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 0.46875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 2.8125*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 101.25*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 632.8125*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 1125.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 632.8125*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 101.25*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 2.8125*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 49.21875*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 393.75*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 738.28125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 393.75*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 49.21875*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 196.875*(d*d*d)*(a*a*a*a)*(p*p)*s - 1575.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 2953.125*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 1575.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 196.875*(d*d*d)*(b*b*b*b)*(p*p)*s + 369.140625*(d*d)*(a*a)*g*(p*p*p) - 885.9375*(d*d)*a*b*g*(p*p*p) + 369.140625*(d*d)*(b*b)*g*(p*p*p) + 738.28125*d*(a*a)*(p*p*p)*s - 1771.875*d*a*b*(p*p*p)*s + 738.28125*d*(b*b)*(p*p*p)*s + 203.02734375*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad665(double a, double b, double p, double d, double s, double g){
	return (-0.1875*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 2.8125*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 9.375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 9.375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 2.8125*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.1875*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 1.3125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 19.6875*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 65.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 65.625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 19.6875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 1.3125*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s + 1.96875*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 29.53125*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 98.4375*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 98.4375*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 29.53125*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 1.96875*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 9.84375*(d*d*d*d)*(a*a*a*a*a)*p*s - 147.65625*(d*d*d*d)*(a*a*a*a)*b*p*s + 492.1875*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 492.1875*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 147.65625*(d*d*d*d)*a*(b*b*b*b)*p*s - 9.84375*(d*d*d*d)*(b*b*b*b*b)*p*s + 59.0625*(d*d*d)*(a*a*a)*g*(p*p) - 265.78125*(d*d*d)*(a*a)*b*g*(p*p) + 265.78125*(d*d*d)*a*(b*b)*g*(p*p) - 59.0625*(d*d*d)*(b*b*b)*g*(p*p) + 177.1875*(d*d)*(a*a*a)*(p*p)*s - 797.34375*(d*d)*(a*a)*b*(p*p)*s + 797.34375*(d*d)*a*(b*b)*(p*p)*s - 177.1875*(d*d)*(b*b*b)*(p*p)*s + 162.421875*d*a*g*(p*p*p) - 162.421875*d*b*g*(p*p*p) + 162.421875*a*(p*p*p)*s - 162.421875*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad666(double a, double b, double p, double d, double s, double g){
	return (0.015625*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 0.5625*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 3.515625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 6.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 3.515625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.5625*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.015625*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 0.09375*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 3.375*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 21.09375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 37.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 21.09375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 3.375*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.09375*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 3.28125*(d*d*d*d)*(a*a*a*a)*g*p - 26.25*(d*d*d*d)*(a*a*a)*b*g*p + 49.21875*(d*d*d*d)*(a*a)*(b*b)*g*p - 26.25*(d*d*d*d)*a*(b*b*b)*g*p + 3.28125*(d*d*d*d)*(b*b*b*b)*g*p + 13.125*(d*d*d)*(a*a*a*a)*p*s - 105.0*(d*d*d)*(a*a*a)*b*p*s + 196.875*(d*d*d)*(a*a)*(b*b)*p*s - 105.0*(d*d*d)*a*(b*b*b)*p*s + 13.125*(d*d*d)*(b*b*b*b)*p*s + 36.9140625*(d*d)*(a*a)*g*(p*p) - 88.59375*(d*d)*a*b*g*(p*p) + 36.9140625*(d*d)*(b*b)*g*(p*p) + 73.828125*d*(a*a)*(p*p)*s - 177.1875*d*a*b*(p*p)*s + 73.828125*d*(b*b)*(p*p)*s + 27.0703125*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad667(double a, double b, double p, double d, double s, double g){
	return (0.046875*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.703125*(d*d*d*d*d)*(a*a*a*a)*b*g + 2.34375*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 2.34375*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.703125*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.046875*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.234375*(d*d*d*d)*(a*a*a*a*a)*s - 3.515625*(d*d*d*d)*(a*a*a*a)*b*s + 11.71875*(d*d*d*d)*(a*a*a)*(b*b)*s - 11.71875*(d*d*d*d)*(a*a)*(b*b*b)*s + 3.515625*(d*d*d*d)*a*(b*b*b*b)*s - 0.234375*(d*d*d*d)*(b*b*b*b*b)*s + 2.8125*(d*d*d)*(a*a*a)*g*p - 12.65625*(d*d*d)*(a*a)*b*g*p + 12.65625*(d*d*d)*a*(b*b)*g*p - 2.8125*(d*d*d)*(b*b*b)*g*p + 8.4375*(d*d)*(a*a*a)*p*s - 37.96875*(d*d)*(a*a)*b*p*s + 37.96875*(d*d)*a*(b*b)*p*s - 8.4375*(d*d)*(b*b*b)*p*s + 11.6015625*d*a*g*(p*p) - 11.6015625*d*b*g*(p*p) + 11.6015625*a*(p*p)*s - 11.6015625*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad668(double a, double b, double p, double d, double s, double g){
	return (0.05859375*(d*d*d*d)*(a*a*a*a)*g - 0.46875*(d*d*d*d)*(a*a*a)*b*g + 0.87890625*(d*d*d*d)*(a*a)*(b*b)*g - 0.46875*(d*d*d*d)*a*(b*b*b)*g + 0.05859375*(d*d*d*d)*(b*b*b*b)*g + 0.234375*(d*d*d)*(a*a*a*a)*s - 1.875*(d*d*d)*(a*a*a)*b*s + 3.515625*(d*d*d)*(a*a)*(b*b)*s - 1.875*(d*d*d)*a*(b*b*b)*s + 0.234375*(d*d*d)*(b*b*b*b)*s + 1.318359375*(d*d)*(a*a)*g*p - 3.1640625*(d*d)*a*b*g*p + 1.318359375*(d*d)*(b*b)*g*p + 2.63671875*d*(a*a)*p*s - 6.328125*d*a*b*p*s + 2.63671875*d*(b*b)*p*s + 1.4501953125*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad669(double a, double b, double p, double d, double s, double g){
	return (0.0390625*(d*d*d)*(a*a*a)*g - 0.17578125*(d*d*d)*(a*a)*b*g + 0.17578125*(d*d*d)*a*(b*b)*g - 0.0390625*(d*d*d)*(b*b*b)*g + 0.1171875*(d*d)*(a*a*a)*s - 0.52734375*(d*d)*(a*a)*b*s + 0.52734375*(d*d)*a*(b*b)*s - 0.1171875*(d*d)*(b*b*b)*s + 0.322265625*d*a*g*p - 0.322265625*d*b*g*p + 0.322265625*a*p*s - 0.322265625*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6610(double a, double b, double p, double d, double s, double g){
	return (0.0146484375*(d*d)*(a*a)*g - 0.03515625*(d*d)*a*b*g + 0.0146484375*(d*d)*(b*b)*g + 0.029296875*d*(a*a)*s - 0.0703125*d*a*b*s + 0.029296875*d*(b*b)*s + 0.0322265625*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6611(double a, double b, double p, double d, double s, double g){
	return 0.0029296875*(d*a*g - d*b*g + a*s - b*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6612(double a, double b, double p, double d, double s, double g){
	return 0.000244140625*g/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad670(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g + 13.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g*p - 21.0 *(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 10.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g*p + 82.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*p*s - 231.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 115.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*p*s + 11.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*(p*p) - 105.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 236.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 157.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 26.25*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*(p*p) + 101.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*(p*p)*s - 945.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 2126.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 1417.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 236.25*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*(p*p)*s + 1.875*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p*p) - 78.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p*p) + 590.625*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p) - 1312.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p) + 984.375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p) - 236.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p) + 13.125*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p*p) + 13.125*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p*p)*s - 551.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p*p)*s + 4134.375*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p)*s - 9187.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p)*s + 6890.625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p)*s - 1653.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p)*s + 91.875*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p*p)*s + 137.8125*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p) - 1378.125*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p) + 3445.3125*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p) - 2756.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p) + 689.0625*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p) - 39.375*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p) + 689.0625*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p)*s - 6890.625*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p)*s + 17226.5625*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p)*s - 13781.25*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p)*s + 3445.3125*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p)*s - 196.875*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p)*s + 1033.59375*(d*d*d)*(a*a*a)*g*(p*p*p*p*p) - 3720.9375*(d*d*d)*(a*a)*b*g*(p*p*p*p*p) + 3100.78125*(d*d*d)*a*(b*b)*g*(p*p*p*p*p) - 590.625*(d*d*d)*(b*b*b)*g*(p*p*p*p*p) + 3100.78125*(d*d)*(a*a*a)*(p*p*p*p*p)*s - 11162.8125*(d*d)*(a*a)*b*(p*p*p*p*p)*s + 9302.34375*(d*d)*a*(b*b)*(p*p*p*p*p)*s - 1771.875*(d*d)*(b*b*b)*(p*p*p*p*p)*s + 1136.953125*d*a*g*(p*p*p*p*p*p) - 974.53125*d*b*g*(p*p*p*p*p*p) + 1136.953125*a*(p*p*p*p*p*p)*s - 974.53125*b*(p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad671(double a, double b, double p, double d, double s, double g){
	return (-3.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g + 3.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 36.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*s + 42.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 15.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*p + 78.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*p - 94.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p + 26.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*p - 150.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*p*s + 787.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*p*s - 945.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s + 262.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*p*s - 11.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p) + 196.875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p) - 787.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 984.375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 393.75*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 39.375*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p) - 90.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p)*s + 1575.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p)*s - 6300.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 7875.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 3150.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 315.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p)*s + 45.9375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p) - 826.875*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 3445.3125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 4593.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 2067.1875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 275.625*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 6.5625*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p) + 275.625*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p)*s - 4961.25*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 20671.875*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 27562.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 12403.125*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 1653.75*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 39.375*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p)*s + 1033.59375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 6201.5625*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 9302.34375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 4134.375*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 442.96875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 4134.375*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 24806.25*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 37209.375*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 16537.5*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 1771.875*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 3410.859375*(d*d)*(a*a)*g*(p*p*p*p*p) - 6821.71875*(d*d)*a*b*g*(p*p*p*p*p) + 2436.328125*(d*d)*(b*b)*g*(p*p*p*p*p) + 6821.71875*d*(a*a)*(p*p*p*p*p)*s - 13643.4375*d*a*b*(p*p*p*p*p)*s + 4872.65625*d*(b*b)*(p*p*p*p*p)*s + 1055.7421875*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad672(double a, double b, double p, double d, double s, double g){
	return (3.75*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g - 10.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g + 5.25*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g + 41.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*s - 115.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*s + 57.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*s + 11.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*p - 105.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 236.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 157.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 26.25*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p + 101.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*p*s - 945.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 2126.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 1417.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 236.25*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s + 2.8125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p) - 118.125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 885.9375*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 1968.75*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 1476.5625*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 354.375*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 19.6875*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) + 19.6875*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p)*s - 826.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 6201.5625*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 13781.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 10335.9375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 2480.625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 137.8125*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s + 275.625*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 2756.25*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 6890.625*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 5512.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 1378.125*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 78.75*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 1378.125*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 13781.25*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 34453.125*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 27562.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 6890.625*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 393.75*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 2583.984375*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 9302.34375*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 7751.953125*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 1476.5625*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 7751.953125*(d*d)*(a*a*a)*(p*p*p*p)*s - 27907.03125*(d*d)*(a*a)*b*(p*p*p*p)*s + 23255.859375*(d*d)*a*(b*b)*(p*p*p*p)*s - 4429.6875*(d*d)*(b*b*b)*(p*p*p*p)*s + 3410.859375*d*a*g*(p*p*p*p*p) - 2923.59375*d*b*g*(p*p*p*p*p) + 3410.859375*a*(p*p*p*p*p)*s - 2923.59375*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad673(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g + 13.125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 15.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 4.375*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g - 25.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*s + 131.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 157.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 43.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s - 3.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 65.625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 262.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 328.125*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 131.25*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 13.125*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 30.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 525.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 2100.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 2625.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 1050.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 105.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s + 22.96875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 413.4375*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 1722.65625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 2296.875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 1033.59375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 137.8125*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 3.28125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 137.8125*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 2480.625*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 10335.9375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 13781.25*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 6201.5625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 826.875*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 19.6875*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 689.0625*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 4134.375*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 6201.5625*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 2756.25*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 295.3125*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 2756.25*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 16537.5*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 24806.25*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 11025.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 1181.25*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 2842.3828125*(d*d)*(a*a)*g*(p*p*p*p) - 5684.765625*(d*d)*a*b*g*(p*p*p*p) + 2030.2734375*(d*d)*(b*b)*g*(p*p*p*p) + 5684.765625*d*(a*a)*(p*p*p*p)*s - 11369.53125*d*a*b*(p*p*p*p)*s + 4060.546875*d*(b*b)*(p*p*p*p)*s + 1055.7421875*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad674(double a, double b, double p, double d, double s, double g){
	return (0.9375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 8.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 19.6875*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 13.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 2.1875*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g + 8.4375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s - 78.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 177.1875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 118.125*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 19.6875*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s + 0.46875*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 19.6875*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 147.65625*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 328.125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 246.09375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 59.0625*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 3.28125*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p + 3.28125*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 137.8125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 1033.59375*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 2296.875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 1722.65625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 413.4375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 22.96875*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s + 68.90625*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 689.0625*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 1722.65625*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 1378.125*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 344.53125*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 19.6875*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 344.53125*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 3445.3125*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 8613.28125*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 6890.625*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 1722.65625*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 98.4375*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 861.328125*(d*d*d)*(a*a*a)*g*(p*p*p) - 3100.78125*(d*d*d)*(a*a)*b*g*(p*p*p) + 2583.984375*(d*d*d)*a*(b*b)*g*(p*p*p) - 492.1875*(d*d*d)*(b*b*b)*g*(p*p*p) + 2583.984375*(d*d)*(a*a*a)*(p*p*p)*s - 9302.34375*(d*d)*(a*a)*b*(p*p*p)*s + 7751.953125*(d*d)*a*(b*b)*(p*p*p)*s - 1476.5625*(d*d)*(b*b*b)*(p*p*p)*s + 1421.19140625*d*a*g*(p*p*p*p) - 1218.1640625*d*b*g*(p*p*p*p) + 1421.19140625*a*(p*p*p*p)*s - 1218.1640625*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad675(double a, double b, double p, double d, double s, double g){
	return (-0.1875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 3.28125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 13.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 16.40625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 6.5625*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 0.65625*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 1.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 26.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 105.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 131.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 52.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 5.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s + 2.296875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 41.34375*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 172.265625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 229.6875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 103.359375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 13.78125*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 0.328125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 13.78125*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 248.0625*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 1033.59375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 1378.125*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 620.15625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 82.6875*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 1.96875*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 103.359375*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 620.15625*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 930.234375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 413.4375*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 44.296875*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 413.4375*(d*d*d)*(a*a*a*a)*(p*p)*s - 2480.625*(d*d*d)*(a*a*a)*b*(p*p)*s + 3720.9375*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 1653.75*(d*d*d)*a*(b*b*b)*(p*p)*s + 177.1875*(d*d*d)*(b*b*b*b)*(p*p)*s + 568.4765625*(d*d)*(a*a)*g*(p*p*p) - 1136.953125*(d*d)*a*b*g*(p*p*p) + 406.0546875*(d*d)*(b*b)*g*(p*p*p) + 1136.953125*d*(a*a)*(p*p*p)*s - 2273.90625*d*a*b*(p*p*p)*s + 812.109375*d*(b*b)*(p*p*p)*s + 263.935546875*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad676(double a, double b, double p, double d, double s, double g){
	return (0.015625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 0.65625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 4.921875*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 10.9375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 8.203125*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 1.96875*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.109375*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g + 0.109375*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 4.59375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 34.453125*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 76.5625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 57.421875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 13.78125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 0.765625*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s + 4.59375*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 45.9375*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 114.84375*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 91.875*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 22.96875*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 1.3125*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 22.96875*(d*d*d*d)*(a*a*a*a*a)*p*s - 229.6875*(d*d*d*d)*(a*a*a*a)*b*p*s + 574.21875*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 459.375*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 114.84375*(d*d*d*d)*a*(b*b*b*b)*p*s - 6.5625*(d*d*d*d)*(b*b*b*b*b)*p*s + 86.1328125*(d*d*d)*(a*a*a)*g*(p*p) - 310.078125*(d*d*d)*(a*a)*b*g*(p*p) + 258.3984375*(d*d*d)*a*(b*b)*g*(p*p) - 49.21875*(d*d*d)*(b*b*b)*g*(p*p) + 258.3984375*(d*d)*(a*a*a)*(p*p)*s - 930.234375*(d*d)*(a*a)*b*(p*p)*s + 775.1953125*(d*d)*a*(b*b)*(p*p)*s - 147.65625*(d*d)*(b*b*b)*(p*p)*s + 189.4921875*d*a*g*(p*p*p) - 162.421875*d*b*g*(p*p*p) + 189.4921875*a*(p*p*p)*s - 162.421875*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad677(double a, double b, double p, double d, double s, double g){
	return (0.0546875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 0.984375*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 4.1015625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 5.46875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 2.4609375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.328125*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.0078125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 0.328125*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 5.90625*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 24.609375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 32.8125*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 14.765625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 1.96875*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.046875*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 4.921875*(d*d*d*d)*(a*a*a*a)*g*p - 29.53125*(d*d*d*d)*(a*a*a)*b*g*p + 44.296875*(d*d*d*d)*(a*a)*(b*b)*g*p - 19.6875*(d*d*d*d)*a*(b*b*b)*g*p + 2.109375*(d*d*d*d)*(b*b*b*b)*g*p + 19.6875*(d*d*d)*(a*a*a*a)*p*s - 118.125*(d*d*d)*(a*a*a)*b*p*s + 177.1875*(d*d*d)*(a*a)*(b*b)*p*s - 78.75*(d*d*d)*a*(b*b*b)*p*s + 8.4375*(d*d*d)*(b*b*b*b)*p*s + 40.60546875*(d*d)*(a*a)*g*(p*p) - 81.2109375*(d*d)*a*b*g*(p*p) + 29.00390625*(d*d)*(b*b)*g*(p*p) + 81.2109375*d*(a*a)*(p*p)*s - 162.421875*d*a*b*(p*p)*s + 58.0078125*d*(b*b)*(p*p)*s + 25.13671875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad678(double a, double b, double p, double d, double s, double g){
	return (0.08203125*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.8203125*(d*d*d*d*d)*(a*a*a*a)*b*g + 2.05078125*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 1.640625*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.41015625*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.0234375*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.41015625*(d*d*d*d)*(a*a*a*a*a)*s - 4.1015625*(d*d*d*d)*(a*a*a*a)*b*s + 10.25390625*(d*d*d*d)*(a*a*a)*(b*b)*s - 8.203125*(d*d*d*d)*(a*a)*(b*b*b)*s + 2.05078125*(d*d*d*d)*a*(b*b*b*b)*s - 0.1171875*(d*d*d*d)*(b*b*b*b*b)*s + 3.076171875*(d*d*d)*(a*a*a)*g*p - 11.07421875*(d*d*d)*(a*a)*b*g*p + 9.228515625*(d*d*d)*a*(b*b)*g*p - 1.7578125*(d*d*d)*(b*b*b)*g*p + 9.228515625*(d*d)*(a*a*a)*p*s - 33.22265625*(d*d)*(a*a)*b*p*s + 27.685546875*(d*d)*a*(b*b)*p*s - 5.2734375*(d*d)*(b*b*b)*p*s + 10.1513671875*d*a*g*(p*p) - 8.701171875*d*b*g*(p*p) + 10.1513671875*a*(p*p)*s - 8.701171875*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad679(double a, double b, double p, double d, double s, double g){
	return (0.068359375*(d*d*d*d)*(a*a*a*a)*g - 0.41015625*(d*d*d*d)*(a*a*a)*b*g + 0.615234375*(d*d*d*d)*(a*a)*(b*b)*g - 0.2734375*(d*d*d*d)*a*(b*b*b)*g + 0.029296875*(d*d*d*d)*(b*b*b*b)*g + 0.2734375*(d*d*d)*(a*a*a*a)*s - 1.640625*(d*d*d)*(a*a*a)*b*s + 2.4609375*(d*d*d)*(a*a)*(b*b)*s - 1.09375*(d*d*d)*a*(b*b*b)*s + 0.1171875*(d*d*d)*(b*b*b*b)*s + 1.1279296875*(d*d)*(a*a)*g*p - 2.255859375*(d*d)*a*b*g*p + 0.8056640625*(d*d)*(b*b)*g*p + 2.255859375*d*(a*a)*p*s - 4.51171875*d*a*b*p*s + 1.611328125*d*(b*b)*p*s + 1.04736328125*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6710(double a, double b, double p, double d, double s, double g){
	return (0.0341796875*(d*d*d)*(a*a*a)*g - 0.123046875*(d*d*d)*(a*a)*b*g + 0.1025390625*(d*d*d)*a*(b*b)*g - 0.01953125*(d*d*d)*(b*b*b)*g + 0.1025390625*(d*d)*(a*a*a)*s - 0.369140625*(d*d)*(a*a)*b*s + 0.3076171875*(d*d)*a*(b*b)*s - 0.05859375*(d*d)*(b*b*b)*s + 0.2255859375*d*a*g*p - 0.193359375*d*b*g*p + 0.2255859375*a*p*s - 0.193359375*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6711(double a, double b, double p, double d, double s, double g){
	return (0.01025390625*(d*d)*(a*a)*g - 0.0205078125*(d*d)*a*b*g + 0.00732421875*(d*d)*(b*b)*g + 0.0205078125*d*(a*a)*s - 0.041015625*d*a*b*s + 0.0146484375*d*(b*b)*s + 0.01904296875*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6712(double a, double b, double p, double d, double s, double g){
	return (0.001708984375*d*a*g - 0.00146484375*d*b*g + 0.001708984375*a*s - 0.00146484375*b*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6713(double a, double b, double p, double d, double s, double g){
	return 0.0001220703125*g/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad680(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g + 14.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*s + 7.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*g*p - 24.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 14.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g*p + 90.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*p*s - 288.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 168.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*p*s + 11.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g*(p*p) - 120.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 315.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 252.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 52.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*(p*p) + 112.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*(p*p)*s - 1200.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 3150.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 2520.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 525.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*(p*p)*s + 1.875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*(p*p*p) - 90.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p*p) + 787.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p*p) - 2100.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p*p) + 1968.75*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p*p) - 630.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p*p) + 52.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p*p) + 15.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(p*p*p)*s - 720.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p*p)*s + 6300.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p*p)*s - 16800.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p*p)*s + 15750.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p*p)*s - 5040.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p*p)*s + 420.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p*p)*s + 183.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p*p) - 2205.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p*p) + 6890.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p*p) - 7350.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p*p) + 2756.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p*p) - 315.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p*p) + 6.5625*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p*p) + 1102.5*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p*p)*s - 13230.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p*p)*s + 41343.75*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p*p)*s - 44100.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p*p)*s + 16537.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p*p)*s - 1890.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p*p)*s + 39.375*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p*p)*s + 2067.1875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p*p) - 9922.5*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p*p) + 12403.125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p*p) - 4725.0*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p*p) + 442.96875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p*p) + 8268.75*(d*d*d)*(a*a*a*a)*(p*p*p*p*p)*s - 39690.0*(d*d*d)*(a*a*a)*b*(p*p*p*p*p)*s + 49612.5*(d*d*d)*(a*a)*(b*b)*(p*p*p*p*p)*s - 18900.0*(d*d*d)*a*(b*b*b)*(p*p*p*p*p)*s + 1771.875*(d*d*d)*(b*b*b*b)*(p*p*p*p*p)*s + 4547.8125*(d*d)*(a*a)*g*(p*p*p*p*p*p) - 7796.25*(d*d)*a*b*g*(p*p*p*p*p*p) + 2436.328125*(d*d)*(b*b)*g*(p*p*p*p*p*p) + 9095.625*d*(a*a)*(p*p*p*p*p*p)*s - 15592.5*d*a*b*(p*p*p*p*p*p)*s + 4872.65625*d*(b*b)*(p*p*p*p*p*p)*s + 1055.7421875*g*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad681(double a, double b, double p, double d, double s, double g){
	return (-3.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*g + 4.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 39.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*s + 52.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 15.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*g*p + 90.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g*p - 126.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 42.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 165.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*p*s + 990.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*p*s - 1386.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 462.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 11.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g*(p*p) + 225.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*(p*p) - 1050.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 1575.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 787.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 105.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 101.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*(p*p)*s + 2025.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*(p*p)*s - 9450.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 14175.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 7087.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 945.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*(p*p)*s + 52.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p*p) - 1102.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p*p) + 5512.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p) - 9187.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p) + 5512.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p) - 1102.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p) + 52.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p*p) + 367.5*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p*p)*s - 7717.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p*p)*s + 38587.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p)*s - 64312.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p)*s + 38587.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p)*s - 7717.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p)*s + 367.5*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p*p)*s + 1653.75*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p) - 12403.125*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p) + 24806.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p) - 16537.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p) + 3543.75*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p) - 177.1875*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p) + 8268.75*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p)*s - 62015.625*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p)*s + 124031.25*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p)*s - 82687.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p)*s + 17718.75*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p)*s - 885.9375*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p)*s + 9095.625*(d*d*d)*(a*a*a)*g*(p*p*p*p*p) - 27286.875*(d*d*d)*(a*a)*b*g*(p*p*p*p*p) + 19490.625*(d*d*d)*a*(b*b)*g*(p*p*p*p*p) - 3248.4375*(d*d*d)*(b*b*b)*g*(p*p*p*p*p) + 27286.875*(d*d)*(a*a*a)*(p*p*p*p*p)*s - 81860.625*(d*d)*(a*a)*b*(p*p*p*p*p)*s + 58471.875*(d*d)*a*(b*b)*(p*p*p*p*p)*s - 9745.3125*(d*d)*(b*b*b)*(p*p*p*p*p)*s + 8445.9375*d*a*g*(p*p*p*p*p*p) - 6334.453125*d*b*g*(p*p*p*p*p*p) + 8445.9375*a*(p*p*p*p*p*p)*s - 6334.453125*b*(p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad682(double a, double b, double p, double d, double s, double g){
	return (3.75*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*g - 12.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g + 7.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g + 45.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*s - 144.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*s + 84.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*s + 11.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g*p - 120.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*p + 315.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*p - 252.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p + 52.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*p + 112.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*p*s - 1200.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*p*s + 3150.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*p*s - 2520.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s + 525.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*p*s + 2.8125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*(p*p) - 135.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p) + 1181.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p) - 3150.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 2953.125*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 945.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 78.75*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p) + 22.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(p*p)*s - 1080.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p)*s + 9450.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p)*s - 25200.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 23625.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 7560.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 630.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p)*s + 367.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p) - 4410.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 13781.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 14700.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 5512.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 630.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 13.125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p) + 2205.0*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p)*s - 26460.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 82687.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 88200.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 33075.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 3780.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 78.75*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p)*s + 5167.96875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 24806.25*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 31007.8125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 11812.5*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 1107.421875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 20671.875*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 99225.0*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 124031.25*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 47250.0*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 4429.6875*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 13643.4375*(d*d)*(a*a)*g*(p*p*p*p*p) - 23388.75*(d*d)*a*b*g*(p*p*p*p*p) + 7308.984375*(d*d)*(b*b)*g*(p*p*p*p*p) + 27286.875*d*(a*a)*(p*p*p*p*p)*s - 46777.5*d*a*b*(p*p*p*p*p)*s + 14617.96875*d*(b*b)*(p*p*p*p*p)*s + 3695.09765625*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad683(double a, double b, double p, double d, double s, double g){
	return (-2.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*g + 15.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g - 21.0 *(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g + 7.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g - 27.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*s + 165.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*s - 231.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*s + 77.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*s - 3.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g*p + 75.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*p - 350.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 525.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 262.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 35.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 33.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*p*s + 675.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*p*s - 3150.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 4725.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 2362.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 315.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s + 26.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p) - 551.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 2756.25*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 4593.75*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 2756.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 551.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 26.25*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) + 183.75*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p)*s - 3858.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 19293.75*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 32156.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 19293.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 3858.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 183.75*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s + 1102.5*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 8268.75*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 16537.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 11025.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 2362.5*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 118.125*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 5512.5*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 41343.75*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 82687.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 55125.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 11812.5*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 590.625*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 7579.6875*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 22739.0625*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 16242.1875*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 2707.03125*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 22739.0625*(d*d)*(a*a*a)*(p*p*p*p)*s - 68217.1875*(d*d)*(a*a)*b*(p*p*p*p)*s + 48726.5625*(d*d)*a*(b*b)*(p*p*p*p)*s - 8121.09375*(d*d)*(b*b*b)*(p*p*p*p)*s + 8445.9375*d*a*g*(p*p*p*p*p) - 6334.453125*d*b*g*(p*p*p*p*p) + 8445.9375*a*(p*p*p*p*p)*s - 6334.453125*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad684(double a, double b, double p, double d, double s, double g){
	return (0.9375*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g - 10.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g + 26.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 21.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 4.375*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g + 9.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*s - 100.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*s + 262.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 210.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 43.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s + 0.46875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*p - 22.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 196.875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 525.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 492.1875*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 157.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 13.125*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p + 3.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*p*s - 180.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 1575.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 4200.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 3937.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 1260.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 105.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s + 91.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 1102.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 3445.3125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 3675.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 1378.125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 157.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 3.28125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 551.25*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 6615.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 20671.875*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 22050.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 8268.75*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 945.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 19.6875*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 1722.65625*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 8268.75*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 10335.9375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 3937.5*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 369.140625*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 6890.625*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 33075.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 41343.75*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 15750.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 1476.5625*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 5684.765625*(d*d)*(a*a)*g*(p*p*p*p) - 9745.3125*(d*d)*a*b*g*(p*p*p*p) + 3045.41015625*(d*d)*(b*b)*g*(p*p*p*p) + 11369.53125*d*(a*a)*(p*p*p*p)*s - 19490.625*d*a*b*(p*p*p*p)*s + 6090.8203125*d*(b*b)*(p*p*p*p)*s + 1847.548828125*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad685(double a, double b, double p, double d, double s, double g){
	return (-0.1875*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g + 3.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 17.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 26.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 13.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 1.75*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 1.6875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*s + 33.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s - 157.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 236.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 118.125*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 15.75*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s + 2.625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 55.125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 275.625*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 459.375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 275.625*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 55.125*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 2.625*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p + 18.375*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 385.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 1929.375*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 3215.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 1929.375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 385.875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 18.375*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s + 165.375*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 1240.3125*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 2480.625*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 1653.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 354.375*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 17.71875*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 826.875*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 6201.5625*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 12403.125*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 8268.75*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 1771.875*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 88.59375*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 1515.9375*(d*d*d)*(a*a*a)*g*(p*p*p) - 4547.8125*(d*d*d)*(a*a)*b*g*(p*p*p) + 3248.4375*(d*d*d)*a*(b*b)*g*(p*p*p) - 541.40625*(d*d*d)*(b*b*b)*g*(p*p*p) + 4547.8125*(d*d)*(a*a*a)*(p*p*p)*s - 13643.4375*(d*d)*(a*a)*b*(p*p*p)*s + 9745.3125*(d*d)*a*(b*b)*(p*p*p)*s - 1624.21875*(d*d)*(b*b*b)*(p*p*p)*s + 2111.484375*d*a*g*(p*p*p*p) - 1583.61328125*d*b*g*(p*p*p*p) + 2111.484375*a*(p*p*p*p)*s - 1583.61328125*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad686(double a, double b, double p, double d, double s, double g){
	return (0.015625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g - 0.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 6.5625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 17.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 16.40625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 5.25*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 0.4375*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g + 0.125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*s - 6.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 52.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 140.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 131.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 42.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 3.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s + 6.125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 73.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 229.6875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 245.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 91.875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 10.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 0.21875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 36.75*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 441.0 *(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 1378.125*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 1470.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 551.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 63.0*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 1.3125*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 172.265625*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 826.875*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 1033.59375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 393.75*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 36.9140625*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 689.0625*(d*d*d)*(a*a*a*a)*(p*p)*s - 3307.5*(d*d*d)*(a*a*a)*b*(p*p)*s + 4134.375*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 1575.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 147.65625*(d*d*d)*(b*b*b*b)*(p*p)*s + 757.96875*(d*d)*(a*a)*g*(p*p*p) - 1299.375*(d*d)*a*b*g*(p*p*p) + 406.0546875*(d*d)*(b*b)*g*(p*p*p) + 1515.9375*d*(a*a)*(p*p*p)*s - 2598.75*d*a*b*(p*p*p)*s + 812.109375*d*(b*b)*(p*p*p)*s + 307.9248046875*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad687(double a, double b, double p, double d, double s, double g){
	return (0.0625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 1.3125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 6.5625*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 10.9375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 6.5625*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 1.3125*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.0625*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g + 0.4375*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 9.1875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 45.9375*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 76.5625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 45.9375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 9.1875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 0.4375*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s + 7.875*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 59.0625*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 118.125*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 78.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 16.875*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 0.84375*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 39.375*(d*d*d*d)*(a*a*a*a*a)*p*s - 295.3125*(d*d*d*d)*(a*a*a*a)*b*p*s + 590.625*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 393.75*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 84.375*(d*d*d*d)*a*(b*b*b*b)*p*s - 4.21875*(d*d*d*d)*(b*b*b*b*b)*p*s + 108.28125*(d*d*d)*(a*a*a)*g*(p*p) - 324.84375*(d*d*d)*(a*a)*b*g*(p*p) + 232.03125*(d*d*d)*a*(b*b)*g*(p*p) - 38.671875*(d*d*d)*(b*b*b)*g*(p*p) + 324.84375*(d*d)*(a*a*a)*(p*p)*s - 974.53125*(d*d)*(a*a)*b*(p*p)*s + 696.09375*(d*d)*a*(b*b)*(p*p)*s - 116.015625*(d*d)*(b*b*b)*(p*p)*s + 201.09375*d*a*g*(p*p*p) - 150.8203125*d*b*g*(p*p*p) + 201.09375*a*(p*p*p)*s - 150.8203125*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad688(double a, double b, double p, double d, double s, double g){
	return (0.109375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 1.3125*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 4.1015625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 4.375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 1.640625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.1875*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.00390625*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 0.65625*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 7.875*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 24.609375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 26.25*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 9.84375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 1.125*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.0234375*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 6.15234375*(d*d*d*d)*(a*a*a*a)*g*p - 29.53125*(d*d*d*d)*(a*a*a)*b*g*p + 36.9140625*(d*d*d*d)*(a*a)*(b*b)*g*p - 14.0625*(d*d*d*d)*a*(b*b*b)*g*p + 1.318359375*(d*d*d*d)*(b*b*b*b)*g*p + 24.609375*(d*d*d)*(a*a*a*a)*p*s - 118.125*(d*d*d)*(a*a*a)*b*p*s + 147.65625*(d*d*d)*(a*a)*(b*b)*p*s - 56.25*(d*d*d)*a*(b*b*b)*p*s + 5.2734375*(d*d*d)*(b*b*b*b)*p*s + 40.60546875*(d*d)*(a*a)*g*(p*p) - 69.609375*(d*d)*a*b*g*(p*p) + 21.7529296875*(d*d)*(b*b)*g*(p*p) + 81.2109375*d*(a*a)*(p*p)*s - 139.21875*d*a*b*(p*p)*s + 43.505859375*d*(b*b)*(p*p)*s + 21.99462890625*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad689(double a, double b, double p, double d, double s, double g){
	return (0.109375*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.8203125*(d*d*d*d*d)*(a*a*a*a)*b*g + 1.640625*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 1.09375*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.234375*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.01171875*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.546875*(d*d*d*d)*(a*a*a*a*a)*s - 4.1015625*(d*d*d*d)*(a*a*a*a)*b*s + 8.203125*(d*d*d*d)*(a*a*a)*(b*b)*s - 5.46875*(d*d*d*d)*(a*a)*(b*b*b)*s + 1.171875*(d*d*d*d)*a*(b*b*b*b)*s - 0.05859375*(d*d*d*d)*(b*b*b*b*b)*s + 3.0078125*(d*d*d)*(a*a*a)*g*p - 9.0234375*(d*d*d)*(a*a)*b*g*p + 6.4453125*(d*d*d)*a*(b*b)*g*p - 1.07421875*(d*d*d)*(b*b*b)*g*p + 9.0234375*(d*d)*(a*a*a)*p*s - 27.0703125*(d*d)*(a*a)*b*p*s + 19.3359375*(d*d)*a*(b*b)*p*s - 3.22265625*(d*d)*(b*b*b)*p*s + 8.37890625*d*a*g*(p*p) - 6.2841796875*d*b*g*(p*p) + 8.37890625*a*(p*p)*s - 6.2841796875*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6810(double a, double b, double p, double d, double s, double g){
	return (0.068359375*(d*d*d*d)*(a*a*a*a)*g - 0.328125*(d*d*d*d)*(a*a*a)*b*g + 0.41015625*(d*d*d*d)*(a*a)*(b*b)*g - 0.15625*(d*d*d*d)*a*(b*b*b)*g + 0.0146484375*(d*d*d*d)*(b*b*b*b)*g + 0.2734375*(d*d*d)*(a*a*a*a)*s - 1.3125*(d*d*d)*(a*a*a)*b*s + 1.640625*(d*d*d)*(a*a)*(b*b)*s - 0.625*(d*d*d)*a*(b*b*b)*s + 0.05859375*(d*d*d)*(b*b*b*b)*s + 0.90234375*(d*d)*(a*a)*g*p - 1.546875*(d*d)*a*b*g*p + 0.4833984375*(d*d)*(b*b)*g*p + 1.8046875*d*(a*a)*p*s - 3.09375*d*a*b*p*s + 0.966796875*d*(b*b)*p*s + 0.733154296875*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6811(double a, double b, double p, double d, double s, double g){
	return (0.02734375*(d*d*d)*(a*a*a)*g - 0.08203125*(d*d*d)*(a*a)*b*g + 0.05859375*(d*d*d)*a*(b*b)*g - 0.009765625*(d*d*d)*(b*b*b)*g + 0.08203125*(d*d)*(a*a*a)*s - 0.24609375*(d*d)*(a*a)*b*s + 0.17578125*(d*d)*a*(b*b)*s - 0.029296875*(d*d)*(b*b*b)*s + 0.15234375*d*a*g*p - 0.1142578125*d*b*g*p + 0.15234375*a*p*s - 0.1142578125*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6812(double a, double b, double p, double d, double s, double g){
	return (0.0068359375*(d*d)*(a*a)*g - 0.01171875*(d*d)*a*b*g + 0.003662109375*(d*d)*(b*b)*g + 0.013671875*d*(a*a)*s - 0.0234375*d*a*b*s + 0.00732421875*d*(b*b)*s + 0.0111083984375*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6813(double a, double b, double p, double d, double s, double g){
	return (0.0009765625*d*a*g - 0.000732421875*d*b*g + 0.0009765625*a*s - 0.000732421875*b*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad6814(double a, double b, double p, double d, double s, double g){
	return 6.103515625e-5*g/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad700(double a, double b, double p, double d, double s, double g){
	return b*(-(d*d*d*d*d*d*d)*(b*b*b*b*b*b)*g - 7.0*(d*d*d*d*d*d)*(b*b*b*b*b*b)*s - 10.5*(d*d*d*d*d)*(b*b*b*b)*g*p - 52.5*(d*d*d*d)*(b*b*b*b)*p*s - 26.25*(d*d*d)*(b*b)*g*(p*p) - 78.75*(d*d)*(b*b)*(p*p)*s - 13.125*d*g*(p*p*p) - 13.125*(p*p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad701(double a, double b, double p, double d, double s, double g){
	return (3.5*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 21.0 *(d*d*d*d*d)*(b*b*b*b*b*b)*s + 26.25*(d*d*d*d)*(b*b*b*b)*g*p + 105.0*(d*d*d)*(b*b*b*b)*p*s + 39.375*(d*d)*(b*b)*g*(p*p) + 78.75*d*(b*b)*(p*p)*s + 6.5625*g*(p*p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad702(double a, double b, double p, double d, double s, double g){
	return b*(-5.25*(d*d*d*d*d)*(b*b*b*b)*g - 26.25*(d*d*d*d)*(b*b*b*b)*s - 26.25*(d*d*d)*(b*b)*g*p - 78.75*(d*d)*(b*b)*p*s - 19.6875*d*g*(p*p) - 19.6875*(p*p)*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad703(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d)*(b*b*b*b)*g + 17.5*(d*d*d)*(b*b*b*b)*s + 13.125*(d*d)*(b*b)*g*p + 26.25*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad704(double a, double b, double p, double d, double s, double g){
	return b*(-2.1875*(d*d*d)*(b*b)*g - 6.5625*(d*d)*(b*b)*s - 3.28125*d*g*p - 3.28125*p*s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad705(double a, double b, double p, double d, double s, double g){
	return (0.65625*d*(b*b)*(d*g + 2*s) + 0.328125*g*p)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad706(double a, double b, double p, double d, double s, double g){
	return 0.109375*b*(-d*g - s)/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad707(double a, double b, double p, double d, double s, double g){
	return 0.0078125*g/(p*p*p*p*p*p*p);
}

inline double MD_Et_grad710(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g - 8.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s - 10.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 3.5*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p - 63.0*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 21.0 *(d*d*d*d*d)*(b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 26.25*(d*d*d*d)*(b*b*b*b)*g*(p*p) - 105.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 105.0*(d*d*d)*(b*b*b*b)*(p*p)*s - 13.125*(d*d)*a*b*g*(p*p*p) + 39.375*(d*d)*(b*b)*g*(p*p*p) - 26.25*d*a*b*(p*p*p)*s + 78.75*d*(b*b)*(p*p*p)*s + 6.5625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad711(double a, double b, double p, double d, double s, double g){
	return (3.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.5*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g + 24.5*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 3.5*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s + 26.25*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 15.75*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 131.25*(d*d*d*d)*a*(b*b*b*b)*p*s - 78.75*(d*d*d*d)*(b*b*b*b*b)*p*s + 39.375*(d*d*d)*a*(b*b)*g*(p*p) - 65.625*(d*d*d)*(b*b*b)*g*(p*p) + 118.125*(d*d)*a*(b*b)*(p*p)*s - 196.875*(d*d)*(b*b*b)*(p*p)*s + 6.5625*d*a*g*(p*p*p) - 45.9375*d*b*g*(p*p*p) + 6.5625*a*(p*p*p)*s - 45.9375*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad712(double a, double b, double p, double d, double s, double g){
	return (-5.25*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 1.75*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g - 31.5*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 10.5*(d*d*d*d*d)*(b*b*b*b*b*b)*s - 26.25*(d*d*d*d)*a*(b*b*b)*g*p + 26.25*(d*d*d*d)*(b*b*b*b)*g*p - 105.0*(d*d*d)*a*(b*b*b)*p*s + 105.0*(d*d*d)*(b*b*b*b)*p*s - 19.6875*(d*d)*a*b*g*(p*p) + 59.0625*(d*d)*(b*b)*g*(p*p) - 39.375*d*a*b*(p*p)*s + 118.125*d*(b*b)*(p*p)*s + 13.125*g*(p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad713(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d)*a*(b*b*b*b)*g - 2.625*(d*d*d*d*d)*(b*b*b*b*b)*g + 21.875*(d*d*d*d)*a*(b*b*b*b)*s - 13.125*(d*d*d*d)*(b*b*b*b*b)*s + 13.125*(d*d*d)*a*(b*b)*g*p - 21.875*(d*d*d)*(b*b*b)*g*p + 39.375*(d*d)*a*(b*b)*p*s - 65.625*(d*d)*(b*b*b)*p*s + 3.28125*d*a*g*(p*p) - 22.96875*d*b*g*(p*p) + 3.28125*a*(p*p)*s - 22.96875*b*(p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad714(double a, double b, double p, double d, double s, double g){
	return (-2.1875*(d*d*d*d)*a*(b*b*b)*g + 2.1875*(d*d*d*d)*(b*b*b*b)*g - 8.75*(d*d*d)*a*(b*b*b)*s + 8.75*(d*d*d)*(b*b*b*b)*s - 3.28125*(d*d)*a*b*g*p + 9.84375*(d*d)*(b*b)*g*p - 6.5625*d*a*b*p*s + 19.6875*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad715(double a, double b, double p, double d, double s, double g){
	return (0.65625*(d*d*d)*a*(b*b)*g - 1.09375*(d*d*d)*(b*b*b)*g + 1.96875*(d*d)*a*(b*b)*s - 3.28125*(d*d)*(b*b*b)*s + 0.328125*d*a*g*p - 2.296875*d*b*g*p + 0.328125*a*p*s - 2.296875*b*p*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad716(double a, double b, double p, double d, double s, double g){
	return (-0.109375*(d*d)*a*b*g + 0.328125*(d*d)*(b*b)*g - 0.21875*d*a*b*s + 0.65625*d*(b*b)*s + 0.21875*g*p)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad717(double a, double b, double p, double d, double s, double g){
	return (0.0078125*d*a*g - 0.0546875*d*b*g + 0.0078125*a*s - 0.0546875*b*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad718(double a, double b, double p, double d, double s, double g){
	return 0.00390625*g/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad720(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g - 9.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s - 10.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 7.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 0.5*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p - 73.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 49.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 3.5*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 52.5*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 15.75*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) - 131.25*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 262.5*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 78.75*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s - 13.125*(d*d*d)*(a*a)*b*g*(p*p*p) + 78.75*(d*d*d)*a*(b*b)*g*(p*p*p) - 65.625*(d*d*d)*(b*b*b)*g*(p*p*p) - 39.375*(d*d)*(a*a)*b*(p*p*p)*s + 236.25*(d*d)*a*(b*b)*(p*p*p)*s - 196.875*(d*d)*(b*b*b)*(p*p*p)*s + 13.125*d*a*g*(p*p*p*p) - 45.9375*d*b*g*(p*p*p*p) + 13.125*a*(p*p*p*p)*s - 45.9375*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad721(double a, double b, double p, double d, double s, double g){
	return (3.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - (d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g + 28.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 8.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + 26.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 31.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 5.25*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 157.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 189.0*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 31.5*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 39.375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 131.25*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 65.625*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 157.5*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 525.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 262.5*(d*d*d)*(b*b*b*b)*(p*p)*s + 6.5625*(d*d)*(a*a)*g*(p*p*p) - 91.875*(d*d)*a*b*g*(p*p*p) + 137.8125*(d*d)*(b*b)*g*(p*p*p) + 13.125*d*(a*a)*(p*p*p)*s - 183.75*d*a*b*(p*p*p)*s + 275.625*d*(b*b)*(p*p*p)*s + 29.53125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad722(double a, double b, double p, double d, double s, double g){
	return (-5.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 3.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.25*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g - 36.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 24.5*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 1.75*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s - 26.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 52.5*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 15.75*(d*d*d*d*d)*(b*b*b*b*b)*g*p - 131.25*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 262.5*(d*d*d*d)*a*(b*b*b*b)*p*s - 78.75*(d*d*d*d)*(b*b*b*b*b)*p*s - 19.6875*(d*d*d)*(a*a)*b*g*(p*p) + 118.125*(d*d*d)*a*(b*b)*g*(p*p) - 98.4375*(d*d*d)*(b*b*b)*g*(p*p) - 59.0625*(d*d)*(a*a)*b*(p*p)*s + 354.375*(d*d)*a*(b*b)*(p*p)*s - 295.3125*(d*d)*(b*b*b)*(p*p)*s + 26.25*d*a*g*(p*p*p) - 91.875*d*b*g*(p*p*p) + 26.25*a*(p*p*p)*s - 91.875*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad723(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 5.25*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 26.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 31.5*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 5.25*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 13.125*(d*d*d*d)*(a*a)*(b*b)*g*p - 43.75*(d*d*d*d)*a*(b*b*b)*g*p + 21.875*(d*d*d*d)*(b*b*b*b)*g*p + 52.5*(d*d*d)*(a*a)*(b*b)*p*s - 175.0*(d*d*d)*a*(b*b*b)*p*s + 87.5*(d*d*d)*(b*b*b*b)*p*s + 3.28125*(d*d)*(a*a)*g*(p*p) - 45.9375*(d*d)*a*b*g*(p*p) + 68.90625*(d*d)*(b*b)*g*(p*p) + 6.5625*d*(a*a)*(p*p)*s - 91.875*d*a*b*(p*p)*s + 137.8125*d*(b*b)*(p*p)*s + 19.6875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad724(double a, double b, double p, double d, double s, double g){
	return (-2.1875*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 4.375*(d*d*d*d*d)*a*(b*b*b*b)*g - 1.3125*(d*d*d*d*d)*(b*b*b*b*b)*g - 10.9375*(d*d*d*d)*(a*a)*(b*b*b)*s + 21.875*(d*d*d*d)*a*(b*b*b*b)*s - 6.5625*(d*d*d*d)*(b*b*b*b*b)*s - 3.28125*(d*d*d)*(a*a)*b*g*p + 19.6875*(d*d*d)*a*(b*b)*g*p - 16.40625*(d*d*d)*(b*b*b)*g*p - 9.84375*(d*d)*(a*a)*b*p*s + 59.0625*(d*d)*a*(b*b)*p*s - 49.21875*(d*d)*(b*b*b)*p*s + 6.5625*d*a*g*(p*p) - 22.96875*d*b*g*(p*p) + 6.5625*a*(p*p)*s - 22.96875*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad725(double a, double b, double p, double d, double s, double g){
	return (0.65625*(d*d*d*d)*(a*a)*(b*b)*g - 2.1875*(d*d*d*d)*a*(b*b*b)*g + 1.09375*(d*d*d*d)*(b*b*b*b)*g + 2.625*(d*d*d)*(a*a)*(b*b)*s - 8.75*(d*d*d)*a*(b*b*b)*s + 4.375*(d*d*d)*(b*b*b*b)*s + 0.328125*(d*d)*(a*a)*g*p - 4.59375*(d*d)*a*b*g*p + 6.890625*(d*d)*(b*b)*g*p + 0.65625*d*(a*a)*p*s - 9.1875*d*a*b*p*s + 13.78125*d*(b*b)*p*s + 2.953125*g*(p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad726(double a, double b, double p, double d, double s, double g){
	return (-0.109375*(d*d*d)*(a*a)*b*g + 0.65625*(d*d*d)*a*(b*b)*g - 0.546875*(d*d*d)*(b*b*b)*g - 0.328125*(d*d)*(a*a)*b*s + 1.96875*(d*d)*a*(b*b)*s - 1.640625*(d*d)*(b*b*b)*s + 0.4375*d*a*g*p - 1.53125*d*b*g*p + 0.4375*a*p*s - 1.53125*b*p*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad727(double a, double b, double p, double d, double s, double g){
	return (0.0078125*(d*d)*(a*a)*g - 0.109375*(d*d)*a*b*g + 0.1640625*(d*d)*(b*b)*g + 0.015625*d*(a*a)*s - 0.21875*d*a*b*s + 0.328125*d*(b*b)*s + 0.140625*g*p)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad728(double a, double b, double p, double d, double s, double g){
	return (0.0078125*d*a*g - 0.02734375*d*b*g + 0.0078125*a*s - 0.02734375*b*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad729(double a, double b, double p, double d, double s, double g){
	return 0.001953125*g/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad730(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g - 10.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*s - 10.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 10.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 1.5*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p - 84.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 84.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 12.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 78.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 47.25*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 5.25*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) - 157.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 472.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 283.5*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 31.5*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s - 13.125*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 118.125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 196.875*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 65.625*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) - 52.5*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 472.5*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 787.5*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 262.5*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 19.6875*(d*d)*(a*a)*g*(p*p*p*p) - 137.8125*(d*d)*a*b*g*(p*p*p*p) + 137.8125*(d*d)*(b*b)*g*(p*p*p*p) + 39.375*d*(a*a)*(p*p*p*p)*s - 275.625*d*a*b*(p*p*p*p)*s + 275.625*d*(b*b)*(p*p*p*p)*s + 29.53125*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad731(double a, double b, double p, double d, double s, double g){
	return (3.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 1.5*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g + 31.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 13.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s + 26.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 47.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 15.75*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 0.75*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p + 183.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 330.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 110.25*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 5.25*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s + 39.375*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 196.875*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 196.875*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 39.375*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 196.875*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 984.375*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 984.375*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 196.875*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 6.5625*(d*d*d)*(a*a*a)*g*(p*p*p) - 137.8125*(d*d*d)*(a*a)*b*g*(p*p*p) + 413.4375*(d*d*d)*a*(b*b)*g*(p*p*p) - 229.6875*(d*d*d)*(b*b*b)*g*(p*p*p) + 19.6875*(d*d)*(a*a*a)*(p*p*p)*s - 413.4375*(d*d)*(a*a)*b*(p*p*p)*s + 1240.3125*(d*d)*a*(b*b)*(p*p*p)*s - 689.0625*(d*d)*(b*b*b)*(p*p*p)*s + 88.59375*d*a*g*(p*p*p*p) - 206.71875*d*b*g*(p*p*p*p) + 88.59375*a*(p*p*p*p)*s - 206.71875*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad732(double a, double b, double p, double d, double s, double g){
	return (-5.25*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 5.25*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 0.75*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g - 42.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 42.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 6.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s - 26.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 78.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 47.25*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 5.25*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p - 157.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 472.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 283.5*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 31.5*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s - 19.6875*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 177.1875*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 295.3125*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 98.4375*(d*d*d*d)*(b*b*b*b)*g*(p*p) - 78.75*(d*d*d)*(a*a*a)*b*(p*p)*s + 708.75*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 1181.25*(d*d*d)*a*(b*b*b)*(p*p)*s + 393.75*(d*d*d)*(b*b*b*b)*(p*p)*s + 39.375*(d*d)*(a*a)*g*(p*p*p) - 275.625*(d*d)*a*b*g*(p*p*p) + 275.625*(d*d)*(b*b)*g*(p*p*p) + 78.75*d*(a*a)*(p*p*p)*s - 551.25*d*a*b*(p*p*p)*s + 551.25*d*(b*b)*(p*p*p)*s + 73.828125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad733(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 7.875*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 2.625*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g + 30.625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 55.125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 18.375*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 0.875*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s + 13.125*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 65.625*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 65.625*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 13.125*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 65.625*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 328.125*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 328.125*(d*d*d*d)*a*(b*b*b*b)*p*s - 65.625*(d*d*d*d)*(b*b*b*b*b)*p*s + 3.28125*(d*d*d)*(a*a*a)*g*(p*p) - 68.90625*(d*d*d)*(a*a)*b*g*(p*p) + 206.71875*(d*d*d)*a*(b*b)*g*(p*p) - 114.84375*(d*d*d)*(b*b*b)*g*(p*p) + 9.84375*(d*d)*(a*a*a)*(p*p)*s - 206.71875*(d*d)*(a*a)*b*(p*p)*s + 620.15625*(d*d)*a*(b*b)*(p*p)*s - 344.53125*(d*d)*(b*b*b)*(p*p)*s + 59.0625*d*a*g*(p*p*p) - 137.8125*d*b*g*(p*p*p) + 59.0625*a*(p*p*p)*s - 137.8125*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad734(double a, double b, double p, double d, double s, double g){
	return (-2.1875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 6.5625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 3.9375*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.4375*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g - 13.125*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 39.375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 23.625*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 2.625*(d*d*d*d*d)*(b*b*b*b*b*b)*s - 3.28125*(d*d*d*d)*(a*a*a)*b*g*p + 29.53125*(d*d*d*d)*(a*a)*(b*b)*g*p - 49.21875*(d*d*d*d)*a*(b*b*b)*g*p + 16.40625*(d*d*d*d)*(b*b*b*b)*g*p - 13.125*(d*d*d)*(a*a*a)*b*p*s + 118.125*(d*d*d)*(a*a)*(b*b)*p*s - 196.875*(d*d*d)*a*(b*b*b)*p*s + 65.625*(d*d*d)*(b*b*b*b)*p*s + 9.84375*(d*d)*(a*a)*g*(p*p) - 68.90625*(d*d)*a*b*g*(p*p) + 68.90625*(d*d)*(b*b)*g*(p*p) + 19.6875*d*(a*a)*(p*p)*s - 137.8125*d*a*b*(p*p)*s + 137.8125*d*(b*b)*(p*p)*s + 24.609375*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad735(double a, double b, double p, double d, double s, double g){
	return (0.65625*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 3.28125*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 3.28125*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.65625*(d*d*d*d*d)*(b*b*b*b*b)*g + 3.28125*(d*d*d*d)*(a*a*a)*(b*b)*s - 16.40625*(d*d*d*d)*(a*a)*(b*b*b)*s + 16.40625*(d*d*d*d)*a*(b*b*b*b)*s - 3.28125*(d*d*d*d)*(b*b*b*b*b)*s + 0.328125*(d*d*d)*(a*a*a)*g*p - 6.890625*(d*d*d)*(a*a)*b*g*p + 20.671875*(d*d*d)*a*(b*b)*g*p - 11.484375*(d*d*d)*(b*b*b)*g*p + 0.984375*(d*d)*(a*a*a)*p*s - 20.671875*(d*d)*(a*a)*b*p*s + 62.015625*(d*d)*a*(b*b)*p*s - 34.453125*(d*d)*(b*b*b)*p*s + 8.859375*d*a*g*(p*p) - 20.671875*d*b*g*(p*p) + 8.859375*a*(p*p)*s - 20.671875*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad736(double a, double b, double p, double d, double s, double g){
	return (-0.109375*(d*d*d*d)*(a*a*a)*b*g + 0.984375*(d*d*d*d)*(a*a)*(b*b)*g - 1.640625*(d*d*d*d)*a*(b*b*b)*g + 0.546875*(d*d*d*d)*(b*b*b*b)*g - 0.4375*(d*d*d)*(a*a*a)*b*s + 3.9375*(d*d*d)*(a*a)*(b*b)*s - 6.5625*(d*d*d)*a*(b*b*b)*s + 2.1875*(d*d*d)*(b*b*b*b)*s + 0.65625*(d*d)*(a*a)*g*p - 4.59375*(d*d)*a*b*g*p + 4.59375*(d*d)*(b*b)*g*p + 1.3125*d*(a*a)*p*s - 9.1875*d*a*b*p*s + 9.1875*d*(b*b)*p*s + 2.4609375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad737(double a, double b, double p, double d, double s, double g){
	return (0.0078125*(d*d*d)*(a*a*a)*g - 0.1640625*(d*d*d)*(a*a)*b*g + 0.4921875*(d*d*d)*a*(b*b)*g - 0.2734375*(d*d*d)*(b*b*b)*g + 0.0234375*(d*d)*(a*a*a)*s - 0.4921875*(d*d)*(a*a)*b*s + 1.4765625*(d*d)*a*(b*b)*s - 0.8203125*(d*d)*(b*b*b)*s + 0.421875*d*a*g*p - 0.984375*d*b*g*p + 0.421875*a*p*s - 0.984375*b*p*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad738(double a, double b, double p, double d, double s, double g){
	return (0.01171875*(d*d)*(a*a)*g - 0.08203125*(d*d)*a*b*g + 0.08203125*(d*d)*(b*b)*g + 0.0234375*d*(a*a)*s - 0.1640625*d*a*b*s + 0.1640625*d*(b*b)*s + 0.087890625*g*p)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad739(double a, double b, double p, double d, double s, double g){
	return (0.005859375*d*a*g - 0.013671875*d*b*g + 0.005859375*a*s - 0.013671875*b*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7310(double a, double b, double p, double d, double s, double g){
	return 0.0009765625*g/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad740(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g - 11.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*s - 10.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 14.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 3.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*p - 94.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 126.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s - 27.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 105.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 94.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 21.0 *(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) - 0.75*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p) - 183.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 735.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 661.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 147.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s - 5.25*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p)*s - 13.125*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 157.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 393.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 262.5*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 39.375*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) - 65.625*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 787.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 1968.75*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 1312.5*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 196.875*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 26.25*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 275.625*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 551.25*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 229.6875*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 78.75*(d*d)*(a*a*a)*(p*p*p*p)*s - 826.875*(d*d)*(a*a)*b*(p*p*p*p)*s + 1653.75*(d*d)*a*(b*b)*(p*p*p*p)*s - 689.0625*(d*d)*(b*b*b)*(p*p*p*p)*s + 118.125*d*a*g*(p*p*p*p*p) - 206.71875*d*b*g*(p*p*p*p*p) + 118.125*a*(p*p*p*p*p)*s - 206.71875*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad741(double a, double b, double p, double d, double s, double g){
	return (3.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g - 2.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g + 35.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s - 20.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*s + 26.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 63.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 31.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 3.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p + 210.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 504.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 252.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 24.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s + 39.375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 262.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 393.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 157.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 13.125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 236.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 1575.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 2362.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 945.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 78.75*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 6.5625*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 183.75*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 826.875*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 918.75*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 229.6875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 26.25*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 735.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 3307.5*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 3675.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 918.75*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 177.1875*(d*d)*(a*a)*g*(p*p*p*p) - 826.875*(d*d)*a*b*g*(p*p*p*p) + 620.15625*(d*d)*(b*b)*g*(p*p*p*p) + 354.375*d*(a*a)*(p*p*p*p)*s - 1653.75*d*a*b*(p*p*p*p)*s + 1240.3125*d*(b*b)*(p*p*p*p)*s + 162.421875*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad742(double a, double b, double p, double d, double s, double g){
	return (-5.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 7.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 1.5*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g - 47.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 63.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 13.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s - 26.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 105.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 94.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 21.0 *(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 0.75*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p - 183.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 735.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 661.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 147.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 5.25*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s - 19.6875*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 236.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 590.625*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 393.75*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 59.0625*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) - 98.4375*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 1181.25*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 2953.125*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 1968.75*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 295.3125*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 52.5*(d*d*d)*(a*a*a)*g*(p*p*p) - 551.25*(d*d*d)*(a*a)*b*g*(p*p*p) + 1102.5*(d*d*d)*a*(b*b)*g*(p*p*p) - 459.375*(d*d*d)*(b*b*b)*g*(p*p*p) + 157.5*(d*d)*(a*a*a)*(p*p*p)*s - 1653.75*(d*d)*(a*a)*b*(p*p*p)*s + 3307.5*(d*d)*a*(b*b)*(p*p*p)*s - 1378.125*(d*d)*(b*b*b)*(p*p*p)*s + 295.3125*d*a*g*(p*p*p*p) - 516.796875*d*b*g*(p*p*p*p) + 295.3125*a*(p*p*p*p)*s - 516.796875*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad743(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 10.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 5.25*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 0.5*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g + 35.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 84.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 42.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 4.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + 13.125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 87.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 131.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 52.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 4.375*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 78.75*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 525.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 787.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 315.0*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 26.25*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 3.28125*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 91.875*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 413.4375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 459.375*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 114.84375*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 13.125*(d*d*d)*(a*a*a*a)*(p*p)*s - 367.5*(d*d*d)*(a*a*a)*b*(p*p)*s + 1653.75*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 1837.5*(d*d*d)*a*(b*b*b)*(p*p)*s + 459.375*(d*d*d)*(b*b*b*b)*(p*p)*s + 118.125*(d*d)*(a*a)*g*(p*p*p) - 551.25*(d*d)*a*b*g*(p*p*p) + 413.4375*(d*d)*(b*b)*g*(p*p*p) + 236.25*d*(a*a)*(p*p*p)*s - 1102.5*d*a*b*(p*p*p)*s + 826.875*d*(b*b)*(p*p*p)*s + 135.3515625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad744(double a, double b, double p, double d, double s, double g){
	return (-2.1875*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 8.75*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 7.875*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 1.75*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.0625*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g - 15.3125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 61.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 55.125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 12.25*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 0.4375*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s - 3.28125*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 39.375*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 98.4375*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 65.625*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 9.84375*(d*d*d*d*d)*(b*b*b*b*b)*g*p - 16.40625*(d*d*d*d)*(a*a*a*a)*b*p*s + 196.875*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 492.1875*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 328.125*(d*d*d*d)*a*(b*b*b*b)*p*s - 49.21875*(d*d*d*d)*(b*b*b*b*b)*p*s + 13.125*(d*d*d)*(a*a*a)*g*(p*p) - 137.8125*(d*d*d)*(a*a)*b*g*(p*p) + 275.625*(d*d*d)*a*(b*b)*g*(p*p) - 114.84375*(d*d*d)*(b*b*b)*g*(p*p) + 39.375*(d*d)*(a*a*a)*(p*p)*s - 413.4375*(d*d)*(a*a)*b*(p*p)*s + 826.875*(d*d)*a*(b*b)*(p*p)*s - 344.53125*(d*d)*(b*b*b)*(p*p)*s + 98.4375*d*a*g*(p*p*p) - 172.265625*d*b*g*(p*p*p) + 98.4375*a*(p*p*p)*s - 172.265625*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad745(double a, double b, double p, double d, double s, double g){
	return (0.65625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 4.375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 6.5625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 2.625*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.21875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 3.9375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 26.25*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 39.375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 15.75*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 1.3125*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 0.328125*(d*d*d*d)*(a*a*a*a)*g*p - 9.1875*(d*d*d*d)*(a*a*a)*b*g*p + 41.34375*(d*d*d*d)*(a*a)*(b*b)*g*p - 45.9375*(d*d*d*d)*a*(b*b*b)*g*p + 11.484375*(d*d*d*d)*(b*b*b*b)*g*p + 1.3125*(d*d*d)*(a*a*a*a)*p*s - 36.75*(d*d*d)*(a*a*a)*b*p*s + 165.375*(d*d*d)*(a*a)*(b*b)*p*s - 183.75*(d*d*d)*a*(b*b*b)*p*s + 45.9375*(d*d*d)*(b*b*b*b)*p*s + 17.71875*(d*d)*(a*a)*g*(p*p) - 82.6875*(d*d)*a*b*g*(p*p) + 62.015625*(d*d)*(b*b)*g*(p*p) + 35.4375*d*(a*a)*(p*p)*s - 165.375*d*a*b*(p*p)*s + 124.03125*d*(b*b)*(p*p)*s + 27.0703125*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad746(double a, double b, double p, double d, double s, double g){
	return (-0.109375*(d*d*d*d*d)*(a*a*a*a)*b*g + 1.3125*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 3.28125*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 2.1875*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.328125*(d*d*d*d*d)*(b*b*b*b*b)*g - 0.546875*(d*d*d*d)*(a*a*a*a)*b*s + 6.5625*(d*d*d*d)*(a*a*a)*(b*b)*s - 16.40625*(d*d*d*d)*(a*a)*(b*b*b)*s + 10.9375*(d*d*d*d)*a*(b*b*b*b)*s - 1.640625*(d*d*d*d)*(b*b*b*b*b)*s + 0.875*(d*d*d)*(a*a*a)*g*p - 9.1875*(d*d*d)*(a*a)*b*g*p + 18.375*(d*d*d)*a*(b*b)*g*p - 7.65625*(d*d*d)*(b*b*b)*g*p + 2.625*(d*d)*(a*a*a)*p*s - 27.5625*(d*d)*(a*a)*b*p*s + 55.125*(d*d)*a*(b*b)*p*s - 22.96875*(d*d)*(b*b*b)*p*s + 9.84375*d*a*g*(p*p) - 17.2265625*d*b*g*(p*p) + 9.84375*a*(p*p)*s - 17.2265625*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad747(double a, double b, double p, double d, double s, double g){
	return (0.0078125*(d*d*d*d)*(a*a*a*a)*g - 0.21875*(d*d*d*d)*(a*a*a)*b*g + 0.984375*(d*d*d*d)*(a*a)*(b*b)*g - 1.09375*(d*d*d*d)*a*(b*b*b)*g + 0.2734375*(d*d*d*d)*(b*b*b*b)*g + 0.03125*(d*d*d)*(a*a*a*a)*s - 0.875*(d*d*d)*(a*a*a)*b*s + 3.9375*(d*d*d)*(a*a)*(b*b)*s - 4.375*(d*d*d)*a*(b*b*b)*s + 1.09375*(d*d*d)*(b*b*b*b)*s + 0.84375*(d*d)*(a*a)*g*p - 3.9375*(d*d)*a*b*g*p + 2.953125*(d*d)*(b*b)*g*p + 1.6875*d*(a*a)*p*s - 7.875*d*a*b*p*s + 5.90625*d*(b*b)*p*s + 1.93359375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad748(double a, double b, double p, double d, double s, double g){
	return (0.015625*(d*d*d)*(a*a*a)*g - 0.1640625*(d*d*d)*(a*a)*b*g + 0.328125*(d*d*d)*a*(b*b)*g - 0.13671875*(d*d*d)*(b*b*b)*g + 0.046875*(d*d)*(a*a*a)*s - 0.4921875*(d*d)*(a*a)*b*s + 0.984375*(d*d)*a*(b*b)*s - 0.41015625*(d*d)*(b*b*b)*s + 0.3515625*d*a*g*p - 0.615234375*d*b*g*p + 0.3515625*a*p*s - 0.615234375*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad749(double a, double b, double p, double d, double s, double g){
	return (0.01171875*(d*d)*(a*a)*g - 0.0546875*(d*d)*a*b*g + 0.041015625*(d*d)*(b*b)*g + 0.0234375*d*(a*a)*s - 0.109375*d*a*b*s + 0.08203125*d*(b*b)*s + 0.0537109375*g*p)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7410(double a, double b, double p, double d, double s, double g){
	return (0.00390625*d*a*g - 0.0068359375*d*b*g + 0.00390625*a*s - 0.0068359375*b*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7411(double a, double b, double p, double d, double s, double g){
	return 0.00048828125*g/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad750(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g - 12.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*s - 10.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p + 17.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*p - 5.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*p - 105.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s + 175.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*p*s - 50.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 131.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 157.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 52.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p) - 3.75*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p) - 210.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 1050.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 1260.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 420.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p)*s - 30.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p)*s - 13.125*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 196.875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 656.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 656.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 196.875*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 13.125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p) - 78.75*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 1181.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 3937.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 3937.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 1181.25*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 78.75*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p)*s + 32.8125*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 459.375*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 1378.125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 1148.4375*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 229.6875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 131.25*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 1837.5*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 5512.5*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 4593.75*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 918.75*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 295.3125*(d*d)*(a*a)*g*(p*p*p*p*p) - 1033.59375*(d*d)*a*b*g*(p*p*p*p*p) + 620.15625*(d*d)*(b*b)*g*(p*p*p*p*p) + 590.625*d*(a*a)*(p*p*p*p*p)*s - 2067.1875*d*a*b*(p*p*p*p*p)*s + 1240.3125*d*(b*b)*(p*p*p*p*p)*s + 162.421875*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad751(double a, double b, double p, double d, double s, double g){
	return (3.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g - 2.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g + 38.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*s - 27.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*s + 26.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 78.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 52.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 7.5*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*p + 236.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 708.75*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 472.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s - 67.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*p*s + 39.375*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 328.125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 656.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 393.75*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 65.625*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) - 1.875*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p) + 275.625*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 2296.875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 4593.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 2756.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 459.375*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s - 13.125*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p)*s + 6.5625*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 229.6875*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 1378.125*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 2296.875*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 1148.4375*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 137.8125*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 32.8125*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 1148.4375*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 6890.625*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 11484.375*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 5742.1875*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 689.0625*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 295.3125*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 2067.1875*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 3100.78125*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 1033.59375*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 885.9375*(d*d)*(a*a*a)*(p*p*p*p)*s - 6201.5625*(d*d)*(a*a)*b*(p*p*p*p)*s + 9302.34375*(d*d)*a*(b*b)*(p*p*p*p)*s - 3100.78125*(d*d)*(b*b*b)*(p*p*p*p)*s + 812.109375*d*a*g*(p*p*p*p*p) - 1136.953125*d*b*g*(p*p*p*p*p) + 812.109375*a*(p*p*p*p*p)*s - 1136.953125*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad752(double a, double b, double p, double d, double s, double g){
	return (-5.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 8.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g - 2.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g - 52.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 87.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s - 25.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*s - 26.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 131.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 157.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 52.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 3.75*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p - 210.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 1050.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 1260.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 420.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 30.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s - 19.6875*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 295.3125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 984.375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 984.375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 295.3125*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 19.6875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) - 118.125*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 1771.875*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 5906.25*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 5906.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 1771.875*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 118.125*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 65.625*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 918.75*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 2756.25*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 2296.875*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 459.375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 262.5*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 3675.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 11025.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 9187.5*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 1837.5*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 738.28125*(d*d)*(a*a)*g*(p*p*p*p) - 2583.984375*(d*d)*a*b*g*(p*p*p*p) + 1550.390625*(d*d)*(b*b)*g*(p*p*p*p) + 1476.5625*d*(a*a)*(p*p*p*p)*s - 5167.96875*d*a*b*(p*p*p*p)*s + 3100.78125*d*(b*b)*(p*p*p*p)*s + 487.265625*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad753(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 13.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 8.75*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 1.25*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g + 39.375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 118.125*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 78.75*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 11.25*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s + 13.125*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 109.375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 218.75*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 131.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 21.875*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 0.625*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p + 91.875*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 765.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 1531.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 918.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 153.125*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 4.375*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s + 3.28125*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 114.84375*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 689.0625*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 1148.4375*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 574.21875*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 68.90625*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 16.40625*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 574.21875*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 3445.3125*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 5742.1875*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 2871.09375*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 344.53125*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 196.875*(d*d*d)*(a*a*a)*g*(p*p*p) - 1378.125*(d*d*d)*(a*a)*b*g*(p*p*p) + 2067.1875*(d*d*d)*a*(b*b)*g*(p*p*p) - 689.0625*(d*d*d)*(b*b*b)*g*(p*p*p) + 590.625*(d*d)*(a*a*a)*(p*p*p)*s - 4134.375*(d*d)*(a*a)*b*(p*p*p)*s + 6201.5625*(d*d)*a*(b*b)*(p*p*p)*s - 2067.1875*(d*d)*(b*b*b)*(p*p*p)*s + 676.7578125*d*a*g*(p*p*p*p) - 947.4609375*d*b*g*(p*p*p*p) + 676.7578125*a*(p*p*p*p)*s - 947.4609375*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad754(double a, double b, double p, double d, double s, double g){
	return (-2.1875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 10.9375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 13.125*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 4.375*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 0.3125*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g - 17.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 87.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 105.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 35.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 2.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s - 3.28125*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 49.21875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 164.0625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 164.0625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 49.21875*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 3.28125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p - 19.6875*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 295.3125*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 984.375*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 984.375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 295.3125*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 19.6875*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 16.40625*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 229.6875*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 689.0625*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 574.21875*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 114.84375*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 65.625*(d*d*d)*(a*a*a*a)*(p*p)*s - 918.75*(d*d*d)*(a*a*a)*b*(p*p)*s + 2756.25*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 2296.875*(d*d*d)*a*(b*b*b)*(p*p)*s + 459.375*(d*d*d)*(b*b*b*b)*(p*p)*s + 246.09375*(d*d)*(a*a)*g*(p*p*p) - 861.328125*(d*d)*a*b*g*(p*p*p) + 516.796875*(d*d)*(b*b)*g*(p*p*p) + 492.1875*d*(a*a)*(p*p*p)*s - 1722.65625*d*a*b*(p*p*p)*s + 1033.59375*d*(b*b)*(p*p*p)*s + 203.02734375*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad755(double a, double b, double p, double d, double s, double g){
	return (0.65625*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 5.46875*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 10.9375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 6.5625*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 1.09375*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.03125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g + 4.59375*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 38.28125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 76.5625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 45.9375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 7.65625*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 0.21875*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s + 0.328125*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 11.484375*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 68.90625*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 114.84375*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 57.421875*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 6.890625*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 1.640625*(d*d*d*d)*(a*a*a*a*a)*p*s - 57.421875*(d*d*d*d)*(a*a*a*a)*b*p*s + 344.53125*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 574.21875*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 287.109375*(d*d*d*d)*a*(b*b*b*b)*p*s - 34.453125*(d*d*d*d)*(b*b*b*b*b)*p*s + 29.53125*(d*d*d)*(a*a*a)*g*(p*p) - 206.71875*(d*d*d)*(a*a)*b*g*(p*p) + 310.078125*(d*d*d)*a*(b*b)*g*(p*p) - 103.359375*(d*d*d)*(b*b*b)*g*(p*p) + 88.59375*(d*d)*(a*a*a)*(p*p)*s - 620.15625*(d*d)*(a*a)*b*(p*p)*s + 930.234375*(d*d)*a*(b*b)*(p*p)*s - 310.078125*(d*d)*(b*b*b)*(p*p)*s + 135.3515625*d*a*g*(p*p*p) - 189.4921875*d*b*g*(p*p*p) + 135.3515625*a*(p*p*p)*s - 189.4921875*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad756(double a, double b, double p, double d, double s, double g){
	return (-0.109375*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 1.640625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 5.46875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 5.46875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 1.640625*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.109375*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g - 0.65625*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 9.84375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 32.8125*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 32.8125*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 9.84375*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.65625*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 1.09375*(d*d*d*d)*(a*a*a*a)*g*p - 15.3125*(d*d*d*d)*(a*a*a)*b*g*p + 45.9375*(d*d*d*d)*(a*a)*(b*b)*g*p - 38.28125*(d*d*d*d)*a*(b*b*b)*g*p + 7.65625*(d*d*d*d)*(b*b*b*b)*g*p + 4.375*(d*d*d)*(a*a*a*a)*p*s - 61.25*(d*d*d)*(a*a*a)*b*p*s + 183.75*(d*d*d)*(a*a)*(b*b)*p*s - 153.125*(d*d*d)*a*(b*b*b)*p*s + 30.625*(d*d*d)*(b*b*b*b)*p*s + 24.609375*(d*d)*(a*a)*g*(p*p) - 86.1328125*(d*d)*a*b*g*(p*p) + 51.6796875*(d*d)*(b*b)*g*(p*p) + 49.21875*d*(a*a)*(p*p)*s - 172.265625*d*a*b*(p*p)*s + 103.359375*d*(b*b)*(p*p)*s + 27.0703125*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad757(double a, double b, double p, double d, double s, double g){
	return (0.0078125*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.2734375*(d*d*d*d*d)*(a*a*a*a)*b*g + 1.640625*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 2.734375*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 1.3671875*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.1640625*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.0390625*(d*d*d*d)*(a*a*a*a*a)*s - 1.3671875*(d*d*d*d)*(a*a*a*a)*b*s + 8.203125*(d*d*d*d)*(a*a*a)*(b*b)*s - 13.671875*(d*d*d*d)*(a*a)*(b*b*b)*s + 6.8359375*(d*d*d*d)*a*(b*b*b*b)*s - 0.8203125*(d*d*d*d)*(b*b*b*b*b)*s + 1.40625*(d*d*d)*(a*a*a)*g*p - 9.84375*(d*d*d)*(a*a)*b*g*p + 14.765625*(d*d*d)*a*(b*b)*g*p - 4.921875*(d*d*d)*(b*b*b)*g*p + 4.21875*(d*d)*(a*a*a)*p*s - 29.53125*(d*d)*(a*a)*b*p*s + 44.296875*(d*d)*a*(b*b)*p*s - 14.765625*(d*d)*(b*b*b)*p*s + 9.66796875*d*a*g*(p*p) - 13.53515625*d*b*g*(p*p) + 9.66796875*a*(p*p)*s - 13.53515625*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad758(double a, double b, double p, double d, double s, double g){
	return (0.01953125*(d*d*d*d)*(a*a*a*a)*g - 0.2734375*(d*d*d*d)*(a*a*a)*b*g + 0.8203125*(d*d*d*d)*(a*a)*(b*b)*g - 0.68359375*(d*d*d*d)*a*(b*b*b)*g + 0.13671875*(d*d*d*d)*(b*b*b*b)*g + 0.078125*(d*d*d)*(a*a*a*a)*s - 1.09375*(d*d*d)*(a*a*a)*b*s + 3.28125*(d*d*d)*(a*a)*(b*b)*s - 2.734375*(d*d*d)*a*(b*b*b)*s + 0.546875*(d*d*d)*(b*b*b*b)*s + 0.87890625*(d*d)*(a*a)*g*p - 3.076171875*(d*d)*a*b*g*p + 1.845703125*(d*d)*(b*b)*g*p + 1.7578125*d*(a*a)*p*s - 6.15234375*d*a*b*p*s + 3.69140625*d*(b*b)*p*s + 1.4501953125*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad759(double a, double b, double p, double d, double s, double g){
	return (0.01953125*(d*d*d)*(a*a*a)*g - 0.13671875*(d*d*d)*(a*a)*b*g + 0.205078125*(d*d*d)*a*(b*b)*g - 0.068359375*(d*d*d)*(b*b*b)*g + 0.05859375*(d*d)*(a*a*a)*s - 0.41015625*(d*d)*(a*a)*b*s + 0.615234375*(d*d)*a*(b*b)*s - 0.205078125*(d*d)*(b*b*b)*s + 0.2685546875*d*a*g*p - 0.3759765625*d*b*g*p + 0.2685546875*a*p*s - 0.3759765625*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7510(double a, double b, double p, double d, double s, double g){
	return (0.009765625*(d*d)*(a*a)*g - 0.0341796875*(d*d)*a*b*g + 0.0205078125*(d*d)*(b*b)*g + 0.01953125*d*(a*a)*s - 0.068359375*d*a*b*s + 0.041015625*d*(b*b)*s + 0.0322265625*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7511(double a, double b, double p, double d, double s, double g){
	return (0.00244140625*d*a*g - 0.00341796875*d*b*g + 0.00244140625*a*s - 0.00341796875*b*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7512(double a, double b, double p, double d, double s, double g){
	return 0.000244140625*g/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad760(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g - 13.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*s - 10.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 21.0 *(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 7.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g*p - 115.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 231.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 82.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 157.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 236.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 105.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 11.25*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*(p*p) - 236.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 1417.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 2126.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 945.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 101.25*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*(p*p)*s - 13.125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p*p) + 236.25*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p) - 984.375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p) + 1312.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p) - 590.625*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p) + 78.75*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p*p) - 1.875*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p*p) - 91.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p*p)*s + 1653.75*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p)*s - 6890.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p)*s + 9187.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p)*s - 4134.375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p)*s + 551.25*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p*p)*s - 13.125*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p*p)*s + 39.375*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p) - 689.0625*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p) + 2756.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p) - 3445.3125*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p) + 1378.125*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p) - 137.8125*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p) + 196.875*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p)*s - 3445.3125*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p)*s + 13781.25*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p)*s - 17226.5625*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p)*s + 6890.625*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p)*s - 689.0625*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p)*s + 590.625*(d*d*d)*(a*a*a)*g*(p*p*p*p*p) - 3100.78125*(d*d*d)*(a*a)*b*g*(p*p*p*p*p) + 3720.9375*(d*d*d)*a*(b*b)*g*(p*p*p*p*p) - 1033.59375*(d*d*d)*(b*b*b)*g*(p*p*p*p*p) + 1771.875*(d*d)*(a*a*a)*(p*p*p*p*p)*s - 9302.34375*(d*d)*(a*a)*b*(p*p*p*p*p)*s + 11162.8125*(d*d)*a*(b*b)*(p*p*p*p*p)*s - 3100.78125*(d*d)*(b*b*b)*(p*p*p*p*p)*s + 974.53125*d*a*g*(p*p*p*p*p*p) - 1136.953125*d*b*g*(p*p*p*p*p*p) + 974.53125*a*(p*p*p*p*p*p)*s - 1136.953125*b*(p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad761(double a, double b, double p, double d, double s, double g){
	return (3.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 3.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 42.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 36.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 26.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*p - 94.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p + 78.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*p - 15.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*p + 262.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*p*s - 945.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s + 787.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*p*s - 150.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*p*s + 39.375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p) - 393.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 984.375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 787.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 196.875*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p) - 11.25*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p) + 315.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p)*s - 3150.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 7875.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 6300.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 1575.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p)*s - 90.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p)*s + 6.5625*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p) - 275.625*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 2067.1875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 4593.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 3445.3125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 826.875*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 45.9375*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p) + 39.375*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p)*s - 1653.75*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 12403.125*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 27562.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 20671.875*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 4961.25*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 275.625*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p)*s + 442.96875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 4134.375*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 9302.34375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 6201.5625*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 1033.59375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 1771.875*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 16537.5*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 37209.375*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 24806.25*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 4134.375*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 2436.328125*(d*d)*(a*a)*g*(p*p*p*p*p) - 6821.71875*(d*d)*a*b*g*(p*p*p*p*p) + 3410.859375*(d*d)*(b*b)*g*(p*p*p*p*p) + 4872.65625*d*(a*a)*(p*p*p*p*p)*s - 13643.4375*d*a*b*(p*p*p*p*p)*s + 6821.71875*d*(b*b)*(p*p*p*p*p)*s + 1055.7421875*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad762(double a, double b, double p, double d, double s, double g){
	return (-5.25*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g + 10.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g - 3.75*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g - 57.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*s + 115.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*s - 41.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*s - 26.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 157.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 236.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 105.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 11.25*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*p - 236.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 1417.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 2126.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 945.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s - 101.25*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*p*s - 19.6875*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 354.375*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 1476.5625*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 1968.75*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 885.9375*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 118.125*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) - 2.8125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p) - 137.8125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 2480.625*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 10335.9375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 13781.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 6201.5625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 826.875*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s - 19.6875*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p)*s + 78.75*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 1378.125*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 5512.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 6890.625*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 2756.25*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 275.625*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 393.75*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 6890.625*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 27562.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 34453.125*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 13781.25*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 1378.125*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 1476.5625*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 7751.953125*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 9302.34375*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 2583.984375*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 4429.6875*(d*d)*(a*a*a)*(p*p*p*p)*s - 23255.859375*(d*d)*(a*a)*b*(p*p*p*p)*s + 27907.03125*(d*d)*a*(b*b)*(p*p*p*p)*s - 7751.953125*(d*d)*(b*b*b)*(p*p*p*p)*s + 2923.59375*d*a*g*(p*p*p*p*p) - 3410.859375*d*b*g*(p*p*p*p*p) + 2923.59375*a*(p*p*p*p*p)*s - 3410.859375*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad763(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 15.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 13.125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g - 2.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g + 43.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 157.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 131.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s - 25.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*s + 13.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 131.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 328.125*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 262.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 65.625*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 3.75*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p + 105.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 1050.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 2625.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 2100.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 525.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 30.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s + 3.28125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 137.8125*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 1033.59375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 2296.875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 1722.65625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 413.4375*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 22.96875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 19.6875*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 826.875*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 6201.5625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 13781.25*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 10335.9375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 2480.625*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 137.8125*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 295.3125*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 2756.25*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 6201.5625*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 4134.375*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 689.0625*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 1181.25*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 11025.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 24806.25*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 16537.5*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 2756.25*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 2030.2734375*(d*d)*(a*a)*g*(p*p*p*p) - 5684.765625*(d*d)*a*b*g*(p*p*p*p) + 2842.3828125*(d*d)*(b*b)*g*(p*p*p*p) + 4060.546875*d*(a*a)*(p*p*p*p)*s - 11369.53125*d*a*b*(p*p*p*p)*s + 5684.765625*d*(b*b)*(p*p*p*p)*s + 1055.7421875*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad764(double a, double b, double p, double d, double s, double g){
	return (-2.1875*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 13.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 19.6875*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 8.75*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 0.9375*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g - 19.6875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 118.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 177.1875*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 78.75*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 8.4375*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s - 3.28125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 59.0625*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 246.09375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 328.125*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 147.65625*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 19.6875*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 0.46875*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p - 22.96875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 413.4375*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 1722.65625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 2296.875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 1033.59375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 137.8125*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 3.28125*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s + 19.6875*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 344.53125*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 1378.125*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 1722.65625*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 689.0625*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 68.90625*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 98.4375*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 1722.65625*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 6890.625*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 8613.28125*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 3445.3125*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 344.53125*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 492.1875*(d*d*d)*(a*a*a)*g*(p*p*p) - 2583.984375*(d*d*d)*(a*a)*b*g*(p*p*p) + 3100.78125*(d*d*d)*a*(b*b)*g*(p*p*p) - 861.328125*(d*d*d)*(b*b*b)*g*(p*p*p) + 1476.5625*(d*d)*(a*a*a)*(p*p*p)*s - 7751.953125*(d*d)*(a*a)*b*(p*p*p)*s + 9302.34375*(d*d)*a*(b*b)*(p*p*p)*s - 2583.984375*(d*d)*(b*b*b)*(p*p*p)*s + 1218.1640625*d*a*g*(p*p*p*p) - 1421.19140625*d*b*g*(p*p*p*p) + 1218.1640625*a*(p*p*p*p)*s - 1421.19140625*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad765(double a, double b, double p, double d, double s, double g){
	return (0.65625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 6.5625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 16.40625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 13.125*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 3.28125*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 0.1875*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g + 5.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 52.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 131.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 105.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 26.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 1.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + 0.328125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 13.78125*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 103.359375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 229.6875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 172.265625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 41.34375*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 2.296875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 1.96875*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 82.6875*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 620.15625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 1378.125*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 1033.59375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 248.0625*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 13.78125*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 44.296875*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 413.4375*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 930.234375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 620.15625*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 103.359375*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 177.1875*(d*d*d)*(a*a*a*a)*(p*p)*s - 1653.75*(d*d*d)*(a*a*a)*b*(p*p)*s + 3720.9375*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 2480.625*(d*d*d)*a*(b*b*b)*(p*p)*s + 413.4375*(d*d*d)*(b*b*b*b)*(p*p)*s + 406.0546875*(d*d)*(a*a)*g*(p*p*p) - 1136.953125*(d*d)*a*b*g*(p*p*p) + 568.4765625*(d*d)*(b*b)*g*(p*p*p) + 812.109375*d*(a*a)*(p*p*p)*s - 2273.90625*d*a*b*(p*p*p)*s + 1136.953125*d*(b*b)*(p*p*p)*s + 263.935546875*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad766(double a, double b, double p, double d, double s, double g){
	return (-0.109375*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 1.96875*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 8.203125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 10.9375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 4.921875*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.65625*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.015625*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g - 0.765625*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 13.78125*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 57.421875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 76.5625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 34.453125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 4.59375*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 0.109375*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s + 1.3125*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 22.96875*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 91.875*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 114.84375*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 45.9375*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 4.59375*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 6.5625*(d*d*d*d)*(a*a*a*a*a)*p*s - 114.84375*(d*d*d*d)*(a*a*a*a)*b*p*s + 459.375*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 574.21875*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 229.6875*(d*d*d*d)*a*(b*b*b*b)*p*s - 22.96875*(d*d*d*d)*(b*b*b*b*b)*p*s + 49.21875*(d*d*d)*(a*a*a)*g*(p*p) - 258.3984375*(d*d*d)*(a*a)*b*g*(p*p) + 310.078125*(d*d*d)*a*(b*b)*g*(p*p) - 86.1328125*(d*d*d)*(b*b*b)*g*(p*p) + 147.65625*(d*d)*(a*a*a)*(p*p)*s - 775.1953125*(d*d)*(a*a)*b*(p*p)*s + 930.234375*(d*d)*a*(b*b)*(p*p)*s - 258.3984375*(d*d)*(b*b*b)*(p*p)*s + 162.421875*d*a*g*(p*p*p) - 189.4921875*d*b*g*(p*p*p) + 162.421875*a*(p*p*p)*s - 189.4921875*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad767(double a, double b, double p, double d, double s, double g){
	return (0.0078125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 0.328125*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 2.4609375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 5.46875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 4.1015625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.984375*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.0546875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 0.046875*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 1.96875*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 14.765625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 32.8125*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 24.609375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 5.90625*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.328125*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 2.109375*(d*d*d*d)*(a*a*a*a)*g*p - 19.6875*(d*d*d*d)*(a*a*a)*b*g*p + 44.296875*(d*d*d*d)*(a*a)*(b*b)*g*p - 29.53125*(d*d*d*d)*a*(b*b*b)*g*p + 4.921875*(d*d*d*d)*(b*b*b*b)*g*p + 8.4375*(d*d*d)*(a*a*a*a)*p*s - 78.75*(d*d*d)*(a*a*a)*b*p*s + 177.1875*(d*d*d)*(a*a)*(b*b)*p*s - 118.125*(d*d*d)*a*(b*b*b)*p*s + 19.6875*(d*d*d)*(b*b*b*b)*p*s + 29.00390625*(d*d)*(a*a)*g*(p*p) - 81.2109375*(d*d)*a*b*g*(p*p) + 40.60546875*(d*d)*(b*b)*g*(p*p) + 58.0078125*d*(a*a)*(p*p)*s - 162.421875*d*a*b*(p*p)*s + 81.2109375*d*(b*b)*(p*p)*s + 25.13671875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad768(double a, double b, double p, double d, double s, double g){
	return (0.0234375*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.41015625*(d*d*d*d*d)*(a*a*a*a)*b*g + 1.640625*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 2.05078125*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.8203125*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.08203125*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.1171875*(d*d*d*d)*(a*a*a*a*a)*s - 2.05078125*(d*d*d*d)*(a*a*a*a)*b*s + 8.203125*(d*d*d*d)*(a*a*a)*(b*b)*s - 10.25390625*(d*d*d*d)*(a*a)*(b*b*b)*s + 4.1015625*(d*d*d*d)*a*(b*b*b*b)*s - 0.41015625*(d*d*d*d)*(b*b*b*b*b)*s + 1.7578125*(d*d*d)*(a*a*a)*g*p - 9.228515625*(d*d*d)*(a*a)*b*g*p + 11.07421875*(d*d*d)*a*(b*b)*g*p - 3.076171875*(d*d*d)*(b*b*b)*g*p + 5.2734375*(d*d)*(a*a*a)*p*s - 27.685546875*(d*d)*(a*a)*b*p*s + 33.22265625*(d*d)*a*(b*b)*p*s - 9.228515625*(d*d)*(b*b*b)*p*s + 8.701171875*d*a*g*(p*p) - 10.1513671875*d*b*g*(p*p) + 8.701171875*a*(p*p)*s - 10.1513671875*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad769(double a, double b, double p, double d, double s, double g){
	return (0.029296875*(d*d*d*d)*(a*a*a*a)*g - 0.2734375*(d*d*d*d)*(a*a*a)*b*g + 0.615234375*(d*d*d*d)*(a*a)*(b*b)*g - 0.41015625*(d*d*d*d)*a*(b*b*b)*g + 0.068359375*(d*d*d*d)*(b*b*b*b)*g + 0.1171875*(d*d*d)*(a*a*a*a)*s - 1.09375*(d*d*d)*(a*a*a)*b*s + 2.4609375*(d*d*d)*(a*a)*(b*b)*s - 1.640625*(d*d*d)*a*(b*b*b)*s + 0.2734375*(d*d*d)*(b*b*b*b)*s + 0.8056640625*(d*d)*(a*a)*g*p - 2.255859375*(d*d)*a*b*g*p + 1.1279296875*(d*d)*(b*b)*g*p + 1.611328125*d*(a*a)*p*s - 4.51171875*d*a*b*p*s + 2.255859375*d*(b*b)*p*s + 1.04736328125*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7610(double a, double b, double p, double d, double s, double g){
	return (0.01953125*(d*d*d)*(a*a*a)*g - 0.1025390625*(d*d*d)*(a*a)*b*g + 0.123046875*(d*d*d)*a*(b*b)*g - 0.0341796875*(d*d*d)*(b*b*b)*g + 0.05859375*(d*d)*(a*a*a)*s - 0.3076171875*(d*d)*(a*a)*b*s + 0.369140625*(d*d)*a*(b*b)*s - 0.1025390625*(d*d)*(b*b*b)*s + 0.193359375*d*a*g*p - 0.2255859375*d*b*g*p + 0.193359375*a*p*s - 0.2255859375*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7611(double a, double b, double p, double d, double s, double g){
	return (0.00732421875*(d*d)*(a*a)*g - 0.0205078125*(d*d)*a*b*g + 0.01025390625*(d*d)*(b*b)*g + 0.0146484375*d*(a*a)*s - 0.041015625*d*a*b*s + 0.0205078125*d*(b*b)*s + 0.01904296875*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7612(double a, double b, double p, double d, double s, double g){
	return (0.00146484375*d*a*g - 0.001708984375*d*b*g + 0.00146484375*a*s - 0.001708984375*b*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7613(double a, double b, double p, double d, double s, double g){
	return 0.0001220703125*g/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad770(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g - 14.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*s - 10.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 24.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 10.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g*p - 126.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 294.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 126.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 183.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 330.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 183.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 26.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*(p*p) - 262.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 1837.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 3307.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 1837.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 262.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*(p*p)*s - 13.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p*p) + 275.625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p*p) - 1378.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p*p) + 2296.875*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p*p) - 1378.125*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p*p) + 275.625*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p*p) - 13.125*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p*p) - 105.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p*p)*s + 2205.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p*p)*s - 11025.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p*p)*s + 18375.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p*p)*s - 11025.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p*p)*s + 2205.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p*p)*s - 105.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p*p)*s + 45.9375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p*p) - 964.6875*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p*p) + 4823.4375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p*p) - 8039.0625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p*p) + 4823.4375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p*p) - 964.6875*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p*p) + 45.9375*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p*p) + 275.625*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p*p)*s - 5788.125*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p*p)*s + 28940.625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p*p)*s - 48234.375*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p*p)*s + 28940.625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p*p)*s - 5788.125*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p*p)*s + 275.625*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p*p)*s + 1033.59375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p*p) - 7235.15625*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p*p) + 13023.28125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p*p) - 7235.15625*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p*p) + 1033.59375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p*p) + 4134.375*(d*d*d)*(a*a*a*a)*(p*p*p*p*p)*s - 28940.625*(d*d*d)*(a*a*a)*b*(p*p*p*p*p)*s + 52093.125*(d*d*d)*(a*a)*(b*b)*(p*p*p*p*p)*s - 28940.625*(d*d*d)*a*(b*b*b)*(p*p*p*p*p)*s + 4134.375*(d*d*d)*(b*b*b*b)*(p*p*p*p*p)*s + 3410.859375*(d*d)*(a*a)*g*(p*p*p*p*p*p) - 7958.671875*(d*d)*a*b*g*(p*p*p*p*p*p) + 3410.859375*(d*d)*(b*b)*g*(p*p*p*p*p*p) + 6821.71875*d*(a*a)*(p*p*p*p*p*p)*s - 15917.34375*d*a*b*(p*p*p*p*p*p)*s + 6821.71875*d*(b*b)*(p*p*p*p*p*p)*s + 1055.7421875*g*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad771(double a, double b, double p, double d, double s, double g){
	return (3.5*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 3.5*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 45.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 45.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 26.25*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g*p - 110.25*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 110.25*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 26.25*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g*p + 288.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*p*s - 1212.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 1212.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 288.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*p*s + 39.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*(p*p) - 459.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 1378.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 1378.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 459.375*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 39.375*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*(p*p) + 354.375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*(p*p)*s - 4134.375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 12403.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 12403.125*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 4134.375*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 354.375*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 6.5625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p*p) - 321.5625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p*p) + 2894.0625*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p) - 8039.0625*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p) + 8039.0625*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p) - 2894.0625*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p) + 321.5625*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p*p) - 6.5625*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p*p) + 45.9375*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p*p)*s - 2250.9375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p*p)*s + 20258.4375*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p)*s - 56273.4375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p)*s + 56273.4375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p)*s - 20258.4375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p)*s + 2250.9375*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p*p)*s - 45.9375*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p*p)*s + 620.15625*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p) - 7235.15625*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p) + 21705.46875*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p) - 21705.46875*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p) + 7235.15625*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p) - 620.15625*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p) + 3100.78125*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p)*s - 36175.78125*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p)*s + 108527.34375*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p)*s - 108527.34375*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p)*s + 36175.78125*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p)*s - 3100.78125*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p)*s + 5684.765625*(d*d*d)*(a*a*a)*g*(p*p*p*p*p) - 23876.015625*(d*d*d)*(a*a)*b*g*(p*p*p*p*p) + 23876.015625*(d*d*d)*a*(b*b)*g*(p*p*p*p*p) - 5684.765625*(d*d*d)*(b*b*b)*g*(p*p*p*p*p) + 17054.296875*(d*d)*(a*a*a)*(p*p*p*p*p)*s - 71628.046875*(d*d)*(a*a)*b*(p*p*p*p*p)*s + 71628.046875*(d*d)*a*(b*b)*(p*p*p*p*p)*s - 17054.296875*(d*d)*(b*b*b)*(p*p*p*p*p)*s + 7390.1953125*d*a*g*(p*p*p*p*p*p) - 7390.1953125*d*b*g*(p*p*p*p*p*p) + 7390.1953125*a*(p*p*p*p*p*p)*s - 7390.1953125*b*(p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad772(double a, double b, double p, double d, double s, double g){
	return (-5.25*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g + 12.25*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 5.25*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g - 63.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*s + 147.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 63.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*s - 26.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*p + 183.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*p - 330.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p + 183.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*p - 26.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*p - 262.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*p*s + 1837.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*p*s - 3307.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s + 1837.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*p*s - 262.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*p*s - 19.6875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p) + 413.4375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p) - 2067.1875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 3445.3125*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 2067.1875*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 413.4375*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p) - 19.6875*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p) - 157.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p)*s + 3307.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p)*s - 16537.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 27562.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 16537.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 3307.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p)*s - 157.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p)*s + 91.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p) - 1929.375*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 9646.875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 16078.125*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 9646.875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 1929.375*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 91.875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p) + 551.25*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p)*s - 11576.25*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 57881.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 96468.75*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 57881.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 11576.25*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 551.25*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p)*s + 2583.984375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 18087.890625*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 32558.203125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 18087.890625*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 2583.984375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 10335.9375*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 72351.5625*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 130232.8125*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 72351.5625*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 10335.9375*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 10232.578125*(d*d)*(a*a)*g*(p*p*p*p*p) - 23876.015625*(d*d)*a*b*g*(p*p*p*p*p) + 10232.578125*(d*d)*(b*b)*g*(p*p*p*p*p) + 20465.15625*d*(a*a)*(p*p*p*p*p)*s - 47752.03125*d*a*b*(p*p*p*p*p)*s + 20465.15625*d*(b*b)*(p*p*p*p*p)*s + 3695.09765625*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad773(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g - 18.375*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g + 18.375*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g - 4.375*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g + 48.125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*s - 202.125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*s + 202.125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*s - 48.125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*s + 13.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*p - 153.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 459.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 459.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 153.125*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 13.125*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*p + 118.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*p*s - 1378.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 4134.375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 4134.375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 1378.125*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s - 118.125*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*p*s + 3.28125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p) - 160.78125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 1447.03125*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 4019.53125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 4019.53125*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 1447.03125*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 160.78125*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) - 3.28125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p) + 22.96875*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p)*s - 1125.46875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 10129.21875*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 28136.71875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 28136.71875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 10129.21875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 1125.46875*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s - 22.96875*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p)*s + 413.4375*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 4823.4375*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 14470.3125*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 14470.3125*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 4823.4375*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 413.4375*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 2067.1875*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 24117.1875*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 72351.5625*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 72351.5625*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 24117.1875*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 2067.1875*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 4737.3046875*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 19896.6796875*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 19896.6796875*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 4737.3046875*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 14211.9140625*(d*d)*(a*a*a)*(p*p*p*p)*s - 59690.0390625*(d*d)*(a*a)*b*(p*p*p*p)*s + 59690.0390625*(d*d)*a*(b*b)*(p*p*p*p)*s - 14211.9140625*(d*d)*(b*b*b)*(p*p*p*p)*s + 7390.1953125*d*a*g*(p*p*p*p*p) - 7390.1953125*d*b*g*(p*p*p*p*p) + 7390.1953125*a*(p*p*p*p*p)*s - 7390.1953125*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad774(double a, double b, double p, double d, double s, double g){
	return (-2.1875*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g + 15.3125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 27.5625*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 15.3125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g - 2.1875*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g - 21.875*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*s + 153.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 275.625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 153.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s - 21.875*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*s - 3.28125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 68.90625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 344.53125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 574.21875*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 344.53125*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 68.90625*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 3.28125*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p - 26.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 551.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 2756.25*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 4593.75*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 2756.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 551.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s + 22.96875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 482.34375*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 2411.71875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 4019.53125*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 2411.71875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 482.34375*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 22.96875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 137.8125*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 2894.0625*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 14470.3125*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 24117.1875*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 14470.3125*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 2894.0625*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 137.8125*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 861.328125*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 6029.296875*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 10852.734375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 6029.296875*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 861.328125*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 3445.3125*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 24117.1875*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 43410.9375*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 24117.1875*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 3445.3125*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 4263.57421875*(d*d)*(a*a)*g*(p*p*p*p) - 9948.33984375*(d*d)*a*b*g*(p*p*p*p) + 4263.57421875*(d*d)*(b*b)*g*(p*p*p*p) + 8527.1484375*d*(a*a)*(p*p*p*p)*s - 19896.6796875*d*a*b*(p*p*p*p)*s + 8527.1484375*d*(b*b)*(p*p*p*p)*s + 1847.548828125*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad775(double a, double b, double p, double d, double s, double g){
	return (0.65625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 7.65625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 22.96875*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 22.96875*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 7.65625*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 0.65625*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g + 5.90625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s - 68.90625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 206.71875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 206.71875*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 68.90625*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 5.90625*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s + 0.328125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 16.078125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 144.703125*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 401.953125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 401.953125*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 144.703125*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 16.078125*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 0.328125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p + 2.296875*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 112.546875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 1012.921875*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 2813.671875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 2813.671875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 1012.921875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 112.546875*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 2.296875*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s + 62.015625*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 723.515625*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 2170.546875*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 2170.546875*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 723.515625*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 62.015625*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 310.078125*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 3617.578125*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 10852.734375*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 10852.734375*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 3617.578125*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 310.078125*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 947.4609375*(d*d*d)*(a*a*a)*g*(p*p*p) - 3979.3359375*(d*d*d)*(a*a)*b*g*(p*p*p) + 3979.3359375*(d*d*d)*a*(b*b)*g*(p*p*p) - 947.4609375*(d*d*d)*(b*b*b)*g*(p*p*p) + 2842.3828125*(d*d)*(a*a*a)*(p*p*p)*s - 11938.0078125*(d*d)*(a*a)*b*(p*p*p)*s + 11938.0078125*(d*d)*a*(b*b)*(p*p*p)*s - 2842.3828125*(d*d)*(b*b*b)*(p*p*p)*s + 1847.548828125*d*a*g*(p*p*p*p) - 1847.548828125*d*b*g*(p*p*p*p) + 1847.548828125*a*(p*p*p*p)*s - 1847.548828125*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad776(double a, double b, double p, double d, double s, double g){
	return (-0.109375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 2.296875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 11.484375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 19.140625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 11.484375*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 2.296875*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 0.109375*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g - 0.875*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 18.375*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 91.875*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 153.125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 91.875*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 18.375*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 0.875*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + 1.53125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 32.15625*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 160.78125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 267.96875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 160.78125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 32.15625*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 1.53125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 9.1875*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 192.9375*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 964.6875*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 1607.8125*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 964.6875*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 192.9375*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 9.1875*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 86.1328125*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 602.9296875*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 1085.2734375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 602.9296875*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 86.1328125*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 344.53125*(d*d*d)*(a*a*a*a)*(p*p)*s - 2411.71875*(d*d*d)*(a*a*a)*b*(p*p)*s + 4341.09375*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 2411.71875*(d*d*d)*a*(b*b*b)*(p*p)*s + 344.53125*(d*d*d)*(b*b*b*b)*(p*p)*s + 568.4765625*(d*d)*(a*a)*g*(p*p*p) - 1326.4453125*(d*d)*a*b*g*(p*p*p) + 568.4765625*(d*d)*(b*b)*g*(p*p*p) + 1136.953125*d*(a*a)*(p*p*p)*s - 2652.890625*d*a*b*(p*p*p)*s + 1136.953125*d*(b*b)*(p*p*p)*s + 307.9248046875*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad777(double a, double b, double p, double d, double s, double g){
	return (0.0078125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 0.3828125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 3.4453125*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 9.5703125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 9.5703125*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 3.4453125*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.3828125*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.0078125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g + 0.0546875*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 2.6796875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 24.1171875*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 66.9921875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 66.9921875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 24.1171875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 2.6796875*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 0.0546875*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s + 2.953125*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 34.453125*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 103.359375*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 103.359375*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 34.453125*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 2.953125*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 14.765625*(d*d*d*d)*(a*a*a*a*a)*p*s - 172.265625*(d*d*d*d)*(a*a*a*a)*b*p*s + 516.796875*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 516.796875*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 172.265625*(d*d*d*d)*a*(b*b*b*b)*p*s - 14.765625*(d*d*d*d)*(b*b*b*b*b)*p*s + 67.67578125*(d*d*d)*(a*a*a)*g*(p*p) - 284.23828125*(d*d*d)*(a*a)*b*g*(p*p) + 284.23828125*(d*d*d)*a*(b*b)*g*(p*p) - 67.67578125*(d*d*d)*(b*b*b)*g*(p*p) + 203.02734375*(d*d)*(a*a*a)*(p*p)*s - 852.71484375*(d*d)*(a*a)*b*(p*p)*s + 852.71484375*(d*d)*a*(b*b)*(p*p)*s - 203.02734375*(d*d)*(b*b*b)*(p*p)*s + 175.95703125*d*a*g*(p*p*p) - 175.95703125*d*b*g*(p*p*p) + 175.95703125*a*(p*p*p)*s - 175.95703125*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad778(double a, double b, double p, double d, double s, double g){
	return (0.02734375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 0.57421875*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 2.87109375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 4.78515625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 2.87109375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.57421875*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.02734375*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 0.1640625*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 3.4453125*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 17.2265625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 28.7109375*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 17.2265625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 3.4453125*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.1640625*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 3.076171875*(d*d*d*d)*(a*a*a*a)*g*p - 21.533203125*(d*d*d*d)*(a*a*a)*b*g*p + 38.759765625*(d*d*d*d)*(a*a)*(b*b)*g*p - 21.533203125*(d*d*d*d)*a*(b*b*b)*g*p + 3.076171875*(d*d*d*d)*(b*b*b*b)*g*p + 12.3046875*(d*d*d)*(a*a*a*a)*p*s - 86.1328125*(d*d*d)*(a*a*a)*b*p*s + 155.0390625*(d*d*d)*(a*a)*(b*b)*p*s - 86.1328125*(d*d*d)*a*(b*b*b)*p*s + 12.3046875*(d*d*d)*(b*b*b*b)*p*s + 30.4541015625*(d*d)*(a*a)*g*(p*p) - 71.0595703125*(d*d)*a*b*g*(p*p) + 30.4541015625*(d*d)*(b*b)*g*(p*p) + 60.908203125*d*(a*a)*(p*p)*s - 142.119140625*d*a*b*(p*p)*s + 60.908203125*d*(b*b)*(p*p)*s + 21.99462890625*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad779(double a, double b, double p, double d, double s, double g){
	return (0.041015625*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.478515625*(d*d*d*d*d)*(a*a*a*a)*b*g + 1.435546875*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 1.435546875*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.478515625*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.041015625*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.205078125*(d*d*d*d)*(a*a*a*a*a)*s - 2.392578125*(d*d*d*d)*(a*a*a*a)*b*s + 7.177734375*(d*d*d*d)*(a*a*a)*(b*b)*s - 7.177734375*(d*d*d*d)*(a*a)*(b*b*b)*s + 2.392578125*(d*d*d*d)*a*(b*b*b*b)*s - 0.205078125*(d*d*d*d)*(b*b*b*b*b)*s + 1.8798828125*(d*d*d)*(a*a*a)*g*p - 7.8955078125*(d*d*d)*(a*a)*b*g*p + 7.8955078125*(d*d*d)*a*(b*b)*g*p - 1.8798828125*(d*d*d)*(b*b*b)*g*p + 5.6396484375*(d*d)*(a*a*a)*p*s - 23.6865234375*(d*d)*(a*a)*b*p*s + 23.6865234375*(d*d)*a*(b*b)*p*s - 5.6396484375*(d*d)*(b*b*b)*p*s + 7.33154296875*d*a*g*(p*p) - 7.33154296875*d*b*g*(p*p) + 7.33154296875*a*(p*p)*s - 7.33154296875*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7710(double a, double b, double p, double d, double s, double g){
	return (0.0341796875*(d*d*d*d)*(a*a*a*a)*g - 0.2392578125*(d*d*d*d)*(a*a*a)*b*g + 0.4306640625*(d*d*d*d)*(a*a)*(b*b)*g - 0.2392578125*(d*d*d*d)*a*(b*b*b)*g + 0.0341796875*(d*d*d*d)*(b*b*b*b)*g + 0.13671875*(d*d*d)*(a*a*a*a)*s - 0.95703125*(d*d*d)*(a*a*a)*b*s + 1.72265625*(d*d*d)*(a*a)*(b*b)*s - 0.95703125*(d*d*d)*a*(b*b*b)*s + 0.13671875*(d*d*d)*(b*b*b*b)*s + 0.6767578125*(d*d)*(a*a)*g*p - 1.5791015625*(d*d)*a*b*g*p + 0.6767578125*(d*d)*(b*b)*g*p + 1.353515625*d*(a*a)*p*s - 3.158203125*d*a*b*p*s + 1.353515625*d*(b*b)*p*s + 0.733154296875*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7711(double a, double b, double p, double d, double s, double g){
	return (0.01708984375*(d*d*d)*(a*a*a)*g - 0.07177734375*(d*d*d)*(a*a)*b*g + 0.07177734375*(d*d*d)*a*(b*b)*g - 0.01708984375*(d*d*d)*(b*b*b)*g + 0.05126953125*(d*d)*(a*a*a)*s - 0.21533203125*(d*d)*(a*a)*b*s + 0.21533203125*(d*d)*a*(b*b)*s - 0.05126953125*(d*d)*(b*b*b)*s + 0.13330078125*d*a*g*p - 0.13330078125*d*b*g*p + 0.13330078125*a*p*s - 0.13330078125*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7712(double a, double b, double p, double d, double s, double g){
	return (0.005126953125*(d*d)*(a*a)*g - 0.011962890625*(d*d)*a*b*g + 0.005126953125*(d*d)*(b*b)*g + 0.01025390625*d*(a*a)*s - 0.02392578125*d*a*b*s + 0.01025390625*d*(b*b)*s + 0.0111083984375*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7713(double a, double b, double p, double d, double s, double g){
	return 0.0008544921875*(d*a*g - d*b*g + a*s - b*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7714(double a, double b, double p, double d, double s, double g){
	return 6.103515625e-5*g/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad780(double a, double b, double p, double d, double s, double g){
	return (-(d*d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g - 15.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*s - 10.5*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 28.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 14.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g*p - 136.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 364.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 182.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 210.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 441.0 *(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 294.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 52.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g*(p*p) - 288.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 2310.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 4851.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 3234.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 577.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*(p*p)*s - 13.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g*(p*p*p) + 315.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*(p*p*p) - 1837.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*(p*p*p) + 3675.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p*p) - 2756.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p*p) + 735.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*(p*p*p) - 52.5*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*(p*p*p) - 118.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*(p*p*p)*s + 2835.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*(p*p*p)*s - 16537.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*(p*p*p)*s + 33075.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p*p)*s - 24806.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p*p)*s + 6615.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*(p*p*p)*s - 472.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*(p*p*p)*s + 52.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p*p*p) - 1286.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p*p*p) + 7717.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p*p) - 16078.125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p*p) + 12862.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p*p) - 3858.75*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p*p) + 367.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p*p*p) - 6.5625*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p*p*p) + 367.5*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p*p*p)*s - 9003.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p*p*p)*s + 54022.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p*p)*s - 112546.875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p*p)*s + 90037.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p*p)*s - 27011.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p*p)*s + 2572.5*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p*p*p)*s - 45.9375*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p*p*p)*s + 1653.75*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p*p) - 14470.3125*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p*p) + 34728.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p*p) - 28940.625*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p*p) + 8268.75*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p*p) - 620.15625*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p*p) + 8268.75*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p*p)*s - 72351.5625*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p*p)*s + 173643.75*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p*p)*s - 144703.125*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p*p)*s + 41343.75*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p*p)*s - 3100.78125*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p*p)*s + 9095.625*(d*d*d)*(a*a*a)*g*(p*p*p*p*p*p) - 31834.6875*(d*d*d)*(a*a)*b*g*(p*p*p*p*p*p) + 27286.875*(d*d*d)*a*(b*b)*g*(p*p*p*p*p*p) - 5684.765625*(d*d*d)*(b*b*b)*g*(p*p*p*p*p*p) + 27286.875*(d*d)*(a*a*a)*(p*p*p*p*p*p)*s - 95504.0625*(d*d)*(a*a)*b*(p*p*p*p*p*p)*s + 81860.625*(d*d)*a*(b*b)*(p*p*p*p*p*p)*s - 17054.296875*(d*d)*(b*b*b)*(p*p*p*p*p*p)*s + 8445.9375*d*a*g*(p*p*p*p*p*p*p) - 7390.1953125*d*b*g*(p*p*p*p*p*p*p) + 8445.9375*a*(p*p*p*p*p*p*p)*s - 7390.1953125*b*(p*p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad781(double a, double b, double p, double d, double s, double g){
	return (3.5*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 4.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 49.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 56.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 26.25*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*g*p - 126.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 147.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 42.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g*p + 315.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*p*s - 1512.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 1764.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 504.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*p*s + 39.375*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g*(p*p) - 525.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 1837.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 2205.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 918.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 105.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*(p*p) + 393.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*(p*p)*s - 5250.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 18375.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 22050.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 9187.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 1050.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 6.5625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*(p*p*p) - 367.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p*p) + 3858.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p*p) - 12862.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p*p) + 16078.125*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p*p) - 7717.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p*p) + 1286.25*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p*p) - 52.5*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p*p) + 52.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(p*p*p)*s - 2940.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p*p)*s + 30870.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p*p)*s - 102900.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p*p)*s + 128625.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p*p)*s - 61740.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p*p)*s + 10290.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p*p)*s - 420.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p*p)*s + 826.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p*p) - 11576.25*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p*p) + 43410.9375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p*p) - 57881.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p*p) + 28940.625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p*p) - 4961.25*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p*p) + 206.71875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p*p) + 4961.25*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p*p)*s - 69457.5*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p*p)*s + 260465.625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p*p)*s - 347287.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p*p)*s + 173643.75*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p*p)*s - 29767.5*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p*p)*s + 1240.3125*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p*p)*s + 11369.53125*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p*p) - 63669.375*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p*p) + 95504.0625*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p*p) - 45478.125*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p*p) + 5684.765625*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p*p) + 45478.125*(d*d*d)*(a*a*a*a)*(p*p*p*p*p)*s - 254677.5*(d*d*d)*(a*a*a)*b*(p*p*p*p*p)*s + 382016.25*(d*d*d)*(a*a)*(b*b)*(p*p*p*p*p)*s - 181912.5*(d*d*d)*a*(b*b*b)*(p*p*p*p*p)*s + 22739.0625*(d*d*d)*(b*b*b*b)*(p*p*p*p*p)*s + 29560.78125*(d*d)*(a*a)*g*(p*p*p*p*p*p) - 59121.5625*(d*d)*a*b*g*(p*p*p*p*p*p) + 22170.5859375*(d*d)*(b*b)*g*(p*p*p*p*p*p) + 59121.5625*d*(a*a)*(p*p*p*p*p*p)*s - 118243.125*d*a*b*(p*p*p*p*p*p)*s + 44341.171875*d*(b*b)*(p*p*p*p*p*p)*s + 7918.06640625*g*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad782(double a, double b, double p, double d, double s, double g){
	return (-5.25*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*g + 14.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 7.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g - 68.25*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*s + 182.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 91.0 *(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*s - 26.25*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*g*p + 210.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g*p - 441.0 *(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 294.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 52.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g*p - 288.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*p*s + 2310.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*p*s - 4851.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 3234.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 577.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*p*s - 19.6875*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g*(p*p) + 472.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*(p*p) - 2756.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 5512.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 4134.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 1102.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 78.75*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*(p*p) - 177.1875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*(p*p)*s + 4252.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*(p*p)*s - 24806.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 49612.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 37209.375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 9922.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 708.75*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 105.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p*p) - 2572.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p*p) + 15435.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p) - 32156.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p) + 25725.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p) - 7717.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p) + 735.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p*p) - 13.125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p*p) + 735.0*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p*p)*s - 18007.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p*p)*s + 108045.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p)*s - 225093.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p)*s + 180075.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p)*s - 54022.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p)*s + 5145.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p*p)*s - 91.875*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p*p)*s + 4134.375*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p) - 36175.78125*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p) + 86821.875*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p) - 72351.5625*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p) + 20671.875*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p) - 1550.390625*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p) + 20671.875*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p)*s - 180878.90625*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p)*s + 434109.375*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p)*s - 361757.8125*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p)*s + 103359.375*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p)*s - 7751.953125*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p)*s + 27286.875*(d*d*d)*(a*a*a)*g*(p*p*p*p*p) - 95504.0625*(d*d*d)*(a*a)*b*g*(p*p*p*p*p) + 81860.625*(d*d*d)*a*(b*b)*g*(p*p*p*p*p) - 17054.296875*(d*d*d)*(b*b*b)*g*(p*p*p*p*p) + 81860.625*(d*d)*(a*a*a)*(p*p*p*p*p)*s - 286512.1875*(d*d)*(a*a)*b*(p*p*p*p*p)*s + 245581.875*(d*d)*a*(b*b)*(p*p*p*p*p)*s - 51162.890625*(d*d)*(b*b*b)*(p*p*p*p*p)*s + 29560.78125*d*a*g*(p*p*p*p*p*p) - 25865.68359375*d*b*g*(p*p*p*p*p*p) + 29560.78125*a*(p*p*p*p*p*p)*s - 25865.68359375*b*(p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad783(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*g - 21.0 *(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g + 24.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 7.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 52.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*s - 252.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*s + 294.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 84.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 13.125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g*p - 175.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*p + 612.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*p - 735.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p + 306.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*p - 35.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*p + 131.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*p*s - 1750.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*p*s + 6125.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*p*s - 7350.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s + 3062.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*p*s - 350.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*p*s + 3.28125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*(p*p) - 183.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p) + 1929.375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p) - 6431.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 8039.0625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 3858.75*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 643.125*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p) - 26.25*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p) + 26.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(p*p)*s - 1470.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p)*s + 15435.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p)*s - 51450.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 64312.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 30870.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 5145.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p)*s - 210.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p)*s + 551.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p) - 7717.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 28940.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 38587.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 19293.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 3307.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 137.8125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p) + 3307.5*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p)*s - 46305.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 173643.75*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 231525.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 115762.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 19845.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 826.875*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p)*s + 9474.609375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 53057.8125*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 79586.71875*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 37898.4375*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 4737.3046875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 37898.4375*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 212231.25*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 318346.875*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 151593.75*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 18949.21875*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 29560.78125*(d*d)*(a*a)*g*(p*p*p*p*p) - 59121.5625*(d*d)*a*b*g*(p*p*p*p*p) + 22170.5859375*(d*d)*(b*b)*g*(p*p*p*p*p) + 59121.5625*d*(a*a)*(p*p*p*p*p)*s - 118243.125*d*a*b*(p*p*p*p*p)*s + 44341.171875*d*(b*b)*(p*p*p*p*p)*s + 9237.744140625*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad784(double a, double b, double p, double d, double s, double g){
	return (-2.1875*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*g + 17.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g - 36.75*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g + 24.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g - 4.375*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g - 24.0625*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*s + 192.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*s - 404.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*s + 269.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*s - 48.125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*s - 3.28125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g*p + 78.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*p - 459.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 918.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 689.0625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 183.75*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 13.125*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*p - 29.53125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*p*s + 708.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*p*s - 4134.375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 8268.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 6201.5625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 1653.75*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s - 118.125*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*p*s + 26.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p) - 643.125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 3858.75*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 8039.0625*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 6431.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 1929.375*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 183.75*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) - 3.28125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p) + 183.75*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p)*s - 4501.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 27011.25*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 56273.4375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 45018.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 13505.625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 1286.25*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s - 22.96875*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p)*s + 1378.125*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 12058.59375*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 28940.625*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 24117.1875*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 6890.625*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 516.796875*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 6890.625*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 60292.96875*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 144703.125*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 120585.9375*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 34453.125*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 2583.984375*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 11369.53125*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 39793.359375*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 34108.59375*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 7105.95703125*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 34108.59375*(d*d)*(a*a*a)*(p*p*p*p)*s - 119380.078125*(d*d)*(a*a)*b*(p*p*p*p)*s + 102325.78125*(d*d)*a*(b*b)*(p*p*p*p)*s - 21317.87109375*(d*d)*(b*b*b)*(p*p*p*p)*s + 14780.390625*d*a*g*(p*p*p*p*p) - 12932.841796875*d*b*g*(p*p*p*p*p) + 14780.390625*a*(p*p*p*p*p)*s - 12932.841796875*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad785(double a, double b, double p, double d, double s, double g){
	return (0.65625*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g - 8.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g + 30.625*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 36.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 15.3125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g - 1.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g + 6.5625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*s - 87.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*s + 306.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 367.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 153.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s - 17.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*s + 0.328125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*p - 18.375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 192.9375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 643.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 803.90625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 385.875*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 64.3125*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 2.625*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p + 2.625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*p*s - 147.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 1543.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 5145.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 6431.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 3087.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 514.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 21.0 *(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s + 82.6875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 1157.625*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 4341.09375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 5788.125*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 2894.0625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 496.125*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 20.671875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 496.125*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 6945.75*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 26046.5625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 34728.75*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 17364.375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 2976.75*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 124.03125*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 1894.921875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 10611.5625*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 15917.34375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 7579.6875*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 947.4609375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 7579.6875*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 42446.25*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 63669.375*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 30318.75*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 3789.84375*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 7390.1953125*(d*d)*(a*a)*g*(p*p*p*p) - 14780.390625*(d*d)*a*b*g*(p*p*p*p) + 5542.646484375*(d*d)*(b*b)*g*(p*p*p*p) + 14780.390625*d*(a*a)*(p*p*p*p)*s - 29560.78125*d*a*b*(p*p*p*p)*s + 11085.29296875*d*(b*b)*(p*p*p*p)*s + 2771.3232421875*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad786(double a, double b, double p, double d, double s, double g){
	return (-0.109375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g + 2.625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 15.3125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 30.625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 22.96875*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 6.125*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 0.4375*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g - 0.984375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*s + 23.625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s - 137.8125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 275.625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 206.71875*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 55.125*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 3.9375*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s + 1.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 42.875*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 257.25*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 535.9375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 428.75*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 128.625*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 12.25*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 0.21875*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p + 12.25*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 300.125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 1800.75*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 3751.5625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 3001.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 900.375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 85.75*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 1.53125*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s + 137.8125*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 1205.859375*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 2894.0625*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 2411.71875*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 689.0625*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 51.6796875*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 689.0625*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 6029.296875*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 14470.3125*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 12058.59375*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 3445.3125*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 258.3984375*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 1515.9375*(d*d*d)*(a*a*a)*g*(p*p*p) - 5305.78125*(d*d*d)*(a*a)*b*g*(p*p*p) + 4547.8125*(d*d*d)*a*(b*b)*g*(p*p*p) - 947.4609375*(d*d*d)*(b*b*b)*g*(p*p*p) + 4547.8125*(d*d)*(a*a*a)*(p*p*p)*s - 15917.34375*(d*d)*(a*a)*b*(p*p*p)*s + 13643.4375*(d*d)*a*(b*b)*(p*p*p)*s - 2842.3828125*(d*d)*(b*b*b)*(p*p*p)*s + 2463.3984375*d*a*g*(p*p*p*p) - 2155.4736328125*d*b*g*(p*p*p*p) + 2463.3984375*a*(p*p*p*p)*s - 2155.4736328125*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad787(double a, double b, double p, double d, double s, double g){
	return (0.0078125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g - 0.4375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 4.59375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 15.3125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 19.140625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 9.1875*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 1.53125*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 0.0625*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g + 0.0625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*s - 3.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 36.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 122.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 153.125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 73.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 12.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 0.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + 3.9375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 55.125*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 206.71875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 275.625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 137.8125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 23.625*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 0.984375*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 23.625*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 330.75*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 1240.3125*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 1653.75*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 826.875*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 141.75*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 5.90625*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 135.3515625*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 757.96875*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 1136.953125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 541.40625*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 67.67578125*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 541.40625*(d*d*d)*(a*a*a*a)*(p*p)*s - 3031.875*(d*d*d)*(a*a*a)*b*(p*p)*s + 4547.8125*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 2165.625*(d*d*d)*a*(b*b*b)*(p*p)*s + 270.703125*(d*d*d)*(b*b*b*b)*(p*p)*s + 703.828125*(d*d)*(a*a)*g*(p*p*p) - 1407.65625*(d*d)*a*b*g*(p*p*p) + 527.87109375*(d*d)*(b*b)*g*(p*p*p) + 1407.65625*d*(a*a)*(p*p*p)*s - 2815.3125*d*a*b*(p*p*p)*s + 1055.7421875*d*(b*b)*(p*p*p)*s + 329.91943359375*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad788(double a, double b, double p, double d, double s, double g){
	return (0.03125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 0.765625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 4.59375*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 9.5703125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 7.65625*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 2.296875*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.21875*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.00390625*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g + 0.21875*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 5.359375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 32.15625*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 66.9921875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 53.59375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 16.078125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 1.53125*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 0.02734375*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s + 4.921875*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 43.06640625*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 103.359375*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 86.1328125*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 24.609375*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 1.845703125*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 24.609375*(d*d*d*d)*(a*a*a*a*a)*p*s - 215.33203125*(d*d*d*d)*(a*a*a*a)*b*p*s + 516.796875*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 430.6640625*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 123.046875*(d*d*d*d)*a*(b*b*b*b)*p*s - 9.228515625*(d*d*d*d)*(b*b*b*b*b)*p*s + 81.2109375*(d*d*d)*(a*a*a)*g*(p*p) - 284.23828125*(d*d*d)*(a*a)*b*g*(p*p) + 243.6328125*(d*d*d)*a*(b*b)*g*(p*p) - 50.7568359375*(d*d*d)*(b*b*b)*g*(p*p) + 243.6328125*(d*d)*(a*a*a)*(p*p)*s - 852.71484375*(d*d)*(a*a)*b*(p*p)*s + 730.8984375*(d*d)*a*(b*b)*(p*p)*s - 152.2705078125*(d*d)*(b*b*b)*(p*p)*s + 175.95703125*d*a*g*(p*p*p) - 153.96240234375*d*b*g*(p*p*p) + 175.95703125*a*(p*p*p)*s - 153.96240234375*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad789(double a, double b, double p, double d, double s, double g){
	return (0.0546875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 0.765625*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 2.87109375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 3.828125*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 1.9140625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.328125*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.013671875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 0.328125*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 4.59375*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 17.2265625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 22.96875*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 11.484375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 1.96875*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.08203125*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 3.759765625*(d*d*d*d)*(a*a*a*a)*g*p - 21.0546875*(d*d*d*d)*(a*a*a)*b*g*p + 31.58203125*(d*d*d*d)*(a*a)*(b*b)*g*p - 15.0390625*(d*d*d*d)*a*(b*b*b)*g*p + 1.8798828125*(d*d*d*d)*(b*b*b*b)*g*p + 15.0390625*(d*d*d)*(a*a*a*a)*p*s - 84.21875*(d*d*d)*(a*a*a)*b*p*s + 126.328125*(d*d*d)*(a*a)*(b*b)*p*s - 60.15625*(d*d*d)*a*(b*b*b)*p*s + 7.51953125*(d*d*d)*(b*b*b*b)*p*s + 29.326171875*(d*d)*(a*a)*g*(p*p) - 58.65234375*(d*d)*a*b*g*(p*p) + 21.99462890625*(d*d)*(b*b)*g*(p*p) + 58.65234375*d*(a*a)*(p*p)*s - 117.3046875*d*a*b*(p*p)*s + 43.9892578125*d*(b*b)*(p*p)*s + 18.328857421875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7810(double a, double b, double p, double d, double s, double g){
	return (0.0546875*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.478515625*(d*d*d*d*d)*(a*a*a*a)*b*g + 1.1484375*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 0.95703125*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.2734375*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.0205078125*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.2734375*(d*d*d*d)*(a*a*a*a*a)*s - 2.392578125*(d*d*d*d)*(a*a*a*a)*b*s + 5.7421875*(d*d*d*d)*(a*a*a)*(b*b)*s - 4.78515625*(d*d*d*d)*(a*a)*(b*b*b)*s + 1.3671875*(d*d*d*d)*a*(b*b*b*b)*s - 0.1025390625*(d*d*d*d)*(b*b*b*b*b)*s + 1.8046875*(d*d*d)*(a*a*a)*g*p - 6.31640625*(d*d*d)*(a*a)*b*g*p + 5.4140625*(d*d*d)*a*(b*b)*g*p - 1.1279296875*(d*d*d)*(b*b*b)*g*p + 5.4140625*(d*d)*(a*a*a)*p*s - 18.94921875*(d*d)*(a*a)*b*p*s + 16.2421875*(d*d)*a*(b*b)*p*s - 3.3837890625*(d*d)*(b*b*b)*p*s + 5.865234375*d*a*g*(p*p) - 5.132080078125*d*b*g*(p*p) + 5.865234375*a*(p*p)*s - 5.132080078125*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7811(double a, double b, double p, double d, double s, double g){
	return (0.0341796875*(d*d*d*d)*(a*a*a*a)*g - 0.19140625*(d*d*d*d)*(a*a*a)*b*g + 0.287109375*(d*d*d*d)*(a*a)*(b*b)*g - 0.13671875*(d*d*d*d)*a*(b*b*b)*g + 0.01708984375*(d*d*d*d)*(b*b*b*b)*g + 0.13671875*(d*d*d)*(a*a*a*a)*s - 0.765625*(d*d*d)*(a*a*a)*b*s + 1.1484375*(d*d*d)*(a*a)*(b*b)*s - 0.546875*(d*d*d)*a*(b*b*b)*s + 0.068359375*(d*d*d)*(b*b*b*b)*s + 0.533203125*(d*d)*(a*a)*g*p - 1.06640625*(d*d)*a*b*g*p + 0.39990234375*(d*d)*(b*b)*g*p + 1.06640625*d*(a*a)*p*s - 2.1328125*d*a*b*p*s + 0.7998046875*d*(b*b)*p*s + 0.4998779296875*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7812(double a, double b, double p, double d, double s, double g){
	return (0.013671875*(d*d*d)*(a*a*a)*g - 0.0478515625*(d*d*d)*(a*a)*b*g + 0.041015625*(d*d*d)*a*(b*b)*g - 0.008544921875*(d*d*d)*(b*b*b)*g + 0.041015625*(d*d)*(a*a*a)*s - 0.1435546875*(d*d)*(a*a)*b*s + 0.123046875*(d*d)*a*(b*b)*s - 0.025634765625*(d*d)*(b*b*b)*s + 0.0888671875*d*a*g*p - 0.0777587890625*d*b*g*p + 0.0888671875*a*p*s - 0.0777587890625*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7813(double a, double b, double p, double d, double s, double g){
	return (0.00341796875*(d*d)*(a*a)*g - 0.0068359375*(d*d)*a*b*g + 0.0025634765625*(d*d)*(b*b)*g + 0.0068359375*d*(a*a)*s - 0.013671875*d*a*b*s + 0.005126953125*d*(b*b)*s + 0.00640869140625*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7814(double a, double b, double p, double d, double s, double g){
	return (0.00048828125*d*a*g - 0.00042724609375*d*b*g + 0.00048828125*a*s - 0.00042724609375*b*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad7815(double a, double b, double p, double d, double s, double g){
	return 3.0517578125e-5*g/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad800(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g + 8.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*s + 14.0*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 84.0*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 52.5*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 210.0*(d*d*d)*(b*b*b*b)*(p*p)*s + 52.5*(d*d)*(b*b)*g*(p*p*p) + 105.0*d*(b*b)*(p*p*p)*s + 6.5625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad801(double a, double b, double p, double d, double s, double g){
	return b*(-4.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b)*g - 28.0*(d*d*d*d*d*d)*(b*b*b*b*b*b)*s - 42.0*(d*d*d*d*d)*(b*b*b*b)*g*p - 210.0*(d*d*d*d)*(b*b*b*b)*p*s - 105.0*(d*d*d)*(b*b)*g*(p*p) - 315.0*(d*d)*(b*b)*(p*p)*s - 52.5*d*g*(p*p*p) - 52.5*(p*p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad802(double a, double b, double p, double d, double s, double g){
	return (7.0*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 42.0*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 52.5*(d*d*d*d)*(b*b*b*b)*g*p + 210.0*(d*d*d)*(b*b*b*b)*p*s + 78.75*(d*d)*(b*b)*g*(p*p) + 157.5*d*(b*b)*(p*p)*s + 13.125*g*(p*p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad803(double a, double b, double p, double d, double s, double g){
	return b*(-7.0*(d*d*d*d*d)*(b*b*b*b)*g - 35.0*(d*d*d*d)*(b*b*b*b)*s - 35.0*(d*d*d)*(b*b)*g*p - 105.0*(d*d)*(b*b)*p*s - 26.25*d*g*(p*p) - 26.25*(p*p)*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad804(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d)*(b*b*b*b)*g + 17.5*(d*d*d)*(b*b*b*b)*s + 13.125*(d*d)*(b*b)*g*p + 26.25*d*(b*b)*p*s + 3.28125*g*(p*p))/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad805(double a, double b, double p, double d, double s, double g){
	return b*(-1.75*(d*d*d)*(b*b)*g - 5.25*(d*d)*(b*b)*s - 2.625*d*g*p - 2.625*p*s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad806(double a, double b, double p, double d, double s, double g){
	return (0.4375*d*(b*b)*(d*g + 2*s) + 0.21875*g*p)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad807(double a, double b, double p, double d, double s, double g){
	return 0.0625*b*(-d*g - s)/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad808(double a, double b, double p, double d, double s, double g){
	return 0.00390625*g/(p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad810(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g + 9.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*s + 14.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 4.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p + 98.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 28.0*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s + 52.5*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 42.0*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 262.5*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 210.0*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 52.5*(d*d*d)*a*(b*b)*g*(p*p*p) - 105.0*(d*d*d)*(b*b*b)*g*(p*p*p) + 157.5*(d*d)*a*(b*b)*(p*p*p)*s - 315.0*(d*d)*(b*b*b)*(p*p*p)*s + 6.5625*d*a*g*(p*p*p*p) - 52.5*d*b*g*(p*p*p*p) + 6.5625*a*(p*p*p*p)*s - 52.5*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad811(double a, double b, double p, double d, double s, double g){
	return (-4.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g + 0.5*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g - 32.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + 4.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*s - 42.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 21.0 *(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p - 252.0*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 126.0*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s - 105.0*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 131.25*(d*d*d*d)*(b*b*b*b)*g*(p*p) - 420.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 525.0*(d*d*d)*(b*b*b*b)*(p*p)*s - 52.5*(d*d)*a*b*g*(p*p*p) + 183.75*(d*d)*(b*b)*g*(p*p*p) - 105.0*d*a*b*(p*p*p)*s + 367.5*d*(b*b)*(p*p*p)*s + 29.53125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad812(double a, double b, double p, double d, double s, double g){
	return (7.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 2.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g + 49.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 14.0*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s + 52.5*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 42.0*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 262.5*(d*d*d*d)*a*(b*b*b*b)*p*s - 210.0*(d*d*d*d)*(b*b*b*b*b)*p*s + 78.75*(d*d*d)*a*(b*b)*g*(p*p) - 157.5*(d*d*d)*(b*b*b)*g*(p*p) + 236.25*(d*d)*a*(b*b)*(p*p)*s - 472.5*(d*d)*(b*b*b)*(p*p)*s + 13.125*d*a*g*(p*p*p) - 105.0*d*b*g*(p*p*p) + 13.125*a*(p*p*p)*s - 105.0*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad813(double a, double b, double p, double d, double s, double g){
	return (-7.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 3.5*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g - 42.0*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 21.0 *(d*d*d*d*d)*(b*b*b*b*b*b)*s - 35.0*(d*d*d*d)*a*(b*b*b)*g*p + 43.75*(d*d*d*d)*(b*b*b*b)*g*p - 140.0*(d*d*d)*a*(b*b*b)*p*s + 175.0*(d*d*d)*(b*b*b*b)*p*s - 26.25*(d*d)*a*b*g*(p*p) + 91.875*(d*d)*(b*b)*g*(p*p) - 52.5*d*a*b*(p*p)*s + 183.75*d*(b*b)*(p*p)*s + 19.6875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad814(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d)*a*(b*b*b*b)*g - 3.5*(d*d*d*d*d)*(b*b*b*b*b)*g + 21.875*(d*d*d*d)*a*(b*b*b*b)*s - 17.5*(d*d*d*d)*(b*b*b*b*b)*s + 13.125*(d*d*d)*a*(b*b)*g*p - 26.25*(d*d*d)*(b*b*b)*g*p + 39.375*(d*d)*a*(b*b)*p*s - 78.75*(d*d)*(b*b*b)*p*s + 3.28125*d*a*g*(p*p) - 26.25*d*b*g*(p*p) + 3.28125*a*(p*p)*s - 26.25*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad815(double a, double b, double p, double d, double s, double g){
	return (-1.75*(d*d*d*d)*a*(b*b*b)*g + 2.1875*(d*d*d*d)*(b*b*b*b)*g - 7.0*(d*d*d)*a*(b*b*b)*s + 8.75*(d*d*d)*(b*b*b*b)*s - 2.625*(d*d)*a*b*g*p + 9.1875*(d*d)*(b*b)*g*p - 5.25*d*a*b*p*s + 18.375*d*(b*b)*p*s + 2.953125*g*(p*p))/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad816(double a, double b, double p, double d, double s, double g){
	return (0.4375*(d*d*d)*a*(b*b)*g - 0.875*(d*d*d)*(b*b*b)*g + 1.3125*(d*d)*a*(b*b)*s - 2.625*(d*d)*(b*b*b)*s + 0.21875*d*a*g*p - 1.75*d*b*g*p + 0.21875*a*p*s - 1.75*b*p*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad817(double a, double b, double p, double d, double s, double g){
	return (-0.0625*(d*d)*a*b*g + 0.21875*(d*d)*(b*b)*g - 0.125*d*a*b*s + 0.4375*d*(b*b)*s + 0.140625*g*p)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad818(double a, double b, double p, double d, double s, double g){
	return (0.00390625*d*a*g - 0.03125*d*b*g + 0.00390625*a*s - 0.03125*b*s)/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad819(double a, double b, double p, double d, double s, double g){
	return 0.001953125*g/(p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad820(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g + 10.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*s + 14.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 8.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p + 0.5*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*p + 112.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 64.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s + 4.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*p*s + 52.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 84.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 21.0 *(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 315.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 504.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 126.0*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 52.5*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 210.0*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 131.25*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 210.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 840.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 525.0*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 6.5625*(d*d)*(a*a)*g*(p*p*p*p) - 105.0*(d*d)*a*b*g*(p*p*p*p) + 183.75*(d*d)*(b*b)*g*(p*p*p*p) + 13.125*d*(a*a)*(p*p*p*p)*s - 210.0*d*a*b*(p*p*p*p)*s + 367.5*d*(b*b)*(p*p*p*p)*s + 29.53125*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad821(double a, double b, double p, double d, double s, double g){
	return (-4.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g + (d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g - 36.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s + 9.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*s - 42.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 42.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 6.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p - 294.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 294.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 42.0*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s - 105.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 262.5*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 105.0*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) - 525.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 1312.5*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 525.0*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s - 52.5*(d*d*d)*(a*a)*b*g*(p*p*p) + 367.5*(d*d*d)*a*(b*b)*g*(p*p*p) - 367.5*(d*d*d)*(b*b*b)*g*(p*p*p) - 157.5*(d*d)*(a*a)*b*(p*p*p)*s + 1102.5*(d*d)*a*(b*b)*(p*p*p)*s - 1102.5*(d*d)*(b*b*b)*(p*p*p)*s + 59.0625*d*a*g*(p*p*p*p) - 236.25*d*b*g*(p*p*p*p) + 59.0625*a*(p*p*p*p)*s - 236.25*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad822(double a, double b, double p, double d, double s, double g){
	return (7.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 4.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g + 0.25*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g + 56.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 32.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + 2.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*s + 52.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 84.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 21.0 *(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 315.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 504.0*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 126.0*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 78.75*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 315.0*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 196.875*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 315.0*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 1260.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 787.5*(d*d*d)*(b*b*b*b)*(p*p)*s + 13.125*(d*d)*(a*a)*g*(p*p*p) - 210.0*(d*d)*a*b*g*(p*p*p) + 367.5*(d*d)*(b*b)*g*(p*p*p) + 26.25*d*(a*a)*(p*p*p)*s - 420.0*d*a*b*(p*p*p)*s + 735.0*d*(b*b)*(p*p*p)*s + 73.828125*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad823(double a, double b, double p, double d, double s, double g){
	return (-7.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 7.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - (d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g - 49.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 49.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 7.0*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s - 35.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 87.5*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 35.0*(d*d*d*d*d)*(b*b*b*b*b)*g*p - 175.0*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 437.5*(d*d*d*d)*a*(b*b*b*b)*p*s - 175.0*(d*d*d*d)*(b*b*b*b*b)*p*s - 26.25*(d*d*d)*(a*a)*b*g*(p*p) + 183.75*(d*d*d)*a*(b*b)*g*(p*p) - 183.75*(d*d*d)*(b*b*b)*g*(p*p) - 78.75*(d*d)*(a*a)*b*(p*p)*s + 551.25*(d*d)*a*(b*b)*(p*p)*s - 551.25*(d*d)*(b*b*b)*(p*p)*s + 39.375*d*a*g*(p*p*p) - 157.5*d*b*g*(p*p*p) + 39.375*a*(p*p*p)*s - 157.5*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad824(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 7.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 1.75*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 26.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 42.0*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 10.5*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 13.125*(d*d*d*d)*(a*a)*(b*b)*g*p - 52.5*(d*d*d*d)*a*(b*b*b)*g*p + 32.8125*(d*d*d*d)*(b*b*b*b)*g*p + 52.5*(d*d*d)*(a*a)*(b*b)*p*s - 210.0*(d*d*d)*a*(b*b*b)*p*s + 131.25*(d*d*d)*(b*b*b*b)*p*s + 3.28125*(d*d)*(a*a)*g*(p*p) - 52.5*(d*d)*a*b*g*(p*p) + 91.875*(d*d)*(b*b)*g*(p*p) + 6.5625*d*(a*a)*(p*p)*s - 105.0*d*a*b*(p*p)*s + 183.75*d*(b*b)*(p*p)*s + 24.609375*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad825(double a, double b, double p, double d, double s, double g){
	return (-1.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 4.375*(d*d*d*d*d)*a*(b*b*b*b)*g - 1.75*(d*d*d*d*d)*(b*b*b*b*b)*g - 8.75*(d*d*d*d)*(a*a)*(b*b*b)*s + 21.875*(d*d*d*d)*a*(b*b*b*b)*s - 8.75*(d*d*d*d)*(b*b*b*b*b)*s - 2.625*(d*d*d)*(a*a)*b*g*p + 18.375*(d*d*d)*a*(b*b)*g*p - 18.375*(d*d*d)*(b*b*b)*g*p - 7.875*(d*d)*(a*a)*b*p*s + 55.125*(d*d)*a*(b*b)*p*s - 55.125*(d*d)*(b*b*b)*p*s + 5.90625*d*a*g*(p*p) - 23.625*d*b*g*(p*p) + 5.90625*a*(p*p)*s - 23.625*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad826(double a, double b, double p, double d, double s, double g){
	return (0.4375*(d*d*d*d)*(a*a)*(b*b)*g - 1.75*(d*d*d*d)*a*(b*b*b)*g + 1.09375*(d*d*d*d)*(b*b*b*b)*g + 1.75*(d*d*d)*(a*a)*(b*b)*s - 7.0*(d*d*d)*a*(b*b*b)*s + 4.375*(d*d*d)*(b*b*b*b)*s + 0.21875*(d*d)*(a*a)*g*p - 3.5*(d*d)*a*b*g*p + 6.125*(d*d)*(b*b)*g*p + 0.4375*d*(a*a)*p*s - 7.0*d*a*b*p*s + 12.25*d*(b*b)*p*s + 2.4609375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad827(double a, double b, double p, double d, double s, double g){
	return (-0.0625*(d*d*d)*(a*a)*b*g + 0.4375*(d*d*d)*a*(b*b)*g - 0.4375*(d*d*d)*(b*b*b)*g - 0.1875*(d*d)*(a*a)*b*s + 1.3125*(d*d)*a*(b*b)*s - 1.3125*(d*d)*(b*b*b)*s + 0.28125*d*a*g*p - 1.125*d*b*g*p + 0.28125*a*p*s - 1.125*b*p*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad828(double a, double b, double p, double d, double s, double g){
	return (0.00390625*(d*d)*(a*a)*g - 0.0625*(d*d)*a*b*g + 0.109375*(d*d)*(b*b)*g + 0.0078125*d*(a*a)*s - 0.125*d*a*b*s + 0.21875*d*(b*b)*s + 0.087890625*g*p)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad829(double a, double b, double p, double d, double s, double g){
	return (0.00390625*d*a*g - 0.015625*d*b*g + 0.00390625*a*s - 0.015625*b*s)/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8210(double a, double b, double p, double d, double s, double g){
	return 0.0009765625*g/(p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad830(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*g + 11.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*s + 14.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 12.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*p + 1.5*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g*p + 126.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s - 108.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*p*s + 13.5*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*p*s + 52.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 126.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 63.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) - 6.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p) + 367.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 882.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 441.0 *(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s - 42.0*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p)*s + 52.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 315.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 393.75*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 105.0*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 262.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 1575.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 1968.75*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 525.0*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 6.5625*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 157.5*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 551.25*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 367.5*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 19.6875*(d*d)*(a*a*a)*(p*p*p*p)*s - 472.5*(d*d)*(a*a)*b*(p*p*p*p)*s + 1653.75*(d*d)*a*(b*b)*(p*p*p*p)*s - 1102.5*(d*d)*(b*b*b)*(p*p*p*p)*s + 88.59375*d*a*g*(p*p*p*p*p) - 236.25*d*b*g*(p*p*p*p*p) + 88.59375*a*(p*p*p*p*p)*s - 236.25*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad831(double a, double b, double p, double d, double s, double g){
	return (-4.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g + 1.5*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g - 40.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*s + 15.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*s - 42.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 63.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 18.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p + 0.75*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*p - 336.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 504.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 144.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s + 6.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*p*s - 105.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 393.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 315.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 52.5*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) - 630.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 2362.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 1890.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 315.0*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s - 52.5*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 551.25*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 1102.5*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 459.375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) - 210.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 2205.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 4410.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 1837.5*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 88.59375*(d*d)*(a*a)*g*(p*p*p*p) - 708.75*(d*d)*a*b*g*(p*p*p*p) + 826.875*(d*d)*(b*b)*g*(p*p*p*p) + 177.1875*d*(a*a)*(p*p*p*p)*s - 1417.5*d*a*b*(p*p*p*p)*s + 1653.75*d*(b*b)*(p*p*p*p)*s + 162.421875*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad832(double a, double b, double p, double d, double s, double g){
	return (7.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 6.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g + 0.75*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g + 63.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 54.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s + 6.75*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*s + 52.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 126.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 63.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 6.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p + 367.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 882.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 441.0 *(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 42.0*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s + 78.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 472.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 590.625*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 157.5*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 393.75*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 2362.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 2953.125*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 787.5*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 13.125*(d*d*d)*(a*a*a)*g*(p*p*p) - 315.0*(d*d*d)*(a*a)*b*g*(p*p*p) + 1102.5*(d*d*d)*a*(b*b)*g*(p*p*p) - 735.0*(d*d*d)*(b*b*b)*g*(p*p*p) + 39.375*(d*d)*(a*a*a)*(p*p*p)*s - 945.0*(d*d)*(a*a)*b*(p*p*p)*s + 3307.5*(d*d)*a*(b*b)*(p*p*p)*s - 2205.0*(d*d)*(b*b*b)*(p*p*p)*s + 221.484375*d*a*g*(p*p*p*p) - 590.625*d*b*g*(p*p*p*p) + 221.484375*a*(p*p*p*p)*s - 590.625*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad833(double a, double b, double p, double d, double s, double g){
	return (-7.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 10.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 3.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g + 0.125*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g - 56.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 84.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 24.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + (d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*s - 35.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 131.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 105.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 17.5*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p - 210.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 787.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 630.0*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 105.0*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 275.625*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 551.25*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 229.6875*(d*d*d*d)*(b*b*b*b)*g*(p*p) - 105.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 1102.5*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 2205.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 918.75*(d*d*d)*(b*b*b*b)*(p*p)*s + 59.0625*(d*d)*(a*a)*g*(p*p*p) - 472.5*(d*d)*a*b*g*(p*p*p) + 551.25*(d*d)*(b*b)*g*(p*p*p) + 118.125*d*(a*a)*(p*p*p)*s - 945.0*d*a*b*(p*p*p)*s + 1102.5*d*(b*b)*(p*p*p)*s + 135.3515625*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad834(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 10.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 5.25*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.5*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g + 30.625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 73.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 36.75*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 3.5*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s + 13.125*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 78.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 98.4375*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 26.25*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 65.625*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 393.75*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 492.1875*(d*d*d*d)*a*(b*b*b*b)*p*s - 131.25*(d*d*d*d)*(b*b*b*b*b)*p*s + 3.28125*(d*d*d)*(a*a*a)*g*(p*p) - 78.75*(d*d*d)*(a*a)*b*g*(p*p) + 275.625*(d*d*d)*a*(b*b)*g*(p*p) - 183.75*(d*d*d)*(b*b*b)*g*(p*p) + 9.84375*(d*d)*(a*a*a)*(p*p)*s - 236.25*(d*d)*(a*a)*b*(p*p)*s + 826.875*(d*d)*a*(b*b)*(p*p)*s - 551.25*(d*d)*(b*b*b)*(p*p)*s + 73.828125*d*a*g*(p*p*p) - 196.875*d*b*g*(p*p*p) + 73.828125*a*(p*p*p)*s - 196.875*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad835(double a, double b, double p, double d, double s, double g){
	return (-1.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 6.5625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 5.25*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g - 10.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 39.375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 31.5*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 5.25*(d*d*d*d*d)*(b*b*b*b*b*b)*s - 2.625*(d*d*d*d)*(a*a*a)*b*g*p + 27.5625*(d*d*d*d)*(a*a)*(b*b)*g*p - 55.125*(d*d*d*d)*a*(b*b*b)*g*p + 22.96875*(d*d*d*d)*(b*b*b*b)*g*p - 10.5*(d*d*d)*(a*a*a)*b*p*s + 110.25*(d*d*d)*(a*a)*(b*b)*p*s - 220.5*(d*d*d)*a*(b*b*b)*p*s + 91.875*(d*d*d)*(b*b*b*b)*p*s + 8.859375*(d*d)*(a*a)*g*(p*p) - 70.875*(d*d)*a*b*g*(p*p) + 82.6875*(d*d)*(b*b)*g*(p*p) + 17.71875*d*(a*a)*(p*p)*s - 141.75*d*a*b*(p*p)*s + 165.375*d*(b*b)*(p*p)*s + 27.0703125*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad836(double a, double b, double p, double d, double s, double g){
	return (0.4375*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 2.625*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 3.28125*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.875*(d*d*d*d*d)*(b*b*b*b*b)*g + 2.1875*(d*d*d*d)*(a*a*a)*(b*b)*s - 13.125*(d*d*d*d)*(a*a)*(b*b*b)*s + 16.40625*(d*d*d*d)*a*(b*b*b*b)*s - 4.375*(d*d*d*d)*(b*b*b*b*b)*s + 0.21875*(d*d*d)*(a*a*a)*g*p - 5.25*(d*d*d)*(a*a)*b*g*p + 18.375*(d*d*d)*a*(b*b)*g*p - 12.25*(d*d*d)*(b*b*b)*g*p + 0.65625*(d*d)*(a*a*a)*p*s - 15.75*(d*d)*(a*a)*b*p*s + 55.125*(d*d)*a*(b*b)*p*s - 36.75*(d*d)*(b*b*b)*p*s + 7.3828125*d*a*g*(p*p) - 19.6875*d*b*g*(p*p) + 7.3828125*a*(p*p)*s - 19.6875*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad837(double a, double b, double p, double d, double s, double g){
	return (-0.0625*(d*d*d*d)*(a*a*a)*b*g + 0.65625*(d*d*d*d)*(a*a)*(b*b)*g - 1.3125*(d*d*d*d)*a*(b*b*b)*g + 0.546875*(d*d*d*d)*(b*b*b*b)*g - 0.25*(d*d*d)*(a*a*a)*b*s + 2.625*(d*d*d)*(a*a)*(b*b)*s - 5.25*(d*d*d)*a*(b*b*b)*s + 2.1875*(d*d*d)*(b*b*b*b)*s + 0.421875*(d*d)*(a*a)*g*p - 3.375*(d*d)*a*b*g*p + 3.9375*(d*d)*(b*b)*g*p + 0.84375*d*(a*a)*p*s - 6.75*d*a*b*p*s + 7.875*d*(b*b)*p*s + 1.93359375*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad838(double a, double b, double p, double d, double s, double g){
	return (0.00390625*(d*d*d)*(a*a*a)*g - 0.09375*(d*d*d)*(a*a)*b*g + 0.328125*(d*d*d)*a*(b*b)*g - 0.21875*(d*d*d)*(b*b*b)*g + 0.01171875*(d*d)*(a*a*a)*s - 0.28125*(d*d)*(a*a)*b*s + 0.984375*(d*d)*a*(b*b)*s - 0.65625*(d*d)*(b*b*b)*s + 0.263671875*d*a*g*p - 0.703125*d*b*g*p + 0.263671875*a*p*s - 0.703125*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad839(double a, double b, double p, double d, double s, double g){
	return (0.005859375*(d*d)*(a*a)*g - 0.046875*(d*d)*a*b*g + 0.0546875*(d*d)*(b*b)*g + 0.01171875*d*(a*a)*s - 0.09375*d*a*b*s + 0.109375*d*(b*b)*s + 0.0537109375*g*p)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8310(double a, double b, double p, double d, double s, double g){
	return (0.0029296875*d*a*g - 0.0078125*d*b*g + 0.0029296875*a*s - 0.0078125*b*s)/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8311(double a, double b, double p, double d, double s, double g){
	return 0.00048828125*g/(p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad840(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*g + 12.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*s + 14.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*p - 16.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*p + 3.0*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g*p + 140.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*p*s - 160.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*p*s + 30.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*p*s + 52.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 168.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 126.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p) - 24.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p) + 0.75*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*(p*p) + 420.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 1344.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 1008.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p)*s - 192.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p)*s + 6.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*(p*p)*s + 52.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 420.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 787.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 420.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 52.5*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p) + 315.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 2520.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 4725.0*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 2520.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 315.0*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p)*s + 6.5625*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 210.0*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 1102.5*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 1470.0*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 459.375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 26.25*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 840.0*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 4410.0*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 5880.0*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 1837.5*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 177.1875*(d*d)*(a*a)*g*(p*p*p*p*p) - 945.0*(d*d)*a*b*g*(p*p*p*p*p) + 826.875*(d*d)*(b*b)*g*(p*p*p*p*p) + 354.375*d*(a*a)*(p*p*p*p*p)*s - 1890.0*d*a*b*(p*p*p*p*p)*s + 1653.75*d*(b*b)*(p*p*p*p*p)*s + 162.421875*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad841(double a, double b, double p, double d, double s, double g){
	return (-4.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g + 2.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*g - 44.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*s + 22.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*s - 42.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 84.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 36.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*p + 3.0*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g*p - 378.0*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 756.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s - 324.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*p*s + 27.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*p*s - 105.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 525.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 630.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 210.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) - 15.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p) - 735.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 3675.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 4410.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 1470.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s - 105.0*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p)*s - 52.5*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 735.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 2205.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 1837.5*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 367.5*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) - 262.5*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 3675.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 11025.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 9187.5*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 1837.5*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 118.125*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 1417.5*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 3307.5*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 1653.75*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 354.375*(d*d)*(a*a*a)*(p*p*p*p)*s - 4252.5*(d*d)*(a*a)*b*(p*p*p*p)*s + 9922.5*(d*d)*a*(b*b)*(p*p*p*p)*s - 4961.25*(d*d)*(b*b*b)*(p*p*p*p)*s + 649.6875*d*a*g*(p*p*p*p*p) - 1299.375*d*b*g*(p*p*p*p*p) + 649.6875*a*(p*p*p*p*p)*s - 1299.375*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad842(double a, double b, double p, double d, double s, double g){
	return (7.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g - 8.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g + 1.5*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g + 70.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s - 80.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*s + 15.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*s + 52.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 168.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 126.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 24.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p + 0.75*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*p + 420.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 1344.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 1008.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 192.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s + 6.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*p*s + 78.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 630.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 1181.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 630.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 78.75*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 472.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 3780.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 7087.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 3780.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 472.5*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 13.125*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 420.0*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 2205.0*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 2940.0*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 918.75*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 52.5*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 1680.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 8820.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 11760.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 3675.0*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 442.96875*(d*d)*(a*a)*g*(p*p*p*p) - 2362.5*(d*d)*a*b*g*(p*p*p*p) + 2067.1875*(d*d)*(b*b)*g*(p*p*p*p) + 885.9375*d*(a*a)*(p*p*p*p)*s - 4725.0*d*a*b*(p*p*p*p)*s + 4134.375*d*(b*b)*(p*p*p*p)*s + 487.265625*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad843(double a, double b, double p, double d, double s, double g){
	return (-7.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 14.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 6.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g + 0.5*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g - 63.0*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 126.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 54.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s + 4.5*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*s - 35.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 175.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 210.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 70.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 5.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p - 245.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 1225.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 1470.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 490.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 35.0*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 367.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 1102.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 918.75*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 183.75*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) - 131.25*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 1837.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 5512.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 4593.75*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 918.75*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 78.75*(d*d*d)*(a*a*a)*g*(p*p*p) - 945.0*(d*d*d)*(a*a)*b*g*(p*p*p) + 2205.0*(d*d*d)*a*(b*b)*g*(p*p*p) - 1102.5*(d*d*d)*(b*b*b)*g*(p*p*p) + 236.25*(d*d)*(a*a*a)*(p*p*p)*s - 2835.0*(d*d)*(a*a)*b*(p*p*p)*s + 6615.0*(d*d)*a*(b*b)*(p*p*p)*s - 3307.5*(d*d)*(b*b*b)*(p*p*p)*s + 541.40625*d*a*g*(p*p*p*p) - 1082.8125*d*b*g*(p*p*p*p) + 541.40625*a*(p*p*p*p)*s - 1082.8125*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad844(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 14.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 10.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 2.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g + 0.0625*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g + 35.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 112.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 84.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 16.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + 0.5*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*s + 13.125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 105.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 196.875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 105.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 13.125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 78.75*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 630.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 1181.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 630.0*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 78.75*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 3.28125*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 105.0*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 551.25*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 735.0*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 229.6875*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 13.125*(d*d*d)*(a*a*a*a)*(p*p)*s - 420.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 2205.0*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 2940.0*(d*d*d)*a*(b*b*b)*(p*p)*s + 918.75*(d*d*d)*(b*b*b*b)*(p*p)*s + 147.65625*(d*d)*(a*a)*g*(p*p*p) - 787.5*(d*d)*a*b*g*(p*p*p) + 689.0625*(d*d)*(b*b)*g*(p*p*p) + 295.3125*d*(a*a)*(p*p*p)*s - 1575.0*d*a*b*(p*p*p)*s + 1378.125*d*(b*b)*(p*p*p)*s + 203.02734375*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad845(double a, double b, double p, double d, double s, double g){
	return (-1.75*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 8.75*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 10.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 3.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.25*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g - 12.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 61.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 73.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 24.5*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 1.75*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s - 2.625*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 36.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 110.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 91.875*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 18.375*(d*d*d*d*d)*(b*b*b*b*b)*g*p - 13.125*(d*d*d*d)*(a*a*a*a)*b*p*s + 183.75*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 551.25*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 459.375*(d*d*d*d)*a*(b*b*b*b)*p*s - 91.875*(d*d*d*d)*(b*b*b*b*b)*p*s + 11.8125*(d*d*d)*(a*a*a)*g*(p*p) - 141.75*(d*d*d)*(a*a)*b*g*(p*p) + 330.75*(d*d*d)*a*(b*b)*g*(p*p) - 165.375*(d*d*d)*(b*b*b)*g*(p*p) + 35.4375*(d*d)*(a*a*a)*(p*p)*s - 425.25*(d*d)*(a*a)*b*(p*p)*s + 992.25*(d*d)*a*(b*b)*(p*p)*s - 496.125*(d*d)*(b*b*b)*(p*p)*s + 108.28125*d*a*g*(p*p*p) - 216.5625*d*b*g*(p*p*p) + 108.28125*a*(p*p*p)*s - 216.5625*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad846(double a, double b, double p, double d, double s, double g){
	return (0.4375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 3.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 6.5625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 3.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.4375*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 2.625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 21.0 *(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 39.375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 21.0 *(d*d*d*d*d)*a*(b*b*b*b*b)*s + 2.625*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 0.21875*(d*d*d*d)*(a*a*a*a)*g*p - 7.0*(d*d*d*d)*(a*a*a)*b*g*p + 36.75*(d*d*d*d)*(a*a)*(b*b)*g*p - 49.0*(d*d*d*d)*a*(b*b*b)*g*p + 15.3125*(d*d*d*d)*(b*b*b*b)*g*p + 0.875*(d*d*d)*(a*a*a*a)*p*s - 28.0*(d*d*d)*(a*a*a)*b*p*s + 147.0*(d*d*d)*(a*a)*(b*b)*p*s - 196.0*(d*d*d)*a*(b*b*b)*p*s + 61.25*(d*d*d)*(b*b*b*b)*p*s + 14.765625*(d*d)*(a*a)*g*(p*p) - 78.75*(d*d)*a*b*g*(p*p) + 68.90625*(d*d)*(b*b)*g*(p*p) + 29.53125*d*(a*a)*(p*p)*s - 157.5*d*a*b*(p*p)*s + 137.8125*d*(b*b)*(p*p)*s + 27.0703125*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad847(double a, double b, double p, double d, double s, double g){
	return (-0.0625*(d*d*d*d*d)*(a*a*a*a)*b*g + 0.875*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 2.625*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 2.1875*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.4375*(d*d*d*d*d)*(b*b*b*b*b)*g - 0.3125*(d*d*d*d)*(a*a*a*a)*b*s + 4.375*(d*d*d*d)*(a*a*a)*(b*b)*s - 13.125*(d*d*d*d)*(a*a)*(b*b*b)*s + 10.9375*(d*d*d*d)*a*(b*b*b*b)*s - 2.1875*(d*d*d*d)*(b*b*b*b*b)*s + 0.5625*(d*d*d)*(a*a*a)*g*p - 6.75*(d*d*d)*(a*a)*b*g*p + 15.75*(d*d*d)*a*(b*b)*g*p - 7.875*(d*d*d)*(b*b*b)*g*p + 1.6875*(d*d)*(a*a*a)*p*s - 20.25*(d*d)*(a*a)*b*p*s + 47.25*(d*d)*a*(b*b)*p*s - 23.625*(d*d)*(b*b*b)*p*s + 7.734375*d*a*g*(p*p) - 15.46875*d*b*g*(p*p) + 7.734375*a*(p*p)*s - 15.46875*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad848(double a, double b, double p, double d, double s, double g){
	return (0.00390625*(d*d*d*d)*(a*a*a*a)*g - 0.125*(d*d*d*d)*(a*a*a)*b*g + 0.65625*(d*d*d*d)*(a*a)*(b*b)*g - 0.875*(d*d*d*d)*a*(b*b*b)*g + 0.2734375*(d*d*d*d)*(b*b*b*b)*g + 0.015625*(d*d*d)*(a*a*a*a)*s - 0.5*(d*d*d)*(a*a*a)*b*s + 2.625*(d*d*d)*(a*a)*(b*b)*s - 3.5*(d*d*d)*a*(b*b*b)*s + 1.09375*(d*d*d)*(b*b*b*b)*s + 0.52734375*(d*d)*(a*a)*g*p - 2.8125*(d*d)*a*b*g*p + 2.4609375*(d*d)*(b*b)*g*p + 1.0546875*d*(a*a)*p*s - 5.625*d*a*b*p*s + 4.921875*d*(b*b)*p*s + 1.4501953125*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad849(double a, double b, double p, double d, double s, double g){
	return (0.0078125*(d*d*d)*(a*a*a)*g - 0.09375*(d*d*d)*(a*a)*b*g + 0.21875*(d*d*d)*a*(b*b)*g - 0.109375*(d*d*d)*(b*b*b)*g + 0.0234375*(d*d)*(a*a*a)*s - 0.28125*(d*d)*(a*a)*b*s + 0.65625*(d*d)*a*(b*b)*s - 0.328125*(d*d)*(b*b*b)*s + 0.21484375*d*a*g*p - 0.4296875*d*b*g*p + 0.21484375*a*p*s - 0.4296875*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8410(double a, double b, double p, double d, double s, double g){
	return (0.005859375*(d*d)*(a*a)*g - 0.03125*(d*d)*a*b*g + 0.02734375*(d*d)*(b*b)*g + 0.01171875*d*(a*a)*s - 0.0625*d*a*b*s + 0.0546875*d*(b*b)*s + 0.0322265625*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8411(double a, double b, double p, double d, double s, double g){
	return (0.001953125*d*a*g - 0.00390625*d*b*g + 0.001953125*a*s - 0.00390625*b*s)/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8412(double a, double b, double p, double d, double s, double g){
	return 0.000244140625*g/(p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad850(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g + 13.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*s + 14.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 20.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g*p + 5.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*g*p + 154.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 220.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*p*s + 55.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*p*s + 52.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 210.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 210.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 60.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*(p*p) + 3.75*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g*(p*p) + 472.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 1890.0*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 1890.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 540.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 33.75*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*(p*p)*s + 52.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p) - 525.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p) + 1312.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p) - 1050.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p) + 262.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p*p) - 15.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p*p) + 367.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p)*s - 3675.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p)*s + 9187.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p)*s - 7350.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p)*s + 1837.5*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p*p)*s - 105.0*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p*p)*s + 6.5625*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p) - 262.5*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p) + 1837.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p) - 3675.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p) + 2296.875*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p) - 367.5*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p) + 32.8125*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p)*s - 1312.5*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p)*s + 9187.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p)*s - 18375.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p)*s + 11484.375*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p)*s - 1837.5*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p)*s + 295.3125*(d*d*d)*(a*a*a)*g*(p*p*p*p*p) - 2362.5*(d*d*d)*(a*a)*b*g*(p*p*p*p*p) + 4134.375*(d*d*d)*a*(b*b)*g*(p*p*p*p*p) - 1653.75*(d*d*d)*(b*b*b)*g*(p*p*p*p*p) + 885.9375*(d*d)*(a*a*a)*(p*p*p*p*p)*s - 7087.5*(d*d)*(a*a)*b*(p*p*p*p*p)*s + 12403.125*(d*d)*a*(b*b)*(p*p*p*p*p)*s - 4961.25*(d*d)*(b*b*b)*(p*p*p*p*p)*s + 812.109375*d*a*g*(p*p*p*p*p*p) - 1299.375*d*b*g*(p*p*p*p*p*p) + 812.109375*a*(p*p*p*p*p*p)*s - 1299.375*b*(p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad851(double a, double b, double p, double d, double s, double g){
	return (-4.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 2.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*g - 48.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 30.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*s - 42.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p + 105.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*p - 60.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*p + 7.5*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g*p - 420.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s + 1050.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*p*s - 600.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*p*s + 75.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*p*s - 105.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 656.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 1050.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 525.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p) - 75.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p) + 1.875*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*(p*p) - 840.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 5250.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 8400.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 4200.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p)*s - 600.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p)*s + 15.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*(p*p)*s - 52.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 918.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 3675.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 4593.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 1837.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 183.75*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p) - 315.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 5512.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 22050.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 27562.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 11025.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 1102.5*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p)*s + 147.65625*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 2362.5*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 8268.75*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 8268.75*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 2067.1875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 590.625*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 9450.0*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 33075.0*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 33075.0*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 8268.75*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 1624.21875*(d*d)*(a*a)*g*(p*p*p*p*p) - 6496.875*(d*d)*a*b*g*(p*p*p*p*p) + 4547.8125*(d*d)*(b*b)*g*(p*p*p*p*p) + 3248.4375*d*(a*a)*(p*p*p*p*p)*s - 12993.75*d*a*b*(p*p*p*p*p)*s + 9095.625*d*(b*b)*(p*p*p*p*p)*s + 1055.7421875*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad852(double a, double b, double p, double d, double s, double g){
	return (7.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g - 10.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g + 2.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*g + 77.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*s - 110.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*s + 27.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*s + 52.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 210.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 210.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 60.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*p + 3.75*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g*p + 472.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 1890.0*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 1890.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s - 540.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*p*s + 33.75*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*p*s + 78.75*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 787.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 1968.75*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 1575.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 393.75*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) - 22.5*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p) + 551.25*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 5512.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 13781.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 11025.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 2756.25*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s - 157.5*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p)*s + 13.125*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 525.0*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 3675.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 7350.0*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 4593.75*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 735.0*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 65.625*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 2625.0*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 18375.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 36750.0*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 22968.75*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 3675.0*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 738.28125*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 5906.25*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 10335.9375*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 4134.375*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 2214.84375*(d*d)*(a*a*a)*(p*p*p*p)*s - 17718.75*(d*d)*(a*a)*b*(p*p*p*p)*s + 31007.8125*(d*d)*a*(b*b)*(p*p*p*p)*s - 12403.125*(d*d)*(b*b*b)*(p*p*p*p)*s + 2436.328125*d*a*g*(p*p*p*p*p) - 3898.125*d*b*g*(p*p*p*p*p) + 2436.328125*a*(p*p*p*p*p)*s - 3898.125*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad853(double a, double b, double p, double d, double s, double g){
	return (-7.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 17.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g - 10.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g + 1.25*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g - 70.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 175.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s - 100.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*s + 12.5*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*s - 35.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 218.75*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 350.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 175.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 25.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p + 0.625*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*p - 280.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 1750.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 2800.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 1400.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 200.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s + 5.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 459.375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 1837.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 2296.875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 918.75*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 91.875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) - 157.5*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 2756.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 11025.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 13781.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 5512.5*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 551.25*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 98.4375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 1575.0*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 5512.5*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 5512.5*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 1378.125*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 393.75*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 6300.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 22050.0*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 22050.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 5512.5*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 1353.515625*(d*d)*(a*a)*g*(p*p*p*p) - 5414.0625*(d*d)*a*b*g*(p*p*p*p) + 3789.84375*(d*d)*(b*b)*g*(p*p*p*p) + 2707.03125*d*(a*a)*(p*p*p*p)*s - 10828.125*d*a*b*(p*p*p*p)*s + 7579.6875*d*(b*b)*(p*p*p*p)*s + 1055.7421875*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad854(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 17.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 17.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 5.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g + 0.3125*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g + 39.375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 157.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 157.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 45.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s + 2.8125*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*s + 13.125*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 131.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 328.125*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 262.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 65.625*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 3.75*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p + 91.875*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 918.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 2296.875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 1837.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 459.375*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s + 3.28125*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 131.25*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 918.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 1837.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 1148.4375*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 183.75*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 16.40625*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 656.25*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 4593.75*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 9187.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 5742.1875*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 918.75*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 246.09375*(d*d*d)*(a*a*a)*g*(p*p*p) - 1968.75*(d*d*d)*(a*a)*b*g*(p*p*p) + 3445.3125*(d*d*d)*a*(b*b)*g*(p*p*p) - 1378.125*(d*d*d)*(b*b*b)*g*(p*p*p) + 738.28125*(d*d)*(a*a*a)*(p*p*p)*s - 5906.25*(d*d)*(a*a)*b*(p*p*p)*s + 10335.9375*(d*d)*a*(b*b)*(p*p*p)*s - 4134.375*(d*d)*(b*b*b)*(p*p*p)*s + 1015.13671875*d*a*g*(p*p*p*p) - 1624.21875*d*b*g*(p*p*p*p) + 1015.13671875*a*(p*p*p*p)*s - 1624.21875*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad855(double a, double b, double p, double d, double s, double g){
	return (-1.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 10.9375*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 17.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 8.75*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 1.25*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g + 0.03125*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g - 14.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 87.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 140.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 70.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 10.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + 0.25*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*s - 2.625*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 45.9375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 183.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 229.6875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 91.875*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 9.1875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p - 15.75*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 275.625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 1102.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 1378.125*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 551.25*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 55.125*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 14.765625*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 236.25*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 826.875*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 826.875*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 206.71875*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 59.0625*(d*d*d)*(a*a*a*a)*(p*p)*s - 945.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 3307.5*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 3307.5*(d*d*d)*a*(b*b*b)*(p*p)*s + 826.875*(d*d*d)*(b*b*b*b)*(p*p)*s + 270.703125*(d*d)*(a*a)*g*(p*p*p) - 1082.8125*(d*d)*a*b*g*(p*p*p) + 757.96875*(d*d)*(b*b)*g*(p*p*p) + 541.40625*d*(a*a)*(p*p*p)*s - 2165.625*d*a*b*(p*p*p)*s + 1515.9375*d*(b*b)*(p*p*p)*s + 263.935546875*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad856(double a, double b, double p, double d, double s, double g){
	return (0.4375*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 4.375*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 10.9375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 8.75*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 2.1875*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g + 3.0625*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 30.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 76.5625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 61.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 15.3125*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 0.875*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s + 0.21875*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 8.75*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 61.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 122.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 76.5625*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 12.25*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 1.09375*(d*d*d*d)*(a*a*a*a*a)*p*s - 43.75*(d*d*d*d)*(a*a*a*a)*b*p*s + 306.25*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 612.5*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 382.8125*(d*d*d*d)*a*(b*b*b*b)*p*s - 61.25*(d*d*d*d)*(b*b*b*b*b)*p*s + 24.609375*(d*d*d)*(a*a*a)*g*(p*p) - 196.875*(d*d*d)*(a*a)*b*g*(p*p) + 344.53125*(d*d*d)*a*(b*b)*g*(p*p) - 137.8125*(d*d*d)*(b*b*b)*g*(p*p) + 73.828125*(d*d)*(a*a*a)*(p*p)*s - 590.625*(d*d)*(a*a)*b*(p*p)*s + 1033.59375*(d*d)*a*(b*b)*(p*p)*s - 413.4375*(d*d)*(b*b*b)*(p*p)*s + 135.3515625*d*a*g*(p*p*p) - 216.5625*d*b*g*(p*p*p) + 135.3515625*a*(p*p*p)*s - 216.5625*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad857(double a, double b, double p, double d, double s, double g){
	return (-0.0625*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 1.09375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 4.375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 5.46875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 2.1875*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.21875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g - 0.375*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 6.5625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 26.25*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 32.8125*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 13.125*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 1.3125*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 0.703125*(d*d*d*d)*(a*a*a*a)*g*p - 11.25*(d*d*d*d)*(a*a*a)*b*g*p + 39.375*(d*d*d*d)*(a*a)*(b*b)*g*p - 39.375*(d*d*d*d)*a*(b*b*b)*g*p + 9.84375*(d*d*d*d)*(b*b*b*b)*g*p + 2.8125*(d*d*d)*(a*a*a*a)*p*s - 45.0*(d*d*d)*(a*a*a)*b*p*s + 157.5*(d*d*d)*(a*a)*(b*b)*p*s - 157.5*(d*d*d)*a*(b*b*b)*p*s + 39.375*(d*d*d)*(b*b*b*b)*p*s + 19.3359375*(d*d)*(a*a)*g*(p*p) - 77.34375*(d*d)*a*b*g*(p*p) + 54.140625*(d*d)*(b*b)*g*(p*p) + 38.671875*d*(a*a)*(p*p)*s - 154.6875*d*a*b*(p*p)*s + 108.28125*d*(b*b)*(p*p)*s + 25.13671875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad858(double a, double b, double p, double d, double s, double g){
	return (0.00390625*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.15625*(d*d*d*d*d)*(a*a*a*a)*b*g + 1.09375*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 2.1875*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 1.3671875*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.21875*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.01953125*(d*d*d*d)*(a*a*a*a*a)*s - 0.78125*(d*d*d*d)*(a*a*a*a)*b*s + 5.46875*(d*d*d*d)*(a*a*a)*(b*b)*s - 10.9375*(d*d*d*d)*(a*a)*(b*b*b)*s + 6.8359375*(d*d*d*d)*a*(b*b*b*b)*s - 1.09375*(d*d*d*d)*(b*b*b*b*b)*s + 0.87890625*(d*d*d)*(a*a*a)*g*p - 7.03125*(d*d*d)*(a*a)*b*g*p + 12.3046875*(d*d*d)*a*(b*b)*g*p - 4.921875*(d*d*d)*(b*b*b)*g*p + 2.63671875*(d*d)*(a*a*a)*p*s - 21.09375*(d*d)*(a*a)*b*p*s + 36.9140625*(d*d)*a*(b*b)*p*s - 14.765625*(d*d)*(b*b*b)*p*s + 7.2509765625*d*a*g*(p*p) - 11.6015625*d*b*g*(p*p) + 7.2509765625*a*(p*p)*s - 11.6015625*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad859(double a, double b, double p, double d, double s, double g){
	return (0.009765625*(d*d*d*d)*(a*a*a*a)*g - 0.15625*(d*d*d*d)*(a*a*a)*b*g + 0.546875*(d*d*d*d)*(a*a)*(b*b)*g - 0.546875*(d*d*d*d)*a*(b*b*b)*g + 0.13671875*(d*d*d*d)*(b*b*b*b)*g + 0.0390625*(d*d*d)*(a*a*a*a)*s - 0.625*(d*d*d)*(a*a*a)*b*s + 2.1875*(d*d*d)*(a*a)*(b*b)*s - 2.1875*(d*d*d)*a*(b*b*b)*s + 0.546875*(d*d*d)*(b*b*b*b)*s + 0.537109375*(d*d)*(a*a)*g*p - 2.1484375*(d*d)*a*b*g*p + 1.50390625*(d*d)*(b*b)*g*p + 1.07421875*d*(a*a)*p*s - 4.296875*d*a*b*p*s + 3.0078125*d*(b*b)*p*s + 1.04736328125*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8510(double a, double b, double p, double d, double s, double g){
	return (0.009765625*(d*d*d)*(a*a*a)*g - 0.078125*(d*d*d)*(a*a)*b*g + 0.13671875*(d*d*d)*a*(b*b)*g - 0.0546875*(d*d*d)*(b*b*b)*g + 0.029296875*(d*d)*(a*a*a)*s - 0.234375*(d*d)*(a*a)*b*s + 0.41015625*(d*d)*a*(b*b)*s - 0.1640625*(d*d)*(b*b*b)*s + 0.1611328125*d*a*g*p - 0.2578125*d*b*g*p + 0.1611328125*a*p*s - 0.2578125*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8511(double a, double b, double p, double d, double s, double g){
	return (0.0048828125*(d*d)*(a*a)*g - 0.01953125*(d*d)*a*b*g + 0.013671875*(d*d)*(b*b)*g + 0.009765625*d*(a*a)*s - 0.0390625*d*a*b*s + 0.02734375*d*(b*b)*s + 0.01904296875*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8512(double a, double b, double p, double d, double s, double g){
	return (0.001220703125*d*a*g - 0.001953125*d*b*g + 0.001220703125*a*s - 0.001953125*b*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8513(double a, double b, double p, double d, double s, double g){
	return 0.0001220703125*g/(p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad860(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g + 14.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*s + 14.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 24.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g*p + 7.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*g*p + 168.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 288.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*p*s + 90.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*p*s + 52.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 252.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 315.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 120.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*(p*p) + 11.25*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g*(p*p) + 525.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 2520.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 3150.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 1200.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 112.5*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*(p*p)*s + 52.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p*p) - 630.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p*p) + 1968.75*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p*p) - 2100.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p*p) + 787.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p*p) - 90.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p*p) + 1.875*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*(p*p*p) + 420.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p*p)*s - 5040.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p*p)*s + 15750.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p*p)*s - 16800.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p*p)*s + 6300.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p*p)*s - 720.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p*p)*s + 15.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*(p*p*p)*s + 6.5625*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p*p) - 315.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p*p) + 2756.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p*p) - 7350.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p*p) + 6890.625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p*p) - 2205.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p*p) + 183.75*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p*p) + 39.375*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p*p)*s - 1890.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p*p)*s + 16537.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p*p)*s - 44100.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p*p)*s + 41343.75*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p*p)*s - 13230.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p*p)*s + 1102.5*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p*p)*s + 442.96875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p*p) - 4725.0*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p*p) + 12403.125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p*p) - 9922.5*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p*p) + 2067.1875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p*p) + 1771.875*(d*d*d)*(a*a*a*a)*(p*p*p*p*p)*s - 18900.0*(d*d*d)*(a*a*a)*b*(p*p*p*p*p)*s + 49612.5*(d*d*d)*(a*a)*(b*b)*(p*p*p*p*p)*s - 39690.0*(d*d*d)*a*(b*b*b)*(p*p*p*p*p)*s + 8268.75*(d*d*d)*(b*b*b*b)*(p*p*p*p*p)*s + 2436.328125*(d*d)*(a*a)*g*(p*p*p*p*p*p) - 7796.25*(d*d)*a*b*g*(p*p*p*p*p*p) + 4547.8125*(d*d)*(b*b)*g*(p*p*p*p*p*p) + 4872.65625*d*(a*a)*(p*p*p*p*p*p)*s - 15592.5*d*a*b*(p*p*p*p*p*p)*s + 9095.625*d*(b*b)*(p*p*p*p*p*p)*s + 1055.7421875*g*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad861(double a, double b, double p, double d, double s, double g){
	return (-4.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 3.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g - 52.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 39.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*s - 42.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 126.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 90.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g*p + 15.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*g*p - 462.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 1386.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 990.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*p*s + 165.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*p*s - 105.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 787.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 1575.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 1050.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 225.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*(p*p) + 11.25*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g*(p*p) - 945.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 7087.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 14175.0*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 9450.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 2025.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 101.25*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*(p*p)*s - 52.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p*p) + 1102.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p) - 5512.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p) + 9187.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p) - 5512.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p) + 1102.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p*p) - 52.5*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p*p) - 367.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p*p)*s + 7717.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p)*s - 38587.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p)*s + 64312.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p)*s - 38587.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p)*s + 7717.5*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p*p)*s - 367.5*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p*p)*s + 177.1875*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p) - 3543.75*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p) + 16537.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p) - 24806.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p) + 12403.125*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p) - 1653.75*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p) + 885.9375*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p)*s - 17718.75*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p)*s + 82687.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p)*s - 124031.25*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p)*s + 62015.625*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p)*s - 8268.75*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p)*s + 3248.4375*(d*d*d)*(a*a*a)*g*(p*p*p*p*p) - 19490.625*(d*d*d)*(a*a)*b*g*(p*p*p*p*p) + 27286.875*(d*d*d)*a*(b*b)*g*(p*p*p*p*p) - 9095.625*(d*d*d)*(b*b*b)*g*(p*p*p*p*p) + 9745.3125*(d*d)*(a*a*a)*(p*p*p*p*p)*s - 58471.875*(d*d)*(a*a)*b*(p*p*p*p*p)*s + 81860.625*(d*d)*a*(b*b)*(p*p*p*p*p)*s - 27286.875*(d*d)*(b*b*b)*(p*p*p*p*p)*s + 6334.453125*d*a*g*(p*p*p*p*p*p) - 8445.9375*d*b*g*(p*p*p*p*p*p) + 6334.453125*a*(p*p*p*p*p*p)*s - 8445.9375*b*(p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad862(double a, double b, double p, double d, double s, double g){
	return (7.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 12.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 3.75*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*g + 84.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 144.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 45.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*s + 52.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*p - 252.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p + 315.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*p - 120.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*p + 11.25*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g*p + 525.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*p*s - 2520.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s + 3150.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*p*s - 1200.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*p*s + 112.5*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*p*s + 78.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p) - 945.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 2953.125*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 3150.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 1181.25*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p) - 135.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p) + 2.8125*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*(p*p) + 630.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p)*s - 7560.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 23625.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 25200.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 9450.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p)*s - 1080.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p)*s + 22.5*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*(p*p)*s + 13.125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p) - 630.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 5512.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 14700.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 13781.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 4410.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 367.5*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p) + 78.75*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p)*s - 3780.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 33075.0*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 88200.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 82687.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 26460.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 2205.0*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p)*s + 1107.421875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 11812.5*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 31007.8125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 24806.25*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 5167.96875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 4429.6875*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 47250.0*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 124031.25*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 99225.0*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 20671.875*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 7308.984375*(d*d)*(a*a)*g*(p*p*p*p*p) - 23388.75*(d*d)*a*b*g*(p*p*p*p*p) + 13643.4375*(d*d)*(b*b)*g*(p*p*p*p*p) + 14617.96875*d*(a*a)*(p*p*p*p*p)*s - 46777.5*d*a*b*(p*p*p*p*p)*s + 27286.875*d*(b*b)*(p*p*p*p*p)*s + 3695.09765625*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad863(double a, double b, double p, double d, double s, double g){
	return (-7.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g + 21.0 *(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g - 15.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g + 2.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*g - 77.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*s + 231.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*s - 165.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*s + 27.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*s - 35.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 262.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 525.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 350.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 75.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*p + 3.75*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g*p - 315.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 2362.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 4725.0*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 3150.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s - 675.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*p*s + 33.75*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 551.25*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 2756.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 4593.75*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 2756.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 551.25*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) - 26.25*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p) - 183.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 3858.75*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 19293.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 32156.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 19293.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 3858.75*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s - 183.75*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p)*s + 118.125*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 2362.5*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 11025.0*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 16537.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 8268.75*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 1102.5*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 590.625*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 11812.5*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 55125.0*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 82687.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 41343.75*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 5512.5*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 2707.03125*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 16242.1875*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 22739.0625*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 7579.6875*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 8121.09375*(d*d)*(a*a*a)*(p*p*p*p)*s - 48726.5625*(d*d)*(a*a)*b*(p*p*p*p)*s + 68217.1875*(d*d)*a*(b*b)*(p*p*p*p)*s - 22739.0625*(d*d)*(b*b*b)*(p*p*p*p)*s + 6334.453125*d*a*g*(p*p*p*p*p) - 8445.9375*d*b*g*(p*p*p*p*p) + 6334.453125*a*(p*p*p*p*p)*s - 8445.9375*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad864(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 21.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 26.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g - 10.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g + 0.9375*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g + 43.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 210.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 262.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s - 100.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*s + 9.375*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*s + 13.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 157.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 492.1875*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 525.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 196.875*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 22.5*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p + 0.46875*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*p + 105.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 1260.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 3937.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 4200.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 1575.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 180.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s + 3.75*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*p*s + 3.28125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 157.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 1378.125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 3675.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 3445.3125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 1102.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 91.875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 19.6875*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 945.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 8268.75*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 22050.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 20671.875*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 6615.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 551.25*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 369.140625*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 3937.5*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 10335.9375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 8268.75*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 1722.65625*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 1476.5625*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 15750.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 41343.75*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 33075.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 6890.625*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 3045.41015625*(d*d)*(a*a)*g*(p*p*p*p) - 9745.3125*(d*d)*a*b*g*(p*p*p*p) + 5684.765625*(d*d)*(b*b)*g*(p*p*p*p) + 6090.8203125*d*(a*a)*(p*p*p*p)*s - 19490.625*d*a*b*(p*p*p*p)*s + 11369.53125*d*(b*b)*(p*p*p*p)*s + 1847.548828125*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad865(double a, double b, double p, double d, double s, double g){
	return (-1.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 13.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 26.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 17.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 3.75*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g + 0.1875*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g - 15.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 118.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 236.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 157.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 33.75*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s + 1.6875*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*s - 2.625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 55.125*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 275.625*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 459.375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 275.625*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 55.125*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 2.625*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p - 18.375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 385.875*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 1929.375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 3215.625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 1929.375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 385.875*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 18.375*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s + 17.71875*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 354.375*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 1653.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 2480.625*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 1240.3125*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 165.375*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 88.59375*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 1771.875*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 8268.75*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 12403.125*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 6201.5625*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 826.875*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 541.40625*(d*d*d)*(a*a*a)*g*(p*p*p) - 3248.4375*(d*d*d)*(a*a)*b*g*(p*p*p) + 4547.8125*(d*d*d)*a*(b*b)*g*(p*p*p) - 1515.9375*(d*d*d)*(b*b*b)*g*(p*p*p) + 1624.21875*(d*d)*(a*a*a)*(p*p*p)*s - 9745.3125*(d*d)*(a*a)*b*(p*p*p)*s + 13643.4375*(d*d)*a*(b*b)*(p*p*p)*s - 4547.8125*(d*d)*(b*b*b)*(p*p*p)*s + 1583.61328125*d*a*g*(p*p*p*p) - 2111.484375*d*b*g*(p*p*p*p) + 1583.61328125*a*(p*p*p*p)*s - 2111.484375*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad866(double a, double b, double p, double d, double s, double g){
	return (0.4375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 5.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 16.40625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 17.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 6.5625*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 0.75*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g + 0.015625*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g + 3.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 42.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 131.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 140.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 52.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 6.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + 0.125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*s + 0.21875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 10.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 91.875*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 245.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 229.6875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 73.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 6.125*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 1.3125*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 63.0*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 551.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 1470.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 1378.125*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 441.0 *(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 36.75*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 36.9140625*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 393.75*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 1033.59375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 826.875*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 172.265625*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 147.65625*(d*d*d)*(a*a*a*a)*(p*p)*s - 1575.0*(d*d*d)*(a*a*a)*b*(p*p)*s + 4134.375*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 3307.5*(d*d*d)*a*(b*b*b)*(p*p)*s + 689.0625*(d*d*d)*(b*b*b*b)*(p*p)*s + 406.0546875*(d*d)*(a*a)*g*(p*p*p) - 1299.375*(d*d)*a*b*g*(p*p*p) + 757.96875*(d*d)*(b*b)*g*(p*p*p) + 812.109375*d*(a*a)*(p*p*p)*s - 2598.75*d*a*b*(p*p*p)*s + 1515.9375*d*(b*b)*(p*p*p)*s + 307.9248046875*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad867(double a, double b, double p, double d, double s, double g){
	return (-0.0625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 1.3125*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 6.5625*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 10.9375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 6.5625*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 1.3125*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.0625*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g - 0.4375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 9.1875*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 45.9375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 76.5625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 45.9375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 9.1875*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 0.4375*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s + 0.84375*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 16.875*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 78.75*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 118.125*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 59.0625*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 7.875*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 4.21875*(d*d*d*d)*(a*a*a*a*a)*p*s - 84.375*(d*d*d*d)*(a*a*a*a)*b*p*s + 393.75*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 590.625*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 295.3125*(d*d*d*d)*a*(b*b*b*b)*p*s - 39.375*(d*d*d*d)*(b*b*b*b*b)*p*s + 38.671875*(d*d*d)*(a*a*a)*g*(p*p) - 232.03125*(d*d*d)*(a*a)*b*g*(p*p) + 324.84375*(d*d*d)*a*(b*b)*g*(p*p) - 108.28125*(d*d*d)*(b*b*b)*g*(p*p) + 116.015625*(d*d)*(a*a*a)*(p*p)*s - 696.09375*(d*d)*(a*a)*b*(p*p)*s + 974.53125*(d*d)*a*(b*b)*(p*p)*s - 324.84375*(d*d)*(b*b*b)*(p*p)*s + 150.8203125*d*a*g*(p*p*p) - 201.09375*d*b*g*(p*p*p) + 150.8203125*a*(p*p*p)*s - 201.09375*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad868(double a, double b, double p, double d, double s, double g){
	return (0.00390625*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 0.1875*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 1.640625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 4.375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 4.1015625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 1.3125*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.109375*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 0.0234375*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 1.125*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 9.84375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 26.25*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 24.609375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 7.875*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.65625*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 1.318359375*(d*d*d*d)*(a*a*a*a)*g*p - 14.0625*(d*d*d*d)*(a*a*a)*b*g*p + 36.9140625*(d*d*d*d)*(a*a)*(b*b)*g*p - 29.53125*(d*d*d*d)*a*(b*b*b)*g*p + 6.15234375*(d*d*d*d)*(b*b*b*b)*g*p + 5.2734375*(d*d*d)*(a*a*a*a)*p*s - 56.25*(d*d*d)*(a*a*a)*b*p*s + 147.65625*(d*d*d)*(a*a)*(b*b)*p*s - 118.125*(d*d*d)*a*(b*b*b)*p*s + 24.609375*(d*d*d)*(b*b*b*b)*p*s + 21.7529296875*(d*d)*(a*a)*g*(p*p) - 69.609375*(d*d)*a*b*g*(p*p) + 40.60546875*(d*d)*(b*b)*g*(p*p) + 43.505859375*d*(a*a)*(p*p)*s - 139.21875*d*a*b*(p*p)*s + 81.2109375*d*(b*b)*(p*p)*s + 21.99462890625*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad869(double a, double b, double p, double d, double s, double g){
	return (0.01171875*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.234375*(d*d*d*d*d)*(a*a*a*a)*b*g + 1.09375*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 1.640625*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.8203125*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.109375*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.05859375*(d*d*d*d)*(a*a*a*a*a)*s - 1.171875*(d*d*d*d)*(a*a*a*a)*b*s + 5.46875*(d*d*d*d)*(a*a*a)*(b*b)*s - 8.203125*(d*d*d*d)*(a*a)*(b*b*b)*s + 4.1015625*(d*d*d*d)*a*(b*b*b*b)*s - 0.546875*(d*d*d*d)*(b*b*b*b*b)*s + 1.07421875*(d*d*d)*(a*a*a)*g*p - 6.4453125*(d*d*d)*(a*a)*b*g*p + 9.0234375*(d*d*d)*a*(b*b)*g*p - 3.0078125*(d*d*d)*(b*b*b)*g*p + 3.22265625*(d*d)*(a*a*a)*p*s - 19.3359375*(d*d)*(a*a)*b*p*s + 27.0703125*(d*d)*a*(b*b)*p*s - 9.0234375*(d*d)*(b*b*b)*p*s + 6.2841796875*d*a*g*(p*p) - 8.37890625*d*b*g*(p*p) + 6.2841796875*a*(p*p)*s - 8.37890625*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8610(double a, double b, double p, double d, double s, double g){
	return (0.0146484375*(d*d*d*d)*(a*a*a*a)*g - 0.15625*(d*d*d*d)*(a*a*a)*b*g + 0.41015625*(d*d*d*d)*(a*a)*(b*b)*g - 0.328125*(d*d*d*d)*a*(b*b*b)*g + 0.068359375*(d*d*d*d)*(b*b*b*b)*g + 0.05859375*(d*d*d)*(a*a*a*a)*s - 0.625*(d*d*d)*(a*a*a)*b*s + 1.640625*(d*d*d)*(a*a)*(b*b)*s - 1.3125*(d*d*d)*a*(b*b*b)*s + 0.2734375*(d*d*d)*(b*b*b*b)*s + 0.4833984375*(d*d)*(a*a)*g*p - 1.546875*(d*d)*a*b*g*p + 0.90234375*(d*d)*(b*b)*g*p + 0.966796875*d*(a*a)*p*s - 3.09375*d*a*b*p*s + 1.8046875*d*(b*b)*p*s + 0.733154296875*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8611(double a, double b, double p, double d, double s, double g){
	return (0.009765625*(d*d*d)*(a*a*a)*g - 0.05859375*(d*d*d)*(a*a)*b*g + 0.08203125*(d*d*d)*a*(b*b)*g - 0.02734375*(d*d*d)*(b*b*b)*g + 0.029296875*(d*d)*(a*a*a)*s - 0.17578125*(d*d)*(a*a)*b*s + 0.24609375*(d*d)*a*(b*b)*s - 0.08203125*(d*d)*(b*b*b)*s + 0.1142578125*d*a*g*p - 0.15234375*d*b*g*p + 0.1142578125*a*p*s - 0.15234375*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8612(double a, double b, double p, double d, double s, double g){
	return (0.003662109375*(d*d)*(a*a)*g - 0.01171875*(d*d)*a*b*g + 0.0068359375*(d*d)*(b*b)*g + 0.00732421875*d*(a*a)*s - 0.0234375*d*a*b*s + 0.013671875*d*(b*b)*s + 0.0111083984375*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8613(double a, double b, double p, double d, double s, double g){
	return (0.000732421875*d*a*g - 0.0009765625*d*b*g + 0.000732421875*a*s - 0.0009765625*b*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8614(double a, double b, double p, double d, double s, double g){
	return 6.103515625e-5*g/(p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad870(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g + 15.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*s + 14.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 28.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g*p + 10.5*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g*p + 182.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 364.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*p*s + 136.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*p*s + 52.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 294.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 441.0 *(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 210.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g*(p*p) + 26.25*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*g*(p*p) + 577.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 3234.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 4851.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 2310.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 288.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*(p*p)*s + 52.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*(p*p*p) - 735.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*(p*p*p) + 2756.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p*p) - 3675.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p*p) + 1837.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*(p*p*p) - 315.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*(p*p*p) + 13.125*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g*(p*p*p) + 472.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*(p*p*p)*s - 6615.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*(p*p*p)*s + 24806.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p*p)*s - 33075.0*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p*p)*s + 16537.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*(p*p*p)*s - 2835.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*(p*p*p)*s + 118.125*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*(p*p*p)*s + 6.5625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p*p*p) - 367.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p*p*p) + 3858.75*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p*p) - 12862.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p*p) + 16078.125*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p*p) - 7717.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p*p) + 1286.25*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p*p*p) - 52.5*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p*p*p) + 45.9375*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p*p*p)*s - 2572.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p*p*p)*s + 27011.25*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p*p)*s - 90037.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p*p)*s + 112546.875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p*p)*s - 54022.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p*p)*s + 9003.75*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p*p*p)*s - 367.5*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p*p*p)*s + 620.15625*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p*p) - 8268.75*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p*p) + 28940.625*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p*p) - 34728.75*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p*p) + 14470.3125*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p*p) - 1653.75*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p*p) + 3100.78125*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p*p)*s - 41343.75*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p*p)*s + 144703.125*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p*p)*s - 173643.75*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p*p)*s + 72351.5625*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p*p)*s - 8268.75*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p*p)*s + 5684.765625*(d*d*d)*(a*a*a)*g*(p*p*p*p*p*p) - 27286.875*(d*d*d)*(a*a)*b*g*(p*p*p*p*p*p) + 31834.6875*(d*d*d)*a*(b*b)*g*(p*p*p*p*p*p) - 9095.625*(d*d*d)*(b*b*b)*g*(p*p*p*p*p*p) + 17054.296875*(d*d)*(a*a*a)*(p*p*p*p*p*p)*s - 81860.625*(d*d)*(a*a)*b*(p*p*p*p*p*p)*s + 95504.0625*(d*d)*a*(b*b)*(p*p*p*p*p*p)*s - 27286.875*(d*d)*(b*b*b)*(p*p*p*p*p*p)*s + 7390.1953125*d*a*g*(p*p*p*p*p*p*p) - 8445.9375*d*b*g*(p*p*p*p*p*p*p) + 7390.1953125*a*(p*p*p*p*p*p*p)*s - 8445.9375*b*(p*p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad871(double a, double b, double p, double d, double s, double g){
	return (-4.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 3.5*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g - 56.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 49.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*s - 42.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 147.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 126.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g*p + 26.25*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*g*p - 504.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 1764.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 1512.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*p*s + 315.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*p*s - 105.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 918.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 2205.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 1837.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 525.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*(p*p) + 39.375*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g*(p*p) - 1050.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 9187.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 22050.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 18375.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 5250.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 393.75*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*(p*p)*s - 52.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p*p) + 1286.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p*p) - 7717.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p*p) + 16078.125*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p*p) - 12862.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p*p) + 3858.75*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p*p) - 367.5*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p*p) + 6.5625*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*(p*p*p) - 420.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p*p)*s + 10290.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p*p)*s - 61740.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p*p)*s + 128625.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p*p)*s - 102900.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p*p)*s + 30870.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p*p)*s - 2940.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p*p)*s + 52.5*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*(p*p*p)*s + 206.71875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p*p) - 4961.25*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p*p) + 28940.625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p*p) - 57881.25*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p*p) + 43410.9375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p*p) - 11576.25*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p*p) + 826.875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p*p) + 1240.3125*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p*p)*s - 29767.5*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p*p)*s + 173643.75*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p*p)*s - 347287.5*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p*p)*s + 260465.625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p*p)*s - 69457.5*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p*p)*s + 4961.25*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p*p)*s + 5684.765625*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p*p) - 45478.125*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p*p) + 95504.0625*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p*p) - 63669.375*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p*p) + 11369.53125*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p*p) + 22739.0625*(d*d*d)*(a*a*a*a)*(p*p*p*p*p)*s - 181912.5*(d*d*d)*(a*a*a)*b*(p*p*p*p*p)*s + 382016.25*(d*d*d)*(a*a)*(b*b)*(p*p*p*p*p)*s - 254677.5*(d*d*d)*a*(b*b*b)*(p*p*p*p*p)*s + 45478.125*(d*d*d)*(b*b*b*b)*(p*p*p*p*p)*s + 22170.5859375*(d*d)*(a*a)*g*(p*p*p*p*p*p) - 59121.5625*(d*d)*a*b*g*(p*p*p*p*p*p) + 29560.78125*(d*d)*(b*b)*g*(p*p*p*p*p*p) + 44341.171875*d*(a*a)*(p*p*p*p*p*p)*s - 118243.125*d*a*b*(p*p*p*p*p*p)*s + 59121.5625*d*(b*b)*(p*p*p*p*p*p)*s + 7918.06640625*g*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad872(double a, double b, double p, double d, double s, double g){
	return (7.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 14.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 5.25*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g + 91.0 *(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 182.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 68.25*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*s + 52.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g*p - 294.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 441.0 *(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 210.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g*p + 26.25*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*g*p + 577.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*p*s - 3234.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 4851.0 *(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 2310.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*p*s + 288.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*p*s + 78.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*(p*p) - 1102.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 4134.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 5512.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 2756.25*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 472.5*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*(p*p) + 19.6875*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g*(p*p) + 708.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*(p*p)*s - 9922.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 37209.375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 49612.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 24806.25*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 4252.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 177.1875*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*(p*p)*s + 13.125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p*p) - 735.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p*p) + 7717.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p) - 25725.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p) + 32156.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p) - 15435.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p) + 2572.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p*p) - 105.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p*p) + 91.875*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p*p)*s - 5145.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p*p)*s + 54022.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p)*s - 180075.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p)*s + 225093.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p)*s - 108045.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p)*s + 18007.5*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p*p)*s - 735.0*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p*p)*s + 1550.390625*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p) - 20671.875*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p) + 72351.5625*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p) - 86821.875*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p) + 36175.78125*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p) - 4134.375*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p) + 7751.953125*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p)*s - 103359.375*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p)*s + 361757.8125*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p)*s - 434109.375*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p)*s + 180878.90625*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p)*s - 20671.875*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p)*s + 17054.296875*(d*d*d)*(a*a*a)*g*(p*p*p*p*p) - 81860.625*(d*d*d)*(a*a)*b*g*(p*p*p*p*p) + 95504.0625*(d*d*d)*a*(b*b)*g*(p*p*p*p*p) - 27286.875*(d*d*d)*(b*b*b)*g*(p*p*p*p*p) + 51162.890625*(d*d)*(a*a*a)*(p*p*p*p*p)*s - 245581.875*(d*d)*(a*a)*b*(p*p*p*p*p)*s + 286512.1875*(d*d)*a*(b*b)*(p*p*p*p*p)*s - 81860.625*(d*d)*(b*b*b)*(p*p*p*p*p)*s + 25865.68359375*d*a*g*(p*p*p*p*p*p) - 29560.78125*d*b*g*(p*p*p*p*p*p) + 25865.68359375*a*(p*p*p*p*p*p)*s - 29560.78125*b*(p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad873(double a, double b, double p, double d, double s, double g){
	return (-7.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g + 24.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 21.0 *(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 4.375*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*g - 84.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*s + 294.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 252.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 52.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*s - 35.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*p + 306.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*p - 735.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p + 612.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*p - 175.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*p + 13.125*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g*p - 350.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*p*s + 3062.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*p*s - 7350.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s + 6125.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*p*s - 1750.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*p*s + 131.25*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p) + 643.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p) - 3858.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 8039.0625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 6431.25*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 1929.375*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p) - 183.75*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p) + 3.28125*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*(p*p) - 210.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p)*s + 5145.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p)*s - 30870.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 64312.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 51450.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 15435.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p)*s - 1470.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p)*s + 26.25*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*(p*p)*s + 137.8125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p) - 3307.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 19293.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 38587.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 28940.625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 7717.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 551.25*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p) + 826.875*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p)*s - 19845.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 115762.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 231525.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 173643.75*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 46305.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 3307.5*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p)*s + 4737.3046875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 37898.4375*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 79586.71875*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 53057.8125*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 9474.609375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 18949.21875*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 151593.75*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 318346.875*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 212231.25*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 37898.4375*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 22170.5859375*(d*d)*(a*a)*g*(p*p*p*p*p) - 59121.5625*(d*d)*a*b*g*(p*p*p*p*p) + 29560.78125*(d*d)*(b*b)*g*(p*p*p*p*p) + 44341.171875*d*(a*a)*(p*p*p*p*p)*s - 118243.125*d*a*b*(p*p*p*p*p)*s + 59121.5625*d*(b*b)*(p*p*p*p*p)*s + 9237.744140625*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad874(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g - 24.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g + 36.75*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g - 17.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g + 2.1875*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*g + 48.125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*s - 269.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*s + 404.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*s - 192.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*s + 24.0625*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*s + 13.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*p - 183.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 689.0625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 918.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 459.375*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 78.75*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*p + 3.28125*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g*p + 118.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*p*s - 1653.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 6201.5625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 8268.75*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 4134.375*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s - 708.75*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*p*s + 29.53125*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*p*s + 3.28125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p) - 183.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 1929.375*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 6431.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 8039.0625*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 3858.75*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 643.125*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) - 26.25*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p) + 22.96875*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p)*s - 1286.25*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 13505.625*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 45018.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 56273.4375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 27011.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 4501.875*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s - 183.75*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p)*s + 516.796875*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 6890.625*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 24117.1875*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 28940.625*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 12058.59375*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 1378.125*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 2583.984375*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 34453.125*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 120585.9375*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 144703.125*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 60292.96875*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 6890.625*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 7105.95703125*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 34108.59375*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 39793.359375*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 11369.53125*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 21317.87109375*(d*d)*(a*a*a)*(p*p*p*p)*s - 102325.78125*(d*d)*(a*a)*b*(p*p*p*p)*s + 119380.078125*(d*d)*a*(b*b)*(p*p*p*p)*s - 34108.59375*(d*d)*(b*b*b)*(p*p*p*p)*s + 12932.841796875*d*a*g*(p*p*p*p*p) - 14780.390625*d*b*g*(p*p*p*p*p) + 12932.841796875*a*(p*p*p*p*p)*s - 14780.390625*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad875(double a, double b, double p, double d, double s, double g){
	return (-1.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g + 15.3125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 36.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 30.625*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g - 8.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g + 0.65625*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g - 17.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*s + 153.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 367.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 306.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s - 87.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*s + 6.5625*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*s - 2.625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 64.3125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 385.875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 803.90625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 643.125*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 192.9375*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 18.375*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p + 0.328125*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*p - 21.0 *(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 514.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 3087.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 6431.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 5145.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 1543.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 147.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s + 2.625*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*p*s + 20.671875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 496.125*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 2894.0625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 5788.125*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 4341.09375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 1157.625*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 82.6875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 124.03125*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 2976.75*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 17364.375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 34728.75*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 26046.5625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 6945.75*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 496.125*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 947.4609375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 7579.6875*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 15917.34375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 10611.5625*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 1894.921875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 3789.84375*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 30318.75*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 63669.375*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 42446.25*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 7579.6875*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 5542.646484375*(d*d)*(a*a)*g*(p*p*p*p) - 14780.390625*(d*d)*a*b*g*(p*p*p*p) + 7390.1953125*(d*d)*(b*b)*g*(p*p*p*p) + 11085.29296875*d*(a*a)*(p*p*p*p)*s - 29560.78125*d*a*b*(p*p*p*p)*s + 14780.390625*d*(b*b)*(p*p*p*p)*s + 2771.3232421875*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad876(double a, double b, double p, double d, double s, double g){
	return (0.4375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 6.125*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 22.96875*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 30.625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 15.3125*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 2.625*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g + 0.109375*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g + 3.9375*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s - 55.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 206.71875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 275.625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 137.8125*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 23.625*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s + 0.984375*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*s + 0.21875*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 12.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 128.625*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 428.75*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 535.9375*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 257.25*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 42.875*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 1.75*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p + 1.53125*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 85.75*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 900.375*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 3001.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 3751.5625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 1800.75*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 300.125*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 12.25*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s + 51.6796875*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 689.0625*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 2411.71875*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 2894.0625*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 1205.859375*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 137.8125*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 258.3984375*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 3445.3125*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 12058.59375*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 14470.3125*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 6029.296875*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 689.0625*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 947.4609375*(d*d*d)*(a*a*a)*g*(p*p*p) - 4547.8125*(d*d*d)*(a*a)*b*g*(p*p*p) + 5305.78125*(d*d*d)*a*(b*b)*g*(p*p*p) - 1515.9375*(d*d*d)*(b*b*b)*g*(p*p*p) + 2842.3828125*(d*d)*(a*a*a)*(p*p*p)*s - 13643.4375*(d*d)*(a*a)*b*(p*p*p)*s + 15917.34375*(d*d)*a*(b*b)*(p*p*p)*s - 4547.8125*(d*d)*(b*b*b)*(p*p*p)*s + 2155.4736328125*d*a*g*(p*p*p*p) - 2463.3984375*d*b*g*(p*p*p*p) + 2155.4736328125*a*(p*p*p*p)*s - 2463.3984375*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad877(double a, double b, double p, double d, double s, double g){
	return (-0.0625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 1.53125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 9.1875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 19.140625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 15.3125*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 4.59375*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 0.4375*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g + 0.0078125*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g - 0.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 12.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 73.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 153.125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 122.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 36.75*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 3.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + 0.0625*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*s + 0.984375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 23.625*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 137.8125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 275.625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 206.71875*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 55.125*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 3.9375*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 5.90625*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 141.75*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 826.875*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 1653.75*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 1240.3125*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 330.75*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 23.625*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 67.67578125*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 541.40625*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 1136.953125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 757.96875*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 135.3515625*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 270.703125*(d*d*d)*(a*a*a*a)*(p*p)*s - 2165.625*(d*d*d)*(a*a*a)*b*(p*p)*s + 4547.8125*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 3031.875*(d*d*d)*a*(b*b*b)*(p*p)*s + 541.40625*(d*d*d)*(b*b*b*b)*(p*p)*s + 527.87109375*(d*d)*(a*a)*g*(p*p*p) - 1407.65625*(d*d)*a*b*g*(p*p*p) + 703.828125*(d*d)*(b*b)*g*(p*p*p) + 1055.7421875*d*(a*a)*(p*p*p)*s - 2815.3125*d*a*b*(p*p*p)*s + 1407.65625*d*(b*b)*(p*p*p)*s + 329.91943359375*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad878(double a, double b, double p, double d, double s, double g){
	return (0.00390625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 0.21875*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 2.296875*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 7.65625*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 9.5703125*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 4.59375*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.765625*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.03125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g + 0.02734375*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 1.53125*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 16.078125*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 53.59375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 66.9921875*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 32.15625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 5.359375*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 0.21875*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s + 1.845703125*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 24.609375*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 86.1328125*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 103.359375*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 43.06640625*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 4.921875*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 9.228515625*(d*d*d*d)*(a*a*a*a*a)*p*s - 123.046875*(d*d*d*d)*(a*a*a*a)*b*p*s + 430.6640625*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 516.796875*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 215.33203125*(d*d*d*d)*a*(b*b*b*b)*p*s - 24.609375*(d*d*d*d)*(b*b*b*b*b)*p*s + 50.7568359375*(d*d*d)*(a*a*a)*g*(p*p) - 243.6328125*(d*d*d)*(a*a)*b*g*(p*p) + 284.23828125*(d*d*d)*a*(b*b)*g*(p*p) - 81.2109375*(d*d*d)*(b*b*b)*g*(p*p) + 152.2705078125*(d*d)*(a*a*a)*(p*p)*s - 730.8984375*(d*d)*(a*a)*b*(p*p)*s + 852.71484375*(d*d)*a*(b*b)*(p*p)*s - 243.6328125*(d*d)*(b*b*b)*(p*p)*s + 153.96240234375*d*a*g*(p*p*p) - 175.95703125*d*b*g*(p*p*p) + 153.96240234375*a*(p*p*p)*s - 175.95703125*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad879(double a, double b, double p, double d, double s, double g){
	return (0.013671875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 0.328125*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 1.9140625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 3.828125*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 2.87109375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.765625*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.0546875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 0.08203125*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 1.96875*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 11.484375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 22.96875*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 17.2265625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 4.59375*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.328125*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 1.8798828125*(d*d*d*d)*(a*a*a*a)*g*p - 15.0390625*(d*d*d*d)*(a*a*a)*b*g*p + 31.58203125*(d*d*d*d)*(a*a)*(b*b)*g*p - 21.0546875*(d*d*d*d)*a*(b*b*b)*g*p + 3.759765625*(d*d*d*d)*(b*b*b*b)*g*p + 7.51953125*(d*d*d)*(a*a*a*a)*p*s - 60.15625*(d*d*d)*(a*a*a)*b*p*s + 126.328125*(d*d*d)*(a*a)*(b*b)*p*s - 84.21875*(d*d*d)*a*(b*b*b)*p*s + 15.0390625*(d*d*d)*(b*b*b*b)*p*s + 21.99462890625*(d*d)*(a*a)*g*(p*p) - 58.65234375*(d*d)*a*b*g*(p*p) + 29.326171875*(d*d)*(b*b)*g*(p*p) + 43.9892578125*d*(a*a)*(p*p)*s - 117.3046875*d*a*b*(p*p)*s + 58.65234375*d*(b*b)*(p*p)*s + 18.328857421875*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8710(double a, double b, double p, double d, double s, double g){
	return (0.0205078125*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.2734375*(d*d*d*d*d)*(a*a*a*a)*b*g + 0.95703125*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 1.1484375*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.478515625*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.0546875*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.1025390625*(d*d*d*d)*(a*a*a*a*a)*s - 1.3671875*(d*d*d*d)*(a*a*a*a)*b*s + 4.78515625*(d*d*d*d)*(a*a*a)*(b*b)*s - 5.7421875*(d*d*d*d)*(a*a)*(b*b*b)*s + 2.392578125*(d*d*d*d)*a*(b*b*b*b)*s - 0.2734375*(d*d*d*d)*(b*b*b*b*b)*s + 1.1279296875*(d*d*d)*(a*a*a)*g*p - 5.4140625*(d*d*d)*(a*a)*b*g*p + 6.31640625*(d*d*d)*a*(b*b)*g*p - 1.8046875*(d*d*d)*(b*b*b)*g*p + 3.3837890625*(d*d)*(a*a*a)*p*s - 16.2421875*(d*d)*(a*a)*b*p*s + 18.94921875*(d*d)*a*(b*b)*p*s - 5.4140625*(d*d)*(b*b*b)*p*s + 5.132080078125*d*a*g*(p*p) - 5.865234375*d*b*g*(p*p) + 5.132080078125*a*(p*p)*s - 5.865234375*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8711(double a, double b, double p, double d, double s, double g){
	return (0.01708984375*(d*d*d*d)*(a*a*a*a)*g - 0.13671875*(d*d*d*d)*(a*a*a)*b*g + 0.287109375*(d*d*d*d)*(a*a)*(b*b)*g - 0.19140625*(d*d*d*d)*a*(b*b*b)*g + 0.0341796875*(d*d*d*d)*(b*b*b*b)*g + 0.068359375*(d*d*d)*(a*a*a*a)*s - 0.546875*(d*d*d)*(a*a*a)*b*s + 1.1484375*(d*d*d)*(a*a)*(b*b)*s - 0.765625*(d*d*d)*a*(b*b*b)*s + 0.13671875*(d*d*d)*(b*b*b*b)*s + 0.39990234375*(d*d)*(a*a)*g*p - 1.06640625*(d*d)*a*b*g*p + 0.533203125*(d*d)*(b*b)*g*p + 0.7998046875*d*(a*a)*p*s - 2.1328125*d*a*b*p*s + 1.06640625*d*(b*b)*p*s + 0.4998779296875*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8712(double a, double b, double p, double d, double s, double g){
	return (0.008544921875*(d*d*d)*(a*a*a)*g - 0.041015625*(d*d*d)*(a*a)*b*g + 0.0478515625*(d*d*d)*a*(b*b)*g - 0.013671875*(d*d*d)*(b*b*b)*g + 0.025634765625*(d*d)*(a*a*a)*s - 0.123046875*(d*d)*(a*a)*b*s + 0.1435546875*(d*d)*a*(b*b)*s - 0.041015625*(d*d)*(b*b*b)*s + 0.0777587890625*d*a*g*p - 0.0888671875*d*b*g*p + 0.0777587890625*a*p*s - 0.0888671875*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8713(double a, double b, double p, double d, double s, double g){
	return (0.0025634765625*(d*d)*(a*a)*g - 0.0068359375*(d*d)*a*b*g + 0.00341796875*(d*d)*(b*b)*g + 0.005126953125*d*(a*a)*s - 0.013671875*d*a*b*s + 0.0068359375*d*(b*b)*s + 0.00640869140625*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8714(double a, double b, double p, double d, double s, double g){
	return (0.00042724609375*d*a*g - 0.00048828125*d*b*g + 0.00042724609375*a*s - 0.00048828125*b*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8715(double a, double b, double p, double d, double s, double g){
	return 3.0517578125e-5*g/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad880(double a, double b, double p, double d, double s, double g){
	return ((d*d*d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g + 16.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*s + 14.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 32.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g*p + 14.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g*p + 196.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 448.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*p*s + 196.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*p*s + 52.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 336.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 588.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 336.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g*(p*p) + 52.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*g*(p*p) + 630.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 4032.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 7056.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 4032.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 630.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(p*p)*s + 52.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g*(p*p*p) - 840.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*(p*p*p) + 3675.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*(p*p*p) - 5880.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*(p*p*p) + 3675.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*(p*p*p) - 840.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*(p*p*p) + 52.5*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g*(p*p*p) + 525.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*(p*p*p)*s - 8400.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*(p*p*p)*s + 36750.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*(p*p*p)*s - 58800.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*(p*p*p)*s + 36750.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*(p*p*p)*s - 8400.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*(p*p*p)*s + 525.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*(p*p*p)*s + 6.5625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*(p*p*p*p) - 420.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p*p*p) + 5145.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p*p*p) - 20580.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p*p*p) + 32156.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p*p*p) - 20580.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p*p*p) + 5145.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p*p*p) - 420.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p*p*p) + 6.5625*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*(p*p*p*p) + 52.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(p*p*p*p)*s - 3360.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p*p*p)*s + 41160.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p*p*p)*s - 164640.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p*p*p)*s + 257250.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p*p*p)*s - 164640.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p*p*p)*s + 41160.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p*p*p)*s - 3360.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p*p*p)*s + 52.5*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*(p*p*p*p)*s + 826.875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p*p*p) - 13230.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p*p*p) + 57881.25*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p*p*p) - 92610.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p*p*p) + 57881.25*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p*p*p) - 13230.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p*p*p) + 826.875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p*p*p) + 4961.25*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p*p*p)*s - 79380.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p*p*p)*s + 347287.5*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p*p*p)*s - 555660.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p*p*p)*s + 347287.5*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p*p*p)*s - 79380.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p*p*p)*s + 4961.25*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p*p*p)*s + 11369.53125*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p*p*p) - 72765.0*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p*p*p) + 127338.75*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p*p*p) - 72765.0*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p*p*p) + 11369.53125*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p*p*p) + 45478.125*(d*d*d)*(a*a*a*a)*(p*p*p*p*p*p)*s - 291060.0*(d*d*d)*(a*a*a)*b*(p*p*p*p*p*p)*s + 509355.0*(d*d*d)*(a*a)*(b*b)*(p*p*p*p*p*p)*s - 291060.0*(d*d*d)*a*(b*b*b)*(p*p*p*p*p*p)*s + 45478.125*(d*d*d)*(b*b*b*b)*(p*p*p*p*p*p)*s + 29560.78125*(d*d)*(a*a)*g*(p*p*p*p*p*p*p) - 67567.5*(d*d)*a*b*g*(p*p*p*p*p*p*p) + 29560.78125*(d*d)*(b*b)*g*(p*p*p*p*p*p*p) + 59121.5625*d*(a*a)*(p*p*p*p*p*p*p)*s - 135135.0*d*a*b*(p*p*p*p*p*p*p)*s + 59121.5625*d*(b*b)*(p*p*p*p*p*p*p)*s + 7918.06640625*g*(p*p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad881(double a, double b, double p, double d, double s, double g){
	return (-4.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 4.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g - 60.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 60.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*s - 42.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 168.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 168.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g*p + 42.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g*p - 546.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 2184.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 2184.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*p*s + 546.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*p*s - 105.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 1050.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 2940.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 2940.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 1050.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g*(p*p) + 105.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*g*(p*p) - 1155.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 11550.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 32340.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 32340.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 11550.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 1155.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*(p*p)*s - 52.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g*(p*p*p) + 1470.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*(p*p*p) - 10290.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*(p*p*p) + 25725.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p*p) - 25725.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p*p) + 10290.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*(p*p*p) - 1470.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*(p*p*p) + 52.5*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g*(p*p*p) - 472.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*(p*p*p)*s + 13230.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*(p*p*p)*s - 92610.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*(p*p*p)*s + 231525.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p*p)*s - 231525.0*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p*p)*s + 92610.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*(p*p*p)*s - 13230.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*(p*p*p)*s + 472.5*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*(p*p*p)*s + 236.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p*p*p) - 6615.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p*p*p) + 46305.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p*p) - 115762.5*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p*p) + 115762.5*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p*p) - 46305.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p*p) + 6615.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p*p*p) - 236.25*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p*p*p) + 1653.75*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p*p*p)*s - 46305.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p*p*p)*s + 324135.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p*p)*s - 810337.5*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p*p)*s + 810337.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p*p)*s - 324135.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p*p)*s + 46305.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p*p*p)*s - 1653.75*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p*p*p)*s + 9095.625*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p*p) - 90956.25*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p*p) + 254677.5*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p*p) - 254677.5*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p*p) + 90956.25*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p*p) - 9095.625*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p*p) + 45478.125*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p*p)*s - 454781.25*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p*p)*s + 1273387.5*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p*p)*s - 1273387.5*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p*p)*s + 454781.25*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p*p)*s - 45478.125*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p*p)*s + 59121.5625*(d*d*d)*(a*a*a)*g*(p*p*p*p*p*p) - 236486.25*(d*d*d)*(a*a)*b*g*(p*p*p*p*p*p) + 236486.25*(d*d*d)*a*(b*b)*g*(p*p*p*p*p*p) - 59121.5625*(d*d*d)*(b*b*b)*g*(p*p*p*p*p*p) + 177364.6875*(d*d)*(a*a*a)*(p*p*p*p*p*p)*s - 709458.75*(d*d)*(a*a)*b*(p*p*p*p*p*p)*s + 709458.75*(d*d)*a*(b*b)*(p*p*p*p*p*p)*s - 177364.6875*(d*d)*(b*b*b)*(p*p*p*p*p*p)*s + 63344.53125*d*a*g*(p*p*p*p*p*p*p) - 63344.53125*d*b*g*(p*p*p*p*p*p*p) + 63344.53125*a*(p*p*p*p*p*p*p)*s - 63344.53125*b*(p*p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad882(double a, double b, double p, double d, double s, double g){
	return (7.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 16.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 7.0*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g + 98.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 224.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 98.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*s + 52.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*g*p - 336.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 588.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 336.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g*p + 52.5*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*g*p + 630.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*p*s - 4032.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 7056.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 4032.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*p*s + 630.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*p*s + 78.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g*(p*p) - 1260.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 5512.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 8820.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 5512.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 1260.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*(p*p) + 78.75*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g*(p*p) + 787.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*(p*p)*s - 12600.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 55125.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 88200.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 55125.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 12600.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 787.5*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*(p*p)*s + 13.125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*(p*p*p) - 840.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p*p) + 10290.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p*p) - 41160.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p*p) + 64312.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p*p) - 41160.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p*p) + 10290.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p*p) - 840.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p*p) + 13.125*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*(p*p*p) + 105.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(p*p*p)*s - 6720.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p*p)*s + 82320.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p*p)*s - 329280.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p*p)*s + 514500.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p*p)*s - 329280.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p*p)*s + 82320.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p*p)*s - 6720.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p*p)*s + 105.0*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*(p*p*p)*s + 2067.1875*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p*p) - 33075.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p*p) + 144703.125*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p*p) - 231525.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p*p) + 144703.125*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p*p) - 33075.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p*p) + 2067.1875*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p*p) + 12403.125*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p*p)*s - 198450.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p*p)*s + 868218.75*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p*p)*s - 1389150.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p*p)*s + 868218.75*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p*p)*s - 198450.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p*p)*s + 12403.125*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p*p)*s + 34108.59375*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p*p) - 218295.0*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p*p) + 382016.25*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p*p) - 218295.0*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p*p) + 34108.59375*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p*p) + 136434.375*(d*d*d)*(a*a*a*a)*(p*p*p*p*p)*s - 873180.0*(d*d*d)*(a*a*a)*b*(p*p*p*p*p)*s + 1528065.0*(d*d*d)*(a*a)*(b*b)*(p*p*p*p*p)*s - 873180.0*(d*d*d)*a*(b*b*b)*(p*p*p*p*p)*s + 136434.375*(d*d*d)*(b*b*b*b)*(p*p*p*p*p)*s + 103462.734375*(d*d)*(a*a)*g*(p*p*p*p*p*p) - 236486.25*(d*d)*a*b*g*(p*p*p*p*p*p) + 103462.734375*(d*d)*(b*b)*g*(p*p*p*p*p*p) + 206925.46875*d*(a*a)*(p*p*p*p*p*p)*s - 472972.5*d*a*b*(p*p*p*p*p*p)*s + 206925.46875*d*(b*b)*(p*p*p*p*p*p)*s + 31672.265625*g*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad883(double a, double b, double p, double d, double s, double g){
	return (-7.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*g + 28.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 28.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 7.0*(d*d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*g - 91.0 *(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*s + 364.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 364.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 91.0 *(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*s - 35.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*g*p + 350.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g*p - 980.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g*p + 980.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g*p - 350.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g*p + 35.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*g*p - 385.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*p*s + 3850.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*p*s - 10780.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*p*s + 10780.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*p*s - 3850.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*p*s + 385.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*p*s - 26.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g*(p*p) + 735.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*(p*p) - 5145.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*(p*p) + 12862.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*(p*p) - 12862.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*(p*p) + 5145.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*(p*p) - 735.0*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*(p*p) + 26.25*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g*(p*p) - 236.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*(p*p)*s + 6615.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*(p*p)*s - 46305.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*(p*p)*s + 115762.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*(p*p)*s - 115762.5*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*(p*p)*s + 46305.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*(p*p)*s - 6615.0*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*(p*p)*s + 236.25*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*(p*p)*s + 157.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p*p) - 4410.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p*p) + 30870.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p*p) - 77175.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p*p) + 77175.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p*p) - 30870.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p*p) + 4410.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p*p) - 157.5*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p*p) + 1102.5*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p*p)*s - 30870.0*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p*p)*s + 216090.0*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p*p)*s - 540225.0*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p*p)*s + 540225.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p*p)*s - 216090.0*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p*p)*s + 30870.0*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p*p)*s - 1102.5*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p*p)*s + 7579.6875*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p*p) - 75796.875*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p*p) + 212231.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p*p) - 212231.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p*p) + 75796.875*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p*p) - 7579.6875*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p*p) + 37898.4375*(d*d*d*d)*(a*a*a*a*a)*(p*p*p*p)*s - 378984.375*(d*d*d*d)*(a*a*a*a)*b*(p*p*p*p)*s + 1061156.25*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p*p)*s - 1061156.25*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p*p)*s + 378984.375*(d*d*d*d)*a*(b*b*b*b)*(p*p*p*p)*s - 37898.4375*(d*d*d*d)*(b*b*b*b*b)*(p*p*p*p)*s + 59121.5625*(d*d*d)*(a*a*a)*g*(p*p*p*p*p) - 236486.25*(d*d*d)*(a*a)*b*g*(p*p*p*p*p) + 236486.25*(d*d*d)*a*(b*b)*g*(p*p*p*p*p) - 59121.5625*(d*d*d)*(b*b*b)*g*(p*p*p*p*p) + 177364.6875*(d*d)*(a*a*a)*(p*p*p*p*p)*s - 709458.75*(d*d)*(a*a)*b*(p*p*p*p*p)*s + 709458.75*(d*d)*a*(b*b)*(p*p*p*p*p)*s - 177364.6875*(d*d)*(b*b*b)*(p*p*p*p*p)*s + 73901.953125*d*a*g*(p*p*p*p*p*p) - 73901.953125*d*b*g*(p*p*p*p*p*p) + 73901.953125*a*(p*p*p*p*p*p)*s - 73901.953125*b*(p*p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad884(double a, double b, double p, double d, double s, double g){
	return (4.375*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*g - 28.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*g + 49.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*g - 28.0*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*g + 4.375*(d*d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*g + 52.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b*b)*s - 336.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b*b)*s + 588.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b*b)*s - 336.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b*b)*s + 52.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b*b)*s + 13.125*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g*p - 210.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g*p + 918.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g*p - 1470.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g*p + 918.75*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g*p - 210.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g*p + 13.125*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g*p + 131.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*p*s - 2100.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*p*s + 9187.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*p*s - 14700.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*p*s + 9187.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*p*s - 2100.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*p*s + 131.25*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*p*s + 3.28125*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*(p*p) - 210.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*(p*p) + 2572.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*(p*p) - 10290.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*(p*p) + 16078.125*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*(p*p) - 10290.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*(p*p) + 2572.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*(p*p) - 210.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*(p*p) + 3.28125*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*(p*p) + 26.25*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(p*p)*s - 1680.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*(p*p)*s + 20580.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*(p*p)*s - 82320.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*(p*p)*s + 128625.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*(p*p)*s - 82320.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*(p*p)*s + 20580.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*(p*p)*s - 1680.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*(p*p)*s + 26.25*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*(p*p)*s + 689.0625*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p*p) - 11025.0*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p*p) + 48234.375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p*p) - 77175.0*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p*p) + 48234.375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p*p) - 11025.0*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p*p) + 689.0625*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p*p) + 4134.375*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p*p)*s - 66150.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p*p)*s + 289406.25*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p*p)*s - 463050.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p*p)*s + 289406.25*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p*p)*s - 66150.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p*p)*s + 4134.375*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p*p)*s + 14211.9140625*(d*d*d*d)*(a*a*a*a)*g*(p*p*p*p) - 90956.25*(d*d*d*d)*(a*a*a)*b*g*(p*p*p*p) + 159173.4375*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p*p) - 90956.25*(d*d*d*d)*a*(b*b*b)*g*(p*p*p*p) + 14211.9140625*(d*d*d*d)*(b*b*b*b)*g*(p*p*p*p) + 56847.65625*(d*d*d)*(a*a*a*a)*(p*p*p*p)*s - 363825.0*(d*d*d)*(a*a*a)*b*(p*p*p*p)*s + 636693.75*(d*d*d)*(a*a)*(b*b)*(p*p*p*p)*s - 363825.0*(d*d*d)*a*(b*b*b)*(p*p*p*p)*s + 56847.65625*(d*d*d)*(b*b*b*b)*(p*p*p*p)*s + 51731.3671875*(d*d)*(a*a)*g*(p*p*p*p*p) - 118243.125*(d*d)*a*b*g*(p*p*p*p*p) + 51731.3671875*(d*d)*(b*b)*g*(p*p*p*p*p) + 103462.734375*d*(a*a)*(p*p*p*p*p)*s - 236486.25*d*a*b*(p*p*p*p*p)*s + 103462.734375*d*(b*b)*(p*p*p*p*p)*s + 18475.48828125*g*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad885(double a, double b, double p, double d, double s, double g){
	return (-1.75*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*g + 17.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*g - 49.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*g + 49.0*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*g - 17.5*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*g + 1.75*(d*d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*g - 19.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b*b)*s + 192.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b*b)*s - 539.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b*b)*s + 539.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b*b)*s - 192.5*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b*b)*s + 19.25*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b*b)*s - 2.625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g*p + 73.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g*p - 514.5*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g*p + 1286.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g*p - 1286.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g*p + 514.5*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g*p - 73.5*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g*p + 2.625*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g*p - 23.625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*p*s + 661.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*p*s - 4630.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*p*s + 11576.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*p*s - 11576.25*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*p*s + 4630.5*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*p*s - 661.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*p*s + 23.625*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*p*s + 23.625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*(p*p) - 661.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*(p*p) + 4630.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*(p*p) - 11576.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*(p*p) + 11576.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*(p*p) - 4630.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*(p*p) + 661.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*(p*p) - 23.625*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*(p*p) + 165.375*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(p*p)*s - 4630.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*(p*p)*s + 32413.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*(p*p)*s - 81033.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*(p*p)*s + 81033.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*(p*p)*s - 32413.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*(p*p)*s + 4630.5*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*(p*p)*s - 165.375*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*(p*p)*s + 1515.9375*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p*p) - 15159.375*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p*p) + 42446.25*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p*p) - 42446.25*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p*p) + 15159.375*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p*p) - 1515.9375*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p*p) + 7579.6875*(d*d*d*d)*(a*a*a*a*a)*(p*p*p)*s - 75796.875*(d*d*d*d)*(a*a*a*a)*b*(p*p*p)*s + 212231.25*(d*d*d*d)*(a*a*a)*(b*b)*(p*p*p)*s - 212231.25*(d*d*d*d)*(a*a)*(b*b*b)*(p*p*p)*s + 75796.875*(d*d*d*d)*a*(b*b*b*b)*(p*p*p)*s - 7579.6875*(d*d*d*d)*(b*b*b*b*b)*(p*p*p)*s + 14780.390625*(d*d*d)*(a*a*a)*g*(p*p*p*p) - 59121.5625*(d*d*d)*(a*a)*b*g*(p*p*p*p) + 59121.5625*(d*d*d)*a*(b*b)*g*(p*p*p*p) - 14780.390625*(d*d*d)*(b*b*b)*g*(p*p*p*p) + 44341.171875*(d*d)*(a*a*a)*(p*p*p*p)*s - 177364.6875*(d*d)*(a*a)*b*(p*p*p*p)*s + 177364.6875*(d*d)*a*(b*b)*(p*p*p*p)*s - 44341.171875*(d*d)*(b*b*b)*(p*p*p*p)*s + 22170.5859375*d*a*g*(p*p*p*p*p) - 22170.5859375*d*b*g*(p*p*p*p*p) + 22170.5859375*a*(p*p*p*p*p)*s - 22170.5859375*b*(p*p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad886(double a, double b, double p, double d, double s, double g){
	return (0.4375*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*g - 7.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*g + 30.625*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*g - 49.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*g + 30.625*(d*d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*g - 7.0*(d*d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*g + 0.4375*(d*d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*g + 4.375*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*(b*b)*s - 70.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b*b)*s + 306.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b*b)*s - 490.0*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b*b)*s + 306.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b*b)*s - 70.0*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b*b)*s + 4.375*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b*b)*s + 0.21875*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g*p - 14.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g*p + 171.5*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g*p - 686.0*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g*p + 1071.875*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g*p - 686.0*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g*p + 171.5*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g*p - 14.0*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g*p + 0.21875*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g*p + 1.75*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*p*s - 112.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*p*s + 1372.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*p*s - 5488.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*p*s + 8575.0*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*p*s - 5488.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*p*s + 1372.0*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*p*s - 112.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*p*s + 1.75*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*p*s + 68.90625*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*(p*p) - 1102.5*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*(p*p) + 4823.4375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*(p*p) - 7717.5*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*(p*p) + 4823.4375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*(p*p) - 1102.5*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*(p*p) + 68.90625*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*(p*p) + 413.4375*(d*d*d*d*d)*(a*a*a*a*a*a)*(p*p)*s - 6615.0*(d*d*d*d*d)*(a*a*a*a*a)*b*(p*p)*s + 28940.625*(d*d*d*d*d)*(a*a*a*a)*(b*b)*(p*p)*s - 46305.0*(d*d*d*d*d)*(a*a*a)*(b*b*b)*(p*p)*s + 28940.625*(d*d*d*d*d)*(a*a)*(b*b*b*b)*(p*p)*s - 6615.0*(d*d*d*d*d)*a*(b*b*b*b*b)*(p*p)*s + 413.4375*(d*d*d*d*d)*(b*b*b*b*b*b)*(p*p)*s + 1894.921875*(d*d*d*d)*(a*a*a*a)*g*(p*p*p) - 12127.5*(d*d*d*d)*(a*a*a)*b*g*(p*p*p) + 21223.125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p*p) - 12127.5*(d*d*d*d)*a*(b*b*b)*g*(p*p*p) + 1894.921875*(d*d*d*d)*(b*b*b*b)*g*(p*p*p) + 7579.6875*(d*d*d)*(a*a*a*a)*(p*p*p)*s - 48510.0*(d*d*d)*(a*a*a)*b*(p*p*p)*s + 84892.5*(d*d*d)*(a*a)*(b*b)*(p*p*p)*s - 48510.0*(d*d*d)*a*(b*b*b)*(p*p*p)*s + 7579.6875*(d*d*d)*(b*b*b*b)*(p*p*p)*s + 8621.89453125*(d*d)*(a*a)*g*(p*p*p*p) - 19707.1875*(d*d)*a*b*g*(p*p*p*p) + 8621.89453125*(d*d)*(b*b)*g*(p*p*p*p) + 17243.7890625*d*(a*a)*(p*p*p*p)*s - 39414.375*d*a*b*(p*p*p*p)*s + 17243.7890625*d*(b*b)*(p*p*p*p)*s + 3695.09765625*g*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad887(double a, double b, double p, double d, double s, double g){
	return (-0.0625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*g + 1.75*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*g - 12.25*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*g + 30.625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*g - 30.625*(d*d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*g + 12.25*(d*d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*g - 1.75*(d*d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*g + 0.0625*(d*d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*g - 0.5625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*b*s + 15.75*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*(b*b)*s - 110.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b*b)*s + 275.625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b*b)*s - 275.625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b*b)*s + 110.25*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b*b)*s - 15.75*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b*b)*s + 0.5625*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b*b)*s + 1.125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g*p - 31.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g*p + 220.5*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g*p - 551.25*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g*p + 551.25*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g*p - 220.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g*p + 31.5*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g*p - 1.125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g*p + 7.875*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*p*s - 220.5*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*p*s + 1543.5*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*p*s - 3858.75*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*p*s + 3858.75*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*p*s - 1543.5*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*p*s + 220.5*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*p*s - 7.875*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*p*s + 108.28125*(d*d*d*d*d)*(a*a*a*a*a)*g*(p*p) - 1082.8125*(d*d*d*d*d)*(a*a*a*a)*b*g*(p*p) + 3031.875*(d*d*d*d*d)*(a*a*a)*(b*b)*g*(p*p) - 3031.875*(d*d*d*d*d)*(a*a)*(b*b*b)*g*(p*p) + 1082.8125*(d*d*d*d*d)*a*(b*b*b*b)*g*(p*p) - 108.28125*(d*d*d*d*d)*(b*b*b*b*b)*g*(p*p) + 541.40625*(d*d*d*d)*(a*a*a*a*a)*(p*p)*s - 5414.0625*(d*d*d*d)*(a*a*a*a)*b*(p*p)*s + 15159.375*(d*d*d*d)*(a*a*a)*(b*b)*(p*p)*s - 15159.375*(d*d*d*d)*(a*a)*(b*b*b)*(p*p)*s + 5414.0625*(d*d*d*d)*a*(b*b*b*b)*(p*p)*s - 541.40625*(d*d*d*d)*(b*b*b*b*b)*(p*p)*s + 1407.65625*(d*d*d)*(a*a*a)*g*(p*p*p) - 5630.625*(d*d*d)*(a*a)*b*g*(p*p*p) + 5630.625*(d*d*d)*a*(b*b)*g*(p*p*p) - 1407.65625*(d*d*d)*(b*b*b)*g*(p*p*p) + 4222.96875*(d*d)*(a*a*a)*(p*p*p)*s - 16891.875*(d*d)*(a*a)*b*(p*p*p)*s + 16891.875*(d*d)*a*(b*b)*(p*p*p)*s - 4222.96875*(d*d)*(b*b*b)*(p*p*p)*s + 2639.35546875*d*a*g*(p*p*p*p) - 2639.35546875*d*b*g*(p*p*p*p) + 2639.35546875*a*(p*p*p*p)*s - 2639.35546875*b*(p*p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad888(double a, double b, double p, double d, double s, double g){
	return (0.00390625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*g - 0.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*g + 3.0625*(d*d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*g - 12.25*(d*d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*g + 19.140625*(d*d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*g - 12.25*(d*d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*g + 3.0625*(d*d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*g - 0.25*(d*d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*g + 0.00390625*(d*d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*g + 0.03125*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a*a)*s - 2.0*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*b*s + 24.5*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*(b*b)*s - 98.0*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b*b)*s + 153.125*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b*b)*s - 98.0*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b*b)*s + 24.5*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b*b)*s - 2.0*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b*b)*s + 0.03125*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b*b)*s + 2.4609375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g*p - 39.375*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g*p + 172.265625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g*p - 275.625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g*p + 172.265625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g*p - 39.375*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g*p + 2.4609375*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g*p + 14.765625*(d*d*d*d*d)*(a*a*a*a*a*a)*p*s - 236.25*(d*d*d*d*d)*(a*a*a*a*a)*b*p*s + 1033.59375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*p*s - 1653.75*(d*d*d*d*d)*(a*a*a)*(b*b*b)*p*s + 1033.59375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*p*s - 236.25*(d*d*d*d*d)*a*(b*b*b*b*b)*p*s + 14.765625*(d*d*d*d*d)*(b*b*b*b*b*b)*p*s + 101.513671875*(d*d*d*d)*(a*a*a*a)*g*(p*p) - 649.6875*(d*d*d*d)*(a*a*a)*b*g*(p*p) + 1136.953125*(d*d*d*d)*(a*a)*(b*b)*g*(p*p) - 649.6875*(d*d*d*d)*a*(b*b*b)*g*(p*p) + 101.513671875*(d*d*d*d)*(b*b*b*b)*g*(p*p) + 406.0546875*(d*d*d)*(a*a*a*a)*(p*p)*s - 2598.75*(d*d*d)*(a*a*a)*b*(p*p)*s + 4547.8125*(d*d*d)*(a*a)*(b*b)*(p*p)*s - 2598.75*(d*d*d)*a*(b*b*b)*(p*p)*s + 406.0546875*(d*d*d)*(b*b*b*b)*(p*p)*s + 615.849609375*(d*d)*(a*a)*g*(p*p*p) - 1407.65625*(d*d)*a*b*g*(p*p*p) + 615.849609375*(d*d)*(b*b)*g*(p*p*p) + 1231.69921875*d*(a*a)*(p*p*p)*s - 2815.3125*d*a*b*(p*p*p)*s + 1231.69921875*d*(b*b)*(p*p*p)*s + 329.91943359375*g*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad889(double a, double b, double p, double d, double s, double g){
	return (0.015625*(d*d*d*d*d*d*d)*(a*a*a*a*a*a*a)*g - 0.4375*(d*d*d*d*d*d*d)*(a*a*a*a*a*a)*b*g + 3.0625*(d*d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*g - 7.65625*(d*d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*g + 7.65625*(d*d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*g - 3.0625*(d*d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*g + 0.4375*(d*d*d*d*d*d*d)*a*(b*b*b*b*b*b)*g - 0.015625*(d*d*d*d*d*d*d)*(b*b*b*b*b*b*b)*g + 0.109375*(d*d*d*d*d*d)*(a*a*a*a*a*a*a)*s - 3.0625*(d*d*d*d*d*d)*(a*a*a*a*a*a)*b*s + 21.4375*(d*d*d*d*d*d)*(a*a*a*a*a)*(b*b)*s - 53.59375*(d*d*d*d*d*d)*(a*a*a*a)*(b*b*b)*s + 53.59375*(d*d*d*d*d*d)*(a*a*a)*(b*b*b*b)*s - 21.4375*(d*d*d*d*d*d)*(a*a)*(b*b*b*b*b)*s + 3.0625*(d*d*d*d*d*d)*a*(b*b*b*b*b*b)*s - 0.109375*(d*d*d*d*d*d)*(b*b*b*b*b*b*b)*s + 3.0078125*(d*d*d*d*d)*(a*a*a*a*a)*g*p - 30.078125*(d*d*d*d*d)*(a*a*a*a)*b*g*p + 84.21875*(d*d*d*d*d)*(a*a*a)*(b*b)*g*p - 84.21875*(d*d*d*d*d)*(a*a)*(b*b*b)*g*p + 30.078125*(d*d*d*d*d)*a*(b*b*b*b)*g*p - 3.0078125*(d*d*d*d*d)*(b*b*b*b*b)*g*p + 15.0390625*(d*d*d*d)*(a*a*a*a*a)*p*s - 150.390625*(d*d*d*d)*(a*a*a*a)*b*p*s + 421.09375*(d*d*d*d)*(a*a*a)*(b*b)*p*s - 421.09375*(d*d*d*d)*(a*a)*(b*b*b)*p*s + 150.390625*(d*d*d*d)*a*(b*b*b*b)*p*s - 15.0390625*(d*d*d*d)*(b*b*b*b*b)*p*s + 58.65234375*(d*d*d)*(a*a*a)*g*(p*p) - 234.609375*(d*d*d)*(a*a)*b*g*(p*p) + 234.609375*(d*d*d)*a*(b*b)*g*(p*p) - 58.65234375*(d*d*d)*(b*b*b)*g*(p*p) + 175.95703125*(d*d)*(a*a*a)*(p*p)*s - 703.828125*(d*d)*(a*a)*b*(p*p)*s + 703.828125*(d*d)*a*(b*b)*(p*p)*s - 175.95703125*(d*d)*(b*b*b)*(p*p)*s + 146.630859375*d*a*g*(p*p*p) - 146.630859375*d*b*g*(p*p*p) + 146.630859375*a*(p*p*p)*s - 146.630859375*b*(p*p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8810(double a, double b, double p, double d, double s, double g){
	return (0.02734375*(d*d*d*d*d*d)*(a*a*a*a*a*a)*g - 0.4375*(d*d*d*d*d*d)*(a*a*a*a*a)*b*g + 1.9140625*(d*d*d*d*d*d)*(a*a*a*a)*(b*b)*g - 3.0625*(d*d*d*d*d*d)*(a*a*a)*(b*b*b)*g + 1.9140625*(d*d*d*d*d*d)*(a*a)*(b*b*b*b)*g - 0.4375*(d*d*d*d*d*d)*a*(b*b*b*b*b)*g + 0.02734375*(d*d*d*d*d*d)*(b*b*b*b*b*b)*g + 0.1640625*(d*d*d*d*d)*(a*a*a*a*a*a)*s - 2.625*(d*d*d*d*d)*(a*a*a*a*a)*b*s + 11.484375*(d*d*d*d*d)*(a*a*a*a)*(b*b)*s - 18.375*(d*d*d*d*d)*(a*a*a)*(b*b*b)*s + 11.484375*(d*d*d*d*d)*(a*a)*(b*b*b*b)*s - 2.625*(d*d*d*d*d)*a*(b*b*b*b*b)*s + 0.1640625*(d*d*d*d*d)*(b*b*b*b*b*b)*s + 2.255859375*(d*d*d*d)*(a*a*a*a)*g*p - 14.4375*(d*d*d*d)*(a*a*a)*b*g*p + 25.265625*(d*d*d*d)*(a*a)*(b*b)*g*p - 14.4375*(d*d*d*d)*a*(b*b*b)*g*p + 2.255859375*(d*d*d*d)*(b*b*b*b)*g*p + 9.0234375*(d*d*d)*(a*a*a*a)*p*s - 57.75*(d*d*d)*(a*a*a)*b*p*s + 101.0625*(d*d*d)*(a*a)*(b*b)*p*s - 57.75*(d*d*d)*a*(b*b*b)*p*s + 9.0234375*(d*d*d)*(b*b*b*b)*p*s + 20.5283203125*(d*d)*(a*a)*g*(p*p) - 46.921875*(d*d)*a*b*g*(p*p) + 20.5283203125*(d*d)*(b*b)*g*(p*p) + 41.056640625*d*(a*a)*(p*p)*s - 93.84375*d*a*b*(p*p)*s + 41.056640625*d*(b*b)*(p*p)*s + 14.6630859375*g*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8811(double a, double b, double p, double d, double s, double g){
	return (0.02734375*(d*d*d*d*d)*(a*a*a*a*a)*g - 0.2734375*(d*d*d*d*d)*(a*a*a*a)*b*g + 0.765625*(d*d*d*d*d)*(a*a*a)*(b*b)*g - 0.765625*(d*d*d*d*d)*(a*a)*(b*b*b)*g + 0.2734375*(d*d*d*d*d)*a*(b*b*b*b)*g - 0.02734375*(d*d*d*d*d)*(b*b*b*b*b)*g + 0.13671875*(d*d*d*d)*(a*a*a*a*a)*s - 1.3671875*(d*d*d*d)*(a*a*a*a)*b*s + 3.828125*(d*d*d*d)*(a*a*a)*(b*b)*s - 3.828125*(d*d*d*d)*(a*a)*(b*b*b)*s + 1.3671875*(d*d*d*d)*a*(b*b*b*b)*s - 0.13671875*(d*d*d*d)*(b*b*b*b*b)*s + 1.06640625*(d*d*d)*(a*a*a)*g*p - 4.265625*(d*d*d)*(a*a)*b*g*p + 4.265625*(d*d*d)*a*(b*b)*g*p - 1.06640625*(d*d*d)*(b*b*b)*g*p + 3.19921875*(d*d)*(a*a*a)*p*s - 12.796875*(d*d)*(a*a)*b*p*s + 12.796875*(d*d)*a*(b*b)*p*s - 3.19921875*(d*d)*(b*b*b)*p*s + 3.9990234375*d*a*g*(p*p) - 3.9990234375*d*b*g*(p*p) + 3.9990234375*a*(p*p)*s - 3.9990234375*b*(p*p)*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8812(double a, double b, double p, double d, double s, double g){
	return (0.01708984375*(d*d*d*d)*(a*a*a*a)*g - 0.109375*(d*d*d*d)*(a*a*a)*b*g + 0.19140625*(d*d*d*d)*(a*a)*(b*b)*g - 0.109375*(d*d*d*d)*a*(b*b*b)*g + 0.01708984375*(d*d*d*d)*(b*b*b*b)*g + 0.068359375*(d*d*d)*(a*a*a*a)*s - 0.4375*(d*d*d)*(a*a*a)*b*s + 0.765625*(d*d*d)*(a*a)*(b*b)*s - 0.4375*(d*d*d)*a*(b*b*b)*s + 0.068359375*(d*d*d)*(b*b*b*b)*s + 0.31103515625*(d*d)*(a*a)*g*p - 0.7109375*(d*d)*a*b*g*p + 0.31103515625*(d*d)*(b*b)*g*p + 0.6220703125*d*(a*a)*p*s - 1.421875*d*a*b*p*s + 0.6220703125*d*(b*b)*p*s + 0.333251953125*g*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8813(double a, double b, double p, double d, double s, double g){
	return (0.0068359375*(d*d*d)*(a*a*a)*g - 0.02734375*(d*d*d)*(a*a)*b*g + 0.02734375*(d*d*d)*a*(b*b)*g - 0.0068359375*(d*d*d)*(b*b*b)*g + 0.0205078125*(d*d)*(a*a*a)*s - 0.08203125*(d*d)*(a*a)*b*s + 0.08203125*(d*d)*a*(b*b)*s - 0.0205078125*(d*d)*(b*b*b)*s + 0.05126953125*d*a*g*p - 0.05126953125*d*b*g*p + 0.05126953125*a*p*s - 0.05126953125*b*p*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8814(double a, double b, double p, double d, double s, double g){
	return (0.001708984375*(d*d)*(a*a)*g - 0.00390625*(d*d)*a*b*g + 0.001708984375*(d*d)*(b*b)*g + 0.00341796875*d*(a*a)*s - 0.0078125*d*a*b*s + 0.00341796875*d*(b*b)*s + 0.003662109375*g*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8815(double a, double b, double p, double d, double s, double g){
	return 0.000244140625*(d*a*g - d*b*g + a*s - b*s)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}

inline double MD_Et_grad8816(double a, double b, double p, double d, double s, double g){
	return 1.52587890625e-5*g/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p);
}


inline double call_MD_Et_grad(
    int idx,
    double a,double b,double p,
    double d,double s,double g)
{
    switch(idx){
        case 0: return MD_Et_grad000(a,b,p,d,s,g);
        case 1: return MD_Et_grad010(a,b,p,d,s,g);
        case 2: return MD_Et_grad011(a,b,p,d,s,g);
        case 3: return MD_Et_grad020(a,b,p,d,s,g);
        case 4: return MD_Et_grad021(a,b,p,d,s,g);
        case 5: return MD_Et_grad022(a,b,p,d,s,g);
        case 6: return MD_Et_grad030(a,b,p,d,s,g);
        case 7: return MD_Et_grad031(a,b,p,d,s,g);
        case 8: return MD_Et_grad032(a,b,p,d,s,g);
        case 9: return MD_Et_grad033(a,b,p,d,s,g);
        case 10: return MD_Et_grad040(a,b,p,d,s,g);
        case 11: return MD_Et_grad041(a,b,p,d,s,g);
        case 12: return MD_Et_grad042(a,b,p,d,s,g);
        case 13: return MD_Et_grad043(a,b,p,d,s,g);
        case 14: return MD_Et_grad044(a,b,p,d,s,g);
        case 15: return MD_Et_grad050(a,b,p,d,s,g);
        case 16: return MD_Et_grad051(a,b,p,d,s,g);
        case 17: return MD_Et_grad052(a,b,p,d,s,g);
        case 18: return MD_Et_grad053(a,b,p,d,s,g);
        case 19: return MD_Et_grad054(a,b,p,d,s,g);
        case 20: return MD_Et_grad055(a,b,p,d,s,g);
        case 21: return MD_Et_grad060(a,b,p,d,s,g);
        case 22: return MD_Et_grad061(a,b,p,d,s,g);
        case 23: return MD_Et_grad062(a,b,p,d,s,g);
        case 24: return MD_Et_grad063(a,b,p,d,s,g);
        case 25: return MD_Et_grad064(a,b,p,d,s,g);
        case 26: return MD_Et_grad065(a,b,p,d,s,g);
        case 27: return MD_Et_grad066(a,b,p,d,s,g);
        case 28: return MD_Et_grad070(a,b,p,d,s,g);
        case 29: return MD_Et_grad071(a,b,p,d,s,g);
        case 30: return MD_Et_grad072(a,b,p,d,s,g);
        case 31: return MD_Et_grad073(a,b,p,d,s,g);
        case 32: return MD_Et_grad074(a,b,p,d,s,g);
        case 33: return MD_Et_grad075(a,b,p,d,s,g);
        case 34: return MD_Et_grad076(a,b,p,d,s,g);
        case 35: return MD_Et_grad077(a,b,p,d,s,g);
        case 36: return MD_Et_grad080(a,b,p,d,s,g);
        case 37: return MD_Et_grad081(a,b,p,d,s,g);
        case 38: return MD_Et_grad082(a,b,p,d,s,g);
        case 39: return MD_Et_grad083(a,b,p,d,s,g);
        case 40: return MD_Et_grad084(a,b,p,d,s,g);
        case 41: return MD_Et_grad085(a,b,p,d,s,g);
        case 42: return MD_Et_grad086(a,b,p,d,s,g);
        case 43: return MD_Et_grad087(a,b,p,d,s,g);
        case 44: return MD_Et_grad088(a,b,p,d,s,g);
        case 45: return MD_Et_grad100(a,b,p,d,s,g);
        case 46: return MD_Et_grad101(a,b,p,d,s,g);
        case 47: return MD_Et_grad110(a,b,p,d,s,g);
        case 48: return MD_Et_grad111(a,b,p,d,s,g);
        case 49: return MD_Et_grad112(a,b,p,d,s,g);
        case 50: return MD_Et_grad120(a,b,p,d,s,g);
        case 51: return MD_Et_grad121(a,b,p,d,s,g);
        case 52: return MD_Et_grad122(a,b,p,d,s,g);
        case 53: return MD_Et_grad123(a,b,p,d,s,g);
        case 54: return MD_Et_grad130(a,b,p,d,s,g);
        case 55: return MD_Et_grad131(a,b,p,d,s,g);
        case 56: return MD_Et_grad132(a,b,p,d,s,g);
        case 57: return MD_Et_grad133(a,b,p,d,s,g);
        case 58: return MD_Et_grad134(a,b,p,d,s,g);
        case 59: return MD_Et_grad140(a,b,p,d,s,g);
        case 60: return MD_Et_grad141(a,b,p,d,s,g);
        case 61: return MD_Et_grad142(a,b,p,d,s,g);
        case 62: return MD_Et_grad143(a,b,p,d,s,g);
        case 63: return MD_Et_grad144(a,b,p,d,s,g);
        case 64: return MD_Et_grad145(a,b,p,d,s,g);
        case 65: return MD_Et_grad150(a,b,p,d,s,g);
        case 66: return MD_Et_grad151(a,b,p,d,s,g);
        case 67: return MD_Et_grad152(a,b,p,d,s,g);
        case 68: return MD_Et_grad153(a,b,p,d,s,g);
        case 69: return MD_Et_grad154(a,b,p,d,s,g);
        case 70: return MD_Et_grad155(a,b,p,d,s,g);
        case 71: return MD_Et_grad156(a,b,p,d,s,g);
        case 72: return MD_Et_grad160(a,b,p,d,s,g);
        case 73: return MD_Et_grad161(a,b,p,d,s,g);
        case 74: return MD_Et_grad162(a,b,p,d,s,g);
        case 75: return MD_Et_grad163(a,b,p,d,s,g);
        case 76: return MD_Et_grad164(a,b,p,d,s,g);
        case 77: return MD_Et_grad165(a,b,p,d,s,g);
        case 78: return MD_Et_grad166(a,b,p,d,s,g);
        case 79: return MD_Et_grad167(a,b,p,d,s,g);
        case 80: return MD_Et_grad170(a,b,p,d,s,g);
        case 81: return MD_Et_grad171(a,b,p,d,s,g);
        case 82: return MD_Et_grad172(a,b,p,d,s,g);
        case 83: return MD_Et_grad173(a,b,p,d,s,g);
        case 84: return MD_Et_grad174(a,b,p,d,s,g);
        case 85: return MD_Et_grad175(a,b,p,d,s,g);
        case 86: return MD_Et_grad176(a,b,p,d,s,g);
        case 87: return MD_Et_grad177(a,b,p,d,s,g);
        case 88: return MD_Et_grad178(a,b,p,d,s,g);
        case 89: return MD_Et_grad180(a,b,p,d,s,g);
        case 90: return MD_Et_grad181(a,b,p,d,s,g);
        case 91: return MD_Et_grad182(a,b,p,d,s,g);
        case 92: return MD_Et_grad183(a,b,p,d,s,g);
        case 93: return MD_Et_grad184(a,b,p,d,s,g);
        case 94: return MD_Et_grad185(a,b,p,d,s,g);
        case 95: return MD_Et_grad186(a,b,p,d,s,g);
        case 96: return MD_Et_grad187(a,b,p,d,s,g);
        case 97: return MD_Et_grad188(a,b,p,d,s,g);
        case 98: return MD_Et_grad189(a,b,p,d,s,g);
        case 99: return MD_Et_grad200(a,b,p,d,s,g);
        case 100: return MD_Et_grad201(a,b,p,d,s,g);
        case 101: return MD_Et_grad202(a,b,p,d,s,g);
        case 102: return MD_Et_grad210(a,b,p,d,s,g);
        case 103: return MD_Et_grad211(a,b,p,d,s,g);
        case 104: return MD_Et_grad212(a,b,p,d,s,g);
        case 105: return MD_Et_grad213(a,b,p,d,s,g);
        case 106: return MD_Et_grad220(a,b,p,d,s,g);
        case 107: return MD_Et_grad221(a,b,p,d,s,g);
        case 108: return MD_Et_grad222(a,b,p,d,s,g);
        case 109: return MD_Et_grad223(a,b,p,d,s,g);
        case 110: return MD_Et_grad224(a,b,p,d,s,g);
        case 111: return MD_Et_grad230(a,b,p,d,s,g);
        case 112: return MD_Et_grad231(a,b,p,d,s,g);
        case 113: return MD_Et_grad232(a,b,p,d,s,g);
        case 114: return MD_Et_grad233(a,b,p,d,s,g);
        case 115: return MD_Et_grad234(a,b,p,d,s,g);
        case 116: return MD_Et_grad235(a,b,p,d,s,g);
        case 117: return MD_Et_grad240(a,b,p,d,s,g);
        case 118: return MD_Et_grad241(a,b,p,d,s,g);
        case 119: return MD_Et_grad242(a,b,p,d,s,g);
        case 120: return MD_Et_grad243(a,b,p,d,s,g);
        case 121: return MD_Et_grad244(a,b,p,d,s,g);
        case 122: return MD_Et_grad245(a,b,p,d,s,g);
        case 123: return MD_Et_grad246(a,b,p,d,s,g);
        case 124: return MD_Et_grad250(a,b,p,d,s,g);
        case 125: return MD_Et_grad251(a,b,p,d,s,g);
        case 126: return MD_Et_grad252(a,b,p,d,s,g);
        case 127: return MD_Et_grad253(a,b,p,d,s,g);
        case 128: return MD_Et_grad254(a,b,p,d,s,g);
        case 129: return MD_Et_grad255(a,b,p,d,s,g);
        case 130: return MD_Et_grad256(a,b,p,d,s,g);
        case 131: return MD_Et_grad257(a,b,p,d,s,g);
        case 132: return MD_Et_grad260(a,b,p,d,s,g);
        case 133: return MD_Et_grad261(a,b,p,d,s,g);
        case 134: return MD_Et_grad262(a,b,p,d,s,g);
        case 135: return MD_Et_grad263(a,b,p,d,s,g);
        case 136: return MD_Et_grad264(a,b,p,d,s,g);
        case 137: return MD_Et_grad265(a,b,p,d,s,g);
        case 138: return MD_Et_grad266(a,b,p,d,s,g);
        case 139: return MD_Et_grad267(a,b,p,d,s,g);
        case 140: return MD_Et_grad268(a,b,p,d,s,g);
        case 141: return MD_Et_grad270(a,b,p,d,s,g);
        case 142: return MD_Et_grad271(a,b,p,d,s,g);
        case 143: return MD_Et_grad272(a,b,p,d,s,g);
        case 144: return MD_Et_grad273(a,b,p,d,s,g);
        case 145: return MD_Et_grad274(a,b,p,d,s,g);
        case 146: return MD_Et_grad275(a,b,p,d,s,g);
        case 147: return MD_Et_grad276(a,b,p,d,s,g);
        case 148: return MD_Et_grad277(a,b,p,d,s,g);
        case 149: return MD_Et_grad278(a,b,p,d,s,g);
        case 150: return MD_Et_grad279(a,b,p,d,s,g);
        case 151: return MD_Et_grad280(a,b,p,d,s,g);
        case 152: return MD_Et_grad281(a,b,p,d,s,g);
        case 153: return MD_Et_grad282(a,b,p,d,s,g);
        case 154: return MD_Et_grad283(a,b,p,d,s,g);
        case 155: return MD_Et_grad284(a,b,p,d,s,g);
        case 156: return MD_Et_grad285(a,b,p,d,s,g);
        case 157: return MD_Et_grad286(a,b,p,d,s,g);
        case 158: return MD_Et_grad287(a,b,p,d,s,g);
        case 159: return MD_Et_grad288(a,b,p,d,s,g);
        case 160: return MD_Et_grad289(a,b,p,d,s,g);
        case 161: return MD_Et_grad2810(a,b,p,d,s,g);
        case 162: return MD_Et_grad300(a,b,p,d,s,g);
        case 163: return MD_Et_grad301(a,b,p,d,s,g);
        case 164: return MD_Et_grad302(a,b,p,d,s,g);
        case 165: return MD_Et_grad303(a,b,p,d,s,g);
        case 166: return MD_Et_grad310(a,b,p,d,s,g);
        case 167: return MD_Et_grad311(a,b,p,d,s,g);
        case 168: return MD_Et_grad312(a,b,p,d,s,g);
        case 169: return MD_Et_grad313(a,b,p,d,s,g);
        case 170: return MD_Et_grad314(a,b,p,d,s,g);
        case 171: return MD_Et_grad320(a,b,p,d,s,g);
        case 172: return MD_Et_grad321(a,b,p,d,s,g);
        case 173: return MD_Et_grad322(a,b,p,d,s,g);
        case 174: return MD_Et_grad323(a,b,p,d,s,g);
        case 175: return MD_Et_grad324(a,b,p,d,s,g);
        case 176: return MD_Et_grad325(a,b,p,d,s,g);
        case 177: return MD_Et_grad330(a,b,p,d,s,g);
        case 178: return MD_Et_grad331(a,b,p,d,s,g);
        case 179: return MD_Et_grad332(a,b,p,d,s,g);
        case 180: return MD_Et_grad333(a,b,p,d,s,g);
        case 181: return MD_Et_grad334(a,b,p,d,s,g);
        case 182: return MD_Et_grad335(a,b,p,d,s,g);
        case 183: return MD_Et_grad336(a,b,p,d,s,g);
        case 184: return MD_Et_grad340(a,b,p,d,s,g);
        case 185: return MD_Et_grad341(a,b,p,d,s,g);
        case 186: return MD_Et_grad342(a,b,p,d,s,g);
        case 187: return MD_Et_grad343(a,b,p,d,s,g);
        case 188: return MD_Et_grad344(a,b,p,d,s,g);
        case 189: return MD_Et_grad345(a,b,p,d,s,g);
        case 190: return MD_Et_grad346(a,b,p,d,s,g);
        case 191: return MD_Et_grad347(a,b,p,d,s,g);
        case 192: return MD_Et_grad350(a,b,p,d,s,g);
        case 193: return MD_Et_grad351(a,b,p,d,s,g);
        case 194: return MD_Et_grad352(a,b,p,d,s,g);
        case 195: return MD_Et_grad353(a,b,p,d,s,g);
        case 196: return MD_Et_grad354(a,b,p,d,s,g);
        case 197: return MD_Et_grad355(a,b,p,d,s,g);
        case 198: return MD_Et_grad356(a,b,p,d,s,g);
        case 199: return MD_Et_grad357(a,b,p,d,s,g);
        case 200: return MD_Et_grad358(a,b,p,d,s,g);
        case 201: return MD_Et_grad360(a,b,p,d,s,g);
        case 202: return MD_Et_grad361(a,b,p,d,s,g);
        case 203: return MD_Et_grad362(a,b,p,d,s,g);
        case 204: return MD_Et_grad363(a,b,p,d,s,g);
        case 205: return MD_Et_grad364(a,b,p,d,s,g);
        case 206: return MD_Et_grad365(a,b,p,d,s,g);
        case 207: return MD_Et_grad366(a,b,p,d,s,g);
        case 208: return MD_Et_grad367(a,b,p,d,s,g);
        case 209: return MD_Et_grad368(a,b,p,d,s,g);
        case 210: return MD_Et_grad369(a,b,p,d,s,g);
        case 211: return MD_Et_grad370(a,b,p,d,s,g);
        case 212: return MD_Et_grad371(a,b,p,d,s,g);
        case 213: return MD_Et_grad372(a,b,p,d,s,g);
        case 214: return MD_Et_grad373(a,b,p,d,s,g);
        case 215: return MD_Et_grad374(a,b,p,d,s,g);
        case 216: return MD_Et_grad375(a,b,p,d,s,g);
        case 217: return MD_Et_grad376(a,b,p,d,s,g);
        case 218: return MD_Et_grad377(a,b,p,d,s,g);
        case 219: return MD_Et_grad378(a,b,p,d,s,g);
        case 220: return MD_Et_grad379(a,b,p,d,s,g);
        case 221: return MD_Et_grad3710(a,b,p,d,s,g);
        case 222: return MD_Et_grad380(a,b,p,d,s,g);
        case 223: return MD_Et_grad381(a,b,p,d,s,g);
        case 224: return MD_Et_grad382(a,b,p,d,s,g);
        case 225: return MD_Et_grad383(a,b,p,d,s,g);
        case 226: return MD_Et_grad384(a,b,p,d,s,g);
        case 227: return MD_Et_grad385(a,b,p,d,s,g);
        case 228: return MD_Et_grad386(a,b,p,d,s,g);
        case 229: return MD_Et_grad387(a,b,p,d,s,g);
        case 230: return MD_Et_grad388(a,b,p,d,s,g);
        case 231: return MD_Et_grad389(a,b,p,d,s,g);
        case 232: return MD_Et_grad3810(a,b,p,d,s,g);
        case 233: return MD_Et_grad3811(a,b,p,d,s,g);
        case 234: return MD_Et_grad400(a,b,p,d,s,g);
        case 235: return MD_Et_grad401(a,b,p,d,s,g);
        case 236: return MD_Et_grad402(a,b,p,d,s,g);
        case 237: return MD_Et_grad403(a,b,p,d,s,g);
        case 238: return MD_Et_grad404(a,b,p,d,s,g);
        case 239: return MD_Et_grad410(a,b,p,d,s,g);
        case 240: return MD_Et_grad411(a,b,p,d,s,g);
        case 241: return MD_Et_grad412(a,b,p,d,s,g);
        case 242: return MD_Et_grad413(a,b,p,d,s,g);
        case 243: return MD_Et_grad414(a,b,p,d,s,g);
        case 244: return MD_Et_grad415(a,b,p,d,s,g);
        case 245: return MD_Et_grad420(a,b,p,d,s,g);
        case 246: return MD_Et_grad421(a,b,p,d,s,g);
        case 247: return MD_Et_grad422(a,b,p,d,s,g);
        case 248: return MD_Et_grad423(a,b,p,d,s,g);
        case 249: return MD_Et_grad424(a,b,p,d,s,g);
        case 250: return MD_Et_grad425(a,b,p,d,s,g);
        case 251: return MD_Et_grad426(a,b,p,d,s,g);
        case 252: return MD_Et_grad430(a,b,p,d,s,g);
        case 253: return MD_Et_grad431(a,b,p,d,s,g);
        case 254: return MD_Et_grad432(a,b,p,d,s,g);
        case 255: return MD_Et_grad433(a,b,p,d,s,g);
        case 256: return MD_Et_grad434(a,b,p,d,s,g);
        case 257: return MD_Et_grad435(a,b,p,d,s,g);
        case 258: return MD_Et_grad436(a,b,p,d,s,g);
        case 259: return MD_Et_grad437(a,b,p,d,s,g);
        case 260: return MD_Et_grad440(a,b,p,d,s,g);
        case 261: return MD_Et_grad441(a,b,p,d,s,g);
        case 262: return MD_Et_grad442(a,b,p,d,s,g);
        case 263: return MD_Et_grad443(a,b,p,d,s,g);
        case 264: return MD_Et_grad444(a,b,p,d,s,g);
        case 265: return MD_Et_grad445(a,b,p,d,s,g);
        case 266: return MD_Et_grad446(a,b,p,d,s,g);
        case 267: return MD_Et_grad447(a,b,p,d,s,g);
        case 268: return MD_Et_grad448(a,b,p,d,s,g);
        case 269: return MD_Et_grad450(a,b,p,d,s,g);
        case 270: return MD_Et_grad451(a,b,p,d,s,g);
        case 271: return MD_Et_grad452(a,b,p,d,s,g);
        case 272: return MD_Et_grad453(a,b,p,d,s,g);
        case 273: return MD_Et_grad454(a,b,p,d,s,g);
        case 274: return MD_Et_grad455(a,b,p,d,s,g);
        case 275: return MD_Et_grad456(a,b,p,d,s,g);
        case 276: return MD_Et_grad457(a,b,p,d,s,g);
        case 277: return MD_Et_grad458(a,b,p,d,s,g);
        case 278: return MD_Et_grad459(a,b,p,d,s,g);
        case 279: return MD_Et_grad460(a,b,p,d,s,g);
        case 280: return MD_Et_grad461(a,b,p,d,s,g);
        case 281: return MD_Et_grad462(a,b,p,d,s,g);
        case 282: return MD_Et_grad463(a,b,p,d,s,g);
        case 283: return MD_Et_grad464(a,b,p,d,s,g);
        case 284: return MD_Et_grad465(a,b,p,d,s,g);
        case 285: return MD_Et_grad466(a,b,p,d,s,g);
        case 286: return MD_Et_grad467(a,b,p,d,s,g);
        case 287: return MD_Et_grad468(a,b,p,d,s,g);
        case 288: return MD_Et_grad469(a,b,p,d,s,g);
        case 289: return MD_Et_grad4610(a,b,p,d,s,g);
        case 290: return MD_Et_grad470(a,b,p,d,s,g);
        case 291: return MD_Et_grad471(a,b,p,d,s,g);
        case 292: return MD_Et_grad472(a,b,p,d,s,g);
        case 293: return MD_Et_grad473(a,b,p,d,s,g);
        case 294: return MD_Et_grad474(a,b,p,d,s,g);
        case 295: return MD_Et_grad475(a,b,p,d,s,g);
        case 296: return MD_Et_grad476(a,b,p,d,s,g);
        case 297: return MD_Et_grad477(a,b,p,d,s,g);
        case 298: return MD_Et_grad478(a,b,p,d,s,g);
        case 299: return MD_Et_grad479(a,b,p,d,s,g);
        case 300: return MD_Et_grad4710(a,b,p,d,s,g);
        case 301: return MD_Et_grad4711(a,b,p,d,s,g);
        case 302: return MD_Et_grad480(a,b,p,d,s,g);
        case 303: return MD_Et_grad481(a,b,p,d,s,g);
        case 304: return MD_Et_grad482(a,b,p,d,s,g);
        case 305: return MD_Et_grad483(a,b,p,d,s,g);
        case 306: return MD_Et_grad484(a,b,p,d,s,g);
        case 307: return MD_Et_grad485(a,b,p,d,s,g);
        case 308: return MD_Et_grad486(a,b,p,d,s,g);
        case 309: return MD_Et_grad487(a,b,p,d,s,g);
        case 310: return MD_Et_grad488(a,b,p,d,s,g);
        case 311: return MD_Et_grad489(a,b,p,d,s,g);
        case 312: return MD_Et_grad4810(a,b,p,d,s,g);
        case 313: return MD_Et_grad4811(a,b,p,d,s,g);
        case 314: return MD_Et_grad4812(a,b,p,d,s,g);
        case 315: return MD_Et_grad500(a,b,p,d,s,g);
        case 316: return MD_Et_grad501(a,b,p,d,s,g);
        case 317: return MD_Et_grad502(a,b,p,d,s,g);
        case 318: return MD_Et_grad503(a,b,p,d,s,g);
        case 319: return MD_Et_grad504(a,b,p,d,s,g);
        case 320: return MD_Et_grad505(a,b,p,d,s,g);
        case 321: return MD_Et_grad510(a,b,p,d,s,g);
        case 322: return MD_Et_grad511(a,b,p,d,s,g);
        case 323: return MD_Et_grad512(a,b,p,d,s,g);
        case 324: return MD_Et_grad513(a,b,p,d,s,g);
        case 325: return MD_Et_grad514(a,b,p,d,s,g);
        case 326: return MD_Et_grad515(a,b,p,d,s,g);
        case 327: return MD_Et_grad516(a,b,p,d,s,g);
        case 328: return MD_Et_grad520(a,b,p,d,s,g);
        case 329: return MD_Et_grad521(a,b,p,d,s,g);
        case 330: return MD_Et_grad522(a,b,p,d,s,g);
        case 331: return MD_Et_grad523(a,b,p,d,s,g);
        case 332: return MD_Et_grad524(a,b,p,d,s,g);
        case 333: return MD_Et_grad525(a,b,p,d,s,g);
        case 334: return MD_Et_grad526(a,b,p,d,s,g);
        case 335: return MD_Et_grad527(a,b,p,d,s,g);
        case 336: return MD_Et_grad530(a,b,p,d,s,g);
        case 337: return MD_Et_grad531(a,b,p,d,s,g);
        case 338: return MD_Et_grad532(a,b,p,d,s,g);
        case 339: return MD_Et_grad533(a,b,p,d,s,g);
        case 340: return MD_Et_grad534(a,b,p,d,s,g);
        case 341: return MD_Et_grad535(a,b,p,d,s,g);
        case 342: return MD_Et_grad536(a,b,p,d,s,g);
        case 343: return MD_Et_grad537(a,b,p,d,s,g);
        case 344: return MD_Et_grad538(a,b,p,d,s,g);
        case 345: return MD_Et_grad540(a,b,p,d,s,g);
        case 346: return MD_Et_grad541(a,b,p,d,s,g);
        case 347: return MD_Et_grad542(a,b,p,d,s,g);
        case 348: return MD_Et_grad543(a,b,p,d,s,g);
        case 349: return MD_Et_grad544(a,b,p,d,s,g);
        case 350: return MD_Et_grad545(a,b,p,d,s,g);
        case 351: return MD_Et_grad546(a,b,p,d,s,g);
        case 352: return MD_Et_grad547(a,b,p,d,s,g);
        case 353: return MD_Et_grad548(a,b,p,d,s,g);
        case 354: return MD_Et_grad549(a,b,p,d,s,g);
        case 355: return MD_Et_grad550(a,b,p,d,s,g);
        case 356: return MD_Et_grad551(a,b,p,d,s,g);
        case 357: return MD_Et_grad552(a,b,p,d,s,g);
        case 358: return MD_Et_grad553(a,b,p,d,s,g);
        case 359: return MD_Et_grad554(a,b,p,d,s,g);
        case 360: return MD_Et_grad555(a,b,p,d,s,g);
        case 361: return MD_Et_grad556(a,b,p,d,s,g);
        case 362: return MD_Et_grad557(a,b,p,d,s,g);
        case 363: return MD_Et_grad558(a,b,p,d,s,g);
        case 364: return MD_Et_grad559(a,b,p,d,s,g);
        case 365: return MD_Et_grad5510(a,b,p,d,s,g);
        case 366: return MD_Et_grad560(a,b,p,d,s,g);
        case 367: return MD_Et_grad561(a,b,p,d,s,g);
        case 368: return MD_Et_grad562(a,b,p,d,s,g);
        case 369: return MD_Et_grad563(a,b,p,d,s,g);
        case 370: return MD_Et_grad564(a,b,p,d,s,g);
        case 371: return MD_Et_grad565(a,b,p,d,s,g);
        case 372: return MD_Et_grad566(a,b,p,d,s,g);
        case 373: return MD_Et_grad567(a,b,p,d,s,g);
        case 374: return MD_Et_grad568(a,b,p,d,s,g);
        case 375: return MD_Et_grad569(a,b,p,d,s,g);
        case 376: return MD_Et_grad5610(a,b,p,d,s,g);
        case 377: return MD_Et_grad5611(a,b,p,d,s,g);
        case 378: return MD_Et_grad570(a,b,p,d,s,g);
        case 379: return MD_Et_grad571(a,b,p,d,s,g);
        case 380: return MD_Et_grad572(a,b,p,d,s,g);
        case 381: return MD_Et_grad573(a,b,p,d,s,g);
        case 382: return MD_Et_grad574(a,b,p,d,s,g);
        case 383: return MD_Et_grad575(a,b,p,d,s,g);
        case 384: return MD_Et_grad576(a,b,p,d,s,g);
        case 385: return MD_Et_grad577(a,b,p,d,s,g);
        case 386: return MD_Et_grad578(a,b,p,d,s,g);
        case 387: return MD_Et_grad579(a,b,p,d,s,g);
        case 388: return MD_Et_grad5710(a,b,p,d,s,g);
        case 389: return MD_Et_grad5711(a,b,p,d,s,g);
        case 390: return MD_Et_grad5712(a,b,p,d,s,g);
        case 391: return MD_Et_grad580(a,b,p,d,s,g);
        case 392: return MD_Et_grad581(a,b,p,d,s,g);
        case 393: return MD_Et_grad582(a,b,p,d,s,g);
        case 394: return MD_Et_grad583(a,b,p,d,s,g);
        case 395: return MD_Et_grad584(a,b,p,d,s,g);
        case 396: return MD_Et_grad585(a,b,p,d,s,g);
        case 397: return MD_Et_grad586(a,b,p,d,s,g);
        case 398: return MD_Et_grad587(a,b,p,d,s,g);
        case 399: return MD_Et_grad588(a,b,p,d,s,g);
        case 400: return MD_Et_grad589(a,b,p,d,s,g);
        case 401: return MD_Et_grad5810(a,b,p,d,s,g);
        case 402: return MD_Et_grad5811(a,b,p,d,s,g);
        case 403: return MD_Et_grad5812(a,b,p,d,s,g);
        case 404: return MD_Et_grad5813(a,b,p,d,s,g);
        case 405: return MD_Et_grad600(a,b,p,d,s,g);
        case 406: return MD_Et_grad601(a,b,p,d,s,g);
        case 407: return MD_Et_grad602(a,b,p,d,s,g);
        case 408: return MD_Et_grad603(a,b,p,d,s,g);
        case 409: return MD_Et_grad604(a,b,p,d,s,g);
        case 410: return MD_Et_grad605(a,b,p,d,s,g);
        case 411: return MD_Et_grad606(a,b,p,d,s,g);
        case 412: return MD_Et_grad610(a,b,p,d,s,g);
        case 413: return MD_Et_grad611(a,b,p,d,s,g);
        case 414: return MD_Et_grad612(a,b,p,d,s,g);
        case 415: return MD_Et_grad613(a,b,p,d,s,g);
        case 416: return MD_Et_grad614(a,b,p,d,s,g);
        case 417: return MD_Et_grad615(a,b,p,d,s,g);
        case 418: return MD_Et_grad616(a,b,p,d,s,g);
        case 419: return MD_Et_grad617(a,b,p,d,s,g);
        case 420: return MD_Et_grad620(a,b,p,d,s,g);
        case 421: return MD_Et_grad621(a,b,p,d,s,g);
        case 422: return MD_Et_grad622(a,b,p,d,s,g);
        case 423: return MD_Et_grad623(a,b,p,d,s,g);
        case 424: return MD_Et_grad624(a,b,p,d,s,g);
        case 425: return MD_Et_grad625(a,b,p,d,s,g);
        case 426: return MD_Et_grad626(a,b,p,d,s,g);
        case 427: return MD_Et_grad627(a,b,p,d,s,g);
        case 428: return MD_Et_grad628(a,b,p,d,s,g);
        case 429: return MD_Et_grad630(a,b,p,d,s,g);
        case 430: return MD_Et_grad631(a,b,p,d,s,g);
        case 431: return MD_Et_grad632(a,b,p,d,s,g);
        case 432: return MD_Et_grad633(a,b,p,d,s,g);
        case 433: return MD_Et_grad634(a,b,p,d,s,g);
        case 434: return MD_Et_grad635(a,b,p,d,s,g);
        case 435: return MD_Et_grad636(a,b,p,d,s,g);
        case 436: return MD_Et_grad637(a,b,p,d,s,g);
        case 437: return MD_Et_grad638(a,b,p,d,s,g);
        case 438: return MD_Et_grad639(a,b,p,d,s,g);
        case 439: return MD_Et_grad640(a,b,p,d,s,g);
        case 440: return MD_Et_grad641(a,b,p,d,s,g);
        case 441: return MD_Et_grad642(a,b,p,d,s,g);
        case 442: return MD_Et_grad643(a,b,p,d,s,g);
        case 443: return MD_Et_grad644(a,b,p,d,s,g);
        case 444: return MD_Et_grad645(a,b,p,d,s,g);
        case 445: return MD_Et_grad646(a,b,p,d,s,g);
        case 446: return MD_Et_grad647(a,b,p,d,s,g);
        case 447: return MD_Et_grad648(a,b,p,d,s,g);
        case 448: return MD_Et_grad649(a,b,p,d,s,g);
        case 449: return MD_Et_grad6410(a,b,p,d,s,g);
        case 450: return MD_Et_grad650(a,b,p,d,s,g);
        case 451: return MD_Et_grad651(a,b,p,d,s,g);
        case 452: return MD_Et_grad652(a,b,p,d,s,g);
        case 453: return MD_Et_grad653(a,b,p,d,s,g);
        case 454: return MD_Et_grad654(a,b,p,d,s,g);
        case 455: return MD_Et_grad655(a,b,p,d,s,g);
        case 456: return MD_Et_grad656(a,b,p,d,s,g);
        case 457: return MD_Et_grad657(a,b,p,d,s,g);
        case 458: return MD_Et_grad658(a,b,p,d,s,g);
        case 459: return MD_Et_grad659(a,b,p,d,s,g);
        case 460: return MD_Et_grad6510(a,b,p,d,s,g);
        case 461: return MD_Et_grad6511(a,b,p,d,s,g);
        case 462: return MD_Et_grad660(a,b,p,d,s,g);
        case 463: return MD_Et_grad661(a,b,p,d,s,g);
        case 464: return MD_Et_grad662(a,b,p,d,s,g);
        case 465: return MD_Et_grad663(a,b,p,d,s,g);
        case 466: return MD_Et_grad664(a,b,p,d,s,g);
        case 467: return MD_Et_grad665(a,b,p,d,s,g);
        case 468: return MD_Et_grad666(a,b,p,d,s,g);
        case 469: return MD_Et_grad667(a,b,p,d,s,g);
        case 470: return MD_Et_grad668(a,b,p,d,s,g);
        case 471: return MD_Et_grad669(a,b,p,d,s,g);
        case 472: return MD_Et_grad6610(a,b,p,d,s,g);
        case 473: return MD_Et_grad6611(a,b,p,d,s,g);
        case 474: return MD_Et_grad6612(a,b,p,d,s,g);
        case 475: return MD_Et_grad670(a,b,p,d,s,g);
        case 476: return MD_Et_grad671(a,b,p,d,s,g);
        case 477: return MD_Et_grad672(a,b,p,d,s,g);
        case 478: return MD_Et_grad673(a,b,p,d,s,g);
        case 479: return MD_Et_grad674(a,b,p,d,s,g);
        case 480: return MD_Et_grad675(a,b,p,d,s,g);
        case 481: return MD_Et_grad676(a,b,p,d,s,g);
        case 482: return MD_Et_grad677(a,b,p,d,s,g);
        case 483: return MD_Et_grad678(a,b,p,d,s,g);
        case 484: return MD_Et_grad679(a,b,p,d,s,g);
        case 485: return MD_Et_grad6710(a,b,p,d,s,g);
        case 486: return MD_Et_grad6711(a,b,p,d,s,g);
        case 487: return MD_Et_grad6712(a,b,p,d,s,g);
        case 488: return MD_Et_grad6713(a,b,p,d,s,g);
        case 489: return MD_Et_grad680(a,b,p,d,s,g);
        case 490: return MD_Et_grad681(a,b,p,d,s,g);
        case 491: return MD_Et_grad682(a,b,p,d,s,g);
        case 492: return MD_Et_grad683(a,b,p,d,s,g);
        case 493: return MD_Et_grad684(a,b,p,d,s,g);
        case 494: return MD_Et_grad685(a,b,p,d,s,g);
        case 495: return MD_Et_grad686(a,b,p,d,s,g);
        case 496: return MD_Et_grad687(a,b,p,d,s,g);
        case 497: return MD_Et_grad688(a,b,p,d,s,g);
        case 498: return MD_Et_grad689(a,b,p,d,s,g);
        case 499: return MD_Et_grad6810(a,b,p,d,s,g);
        case 500: return MD_Et_grad6811(a,b,p,d,s,g);
        case 501: return MD_Et_grad6812(a,b,p,d,s,g);
        case 502: return MD_Et_grad6813(a,b,p,d,s,g);
        case 503: return MD_Et_grad6814(a,b,p,d,s,g);
        case 504: return MD_Et_grad700(a,b,p,d,s,g);
        case 505: return MD_Et_grad701(a,b,p,d,s,g);
        case 506: return MD_Et_grad702(a,b,p,d,s,g);
        case 507: return MD_Et_grad703(a,b,p,d,s,g);
        case 508: return MD_Et_grad704(a,b,p,d,s,g);
        case 509: return MD_Et_grad705(a,b,p,d,s,g);
        case 510: return MD_Et_grad706(a,b,p,d,s,g);
        case 511: return MD_Et_grad707(a,b,p,d,s,g);
        case 512: return MD_Et_grad710(a,b,p,d,s,g);
        case 513: return MD_Et_grad711(a,b,p,d,s,g);
        case 514: return MD_Et_grad712(a,b,p,d,s,g);
        case 515: return MD_Et_grad713(a,b,p,d,s,g);
        case 516: return MD_Et_grad714(a,b,p,d,s,g);
        case 517: return MD_Et_grad715(a,b,p,d,s,g);
        case 518: return MD_Et_grad716(a,b,p,d,s,g);
        case 519: return MD_Et_grad717(a,b,p,d,s,g);
        case 520: return MD_Et_grad718(a,b,p,d,s,g);
        case 521: return MD_Et_grad720(a,b,p,d,s,g);
        case 522: return MD_Et_grad721(a,b,p,d,s,g);
        case 523: return MD_Et_grad722(a,b,p,d,s,g);
        case 524: return MD_Et_grad723(a,b,p,d,s,g);
        case 525: return MD_Et_grad724(a,b,p,d,s,g);
        case 526: return MD_Et_grad725(a,b,p,d,s,g);
        case 527: return MD_Et_grad726(a,b,p,d,s,g);
        case 528: return MD_Et_grad727(a,b,p,d,s,g);
        case 529: return MD_Et_grad728(a,b,p,d,s,g);
        case 530: return MD_Et_grad729(a,b,p,d,s,g);
        case 531: return MD_Et_grad730(a,b,p,d,s,g);
        case 532: return MD_Et_grad731(a,b,p,d,s,g);
        case 533: return MD_Et_grad732(a,b,p,d,s,g);
        case 534: return MD_Et_grad733(a,b,p,d,s,g);
        case 535: return MD_Et_grad734(a,b,p,d,s,g);
        case 536: return MD_Et_grad735(a,b,p,d,s,g);
        case 537: return MD_Et_grad736(a,b,p,d,s,g);
        case 538: return MD_Et_grad737(a,b,p,d,s,g);
        case 539: return MD_Et_grad738(a,b,p,d,s,g);
        case 540: return MD_Et_grad739(a,b,p,d,s,g);
        case 541: return MD_Et_grad7310(a,b,p,d,s,g);
        case 542: return MD_Et_grad740(a,b,p,d,s,g);
        case 543: return MD_Et_grad741(a,b,p,d,s,g);
        case 544: return MD_Et_grad742(a,b,p,d,s,g);
        case 545: return MD_Et_grad743(a,b,p,d,s,g);
        case 546: return MD_Et_grad744(a,b,p,d,s,g);
        case 547: return MD_Et_grad745(a,b,p,d,s,g);
        case 548: return MD_Et_grad746(a,b,p,d,s,g);
        case 549: return MD_Et_grad747(a,b,p,d,s,g);
        case 550: return MD_Et_grad748(a,b,p,d,s,g);
        case 551: return MD_Et_grad749(a,b,p,d,s,g);
        case 552: return MD_Et_grad7410(a,b,p,d,s,g);
        case 553: return MD_Et_grad7411(a,b,p,d,s,g);
        case 554: return MD_Et_grad750(a,b,p,d,s,g);
        case 555: return MD_Et_grad751(a,b,p,d,s,g);
        case 556: return MD_Et_grad752(a,b,p,d,s,g);
        case 557: return MD_Et_grad753(a,b,p,d,s,g);
        case 558: return MD_Et_grad754(a,b,p,d,s,g);
        case 559: return MD_Et_grad755(a,b,p,d,s,g);
        case 560: return MD_Et_grad756(a,b,p,d,s,g);
        case 561: return MD_Et_grad757(a,b,p,d,s,g);
        case 562: return MD_Et_grad758(a,b,p,d,s,g);
        case 563: return MD_Et_grad759(a,b,p,d,s,g);
        case 564: return MD_Et_grad7510(a,b,p,d,s,g);
        case 565: return MD_Et_grad7511(a,b,p,d,s,g);
        case 566: return MD_Et_grad7512(a,b,p,d,s,g);
        case 567: return MD_Et_grad760(a,b,p,d,s,g);
        case 568: return MD_Et_grad761(a,b,p,d,s,g);
        case 569: return MD_Et_grad762(a,b,p,d,s,g);
        case 570: return MD_Et_grad763(a,b,p,d,s,g);
        case 571: return MD_Et_grad764(a,b,p,d,s,g);
        case 572: return MD_Et_grad765(a,b,p,d,s,g);
        case 573: return MD_Et_grad766(a,b,p,d,s,g);
        case 574: return MD_Et_grad767(a,b,p,d,s,g);
        case 575: return MD_Et_grad768(a,b,p,d,s,g);
        case 576: return MD_Et_grad769(a,b,p,d,s,g);
        case 577: return MD_Et_grad7610(a,b,p,d,s,g);
        case 578: return MD_Et_grad7611(a,b,p,d,s,g);
        case 579: return MD_Et_grad7612(a,b,p,d,s,g);
        case 580: return MD_Et_grad7613(a,b,p,d,s,g);
        case 581: return MD_Et_grad770(a,b,p,d,s,g);
        case 582: return MD_Et_grad771(a,b,p,d,s,g);
        case 583: return MD_Et_grad772(a,b,p,d,s,g);
        case 584: return MD_Et_grad773(a,b,p,d,s,g);
        case 585: return MD_Et_grad774(a,b,p,d,s,g);
        case 586: return MD_Et_grad775(a,b,p,d,s,g);
        case 587: return MD_Et_grad776(a,b,p,d,s,g);
        case 588: return MD_Et_grad777(a,b,p,d,s,g);
        case 589: return MD_Et_grad778(a,b,p,d,s,g);
        case 590: return MD_Et_grad779(a,b,p,d,s,g);
        case 591: return MD_Et_grad7710(a,b,p,d,s,g);
        case 592: return MD_Et_grad7711(a,b,p,d,s,g);
        case 593: return MD_Et_grad7712(a,b,p,d,s,g);
        case 594: return MD_Et_grad7713(a,b,p,d,s,g);
        case 595: return MD_Et_grad7714(a,b,p,d,s,g);
        case 596: return MD_Et_grad780(a,b,p,d,s,g);
        case 597: return MD_Et_grad781(a,b,p,d,s,g);
        case 598: return MD_Et_grad782(a,b,p,d,s,g);
        case 599: return MD_Et_grad783(a,b,p,d,s,g);
        case 600: return MD_Et_grad784(a,b,p,d,s,g);
        case 601: return MD_Et_grad785(a,b,p,d,s,g);
        case 602: return MD_Et_grad786(a,b,p,d,s,g);
        case 603: return MD_Et_grad787(a,b,p,d,s,g);
        case 604: return MD_Et_grad788(a,b,p,d,s,g);
        case 605: return MD_Et_grad789(a,b,p,d,s,g);
        case 606: return MD_Et_grad7810(a,b,p,d,s,g);
        case 607: return MD_Et_grad7811(a,b,p,d,s,g);
        case 608: return MD_Et_grad7812(a,b,p,d,s,g);
        case 609: return MD_Et_grad7813(a,b,p,d,s,g);
        case 610: return MD_Et_grad7814(a,b,p,d,s,g);
        case 611: return MD_Et_grad7815(a,b,p,d,s,g);
        case 612: return MD_Et_grad800(a,b,p,d,s,g);
        case 613: return MD_Et_grad801(a,b,p,d,s,g);
        case 614: return MD_Et_grad802(a,b,p,d,s,g);
        case 615: return MD_Et_grad803(a,b,p,d,s,g);
        case 616: return MD_Et_grad804(a,b,p,d,s,g);
        case 617: return MD_Et_grad805(a,b,p,d,s,g);
        case 618: return MD_Et_grad806(a,b,p,d,s,g);
        case 619: return MD_Et_grad807(a,b,p,d,s,g);
        case 620: return MD_Et_grad808(a,b,p,d,s,g);
        case 621: return MD_Et_grad810(a,b,p,d,s,g);
        case 622: return MD_Et_grad811(a,b,p,d,s,g);
        case 623: return MD_Et_grad812(a,b,p,d,s,g);
        case 624: return MD_Et_grad813(a,b,p,d,s,g);
        case 625: return MD_Et_grad814(a,b,p,d,s,g);
        case 626: return MD_Et_grad815(a,b,p,d,s,g);
        case 627: return MD_Et_grad816(a,b,p,d,s,g);
        case 628: return MD_Et_grad817(a,b,p,d,s,g);
        case 629: return MD_Et_grad818(a,b,p,d,s,g);
        case 630: return MD_Et_grad819(a,b,p,d,s,g);
        case 631: return MD_Et_grad820(a,b,p,d,s,g);
        case 632: return MD_Et_grad821(a,b,p,d,s,g);
        case 633: return MD_Et_grad822(a,b,p,d,s,g);
        case 634: return MD_Et_grad823(a,b,p,d,s,g);
        case 635: return MD_Et_grad824(a,b,p,d,s,g);
        case 636: return MD_Et_grad825(a,b,p,d,s,g);
        case 637: return MD_Et_grad826(a,b,p,d,s,g);
        case 638: return MD_Et_grad827(a,b,p,d,s,g);
        case 639: return MD_Et_grad828(a,b,p,d,s,g);
        case 640: return MD_Et_grad829(a,b,p,d,s,g);
        case 641: return MD_Et_grad8210(a,b,p,d,s,g);
        case 642: return MD_Et_grad830(a,b,p,d,s,g);
        case 643: return MD_Et_grad831(a,b,p,d,s,g);
        case 644: return MD_Et_grad832(a,b,p,d,s,g);
        case 645: return MD_Et_grad833(a,b,p,d,s,g);
        case 646: return MD_Et_grad834(a,b,p,d,s,g);
        case 647: return MD_Et_grad835(a,b,p,d,s,g);
        case 648: return MD_Et_grad836(a,b,p,d,s,g);
        case 649: return MD_Et_grad837(a,b,p,d,s,g);
        case 650: return MD_Et_grad838(a,b,p,d,s,g);
        case 651: return MD_Et_grad839(a,b,p,d,s,g);
        case 652: return MD_Et_grad8310(a,b,p,d,s,g);
        case 653: return MD_Et_grad8311(a,b,p,d,s,g);
        case 654: return MD_Et_grad840(a,b,p,d,s,g);
        case 655: return MD_Et_grad841(a,b,p,d,s,g);
        case 656: return MD_Et_grad842(a,b,p,d,s,g);
        case 657: return MD_Et_grad843(a,b,p,d,s,g);
        case 658: return MD_Et_grad844(a,b,p,d,s,g);
        case 659: return MD_Et_grad845(a,b,p,d,s,g);
        case 660: return MD_Et_grad846(a,b,p,d,s,g);
        case 661: return MD_Et_grad847(a,b,p,d,s,g);
        case 662: return MD_Et_grad848(a,b,p,d,s,g);
        case 663: return MD_Et_grad849(a,b,p,d,s,g);
        case 664: return MD_Et_grad8410(a,b,p,d,s,g);
        case 665: return MD_Et_grad8411(a,b,p,d,s,g);
        case 666: return MD_Et_grad8412(a,b,p,d,s,g);
        case 667: return MD_Et_grad850(a,b,p,d,s,g);
        case 668: return MD_Et_grad851(a,b,p,d,s,g);
        case 669: return MD_Et_grad852(a,b,p,d,s,g);
        case 670: return MD_Et_grad853(a,b,p,d,s,g);
        case 671: return MD_Et_grad854(a,b,p,d,s,g);
        case 672: return MD_Et_grad855(a,b,p,d,s,g);
        case 673: return MD_Et_grad856(a,b,p,d,s,g);
        case 674: return MD_Et_grad857(a,b,p,d,s,g);
        case 675: return MD_Et_grad858(a,b,p,d,s,g);
        case 676: return MD_Et_grad859(a,b,p,d,s,g);
        case 677: return MD_Et_grad8510(a,b,p,d,s,g);
        case 678: return MD_Et_grad8511(a,b,p,d,s,g);
        case 679: return MD_Et_grad8512(a,b,p,d,s,g);
        case 680: return MD_Et_grad8513(a,b,p,d,s,g);
        case 681: return MD_Et_grad860(a,b,p,d,s,g);
        case 682: return MD_Et_grad861(a,b,p,d,s,g);
        case 683: return MD_Et_grad862(a,b,p,d,s,g);
        case 684: return MD_Et_grad863(a,b,p,d,s,g);
        case 685: return MD_Et_grad864(a,b,p,d,s,g);
        case 686: return MD_Et_grad865(a,b,p,d,s,g);
        case 687: return MD_Et_grad866(a,b,p,d,s,g);
        case 688: return MD_Et_grad867(a,b,p,d,s,g);
        case 689: return MD_Et_grad868(a,b,p,d,s,g);
        case 690: return MD_Et_grad869(a,b,p,d,s,g);
        case 691: return MD_Et_grad8610(a,b,p,d,s,g);
        case 692: return MD_Et_grad8611(a,b,p,d,s,g);
        case 693: return MD_Et_grad8612(a,b,p,d,s,g);
        case 694: return MD_Et_grad8613(a,b,p,d,s,g);
        case 695: return MD_Et_grad8614(a,b,p,d,s,g);
        case 696: return MD_Et_grad870(a,b,p,d,s,g);
        case 697: return MD_Et_grad871(a,b,p,d,s,g);
        case 698: return MD_Et_grad872(a,b,p,d,s,g);
        case 699: return MD_Et_grad873(a,b,p,d,s,g);
        case 700: return MD_Et_grad874(a,b,p,d,s,g);
        case 701: return MD_Et_grad875(a,b,p,d,s,g);
        case 702: return MD_Et_grad876(a,b,p,d,s,g);
        case 703: return MD_Et_grad877(a,b,p,d,s,g);
        case 704: return MD_Et_grad878(a,b,p,d,s,g);
        case 705: return MD_Et_grad879(a,b,p,d,s,g);
        case 706: return MD_Et_grad8710(a,b,p,d,s,g);
        case 707: return MD_Et_grad8711(a,b,p,d,s,g);
        case 708: return MD_Et_grad8712(a,b,p,d,s,g);
        case 709: return MD_Et_grad8713(a,b,p,d,s,g);
        case 710: return MD_Et_grad8714(a,b,p,d,s,g);
        case 711: return MD_Et_grad8715(a,b,p,d,s,g);
        case 712: return MD_Et_grad880(a,b,p,d,s,g);
        case 713: return MD_Et_grad881(a,b,p,d,s,g);
        case 714: return MD_Et_grad882(a,b,p,d,s,g);
        case 715: return MD_Et_grad883(a,b,p,d,s,g);
        case 716: return MD_Et_grad884(a,b,p,d,s,g);
        case 717: return MD_Et_grad885(a,b,p,d,s,g);
        case 718: return MD_Et_grad886(a,b,p,d,s,g);
        case 719: return MD_Et_grad887(a,b,p,d,s,g);
        case 720: return MD_Et_grad888(a,b,p,d,s,g);
        case 721: return MD_Et_grad889(a,b,p,d,s,g);
        case 722: return MD_Et_grad8810(a,b,p,d,s,g);
        case 723: return MD_Et_grad8811(a,b,p,d,s,g);
        case 724: return MD_Et_grad8812(a,b,p,d,s,g);
        case 725: return MD_Et_grad8813(a,b,p,d,s,g);
        case 726: return MD_Et_grad8814(a,b,p,d,s,g);
        case 727: return MD_Et_grad8815(a,b,p,d,s,g);
        case 728: return MD_Et_grad8816(a,b,p,d,s,g);
        default: return 0.0;
    }
}

// MD法におけるE'(grad)に関して，(i,l,t)の対応するデバイス関数を選択
inline double Et_grad_NonRecursion(int i, int l, int t, double alpha, double beta, double dist){
    if( i<0 || l<0 || t<0 || i+l<t){ // 範囲外の処理
        return 0.0;
    }else{
        return call_MD_Et_grad(4*i*(10+i) + (i+l)*(i+l+1)/2 + t, alpha, beta, alpha+beta, dist, sycl::exp(-alpha*beta/(alpha+beta)*dist*dist), -2*alpha*beta/(alpha+beta)*dist*sycl::exp(-(alpha*beta/(alpha+beta))*dist*dist));
        }
}


}
