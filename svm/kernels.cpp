/*
kernels.cpp

definition of kernel functions
*/

#include "kernels.h"
#include <numeric>
#include <math.h>
#include <bits/stdc++.h>


double linearKernel(std::vector<double> u, std::vector<double> v, kernelPars pars){

    return std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
}

double polyKernel(std::vector<double> u, std::vector<double> v, kernelPars pars){

    double dot = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
    return pow(pars.gamma*dot + pars.r, pars.d);
}

double rbfKernel(std::vector<double> u, std::vector<double> v, kernelPars pars){

    double uu = std::inner_product(u.begin(), u.end(), u.begin(), 0.0);
    double vv = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double uv = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
    double norm = uu - 2*uv + vv; // l2 squared norm ||u-v||^2 = xx - 2xy + yy
    return exp(-norm*pars.gamma);
}

double tanhKernel(std::vector<double> u, std::vector<double> v, kernelPars pars){

    double dot = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
    return tanh(dot*pars.gamma + pars.r);
}
