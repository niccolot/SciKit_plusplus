/*
kernels.cpp

definitions of kernel functions
*/

#include "kernels.h"
#include "utils.h"
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

    std::vector<double> x(u.size(), 0);
    vectDiff(u,v,x);
    double y = std::inner_product(x.begin(), x.end(), x.begin(), 0.0); // l2 squared norm ||u-v||^2
    return exp(-y*pars.gamma);
}

double tanhKernel(std::vector<double> u, std::vector<double> v, kernelPars pars){

    double dot = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
    return tanh(dot*pars.gamma + pars.r);
}
