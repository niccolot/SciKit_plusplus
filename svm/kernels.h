/*
kernels.h

declarations of various kernel functions 
*/

#ifndef KERNELS_H
#define KERNELS_H

#include <vector>

typedef struct{
    double gamma;
    double r;
    int d;
} kernelPars; // parameters for kernel functions

double linearKernel(std::vector<double> u, std::vector<double> v, kernelPars pars);
double polyKernel(std::vector<double> u, std::vector<double> v, kernelPars pars);
double rbfKernel(std::vector<double> u, std::vector<double> v, kernelPars pars);
double tanhKernel(std::vector<double> u, std::vector<double> v, kernelPars pars);
#endif
