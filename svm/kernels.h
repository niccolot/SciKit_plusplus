/*
kernels.h

declarations of various kernel functions and struct containing the hyperparameters
*/

#ifndef KERNELS_H
#define KERNELS_H

#include <vector>
#include <unordered_map>

using std::size_t;

// parameters for kernel functions
typedef struct{
    double gamma;
    double r {0.0};
    int d {3};
} kernelPars; 

// lookup table for kernel functions values
typedef std::unordered_map<size_t, std::unordered_map<size_t, double>> LookUpTable;

double linearKernel(const std::vector<double>& u, const std::vector<double>& v, const kernelPars& pars);
double polyKernel(const std::vector<double>& u, const std::vector<double>& v, const kernelPars& pars);
double rbfKernel(const std::vector<double>& u, const std::vector<double>& v, const kernelPars& pars);
double tanhKernel(const std::vector<double>& u, const std::vector<double>& v, const kernelPars& pars);
#endif
