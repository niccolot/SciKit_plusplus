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

double linearKernel(std::vector<double> u, std::vector<double> v, kernelPars pars);
double polyKernel(std::vector<double> u, std::vector<double> v, kernelPars pars);
double rbfKernel(std::vector<double> u, std::vector<double> v, kernelPars pars);
double tanhKernel(std::vector<double> u, std::vector<double> v, kernelPars pars);
#endif
