/*
smo_utils.cpp

definitions of utility functions regarding the sequential minimal optimization (SMO) algorithm, 
in particular the heuristic choice method and the local objective function calculator
*/

#include "smo_utils.h"
#include "data_utils.h"
#include <unordered_map>
#include <vector>


size_t heuristic_choice(size_t i2, std::vector<float> &E, size_t n_samples){

    double maxDiff = 0;
    size_t index = 0;
    if(E[i2] > 0){
        for(size_t j = 0; j < n_samples; j++){
            if(E[j] < maxDiff && i2 != j){
                    index = j;
                    maxDiff = E[j];
                }
            }
        }
        else{
            for(size_t j = 0; j < n_samples; j++){
                if(E[j] > maxDiff && i2 != j){
                    index = j;
                    maxDiff = E[j];
                }
            }
        }
    return index;
}

double objective_func(
    
    size_t i1, size_t i2, double a1, double a2,
    DataSet& dataset,
    std::vector<double>& alpha,
    std::vector<double>& w,
    std::unordered_map<size_t, size_t>& K_lookup)
{

    int y1 = dataset.labels[i1];
    int y2 = dataset.labels[i2];
    double k11 = K_lookup[(i1,i1)];
    double k12 = K_lookup[(i1,i2)];
    double k21 = K_lookup[(i2,i1)];
    double k22 = K_lookup[(i2,i2)];

    double obj_val = k11 * alpha[i1] * a1 * y1 * y1 + k12 * alpha[i1] * a2 * y1 * y2 + k21 * a2 * alpha[i1] * y2 * y1 + k22 * a2 * a2 * y2 * y2;

    return alpha[i1] + a2 - 0.5*obj_val;
}




