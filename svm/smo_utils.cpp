/*
smo_utils.cpp

definitions of utility functions regarding the sequential minimal optimization (SMO) algorithm, 
in particular the heuristic choice method and the local objective function calculator

    heuristic_choice() implements the choice of the second alpha to be optimized as illustrated in Platt's article
    objectiv_fucn() calculates a local objective function to be used when eta > 0 (refer to Platt's article)
*/

#include "smo_utils.h"
#include "data_utils.h"
#include "kernels.h"
#include <unordered_map>
#include <vector>

using std::size_t;


size_t heuristic_choice(size_t i2, std::vector<double>& E, size_t n_samples){
    /**
     * @param i2 index of the first alpha to be optimized
     * @param E erorr cache 
     * @param n_samples number of datapoints in the dataset 
     * @return the index of the chosen alpha to be optimized along with alpha[i2]
    */

    double maxDiff = 0;
    size_t index = 0;
    if(E[i2] > 0){
        for(size_t j = 0; j < n_samples; ++j){
            if(E[j] < maxDiff && i2 != j){
                index = j;
                maxDiff = E[j];
            }
        }
    }
    else{
        for(size_t j = 0; j < n_samples; ++j){
            if(E[j] > maxDiff && i2 != j){
                index = j;
                maxDiff = E[j];
            }
        }
    }

    return index;
}

double objective_func(
    /**
     * @param i1, i2 indexes of the alphas
     * @param a1,a2 alpha values of i1,i2
     * @param dataset struct containig the training dataset
     * @param alpha alphas vector
     * @param K_lookup kernel lookup table
     * @return value of the objective function
    */
    
    size_t i1, size_t i2, double a1, double a2,
    DataSet& dataset,
    std::vector<double>& alpha,
    LookUpTable& K_lookup)
{

    int y1 = dataset.labels[i1];
    int y2 = dataset.labels[i2];
    double k11 = K_lookup[i1][i1];
    double k12 = K_lookup[i1][i2];
    double k21 = K_lookup[i2][i1];
    double k22 = K_lookup[i2][i2];

    double obj_val = k11 * alpha[i1] * a1 * y1 * y1 + k12 * alpha[i1] * a2 * y1 * y2 + k21 * a2 * alpha[i1] * y2 * y1 + k22 * a2 * a2 * y2 * y2;

    return alpha[i1] + a2 - 0.5*obj_val;
}




