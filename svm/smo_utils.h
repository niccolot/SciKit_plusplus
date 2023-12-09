/*
smo_utils.h

header with utility functions regarding the sequential minimal optimization (SMO)
algorithm
*/

#ifndef SMO_UTILS_H
#define SMO_UTILS_H

#include "data_utils.h"
#include <vector>
#include <algorithm>
#include <unordered_map>


template <typename T> 
void addVect(std::vector<T> const& x, std::vector<T> const& y, std::vector<T>& diff){

    assert(x.size() == y.size());

    size_t len = x.size();
    for(size_t i = 0; i<len; ++i){
        diff.push_back(x[i] - y[i]);
    }
    return;
}

template <typename T, typename K>
void vectByScalar(std::vector<T> &v, K k){
    std::transform(v.begin(), v.end(), v.begin(), [k](T &c){ return c*k; });
}

size_t heuristic_choice(size_t i2, std::vector<double>& E, size_t n_samples);

double objective_func(

    size_t i1, size_t i2, double a1, double a2,
    DataSet& dataset,
    std::vector<double>& alpha,
    std::vector<double>& w,
    std::unordered_map<size_t, size_t>& K_lookup);

#endif
