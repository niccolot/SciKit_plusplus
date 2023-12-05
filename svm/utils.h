/*
utils.h

declaration of utility functions
*/

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <cassert>

using index_t = std::vector<double>::size_type;

typedef struct{
    std::vector<std::vector<double>> features;
    std::vector<double> labels;
} DataSet;

template <typename T> 
void vectDiff(std::vector<T> const& x, std::vector<T> const& y, std::vector<T>& diff){

    assert(x.size() == y.size());

    for(index_t i = 0; i<x.size(); ++i){
        diff.push_back(x.at(i) - y.at(i));
    }
    return;
}

void load_dataset(std::string filename, DataSet& dataset, bool label_first_col=false, bool train=false);

#endif
