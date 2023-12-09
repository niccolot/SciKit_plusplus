/*
utils.h

declaration of utility functions
*/

#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <iostream>
#include <vector>
#include <cassert>


typedef struct{
    std::vector<std::vector<double>> features; // shape (n_points, n_features)
    std::vector<int> labels;
} DataSet;

void load_dataset(std::string filename, DataSet& dataset, bool label_first_col=false, bool train=false, bool shuffle=true, int seed=25);
void appendToCSV(const std::string& filename, const std::vector<double>& data, int label);

#endif
