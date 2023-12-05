/*
k_means.h

k_means clustering model 

the implementation contains two different centroid's
initialization, random and kmeans++ algorithm, and
two different clustering algorithms: lloyd's and elkan's  

usage: 
    instanciate the model, load the dataset into it
    and choose the initialization strategy by calling the constructor.
    The dataset has to be a .csv file formatted as (x, y, ...) with an arbitrary
    number of features.

    Clusterize the dataset with the 'fit' method with the chosen algorithm 
    and write the clusterized dataset with the 'write_to_file' method which
    produce a .csv file formatted as (x, y, ..., c) with 'c' the label of the cluster 
*/

#ifndef K_MEANS_H
#define K_MEANS_H
#include "point.h"
#include <iostream>
#include <vector>

class K_means{

    private:

        std::vector<Point> centroids;
        std::vector<Point> points; // dataset's points
        int m_k;

    public:

        K_means() = delete;
        K_means(std::string file_name, int k_val, std::string init="kmeans++");
        void fit(int epochs, std::string algorithm="lloyd");
        void write_to_file(std::string output_file);
};
#endif
