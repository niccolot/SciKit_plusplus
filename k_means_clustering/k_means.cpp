/*
k_means.cpp

Implementation of the k_means clustering model as a class defined
in the file k_means.hpp
*/

#include "point.h"
#include "k_means.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cmath>

#include <iterator>
#include <bits/stdc++.h>

using index_t = std::vector<double>::size_type; 


K_means::K_means(std::string file_name, int k_val, std::string init){

    // read file and codify the datapoints in the 'points' member
    if(init!="kmeans++" && init!="random"){
        std::cout<<"ERROR: invalid 'init' value, needs to be either kmeans++ or random";
        exit(EXIT_FAILURE);
    }

    if(k_val<0){
        std::cout<<"ERROR: invalid 'k' value, needs to be a positive int";
        exit(EXIT_FAILURE);
    }
 
    std::ifstream file(file_name);

    m_k = k_val;

    set_points(points, file);

    index_t n = points.size();
    srand(time(0));

    // random (i.e. 'Forgy') cluster initialization
    if(init == "random"){
        
        for (int i = 0; i < m_k; ++i){
            centroids.push_back(points.at(rand() % n));
        }
    }

    // kmeans++ initialization algorithm
    if(init == "kmeans++"){
        k_means_pp_init(centroids, points, m_k, n);
    }
}


void K_means::fit(int epochs, std::string algorithm){

    if(algorithm!="lloyd" && algorithm!="elkan"){
        std::cout<<"ERROR: invalid 'algorithm' value, needs to be either lloyd or elkan";
        exit(EXIT_FAILURE);
    }

    if(epochs<0){
        std::cout<<"ERROR: invalid 'epochs' value, needs to be a positive int";
        exit(EXIT_FAILURE);
    }

    index_t n_features = points.at(0).coordinates().size(); // number of datapoint's features

    //lloyd's algorithm
    if(algorithm == "lloyd"){

	    for (int i = 0; i < epochs; ++i) {

		    lloyd_step(centroids, points, m_k, n_features);
		}
		return;
    }

    if(algorithm == "elkan"){

        // lower and upper bounds for the elkan's algorithm
        // given point x, centroid c and d(,) a distance: 
        // lower_bounds(c,x) <= d(c,x) <= upper_bounds(x)
        std::vector<std::vector<double>> lower_bounds(m_k, std::vector<double>(points.size(), 0.0)); 
        std::vector<double> upper_bounds;
        std::vector<std::vector<double>> centroid_centroid_distance; // distances between centroids

        elkan_init(lower_bounds, upper_bounds, centroid_centroid_distance, centroids, points, m_k);
        
        // repeat untill convergence
        for(int epoch=0; epoch<epochs; ++epoch){

            elkan_step(lower_bounds, upper_bounds, centroid_centroid_distance, centroids, points, m_k, n_features);
        }
        return;
    }
}


void K_means::write_to_file(std::string output_file){

    std::ofstream myfile;
    myfile.open(output_file);
    int n_features = points.at(0).coordinates().size();

    for (std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it) {
        
        for(int i = 0; i < n_features; ++i){
            myfile<<it->coordinates().at(i)<<",";
        }
        myfile<<it->cluster()<<"\n";
    }
    myfile.close();

    return;
}
