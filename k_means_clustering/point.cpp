/*
point.cpp

Implementation of the members functions of the Point class
defined in point.hpp
*/

#include "point.h"
#include <iostream>
#include <vector>
#include <cmath> 

using index_t = std::vector<double>::size_type;


Point::Point(index_t dim){
    /*
    dim is the dimensionality of the datapoint 
    */

    coords = std::vector<double>(dim, 0.0); // zero initialization
    cluster_idx = -1; // no cluster

    // at first the point is set to be infinitely far from every cluster
    min_dist_val = __DBL_MAX__; 
}

double Point::l2_distance(Point p){

    if(p.coordinates().size() != coords.size()){
        std::cout<<"ERROR: points do not have the same dimensionality";
        exit(EXIT_FAILURE);
    }

    double d = 0.0;

    for(index_t i=0; i<p.coordinates().size(); ++i){

        d += pow(p.coordinates().at(i) - coords.at(i), 2); 
    }
    
    return d;
}
