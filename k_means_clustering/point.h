/*
point.h

Implementation of the point data structure for the k_means 
clustering algorithm.

This implementation contains members for storing the coordinate
of the point in feature space, the point's cluster id and the distace
between the point and it's cluster's centroid's coordinates
*/

#ifndef POINT_H
#define POINT_H
#include <iostream>
#include <vector>

using index_t = std::vector<double>::size_type; 


class Point{

    private:

        std::vector<double> coords; // point coordinates
        int cluster_idx; // cluster the points belongs to 
        double min_dist_val; // squared distance between the point and the nearest cluster

    public:

        // constructors
        Point() : cluster_idx(-1), min_dist_val(__DBL_MAX__) {}
        Point(index_t dim); 

        int& cluster() {return cluster_idx;}
        double& min_dist() {return min_dist_val;}
        std::vector<double>& coordinates() {return coords;}
        double l2_distance(Point p);
  
};
#endif
