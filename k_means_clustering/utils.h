/*
utils.h

declaration for the utility functions:

    set_points() reads the datapoitns from the .csv dataset and codify them in the 'points' class member

    k_means_pp_init() implements the kmeans++ centroid initialization algorithm

    lloyd_step() performs a single training step with the lloyd's algorithm

    elkan_init() performs the initialization of the elkan's training routine

    elkan_step() performs a single training step with the elkan's algorithm
*/

#ifndef UTILS_H
#define UTILS_H
#include "k_means.h"
#include "point.h"

using index_t = std::vector<double>::size_type;


void set_points(std::vector<Point>& points, std::ifstream& file);
void k_means_pp_init(std::vector<Point>& centroids, std::vector<Point>& points, int& k, index_t n);
void lloyd_step(std::vector<Point>& centroids, std::vector<Point>& points, int& k, index_t& n_features);

void elkan_init(std::vector<std::vector<double>>& lower_bounds, 
                std::vector<double>& upper_bounds, 
                std::vector<std::vector<double>>& centroid_centroid_distance,
                std::vector<Point>& centroids,
                std::vector<Point>& points,
                int& k);

void elkan_step(std::vector<std::vector<double>>& lower_bounds, 
                std::vector<double>& upper_bounds, 
                std::vector<std::vector<double>>& centroid_centroid_distance,
                std::vector<Point>& centroids,
                std::vector<Point>& points,
                int& k,
                index_t& n_features);
#endif
