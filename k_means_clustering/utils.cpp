/*
utils.cpp

definition of the utility functions declared in utils.h
*/

#include "utils.h"
#include "k_means.h"
#include "point.h"
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>

using index_t = std::vector<double>::size_type;


void set_points(std::vector<Point>& points, std::ifstream& file){
    /*
    points: datapoints
    file_name: name of the .csv dataset file
    */

    std::string line;

    // load the dataset into 'points' vector
    while (getline(file, line)) {

        Point current_point;
        std::stringstream lineStream(line);
        std::string cell;

        while(getline(lineStream, cell, ',')){  
            current_point.coordinates().push_back(stod(cell));
        }
        points.push_back(current_point);
    }
    file.close();
}


void k_means_pp_init(std::vector<Point>& centroids, std::vector<Point>& points, int& k, index_t n){
    /*
    centroids: coordinates of the centroids
    points: datapoints
    k: number of clusters
    n: number of datapoints
    */


    // first cluster chosen at random
    centroids.push_back(points.at(rand() % n));

    // following cluster chosen by maximising the distance between the clusters
    for(int i = 0; i < k-1; ++i){

        std::vector<double> distances;

        for(index_t j=0; j<n; ++j){
                
            double d = __DBL_MAX__;
            for(index_t m=0; m<centroids.size(); ++m){
                double temp = centroids.at(m).l2_distance(points.at(j));

                if(temp<d){
                    d = temp;
                }
            }
            distances.push_back(d);
        }

        Point next_centroid;
        index_t index = std::distance(distances.begin(), std::max_element(distances.begin(), distances.end()));
        next_centroid = points.at(index);
        centroids.push_back(next_centroid);        
    }
}


void lloyd_step(std::vector<Point>& centroids, std::vector<Point>& points, int& k, index_t& n_features){
    /*
    centroids: coordinates of the centroids
    points: datapoints
    k: number of clusters
    n_features: number of datapoint's features
    */

    // For each centroid, compute distance from centroid to each point
	// and update point's cluster if necessary
	for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c){

		long long int clusterId = c - begin(centroids);

		for (std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it){

			Point p = *it;
			double dist = c->l2_distance(p);

			if (dist < p.min_dist()){
				p.min_dist() = dist;
				p.cluster() = clusterId;
			}
			*it = p;
		}
	}
			
	std::vector<int> nPoints; // number of points belonging to each cluster
			
	// coordinates of the mass center of each cluster, shape = (num_clusters, num_features)
	std::vector<std::vector<double>> mass_centers;  

	for(int j = 0; j < k; ++j){
		nPoints.push_back(0);
		mass_centers.push_back(std::vector<double>(n_features, 0.0));
	}

	// Iterate over points to append data to centroids
	for (std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it){

		int clusterId = it->cluster();
		nPoints.at(clusterId) += 1;
				
		for(index_t m = 0; m < n_features; ++m){
			mass_centers.at(clusterId).at(m) += it->coordinates().at(m);
		}

		it->min_dist() = __DBL_MAX__;  // reset distance
	}

	// Compute the new centroids
    // 
    // lloyd's algorithm is acutally implemented by using 
    // the cluster's mass centers as centroids and not by integration
	for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c){

		index_t clusterId = c - begin(centroids);

		for(index_t j = 0; j < n_features; ++j){
			c->coordinates().at(j) = mass_centers.at(clusterId).at(j) / nPoints.at(clusterId); 
		}
	}
}


void elkan_init(std::vector<std::vector<double>>& lower_bounds, 
                std::vector<double>& upper_bounds, 
                std::vector<std::vector<double>>& centroid_centroid_distance,
                std::vector<Point>& centroids,
                std::vector<Point>& points,
                int& k)
{
    /*
    lower_bounds: elkans' algorithm distance's lower bounds
    upper_bounds: elkans' algorithm distance's upper bounds 
    centroid_centroid_distances: distance between each centroid
    centroids: coordinates of the centroids
    points: datapoints
    k: number of clusters
    */

    for(int i=0; i<k; ++i){
            std::vector<double> temp;
            for(int j=0; j<k; ++j){
                temp.push_back(centroids.at(i).l2_distance(centroids.at(j)));
            }
            centroid_centroid_distance.push_back(temp);
        }

        // at first give a random centroid to each point
        for(std::vector<Point>::iterator it=points.begin(); it != points.end(); ++it){
            it->cluster() = rand() % k;
        }

        // elkan's initialization
        for(std::vector<Point>::iterator it=points.begin(); it != points.end(); ++it){
            
            int point_id = it - points.begin();
            std::vector<double> temp;
            
            for(std::vector<Point>::iterator c = centroids.begin(); c != centroids.end(); ++c){
                
                int centroid_id = c - centroids.begin();
                double dist = it->l2_distance(centroids.at(centroid_id));
                temp.push_back(dist);
                lower_bounds.at(centroid_id).at(point_id) = dist;
            }

            int new_cluster = std::distance(temp.begin(), std::min_element(temp.begin(), temp.end()));
            it->cluster() = new_cluster;
            upper_bounds.push_back(temp.at(new_cluster));
        }
}


void elkan_step(std::vector<std::vector<double>>& lower_bounds, 
                std::vector<double>& upper_bounds, 
                std::vector<std::vector<double>>& centroid_centroid_distance,
                std::vector<Point>& centroids,
                std::vector<Point>& points,
                int& k,
                index_t& n_features)
{
    /*
    lower_bounds: elkans' algorithm distance's lower bounds
    upper_bounds: elkans' algorithm distance's upper bounds 
    centroid_centroid_distances: distance between each centroid
    centroids: coordinates of the centroids
    points: datapoints
    k: number of clusters
    n_features: number of datapoint's features 
    */

    for(int i=0; i<k; ++i){
        for(int j=0; j<k; ++j){
            centroid_centroid_distance.at(i).at(j) = centroids.at(i).l2_distance(centroids.at(j));
        }
    }

	std::vector<double> s;
	
	for(int i=0; i<k; ++i){
		double min_temp =  __DBL_MAX__;
		for(int j=0; j<k; ++j){
			double dist = centroid_centroid_distance.at(i).at(j);
			if(i!=j && dist<min_temp){
				min_temp = dist;
			}
		}
		s.push_back(0.5*min_temp);
	}
	
	std::vector<int> temp_points;
	
	for(std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it){
		
		int point_id = it - points.begin();
	
		if(upper_bounds.at(point_id) > s.at(points.at(point_id).cluster())){
			temp_points.push_back(point_id);
		}
	}
	
	for(std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it){
	
		int point_id = it - points.begin();
	
		for(std::vector<Point>::iterator itc = centroids.begin(); itc != centroids.end(); ++itc){
	
			int centroid_id = itc - centroids.begin();
	
			if(centroid_id != points.at(point_id).cluster() && 
				upper_bounds.at(point_id) > lower_bounds.at(centroid_id).at(point_id) &&
				upper_bounds.at(point_id) > 0.5*centroid_centroid_distance.at(points.at(point_id).cluster()).at(centroid_id))
			{
				double d = it->l2_distance(centroids.at(points.at(point_id).cluster()));
				upper_bounds.at(point_id) = d;
	
				if(d > lower_bounds.at(centroid_id).at(point_id) || 
					d > 0.5*centroid_centroid_distance.at(points.at(point_id).cluster()).at(centroid_id))
				{
					double d1 = it->l2_distance(*itc);
					if(d1 < d){
						points.at(point_id).cluster() = centroid_id;
						upper_bounds.at(point_id) = d;
					}
				}
			}
		}
	} 
	
	std::vector<std::vector<int>> points_per_cluster; // idxs of the points belonging to each cluster
	std::vector<Point> means; // means(c) is the mean of the points assigned to c, shape = (k, num_features)
	
	
	for(std::vector<Point>::iterator itc = centroids.begin(); itc != centroids.end(); ++itc){
		
		int cluster_idx = itc - centroids.begin();
		std::vector<int> temp_vec;
		for(std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it){
	
			int point_idx = it - points.begin();
			if(it->cluster() == cluster_idx){
				temp_vec.push_back(point_idx);
			}
		}
		points_per_cluster.push_back(temp_vec);
	}
	
	for(int c=0; c<k; ++c){ // for every cluster
		Point p;
		for(index_t j=0; j<n_features; ++j){ // for every feature
			
			double sum = 0.0;
			for(std::vector<int>::iterator it = points_per_cluster.at(c).begin(); it!= points_per_cluster.at(c).end(); ++it){
				int point_idx = it - points_per_cluster.at(c).begin();
				sum += points.at(point_idx).coordinates().at(j);
			}
			p.coordinates().push_back(sum/static_cast<double>(points_per_cluster.at(c).size()));
		}
		means.push_back(p);
	}
	
	for(std::vector<Point>::iterator itc = centroids.begin(); itc != centroids.end(); ++itc){
		
		int cluster_idx = itc - centroids.begin();
		Point c = *itc;
		
		for(std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it){
			
			int point_idx = it - points.begin();
			double lb = lower_bounds.at(cluster_idx).at(point_idx);
			lower_bounds.at(cluster_idx).at(point_idx) = std::max(0.0, lb - c.l2_distance(means.at(cluster_idx)));
		}
	}
	
	for(std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it){
	
		int point_idx = it - points.begin();
		upper_bounds.at(point_idx) += centroids.at(it->cluster()).l2_distance(means.at(it->cluster()));
	}
	
	for(std::vector<Point>::iterator itc = centroids.begin(); itc != centroids.end(); ++itc){
	
		int centroid_idx = itc - centroids.begin();
		centroids.at(centroid_idx) = means.at(centroid_idx);
	}
}

