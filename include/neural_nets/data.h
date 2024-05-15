#ifndef DATA_H
#define DATA_H

#include <eigen3/Eigen/Dense>
#include <vector>


std::vector<Eigen::MatrixXf> generateXORdataset() {
    /**
     * generates the XOR dataset:
     * 
     * [{0,0}, {0,1}, {1,0}, {1,1}]
    */

    std::vector<Eigen::MatrixXf> dataset{
        Eigen::Map<Eigen::MatrixXf>(new float[2]{0, 0}, 1, 2),
        Eigen::Map<Eigen::MatrixXf>(new float[2]{0, 1}, 1, 2),
        Eigen::Map<Eigen::MatrixXf>(new float[2]{1, 0}, 1, 2),
        Eigen::Map<Eigen::MatrixXf>(new float[2]{1, 1}, 1, 2)
    };

    return dataset;
}

std::vector<Eigen::MatrixXf> generateXORlabels() {
    /**
     * generates the labels for the XOR dataset
     * 
     * [{0}, {1}, {1}, {0}]
    */
   
    std::vector<Eigen::MatrixXf> labels{
        Eigen::Map<Eigen::MatrixXf>(new float[1]{0}, 1, 1),
        Eigen::Map<Eigen::MatrixXf>(new float[1]{1}, 1, 1),
        Eigen::Map<Eigen::MatrixXf>(new float[1]{1}, 1, 1),
        Eigen::Map<Eigen::MatrixXf>(new float[1]{0}, 1, 1)
    };

    return labels;
}

#endif