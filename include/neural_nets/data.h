#ifndef DATA_H
#define DATA_H

#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

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


void readCSV(const std::string& filename, std::vector<Eigen::MatrixXf>& dataPoints, std::vector<Eigen::MatrixXf>& labels, bool labelsInFirstColumn) {
    // Open the CSV file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Read data from CSV file
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;
        Eigen::MatrixXf dataPoint(1, 0);
        Eigen::MatrixXf label(1, 0);
        while (std::getline(ss, cell, ',')) {
            float val;
            std::istringstream iss(cell);
            if (iss >> val) {
                if ((labelsInFirstColumn && col == 0) || (!labelsInFirstColumn && col == 0)) {
                    label.conservativeResize(1, label.cols() + 1);
                    label(0, label.cols() - 1) = val;
                } else {
                    dataPoint.conservativeResize(1, dataPoint.cols() + 1);
                    dataPoint(0, dataPoint.cols() - 1) = val;
                }
            }
            col++;
        }
        labels.push_back(label);
        dataPoints.push_back(dataPoint);
    }

    // Close the file
    file.close();
}


void normalizeDataAndLabels(std::vector<Eigen::MatrixXf>& dataPoints, std::vector<Eigen::MatrixXf>& labels) {
    // Calculate mean and variance for data points
    for (size_t i = 0; i < dataPoints.size(); ++i) {
        Eigen::MatrixXf& dataPoint = dataPoints[i];
        float mean = dataPoint.mean();
        float var = (dataPoint.array() - mean).square().sum() / dataPoint.size();
        dataPoint = (dataPoint.array() - mean) / sqrt(var);
    }

    // Flatten labels vector and calculate mean and variance
    Eigen::MatrixXf allLabels(labels.size(), labels[0].size());
    for (size_t i = 0; i < labels.size(); ++i) {
        allLabels.row(i) = labels[i];
    }
    float mean = allLabels.mean();
    float var = (allLabels.array() - mean).square().sum() / allLabels.size();

    // Normalize labels
    for (size_t i = 0; i < labels.size(); ++i) {
        labels[i] = (labels[i].array() - mean) / sqrt(var);
    }
}

#endif