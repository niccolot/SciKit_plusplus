/*
utils.cpp

definition of functions only declared in utils.h
*/

#include "data_utils.h"
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

using std::size_t;


void load_dataset(std::string filename, DataSet& dataset, bool label_first_col, bool train, bool shuffle, int seed){
    /**
     * takes the dataset path as input and loads it into the model
     * 
     * the dataset has to be a .csv file with binary labels as +-1 
     * in the first or last column and numerical features in the others if is the train dataset,
     * only numerical features if is the test dataset 
     * 
     * @param filename string with filepath
     * @param dataset reference to the DataSet struct where to store the datapoints and labels
     * @param label_first_col whether the label column is the first or the last, default is false
     * @param train whether is the train dataset or not, in order to append also the labels to
     *  the labels member of the dataset, default is true
     * @param shuffle whether to shuffle the input dataset, default is true
     * @param seed the seed for the shuffling
    */

    std::ifstream file(filename);
    std::string line;

    // Read each line from the file
    while (std::getline(file, line)) {
        std::vector<double> feature_row;
        std::istringstream iss(line);
        std::string token;

        // Parse each comma-separated value in the line
        while (std::getline(iss, token, ',')) {
            // Convert the token to double and add it to the feature_row
            feature_row.push_back(std::stod(token));
        }

        if(train){
            // Determine the index for labels based on label_at_first_column
            size_t label_index = label_first_col ? 0 : feature_row.size() - 1;

            // Convert the label token to an integer and add it to the labels vector
            dataset.labels.push_back((feature_row[label_index]));

            // Remove the label from feature_row if it's not needed
            if (label_first_col){
                feature_row.erase(feature_row.begin());
            }else{
            feature_row.pop_back();
            }
        }

        // Add the feature_row to the features vector
        dataset.features.push_back(feature_row);
    }

    if(shuffle){
        std::default_random_engine gen(seed);
        std::shuffle(dataset.features.begin(), dataset.features.end(), gen);
    }
}


void appendToCSV(const std::string& filename, const std::vector<double>& data, int label){
    /**
     * append a datapoint and label to a .csv, the label will be appended 
     * on the last column
     * 
     * @param filename file in which to save the classified dataset
     * @param data feature vector
     * @param label label predicted to be appended in the last column 
    */

    // Open the file in append mode
    std::ofstream file;
    file.open(filename, std::ios::app);

    // Check if the file is successfully opened
    if (!file.is_open()) {
        std::cerr << "Error opening the file: " << filename << std::endl;
        return;
    }

    // Write each element of the vector to a new column
    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i];
        if (i < data.size() - 1) {
            file << ",";
        }
    }

    // Write the label in the last column
    file << "," << label << "\n";

    // Close the file
    file.close();
}


double variance(const std::vector<std::vector<double>>& points){

    double mean_of_square = 0.0;
    double square_of_mean = 0.0;
    size_t n_points = points.size();
    double inverse = 1/(double)n_points;

    for(auto point : points){
        for(auto x : point){
            mean_of_square += x*x;
            square_of_mean += x;
        }
    }
    square_of_mean *= inverse;
    square_of_mean *= square_of_mean;

    

    double var = mean_of_square*inverse - square_of_mean;
    return var;
}


