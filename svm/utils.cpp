/*
utils.cpp

definition of functions only declared in utils.h
*/

#include "utils.h"
#include <fstream>
#include <sstream>


void load_dataset(std::string filename, DataSet& dataset, bool label_first_col, bool train){
    /**
     * takes the dataset path as input and loads it into the model
     * 
     * the dataset has to be a .csv file with binary labels as +-1 
     * in the first or last column and numerical features in the others if is the train dataset,
     * only numerical features if is the test dataset 
     * 
     * @param filename string with filepath
     * @param dataset reference to the DataSet struct where to store the datapoints and labels
     * @param label_first_col whether the label column is the first or the last
     * @param train whether is the train dataset or not, in order to append also the labels to
     *  the labels member of the dataset
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
}


