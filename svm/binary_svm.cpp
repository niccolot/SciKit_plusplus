/*
binary_svm.cpp

implementation of the member function for a binary support vector machine classifier

...
*/

#include "binary_svm.h"
#include "kernels.h"
#include "utils.h"
#include <cassert>


BinarySVM::BinarySVM(kernelPars kernel_pars, std::string kernel, double C){
    /**
     * Support vector machine (SVM) constructor
     * 
     * @param kernel_pars struct containing the parameter for the kernel functions 
     * @param kernel whihc kernel function to use, linear kernel doesn't need parameters
     * @param C soft margin hyperparameter
    */

    assert(kernel_pars.gamma >= 0.0);
    assert(kernel_pars.d >= 0);
    assert(kernel=="linear" || kernel=="poly" || kernel=="rbf" || kernel=="tanh");

    if(kernel=="rbf"){
        this->K = &rbfKernel;
    }
    else if(kernel=="linear"){
        this->K = &linearKernel;
    }
    else if(kernel=="poly"){
        this->K = &polyKernel;
    }
    else if(kernel=="tanh"){
        this->K = &tanhKernel;
    }

    m_C = C;
}

void BinarySVM::fit(int epochs, std::string filename, bool label_first_col){
    /**
     * @param epochs train epochs
     * @param filename path to train dataset
     * @param label_first_col whether the labels are on the first or last column in 
     *  the train dataset
    */

    DataSet dataset;
    load_dataset(filename, dataset, label_first_col, true);
    std::vector<double> m_alpha(dataset.features[0].size(), 0.0); // lagrange multipliers for dual problem
    
}



void BinarySVM::predict(std::string filename, std::string outfile){
    /**
     * @param filename path to test dataset
     * @param outfile path to new file which will be a copy of the 
     *  test dataset but with another column containing the labels predicted
    */

    DataSet dataset;
    load_dataset(filename, dataset);



   
}
