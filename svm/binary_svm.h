/*
binary_svm.h

Support vector machine for binary classification
*/

#ifndef BINARY_SVM_H
#define BINARY_SVM_H

#include "kernels.h"
#include "data_utils.h"
#include <string>
#include <vector>
#include <numeric>

using std::size_t;


class BinarySVM{

private:
    
    std::vector<double> m_w; // weight vector
    double m_b; // bias
    double m_C; // regularization, C->\infty is the hard margin SVM
    std::string m_gamma; // gamma kernel hyperparameter selection
    kernelPars m_kernel_pars; 
    double (*K)(std::vector<double>, std::vector<double>, kernelPars); // pointer to kernel function
    double f(const std::vector<double>& x) {return std::inner_product(x.begin(), x.end(), m_w.begin(), -m_b);} // f(x) = w*x - b
    double m_train_acc;
    double m_test_acc;
    
    int takeStep(    
        size_t i1, size_t i2,
        DataSet& dataset, 
        std::vector<double>& E, 
        std::vector<double>& alpha,
        LookUpTable& K_lookup);
    
    int examineExample(
        size_t i2, 
        DataSet& dataset, 
        std::vector<double>& E, 
        std::vector<double>& alpha, 
        LookUpTable& K_lookup);

public:

    BinarySVM() = delete;
    BinarySVM(kernelPars kernel_pars, std::string kernel="rbf", double C=1.0, std::string gamma="auto");
    double fit(int max_epochs, std::string filename, bool label_first_col=false, bool shuffle=true, int seed=25);
    double predict(std::string filename, std::string outfile);
    double get_train_acc() const {return m_train_acc;}
    double get_test_acc() const {return m_test_acc;}
};
#endif
