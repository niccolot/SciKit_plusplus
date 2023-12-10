/*
binary_svm.h

Support vector machine for binary classification
*/

#ifndef BINARY_SVM_H
#define BINARY_SVM_H

#include "kernels.h"
#include <string>
#include <vector>


class BinarySVM{

private:
    
    std::vector<double> m_w; // weight vector
    double m_b; // bias
    double m_C; // regularization, C->\infty is the hard margin SVM
    double m_gamma; // kernel hyperparameter
    kernelPars m_kernel_pars; 
    double (*K)(std::vector<double>, std::vector<double>, kernelPars); // pointer to kernel function
    double f(const std::vector<double>& x) {return std::inner_product(x.begin(), x.end(), m_w.begin(), -m_b);} // f(x) = w*x - b
    double m_train_acc;
    double m_val_acc;
    
    int takeStep(    
        size_t i1, size_t i2,
        const DataSet& dataset, 
        std::vector<double>& E, 
        std::vector<double>& alpha,
        std::unordered_map<size_t, size_t>& K_lookup);
    
    int examineExample(
        size_t i2, 
        DataSet& dataset, 
        std::vector<double>& E, 
        std::vector<double>& alpha, 
        std::unordered_map<size_t, size_t>& K_lookup);

public:

    BinarySVM() = delete;
    BinarySVM(kernelPars kernel_pars, std::string kernel="rbf", double C=1.0, std::string gamma="auto");
    void fit(int max_epochs, std::string filename, bool label_first_col=false, bool shuffle=true, int seed=25);
    void predict(std::string filename, std::string outfile);
    double const get_train_acc() {return m_train_acc;}
    double const get_val_acc() {return m_val_acc;}
};
#endif
