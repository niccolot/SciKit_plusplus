/*
binary_svm.h

Support vector machine for binary classification
*/

#ifndef BINARY_SVM_H
#define BINARY_SVM_H

#include "kernels.h"
#include "data_utils.h"
#include <string>
#include <string_view>
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
    double (*K)(const std::vector<double>&, const std::vector<double>&, const kernelPars&); // pointer to kernel function
    double f(const std::vector<double>& x) {return std::inner_product(x.begin(), x.end(), m_w.begin(), -m_b);} // f(x) = w*x - b
    double m_train_acc;
    double m_test_acc;

    int examineExample(
        size_t i2, 
        const DataSet& dataset, 
        std::vector<double>& E, 
        std::vector<double>& alpha, 
        const LookUpTable& K_lookup);

    int takeStep(    
        size_t i1, size_t i2,
        const DataSet& dataset, 
        std::vector<double>& E, 
        std::vector<double>& alpha,
        const LookUpTable& K_lookup);
    
public:

    BinarySVM() = delete;
    BinarySVM(kernelPars& kernel_pars, std::string_view kernel="poly", double C=1.0, std::string_view gamma="scale");
    double fit(int max_epochs, std::string& filename, bool label_first_col=false, bool shuffle=true, int seed=25);
    double predict(std::string& filename, std::string& outfile);
    double get_train_acc() const {return m_train_acc;}
    double get_test_acc() const {return m_test_acc;}
};
#endif
