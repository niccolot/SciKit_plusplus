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
    double m_C; // regularization
    double (*K)(std::vector<double>, std::vector<double>, kernelPars); // pointer to kernel function

public:

    BinarySVM() = delete;
    BinarySVM(kernelPars kernel_pars, std::string kernel="rbf", double C=1.0);
    void fit(int epochs, std::string filename, bool label_first_col);
    void predict(std::string filename, std::string outfile);
};
#endif
