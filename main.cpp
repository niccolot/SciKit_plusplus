#include "svm/data_utils.h"
#include "svm/smo_utils.h"
#include "svm/binary_svm.h"
#include "svm/kernels.h"
#include <iostream>
#include <string>

using std::size_t;



int main(){
    
    std::string inputFile = "svm/wdbc_data_formatted.csv";
    kernelPars kernel_pars = {1.0, 0.0, 3};
    BinarySVM svm(kernel_pars);
    double acc = svm.fit(50,inputFile);
    return 0;
}


