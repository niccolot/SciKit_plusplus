# Support vector machine classifier (SVM)

Implementation of binary softmargin SVM optimized with the [SMO algorithm ](https://www.researchgate.net/publication/2624239_Sequential_Minimal_Optimization_A_Fast_Algorithm_for_Training_Support_Vector_Machines). The kernels available are 

* Linear
* Polynomial
* Gaussian (rbf)
* Tanh

The dataset has to be a `.csv` file with numerical data and +-1 label on the first or last column

## Contents

* `binary_svm.h`: declaration of the SVM class 
* `kernels.h`: declaration of the kernel functions
* `smo_utils.h`: declaration of some functions used in the SMO optimization routine
* `data_utils.h`: auxiliary function in order to load, test and manipulate the dataset


