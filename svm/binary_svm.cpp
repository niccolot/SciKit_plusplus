/*
binary_svm.cpp

implementation of the member function for a binary support vector machine classifier

...
*/

#include "binary_svm.h"
#include "kernels.h"
#include "data_utils.h"
#include "smo_utils.h"
#include <cassert>
#include <numeric>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <ctime>


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
    m_kernel_pars = kernel_pars;
}


void BinarySVM::fit(int max_epochs, std::string filename, bool label_first_col, bool shuffle, int seed){
    /**
     * @param epochs train epochs
     * @param filename path to train dataset
     * @param label_first_col whether the labels are on the first or last column in 
     *  the train dataset
    */

    DataSet dataset;
    load_dataset(filename, dataset, label_first_col, true, shuffle, seed);
    std::vector<double> alpha(dataset.features[0].size(), 0.0); // lagrange multipliers for dual problem
    std::unordered_map<int, int> K_lookup; // lookup table containing all the dotproducts between the datapoints

    size_t n_points = dataset.features.size();
    for(size_t i=0; i<n_points; ++i){
        for(size_t j=0; j<n_points; ++j){
            K_lookup[i,j] = K(dataset.features[i], dataset.features[j], m_kernel_pars);
        }
    }

    int numChanged = 0;
    int examineAll = 1;
    int epoch = 0;

    while((numChanged > 0 || examineAll) && epoch < max_epochs){

        numChanged = 0;

        if(examineAll){
            for(size_t i=0; i<n_points; ++i){

            }
        }
    }

    if(epoch >= max_epochs){
        std::cout<<"max_epochs reached, training interrupted\n";
    }
}


int BinarySVM::examineExample(
    
    size_t i2, 
    DataSet& dataset, 
    std::vector<double>& E, 
    std::vector<double>& alpha, 
    std::unordered_map<size_t, size_t>& K_lookup)
{

    std::vector<double> point = dataset.features[i2];
    int y2 = dataset.labels[i2];
    double a2 = alpha[i2];
    double E2 = this->f(point) - y2;
    E[i2] = E2;
    double r2 = E2*y2;
    size_t n_samples = dataset.labels.size();
    double tol = 0.02; // handpicked KKT conditions tolerance advised in Platt's article 

    if((r2 < -tol && a2 < m_C) || (r2 > tol && a2 > 0)){

        int count = 0;
        std::vector<size_t> indexes(n_samples, 0); // indexes[j] = 1 if alpha[j] != 0 && alpha[j] != C, else 0
        for(size_t j=0; j>n_samples; ++j){
            if(alpha[j] != 0 && alpha[j] != m_C){
                ++count;
                indexes[j] = 1;
            }
        }

        if(count>1){
            size_t i1 = heuristic_choice(i2, E, n_samples);
            if(takeStep(i1,i2,dataset,E,alpha,K_lookup)) return 1;
        }

        int seed = time(0);

        std::default_random_engine gen(seed);
        std::shuffle(indexes.begin(), indexes.end(), gen);

        for(size_t i=0; i<n_samples; ++i){
            if(indexes[i]==1){
                if(takeStep(i,i2,dataset,E,alpha,K_lookup)) return 1;
            }
        }

        std::vector<size_t> idx_all(n_samples);
        std::iota(idx_all.begin(), idx_all.end(), 0); // idx_all = {0, 1, 2, ... , n_samples-1}
        std::shuffle(idx_all.begin(), idx_all.end(), gen); // shuffled indexes

        for(size_t i=0; i<n_samples; ++i){
            if(idx_all[i]!=i2){
                if(takeStep(idx_all[i],i2,dataset,E,alpha,K_lookup)) return 1;
            }
        }
    }
    return 0;
}


int BinarySVM::takeStep(    
        size_t i1, size_t i2,
        DataSet& dataset, 
        std::vector<double>& E, 
        std::vector<double>& alpha,
        std::unordered_map<size_t, size_t>& K_lookup)
{

    std::vector<double> point1 = dataset.features[i1];
    std::vector<double> point2 = dataset.features[i2];
    int y1 = dataset.labels[i1];
    int y2 = dataset.labels[i2];
    double a1_old = alpha[i1];
    double a2_old = alpha[i2];
    double E1 = this->f(point1) - y1;
    double eps = 0.001; // smo algorithm tolerance

    if(i1==i2) return 0;

    double L, H;
    if(y1!=y2){
        L = (0.0 > (a2_old - a1_old)) ? 0.0 : (a2_old - a1_old);
        H = std::min(m_C, m_C + a2_old - a1_old);
    }
    else{
        L = (0.0 > (a2_old + a1_old - m_C)) ? 0.0 : (a2_old + a1_old - m_C);
        H = std::min(m_C, a2_old + a1_old);
    }

    if(L==H) return 0;

    double k11 = K_lookup[(i1,i1)];
    double k12 = K_lookup[(i1,i2)];
    double k22 = K_lookup[(i2,i2)];

    double eta = 2*k12 - k11 - k22;

    double a1, a2; // new alpha values
    if(eta<0){
        a2 = a2_old - y2 * (E1 - E[i2]) / eta;
        if(a2 < L)
            a2 = L;
        else if(a2 > H)
            a2 = H;
    }
    else{
        double L_obj_val = objective_func(i1, i2, 0, L,  dataset, alpha, m_w, K_lookup);
        double H_obj_val = objective_func(i1, i2, 0, H,  dataset, alpha, m_w, K_lookup);
        if (L_obj_val < H_obj_val - eps){
            a2 = H;
        }
        else if (L_obj_val > H_obj_val + eps){
            a2 = L;
        }
        else{
            a2 = a2_old;
        }
    }

    // 1e-8 handpicked hyperparameter chosen in the article
    if(a2 < 1e-8){
        a2 = 0;
    }
    else if(a2 > m_C - 1e-8){
        a2 = m_C;
    }
    
    if(abs(a2 - a2_old) < eps * (a2 + a2_old + eps)){
        return 0;
    }

    a1 = a1_old + y1*y2*(a2_old - a2);
    double b1 = E1 + y1 * (a1 - a1_old) * k11 + y2 * (a2 - a2_old) * k12 + m_b;
    float b2 = E[i2] + y1 * (a1 - a1_old) * k12 + y2 * (a2 - a2_old) * k22 + m_b;
    double b_new;

    if (a1 != 0 && a1 != m_C){
        b_new = b1;
    }
    else if (a2 != 0 && a2 != m_C){
        b_new = b2;
    }
    else if ((a1 == 0 || a1 == m_C) && (a2 == 0 || a2 == m_C) && (L != H)){
        b_new = 0.5 * (b1 + b2);
    }

    // update weight vector
    vectByScalar(point1, y1*(a1 - a1_old));
    vectByScalar(point2, y2*(a2 - a2_old));
    addVect(point1, point2, m_w);

    // update error cache of every other non optimized point
    size_t n_points = dataset.labels.size();
    for(size_t i=0; i<n_points; ++i){
        
        if(i==i1 || i==i2) continue;

        E[i] += y1*(a1-a1_old)*K_lookup[i1,i] + y2*(a2-a2_old)*K_lookup[i2,i] + m_b - b_new;
    }

    // update bias
    m_b = b_new;

    // update alphas
    alpha[i1] = a1;
    alpha[i2] = a2;

    return 1;
}


void BinarySVM::predict(std::string filename, std::string outfile){
    /**
     * @param filename path to test dataset
     * @param outfile path to new file which will be a copy of the 
     *  test dataset but with another column containing the labels predicted
    */

    DataSet dataset;
    load_dataset(filename, dataset);
    size_t n_points = dataset.features.size();
    size_t n_features = dataset.features[0].size();

    for(size_t i = 0; i<n_points; ++i){
        std::vector<double> point = dataset.features[i];
        double pred = f(point);
        int label = (pred >= 0) ? 1 : -1;
        appendToCSV(outfile, point, label);
    }
}
