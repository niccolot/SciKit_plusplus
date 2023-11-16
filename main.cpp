#include "k_means_clustering/point.h"
#include "k_means_clustering/k_means.h"
#include <iostream>

int main(){
    
    int epochs = 100;
    int k = 5;
    std::string input_file = "mall_data_2d.csv";
    std::string output_file = "output1.csv";
    
    K_means k_means(input_file, k, "kmeans++");
    k_means.fit(epochs, "elkan");
    k_means.write_to_file(output_file);

    return 0;
}


