#include "sequential.h"
#include "layers.h"
#include "activations.h"
#include <eigen3/Eigen/Dense>

int main(){

    Eigen::MatrixXf xData{
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };
    

    Eigen::MatrixXf yData{{0,1,1,0}};
    
    neuralNets::Sequential model;

    model.add(new neuralNets::Layers::Linear(2,3));
    model.add(new neuralNets::Activations::ReLU());
    model.add(new neuralNets::Layers::Linear(3,1));
 
    neuralNets::Losses::MSE mse;
    model.set_loss(&mse);
    
    return 0;
}