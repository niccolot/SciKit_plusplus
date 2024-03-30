#include "losses.h"
#include <eigen3/Eigen/Dense>

namespace neuralNets{
namespace Losses{

void MSE::forward(float& loss, const Eigen::MatrixXf& y_true, const Eigen::MatrixXf& pred){
    /**
     * mse loss forward pass
     * 
     * @param y_true true value
     * @param pred value predicted by model
     * @return loss, loss value
    */

    loss = (pred - y_true).squaredNorm() / y_true.rows();
}

void MSE::backward(Eigen::MatrixXf& dloss, const Eigen::MatrixXf& y_true, const Eigen::MatrixXf& pred){
    /**
     * mse backward pass
     * 
     * @param y_true true value
     * @param pred value predicted by model
     * @return dloss derivative of loss wrt pred
    */

    dloss = 2.f*(pred - y_true)/y_true.rows();
}
}; // namespace Losses
}; // namespace neuralNets

