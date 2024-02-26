#include "losses.h"
#include <eigen3/Eigen/Dense>

namespace neuralNets{
namespace Losses{

void MSE::forward(float& loss, const Eigen::MatrixXf& pred, const Eigen::MatrixXf& y_true){

    loss = (pred - y_true).squaredNorm() / y_true.rows();
}

void MSE::backward(Eigen::MatrixXf& dloss, const Eigen::MatrixXf& pred, const Eigen::MatrixXf& y_true){

    dloss = 2.f*(pred - y_true)/y_true.rows();
}
}; // namespace Losses
}; // namespace neuralNets

