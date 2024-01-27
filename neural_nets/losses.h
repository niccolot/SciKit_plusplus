#ifndef LOSSES_H
#define LOSSES_H

#include "module.h"
#include <eigen3/Eigen/Dense>

namespace neuralNets{
namespace Losses{

class MSE{

public:

void forward(float& loss, const Eigen::MatrixXf& pred, const Eigen::MatrixXf& y_true);
void backward(Eigen::MatrixXf& dloss, const Eigen::MatrixXf& pred, const Eigen::MatrixXf& y_true);

}; // class MSE
}; // namespace Losses
}; // namespace neuralNets
#endif