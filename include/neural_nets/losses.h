#ifndef LOSSES_H
#define LOSSES_H

#include <eigen3/Eigen/Dense>


namespace neuralNets{
namespace Losses{

class LossModule{

public:

virtual void forward(float& loss, const Eigen::MatrixXf& y_true, const Eigen::MatrixXf& pred) = 0;
virtual void backward(Eigen::MatrixXf& dloss, const Eigen::MatrixXf& y_true, const Eigen::MatrixXf& pred) = 0;
}; // class LossModule


class MSE : public LossModule{

public:

virtual void forward(float& loss, const Eigen::MatrixXf& y_true, const Eigen::MatrixXf& pred) override;
virtual void backward(Eigen::MatrixXf& dloss, const Eigen::MatrixXf& y_true, const Eigen::MatrixXf& pred) override;

}; // class MSE

}; // namespace Losses
}; // namespace neuralNets
#endif