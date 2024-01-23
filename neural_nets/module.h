#include <eigen3/Eigen/Dense>

#ifndef MODULE_H
#define MODULE_H

namespace neuralNets{

class Module{

public:

virtual void forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x) = 0;

virtual void backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err) = 0;
}; // class Module
}; // namespace neuralNets

#endif