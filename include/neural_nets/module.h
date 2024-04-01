#ifndef MODULE_H
#define MODULE_H

#include <eigen3/Eigen/Dense>
#include <string>

namespace neuralNets{

class Module{

public:

virtual ~Module() = default;
virtual void forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x) = 0;
virtual void backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err) = 0;
virtual const Eigen::MatrixXf& _get_output() const = 0;
virtual const std::string& get_name() const = 0;
virtual void _set_weights_bias(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b) = 0;
virtual void _set_lr(float lr) = 0;
}; // class Module
}; // namespace neuralNets
#endif