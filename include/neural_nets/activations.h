#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "layers.h"
#include "module.h"
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <string>

namespace neuralNets{
namespace Activations{

class ReLU : public Module{

private:

Eigen::MatrixXf m_forward_input;
Eigen::MatrixXf m_forward_output;
std::string name = "ReLU";

public:

ReLU() = default;
~ReLU() = default;
virtual void forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x) override;
virtual void backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err) override;
virtual void _set_weights_bias(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b) override;
virtual const Eigen::MatrixXf& _get_output() const {return m_forward_output;}
const std::string& get_name() const {return name;}

}; // class ReLU


class Sigmoid : public Module{

private:

Eigen::MatrixXf m_forward_input;
Eigen::MatrixXf m_forward_output;
static float m_sigmoid(float x) { return 1/(1+exp(-x)); } // sigma(x)
static float m_sigmoid_prime(float x) { return m_sigmoid(x)*(1-m_sigmoid(x)); } // sigma'(x)
std::string name = "Sigmoid";



public:

Sigmoid() = default;
~Sigmoid() = default;
virtual void forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x) override;
virtual void backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err) override;
virtual void _set_weights_bias(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b) override;
virtual const Eigen::MatrixXf& _get_output() const {return m_forward_output;}
const std::string& get_name() const {return name;}

}; // class Sigmoid

}; // namespace Activations
}; // namespace neuralNets

#endif