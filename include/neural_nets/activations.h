#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "layers.h"
#include <eigen3/Eigen/Dense>
#include <cmath>

namespace neuralNets{
namespace Activations{

class ReLU : public Module{

private:

Eigen::MatrixXf m_forward_input;

public:

ReLU() = default;
~ReLU() = default;
virtual void forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x) override;
virtual void backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err) override;
}; // class ReLU


class Sigmoid : public Module{

private:

Eigen::MatrixXf m_forward_input;
static float m_sigmoid(float x) { return 1/(1+exp(-x)); } // sigma(x)
static float m_sigmoid_prime(float x) { return m_sigmoid(x)*(1-m_sigmoid(x)); } // sigma'(x)



public:

Sigmoid() = default;
~Sigmoid() = default;
virtual void forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x) override;
virtual void backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err) override;

}; // class Sigmoid

}; // namespace Activations
}; // namespace neuralNets

#endif