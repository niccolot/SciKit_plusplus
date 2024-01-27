#include "activations.h"
#include <eigen3/Eigen/Dense>


namespace neuralNets{
namespace Activations{

float sigmoid(float x) { return 1/(1+exp(-x)); } // sigma(x)
float sigmoid_prime(float x) { return sigmoid(x)*(1-sigmoid(x)); } // sigma'(x)

void ReLU::forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x){

    m_forward_input = x;
    out = (x.array() < 0).select(0, x); // ReLU(x) = max{0,x}
}

void ReLU::backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err){

    // in_err = out_err * ReLU'(forward_input)
    // with * the element wise product
    in_err = (m_forward_input.array() < 0).select(0, out_err); 
}


void Sigmoid::forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x){

    m_forward_input = x;
    out = x.unaryExpr(&m_sigmoid);
}

void Sigmoid::backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err){

    in_err = out_err.cwiseProduct(m_forward_input.unaryExpr(&m_sigmoid_prime));
}
}; // namespace Activations
}; // namespace neuralNets