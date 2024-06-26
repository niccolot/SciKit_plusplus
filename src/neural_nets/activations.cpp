#include "activations.h"
#include <eigen3/Eigen/Dense>


namespace neuralNets{
namespace Activations{


/*ReLU ACTIVATION*/
void ReLU::forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x){
    /**
     * relu activation forward pass
     * 
     * @param x input tensor
     * @return out output tensor
    */
    m_forward_input = x;
    out = (x.array() < 0.f).select(0.f, x); // ReLU(x) = max{0,x}
    m_forward_output = out;
}

void ReLU::backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err){
    /**
     * relu activation backward pass
     * 
     * @param out_err derivative of error wrt the output given by subsequent layer in backpropagation
     * @return in_err derivative of error wrt the input
    */

    // in_err = out_err * ReLU'(forward_input)
    // with * the element wise product
    // and ReLU' just the step function
    in_err = (m_forward_input.array() < 0.f).select(0.f, out_err); 
}

void ReLU::_set_weights_bias(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b){}
void ReLU::_set_lr(float lr){}



/*SIGMOID ACTIVATION*/
void Sigmoid::forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x){
    /**
     * sigmoid activation forward pass
     * 
     * @param x input tensor
     * @return out output tensor
    */

    m_forward_input = x;
    out = x.unaryExpr(&m_sigmoid);
    m_forward_output = out;
}

void Sigmoid::backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err){
    /**
     * sigmoid activation backward pass
     * 
     * @param out_err derivative of error wrt the output given by subsequent layer in backpropagation
     * @return in_err derivative of error wrt the input
    */

    in_err = out_err.cwiseProduct(m_forward_input.unaryExpr(&m_sigmoid_prime));
}


void Sigmoid::_set_weights_bias(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b){}
void Sigmoid::_set_lr(float lr){}



/*SOFTMAX ACTIVATION*/
void Softmax::forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x){
    /**
     * softmax activation forward pass
     * 
     * @param x input tensor
     * @return out output tensor
    */

    m_forward_input = x;

    Eigen::MatrixXf expX = x.array().exp();

    int cols = x.cols();
    for(int i=0; i<cols; ++i){
        out.col(i) = expX.col(i) / expX.row(i).sum();
    }

    m_forward_output = out;

}

void Softmax::backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err){
    /**
     * softmax activation backward pass
     * 
     * @param out_err derivative of error wrt the output given by subsequent layer in backpropagation
     * @return in_err derivative of error wrt the input
    */
    
    in_err = out_err.array() * (m_forward_output.array() * (1.f - m_forward_output.array()).array()).array();

}

void Softmax::_set_weights_bias(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b){}
void Softmax::_set_lr(float lr){}

}; // namespace Activations
}; // namespace neuralNets