#include "layers.h"
#include <eigen3/Eigen/Dense>
#include <cassert>

namespace neuralNets{
namespace Layers{


Linear::Linear(int inNodes, int outNodes, std::string_view init, bool bias){

    assert((inNodes>=0) && (outNodes>=0));

    m_inNodes = inNodes;
    m_outNodes = outNodes;
    m_bias_bool = bias;

    // uniform [-1, 1] initialization
    if(init == "random"){
        m_weights = Eigen::MatrixXf::Random(inNodes, outNodes);
        m_biasWeights = Eigen::MatrixXf::Zero(1, outNodes);
        if(bias){
            m_biasWeights = Eigen::MatrixXf::Random(1, outNodes);
        }
    }
}


void Linear::forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x){

    assert(x.cols() == m_inNodes);

    m_forward_input = x;
    out = x*m_weights + m_biasWeights;
    m_forward_output = out;
}


void Linear::backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err){

    auto dEdW = m_forward_input.transpose() * out_err;

    // weight and bias update
    m_weights -= m_lr*dEdW;
    m_biasWeights -= m_lr*out_err; // dEdB = out_err

    in_err = out_err * m_weights.transpose(); // dEdX
}


void Linear::_set_weights_bias(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b){

    m_weights = w;
    m_biasWeights = b;
}

}; // namespace Layers
}; // namespace neuralNets