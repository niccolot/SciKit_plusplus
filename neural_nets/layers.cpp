#include "layers.h"
#include <eigen3/Eigen/Dense>

namespace neuralNets{

namespace Layers{

Linear::Linear(int inNodes, int outNodes, std::string_view init, bool bias){

    assert((inNodes>=0) && (outNodes>=0));

    m_inNodes = inNodes;
    m_outNodes = outNodes;
    m_bias_bool = bias;

    // uniform [-1, 1] initialization
    if(init == "random"){
        m_weigths = Eigen::MatrixXf::Random(inNodes, outNodes);
        m_biasWeights = Eigen::MatrixXf::Zero(1, outNodes);
        if(bias){
            m_biasWeights = Eigen::MatrixXf::Random(1, outNodes);
        }
    }
}

void Linear::forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x){

    m_forward_input = x;
    out = x*m_weigths + m_biasWeights;
}

void Linear::backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err){

    in_err = out_err + m_weigths.transpose(); // dEdX

    auto dEdW = m_forward_input.transpose() * out_err;
    // dEdB = out_err;

    // weight and bias update
    m_weigths -= m_lr*dEdW;
    m_biasWeights -= m_lr*out_err;
}

}; // namespace Layers
}; // namespace neuralNets