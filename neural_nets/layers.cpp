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
        m_weigths = Eigen::MatrixXf::Identity(outNodes, inNodes);
        m_biasWeights = Eigen::MatrixXf::Zero(outNodes, 1);
        if(bias){
            m_biasWeights = Eigen::MatrixXf::Ones(outNodes, 1);
        }
    }
}

void Linear::forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x){
    out = m_weigths*x + m_biasWeights;
}

}; // namespace Layers

}; // namespace neuralNets