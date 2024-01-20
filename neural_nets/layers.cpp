#include "layers.h"
#include <eigen3/Eigen/Dense>

namespace neuralNets{
namespace Layers{

Linear::Linear(int inNodes, int outNodes, std::string_view init, bool bias){

    assert((inNodes>=0) && (outNodes>=0));

    m_inNodes = inNodes;
    m_outNodes = outNodes;

    // uniform [-1, 1] initialization
    if(init == "random"){
        m_weigths = Eigen::MatrixXf::Identity(outNodes, inNodes);
        if(bias){
            m_bias = Eigen::VectorXf::Zero(outNodes);
        }
    }
}

void Linear::forward(Eigen::VectorXf& out, const Eigen::VectorXf& x){
    out = m_weigths*x + m_bias;
}


}; // namespace Layers
}; // namespace neuralNets