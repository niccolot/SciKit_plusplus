#ifndef LAYERS_H
#define LAYERS_H

#include <eigen3/Eigen/Dense>

namespace neuralNets{
namespace Layers{

class Linear{

private:

int m_inNodes, m_outNodes;
Eigen::MatrixXf m_weigths;
Eigen::VectorXf m_bias;
    
public:

Linear() = delete;
Linear(int inNodes, int outNodes, std::string_view init="random", bool bias=true);
void forward(Eigen::VectorXf& out, const Eigen::VectorXf& x);
   
}; // class Linear
}; // namespace Layers
}; // namespace neuralNets
#endif