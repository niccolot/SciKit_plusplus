#ifndef LAYERS_H
#define LAYERS_H

#include <eigen3/Eigen/Dense>

namespace neuralNets{

class Module{

public:

virtual ~Module() = default;
virtual void forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x) = 0;
virtual void backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err) = 0;
}; // class Module

namespace Layers{



class Linear : public Module{

private:

int m_inNodes, m_outNodes;
Eigen::MatrixXf m_weigths;
Eigen::MatrixXf m_biasWeights;
Eigen::MatrixXf m_forward_input;
bool m_bias_bool;
float m_lr = 0.01;
    
public:

Linear() = delete;
~Linear() = default;
Linear(int inNodes, int outNodes, std::string_view init="random", bool bias=true);
virtual void forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x) override;
virtual void backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err) override;
void _set_weights_bias(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b);
   
}; // class Linear

}; // namespace Layers
}; // namespace neuralNets
#endif