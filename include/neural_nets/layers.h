#ifndef LAYERS_H
#define LAYERS_H

#include "module.h"
#include <eigen3/Eigen/Dense>
#include <string>

namespace neuralNets{
namespace Layers{

class Linear : public Module{

private:

int m_inNodes, m_outNodes;
Eigen::MatrixXf m_weights;
Eigen::MatrixXf m_biasWeights;
Eigen::MatrixXf m_forward_input;
Eigen::MatrixXf m_forward_output;
std::string name = "Linear";
bool m_bias_bool;
float m_lr = 0.1;
    
public:

Linear() = delete;
~Linear() = default;
Linear(int inNodes, int outNodes, std::string_view init="random", bool bias=true);
virtual void forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x) override;
virtual void backward(Eigen::MatrixXf& in_err, const Eigen::MatrixXf& out_err) override;
virtual void _set_weights_bias(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b) override;
const Eigen::MatrixXf& _get_output() const {return m_forward_output;}
const Eigen::MatrixXf& _get_weights() const {return m_weights;}
const Eigen::MatrixXf& _get_bias() const {return m_biasWeights;}
const std::string& get_name() const {return name;}
int _getInNodes() const {return m_inNodes;}
int _getOutNodes() const {return m_outNodes;}
   
}; // class Linear

}; // namespace Layers
}; // namespace neuralNets
#endif