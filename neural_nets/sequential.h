#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "layers.h"
#include "losses.h"
#include <eigen3/Eigen/Dense>

namespace neuralNets{

class Sequential{

private:

std::vector<Module*> m_model;
Losses::LossModule& m_loss;

public:

Sequential();
Sequential(std::vector<Module*>& model, Losses::LossModule& loss) : m_model(model), m_loss(loss){}
~Sequential(){for(auto l : m_model) delete l;}
void add(Module* layer);
void _set_loss(Losses::LossModule& loss);

void forward(Eigen::MatrixXf &x);
void backward(float& loss, Eigen::MatrixXf& y_true, Eigen::MatrixXf& pred);
}; // class Sequential


}; // namespace neuralNets

#endif
