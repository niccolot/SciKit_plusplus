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
void set_loss(Losses::LossModule& loss);
void forward(Eigen::MatrixXf &x);
void backward(float& loss, Eigen::MatrixXf& y_true, Eigen::MatrixXf& pred);


public:

Sequential();
Sequential(std::vector<Module*>& model, Losses::LossModule& loss) : m_model(model), m_loss(loss){}
~Sequential(){for(auto l : m_model) delete l;}
void add(Module* layer);
void predict(std::vector<Eigen::MatrixXf>& pred, const std::vector<Eigen::MatrixXf>& dataset);
void fit(const std::vector<Eigen::MatrixXf>& dataset, const std::vector<Eigen::MatrixXf>& labels, int epochs, float& acc, float& loss);

}; // class Sequential
}; // namespace neuralNets

#endif
