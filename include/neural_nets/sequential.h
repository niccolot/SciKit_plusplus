#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "layers.h"
#include "losses.h"
#include <eigen3/Eigen/Dense>

namespace neuralNets{

class Sequential{

private:

std::vector<Module*> m_model;
Losses::LossModule* m_loss;

void forward(Eigen::MatrixXf &x);
void backward(float& loss, Eigen::MatrixXf& y_true, Eigen::MatrixXf& pred);
void check_inOut_dims();


public:

Sequential() = default;
Sequential(const std::vector<Module*>& model, Losses::LossModule* loss) : m_model(model), m_loss(loss){}
~Sequential() {for(auto l : m_model) delete l;}
void add(Module* layer);
void set_loss(Losses::LossModule* loss);
void predict(std::vector<Eigen::MatrixXf>& pred, const std::vector<Eigen::MatrixXf>& dataset);
void fit(const std::vector<Eigen::MatrixXf>& dataset, const std::vector<Eigen::MatrixXf>& labels, int epochs, float& acc, float& loss);
const Eigen::MatrixXf& _get_output_from_layer(int layer) const;
int count_layers() const;

}; // class Sequential
}; // namespace neuralNets

#endif
