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

public:

Sequential() = default;
Sequential(const std::vector<Module*>& model, Losses::LossModule* loss) : m_model(model), m_loss(loss){}
~Sequential(); 
void add(Module* layer);
void set_loss(Losses::LossModule* loss);
void predict(std::vector<Eigen::MatrixXf>& pred, const std::vector<Eigen::MatrixXf>& dataset);
void fit(const std::vector<Eigen::MatrixXf>& dataset, const std::vector<Eigen::MatrixXf>& labels, int epochs, float& acc, float& loss, bool verbose=false);
const Eigen::MatrixXf& _get_output_from_layer(int layer) const;
void set_W_b_layer(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b, int layer);
int count_layers() const;
void _forward(Eigen::MatrixXf& x);
void _backward(float& loss, const Eigen::MatrixXf& y_true, Eigen::MatrixXf& pred);

}; // class Sequential
}; // namespace neuralNets

#endif
