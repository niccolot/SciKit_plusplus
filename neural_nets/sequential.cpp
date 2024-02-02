#include "sequential.h"
#include <eigen3/Eigen/Dense>

namespace neuralNets{

void Sequential::forward(Eigen::MatrixXf &x){for(auto l : m_model) l->forward(x,x);}

void Sequential::backward(float& loss, Eigen::MatrixXf& y_true, Eigen::MatrixXf& pred){

    m_loss.forward(loss, pred, y_true);

    Eigen::MatrixXf dloss;
    m_loss.backward(dloss, y_true, pred);
    for(auto it = m_model.rbegin(); it != m_model.rend(); ++it){
        (*it)->backward(dloss, dloss);
    }
}

void Sequential::add(Module* layer){m_model.emplace_back(layer);}

void Sequential::_set_loss(Losses::LossModule& loss){m_loss = loss;}
}; // namespace neuralNets