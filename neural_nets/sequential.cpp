#include "sequential.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

namespace neuralNets{

void Sequential::forward(Eigen::MatrixXf &x) {for(const auto& layer : m_model) layer->forward(x,x);}

void Sequential::backward(float& loss, Eigen::MatrixXf& y_true, Eigen::MatrixXf& pred){

    m_loss.forward(loss, pred, y_true);

    Eigen::MatrixXf dloss;
    m_loss.backward(dloss, y_true, pred);
    for(auto it = m_model.rbegin(); it != m_model.rend(); ++it){
        (*it)->backward(dloss, dloss);
    }
}

void Sequential::add(Module* layer){m_model.emplace_back(layer);}

void Sequential::set_loss(Losses::LossModule& loss){m_loss = loss;}

void Sequential::predict(std::vector<Eigen::MatrixXf>& pred, const std::vector<Eigen::MatrixXf>& dataset){

    for(const auto& x : dataset){
        
        Eigen::MatrixXf out;
        out = x;
        for(const auto layer : m_model){
            layer->forward(out,out);
        }

        pred.push_back(out);
    }
}

void Sequential::fit(const std::vector<Eigen::MatrixXf>& dataset, const std::vector<Eigen::MatrixXf>& labels, int epochs, float& acc, float& loss){

    int samples = dataset.size();

    for(int epoch=0; epoch<epochs; ++epoch){

        float err=0;
        for(int i=0; i<samples; ++i){

            Eigen::MatrixXf out;
            Eigen::MatrixXf label;
            out = dataset[i];
            label = labels[i];

            this->forward(out);

            float loss_val;
            this->backward(loss_val, label, out);
            err += loss_val;
        }

        loss = err/samples;

        std::cout<<"Epoch: "<<epoch+1<<"/"<<epochs;
        std::cout<<"Loss: "<<loss<<"\n\n";
    }
}
}; // namespace neuralNets