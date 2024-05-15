#include "sequential.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cassert>


namespace neuralNets{


Sequential::~Sequential(){
    for(auto l : m_model){
        delete l;
        l = nullptr;
    } 
}


void Sequential::_forward(Eigen::MatrixXf& x) {for(const auto& layer : m_model) layer->forward(x,x);}


void Sequential::_backward(float& loss, const Eigen::MatrixXf& y_true, Eigen::MatrixXf& pred){

    m_loss->forward(loss, y_true, pred);

    Eigen::MatrixXf dloss;
    m_loss->backward(dloss, y_true, pred);
    for(auto it = m_model.rbegin(); it != m_model.rend(); ++it){
        (*it)->backward(dloss, dloss);
    }
}


void Sequential::set_loss(Losses::LossModule* loss){m_loss = loss;}


int Sequential::count_layers() const{

    int layers = 0;

    for(const auto layer : m_model) ++layers;

    return layers;
}


void Sequential::add(Module* layer){m_model.emplace_back(layer);}


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


void Sequential::fit(
    const std::vector<Eigen::MatrixXf>& dataset, 
    const std::vector<Eigen::MatrixXf>& labels, 
    int epochs, float& acc, float& loss, float lr, bool verbose){

    int samples = dataset.size();
    _set_lr(lr);

    for(int epoch=0; epoch<epochs; ++epoch){

        float err=0;
        for(int i=0; i<samples; ++i){

            Eigen::MatrixXf x;
            Eigen::MatrixXf label;
            x = dataset[i];
            label = labels[i];
        
            _forward(x);

            float loss_val;
            _backward(loss_val, label, x);
            err += loss_val;
        }

        loss = err/samples;

        if(verbose){
            std::cout<<"Epoch: "<<epoch+1<<"/"<<epochs;
            std::cout<<"  Loss: "<<loss<<"\n\n";
        }
    }
}


/*
void Sequential::fit(
    const std::vector<Eigen::MatrixXf>& dataset, 
    const std::vector<Eigen::MatrixXf>& labels, 
    int epochs, float& acc, float& loss, int batch_size, float lr, bool verbose){

    int samples = dataset.size();
    int batches = samples/batch_size;
    int remainingDataPoints = samples % batch_size;
    int features = dataset[0].cols();
    _set_lr(lr);

    for(int epoch=0; epoch<epochs; ++epoch){

        float err1=0;
        for(int batch_idx=0; batch_idx<batches; ++batch_idx){

            Eigen::MatrixXf xBatch;
            Eigen::MatrixXf labelBatch;

            xBatch.resize(batch_size, features);
            labelBatch.resize(batch_size,1);

            for(int i=0; i<batch_size; ++i){
                xBatch.row(i) = dataset[i+batch_idx*batch_size];
                labelBatch.row(i) = labels[i+batch_idx*batch_size];
            }
        
            _forward(xBatch);

            float loss_val;
            _backward(loss_val, labelBatch, xBatch);
            err1 += loss_val;
        }

        err1 /= batches;

        float err2 = 0;
        
        if(remainingDataPoints != 0){

            for(int i=0; i<remainingDataPoints; ++i){

                Eigen::MatrixXf x;
                Eigen::MatrixXf label;
                x = dataset[i + batches*batch_size + 1];
                label = labels[i + batches*batch_size + 1];
            
                _forward(x);

                float loss_val;
                _backward(loss_val, label, x);
                err2 += loss_val;            
            }
            err2 /= remainingDataPoints;
        }
        
        loss = err1 + err2;

        if(verbose){
            std::cout<<"Epoch: "<<epoch+1<<"/"<<epochs;
            std::cout<<"  Loss: "<<loss<<"\n\n";
        }
    }
}

*/


const Eigen::MatrixXf& Sequential::_get_output_from_layer(int layer) const{

    int totalLayers = count_layers();
    
    assert(layer < totalLayers);

    return m_model[layer]->_get_output();
}


void Sequential::set_W_b_layer(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b, int layer){

    int totalLayers = count_layers();
    
    assert(layer < totalLayers);

    m_model[layer]->_set_weights_bias(w,b);
}

}; // namespace neuralNets