#define CATCH_CONFIG_MAIN
#include "sequential.h"
#include "layers.h"
#include "activations.h"
#include "data.h"
#include <eigen3/Eigen/Dense>
#include <catch2/catch.hpp>


TEST_CASE("XOR dataset", "[xor]"){

    std::vector<Eigen::MatrixXf> dataset = generateXORdataset();
    std::vector<Eigen::MatrixXf> labels = generateXORlabels();
    
    neuralNets::Sequential model;

    model.add(new neuralNets::Layers::Linear(2,3));
    model.add(new neuralNets::Activations::ReLU());
    model.add(new neuralNets::Layers::Linear(3,1));
    model.add(new neuralNets::Activations::Sigmoid());
 
    neuralNets::Losses::MSE mse;
    model.set_loss(&mse);
    float acc, loss;

    model.fit(dataset, labels, 1000, acc, loss, 0.1, false);
    std::vector<Eigen::MatrixXf> pred;
    model.predict(pred, dataset);

    REQUIRE(pred[0].coeff(0,0) < 0.1);
    REQUIRE(pred[1].coeff(0,0) > 0.9);
    REQUIRE(pred[2].coeff(0,0) > 0.9);
    REQUIRE(pred[3].coeff(0,0) < 0.1);
}
