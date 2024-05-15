#define CATCH_CONFIG_MAIN
#include "sequential.h"
#include "layers.h"
#include "activations.h"
#include <catch2/catch_all.hpp>
#include <eigen3/Eigen/Dense>

using namespace neuralNets;


TEST_CASE("Correct layers counting and .add() initialization with default constructor", "[layers_num]"){

    SECTION(".add() method"){

        neuralNets::Sequential model;

        model.add(new neuralNets::Layers::Linear(2,3));
        model.add(new neuralNets::Activations::ReLU());
        model.add(new neuralNets::Layers::Linear(3,4));

        REQUIRE(model.count_layers() == 3);
    }

    SECTION("Inizialization list"){

        Losses::MSE mse;

        std::vector<Module*> layers;
        layers.emplace_back(new Layers::Linear(2,2));
        layers.emplace_back(new Activations::ReLU());
        layers.emplace_back(new Layers::Linear(2,2));

        Sequential model(layers, &mse);

        REQUIRE(model.count_layers() == 3);
    }
}


TEST_CASE("Forward pass", "[forward]"){

    Eigen::MatrixXf w1 = Eigen::MatrixXf::Identity(2,2);
    Eigen::MatrixXf w2 = Eigen::MatrixXf::Identity(2,2);
    Eigen::MatrixXf b1 = Eigen::MatrixXf::Zero(1,2);
    Eigen::MatrixXf b2 = Eigen::MatrixXf::Zero(1,2);

    Losses::MSE mse;

    std::vector<Module*> layers;
    layers.emplace_back(new Layers::Linear(2,2));
    layers.emplace_back(new Activations::ReLU());
    layers.emplace_back(new Layers::Linear(2,2));

    Sequential model(layers, &mse);

    model.set_W_b_layer(w1, b1, 0);
    model.set_W_b_layer(w2, b2, 2);


    SECTION("Identity weight matrix and all positive entries"){

        Eigen::MatrixXf xData1{
            {1,2},
        };

        Eigen::MatrixXf target1{
            {1,2},
        };

        model._forward(xData1);

        Eigen::MatrixXf out1 = model._get_output_from_layer(2);

        REQUIRE(out1.isApprox(target1));
    }

    SECTION("Mixed sign entries"){

        Eigen::MatrixXf xData2{
            {-1,2},
        };

        Eigen::MatrixXf target2{
            {0,2},
        };

        model._forward(xData2);

        Eigen::MatrixXf out2 = model._get_output_from_layer(2);

        REQUIRE(out2.isApprox(target2));
    }
}

