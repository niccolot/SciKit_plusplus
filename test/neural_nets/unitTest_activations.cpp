#define CATCH_CONFIG_MAIN
#include "activations.h"
#include <catch2/catch_all.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>

TEST_CASE("ReLU activation function", "[ReLU_layer]"){

    SECTION("Forward test"){

        neuralNets::Activations::ReLU relu;

        Eigen::MatrixXf x{
            {-9.f, -5.f, 0.f, 1.f, 2.f, 8.f},
            {9.f, -5.f, 0.f, 1.f, -2.f, 8.f},
            {-4.f, 5.f, 0.f, 1.f, 2.f, -8.f},
            {-2.f, -5.f, 0.f, 1.f, -2.f, 8.f},
        };

        Eigen::MatrixXf target{
            {0.f, 0.f, 0.f, 1.f, 2.f, 8.f},
            {9.f, 0.f, 0.f, 1.f, 0.f, 8.f},
            {0.f, 5.f, 0.f, 1.f, 2.f, 0.f},
            {0.f, 0.f, 0.f, 1.f, 0.f, 8.f},
        };

        Eigen::MatrixXf out;
        relu.forward(out, x);

        REQUIRE(out.isApprox(target));
    }

    SECTION("Backward test"){

        neuralNets::Activations::ReLU relu;

        Eigen::MatrixXf forwardX{
            {3.f, -5.f, 0.f, 1.f, 2.f, 7.f},
            {9.f, -5.f, 0.f, 1.f, -2.f, -8.f},
            {-4.f, 5.f, 8.f, -1.f, 2.f, -8.f},
            {-2.f, -5.f, 0.f, 4.f, -2.f, 8.f},
        };

        // backward input
        Eigen::MatrixXf backwardX{
            {7.f, 7.f, 0.f, 1.f, -4.f, 7.f},
            {-4.f, -9.f, 3.f, 1.f, -2.f, -8.f},
            {8.f, -4.f, -8.f, -4.f, 2.f, 2.f},
            {-2.f, -6.f, 0.f, 4.f, 2.f, -8.f},
        };
        Eigen::MatrixXf backwardTarget{
            {7.f, 0.f, 0.f, 1.f, -4.f, 7.f},
            {-4.f, 0.f, 3.f, 1.f, 0.f, 0.f},
            {0.f, -4.f, -8.f, 0.f, 2.f, 0.f},
            {0.f, 0.f, 0.f, 4.f, 0.f, -8.f},
        };

        Eigen::MatrixXf out;
        relu.forward(out, forwardX);
        relu.backward(out, backwardX);

        REQUIRE(out.isApprox(backwardTarget));
    }
}


TEST_CASE("Sigmoid activation function", "[Sigmoid_layer]"){

    SECTION("Forward test"){

        neuralNets::Activations::Sigmoid s;

        Eigen::MatrixXf x{
            {-0.458f, -5.f},
            {9.f, 0.5f},
        };

        Eigen::MatrixXf target{
            {0.38746f, 0.00669f},
            {0.999876f, 0.622459f},
        };

        Eigen::MatrixXf out;
        s.forward(out, x);

        REQUIRE(out.isApprox(target));
    }

    SECTION("Backward test"){

        neuralNets::Activations::Sigmoid s;

        Eigen::MatrixXf forwardX{
            {-0.458f, -5.f},
            {9.f, 0.5f},
        };

        // backward input
        Eigen::MatrixXf backwardX{
            {0.1f, 10.f},
            {-10.f, -0.1f},
        };
        Eigen::MatrixXf backwardTarget{
            {0.0237335f, 0.0664806f},
            {-0.00123366f, -0.0235004f},
        };

        Eigen::MatrixXf out;
        s.forward(out, forwardX);
        s.backward(out, backwardX);

        REQUIRE(out.isApprox(backwardTarget));
    }
}
