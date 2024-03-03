#define CATCH_CONFIG_MAIN
#include "losses.h"
#include <catch2/catch.hpp>
#include <eigen3/Eigen/Dense>

TEST_CASE("MSE test", "[mse]"){

    SECTION("Forward test"){

        neuralNets::Losses::MSE mse;

        Eigen::MatrixXf y{
            {1.f, 0.f},
            {1.f, 0.f},
            {0.f, 1.f},
        };

        Eigen::MatrixXf yPred{
            {0.4f, 0.6f},
            {0.2f, 0.8f},
            {0.9f, 0.1f},
        };

        float target = 1.20667f;

        float out;
        mse.forward(out, yPred, y);

        REQUIRE_THAT(out, Catch::Matchers::WithinRel(target, 0.001f));
        REQUIRE(1==1);

    }
}