#define CATCH_CONFIG_MAIN
#include "layers.h"
#include "activations.h"
#include <catch2/catch.hpp>
#include <eigen3/Eigen/Dense>


TEST_CASE("Linear layer shape test", "[linear_layer_shape_test]"){

    SECTION("Test1: inDim 3, outDim 4"){

        int inDim = 3;
        int outDim = 4;
        neuralNets::Layers::Linear linear(inDim, outDim);

        Eigen::MatrixXf x = Eigen::MatrixXf::Ones(1,inDim);

        Eigen::MatrixXf out;

        linear.forward(out,x);

        REQUIRE(out.rows() == x.rows());
        REQUIRE(out.cols() == outDim);
    }

    SECTION("Test2: inDim 1, outDim 5"){

        int inDim = 1;
        int outDim = 5;
        neuralNets::Layers::Linear linear(inDim, outDim);

        Eigen::MatrixXf x = Eigen::MatrixXf::Ones(1,inDim);

        Eigen::MatrixXf out;

        linear.forward(out,x);

        REQUIRE(out.rows() == x.rows());
        REQUIRE(out.cols() == outDim);
    }

    SECTION("Test3: inDim 4, outDim 1"){

        int inDim = 4;
        int outDim = 1;
        neuralNets::Layers::Linear linear(inDim, outDim);

        Eigen::MatrixXf x = Eigen::MatrixXf::Ones(1,inDim);

        Eigen::MatrixXf out;

        linear.forward(out,x);

        REQUIRE(out.rows() == x.rows());
        REQUIRE(out.cols() == outDim);
    }
}


TEST_CASE("Linear layer forward test", "[linear_layer_forward_test]"){

    SECTION("Test1: identity matrix as weight and null bias vector"){

        int inDim = 3;
        int outDim = 4;

        neuralNets::Layers::Linear linear(inDim,outDim);

        Eigen::MatrixXf w = Eigen::MatrixXf::Identity(inDim,outDim);
        Eigen::MatrixXf b = Eigen::MatrixXf::Zero(1,outDim);

        linear._set_weights_bias(w,b);

        Eigen::MatrixXf x{{1.f, 2.f, 3.f}};
        Eigen::MatrixXf target{{1.f, 2.f, 3.f, 0.f}};
        
        Eigen::MatrixXf out;

        linear.forward(out,x);

        REQUIRE(out.isApprox(target));
    }

    SECTION("Test2: non-trivial weights and bias"){

        int inDim = 2;
        int outDim = 4;

        neuralNets::Layers::Linear linear(inDim,outDim);

        Eigen::MatrixXf w{
            {0.5f, 0.1f, -0.5f, 0.1f},
            {0.09f, -0.5f, 0.1f, 0.09f},
        };
        
        Eigen::MatrixXf b{{0.2f, -1.f, 0.f, 0.5f}};

        linear._set_weights_bias(w,b);

        Eigen::MatrixXf x{{-9.f, -5.f}};

        Eigen::MatrixXf target{{-4.75f, 0.6f, 4.f, -0.85f},};
        
        Eigen::MatrixXf out;

        linear.forward(out,x);

        REQUIRE(out.isApprox(target));
    }

}


TEST_CASE("Linear layer backward test", "[linear_layer_backward_test]"){

    int inDim = 2;
    int outDim = 4;

    neuralNets::Layers::Linear linear(inDim,outDim);

    Eigen::MatrixXf w{
        {0.5f, 0.1f, -0.5f, 0.1f},
        {0.09f, -0.5f, 0.1f, 0.09f},
    };

    Eigen::MatrixXf b{{0.2f, -1.f, 0.f, 0.5f}};

    Eigen::MatrixXf x{{-9.f, -5.f}};

    linear._set_weights_bias(w,b);

    Eigen::MatrixXf out;
    Eigen::MatrixXf backwardInput{{0.f, -2.f, 1.f, 0.f}};

    linear.forward(out, x);
    linear.backward(out, backwardInput);

    SECTION("Backpropagation test"){
        
        Eigen::MatrixXf backwardTarget{{-0.7f, 1.1f}};
        
        // test correct error propagation
        REQUIRE(out.isApprox(backwardTarget));
    }

    SECTION("Parameters update test"){

        Eigen::MatrixXf updatedWeights = linear._get_weights();
        Eigen::MatrixXf updatedBias = linear._get_bias();

        Eigen::MatrixXf updatedWeightsTarget{
            {0.5f, -0.08f, -0.41f, 0.1f},
            {0.09f, -0.6f, 0.15f, 0.09f},
        };
        Eigen::MatrixXf updatedBiasTarget{
            {0.2f, -0.98f, -0.01f, 0.5f},
        };

        // test correct parameters update
        REQUIRE(updatedWeights.isApprox(updatedWeightsTarget));
        REQUIRE(updatedBias.isApprox(updatedBiasTarget));
    }
}






