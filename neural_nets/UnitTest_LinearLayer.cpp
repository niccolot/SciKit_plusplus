#include "UnitTest_LinearLayer.h"
#include "layers.h"
#include <iostream>
#include <eigen3/Eigen/Dense>


namespace neuralNets{
namespace Layers{
namespace UnitTests{

void UnitTest_LinearLayer::shapeTest(){

    std::cout<<"Shape test: ";

    int inDim1 = 3;
    int outDim1 = 4;
    int inDim2 = 1;
    int outDim2 = 5;
    int inDim3 = 4;
    int outDim3 = 1;

    Linear linear1(inDim1, outDim1);
    Linear linear2(inDim2, outDim2);
    Linear linear3(inDim3, outDim3);

    Eigen::MatrixXf x1 = Eigen::MatrixXf::Ones(1,inDim1);
    Eigen::MatrixXf x2 = Eigen::MatrixXf::Ones(1,inDim2);
    Eigen::MatrixXf x3 = Eigen::MatrixXf::Ones(1,inDim3);

    Eigen::MatrixXf out;

    linear1.forward(out,x1);

    if(out.rows() != x1.rows()){
        std::cout<<"Output rows number error:\n"
                <<"Expected: "<<x1.rows()<<"\n"
                <<"Got: "<<out.rows()<<"\n";
        return;
    }

    if(out.cols() != outDim1){
        std::cout<<"Output columns number error:\n"
                <<"Expected: "<<outDim1<<"\n"
                <<"Got: "<<out.cols()<<"\n";
        return;
    }

    linear2.forward(out,x2);

    if(out.rows() != x2.rows()){
        std::cout<<"Output rows number error:\n"
                <<"Expected: "<<x2.rows()<<"\n"
                <<"Got: "<<out.rows()<<"\n";
        return;
    }

    if(out.cols() != outDim2){
        std::cout<<"Output columns number error:\n"
                <<"Expected: "<<outDim2<<"\n"
                <<"Got: "<<out.cols()<<"\n";
        return;
    }

    linear3.forward(out,x3);

    if(out.rows() != x3.rows()){
        std::cout<<"Output rows number error:\n"
                <<"Expected: "<<x3.rows()<<"\n"
                <<"Got: "<<out.rows()<<"\n";
        return;
    }

    if(out.cols() != outDim3){
        std::cout<<"Output columns number error:\n"
                <<"Expected: "<<outDim3<<"\n"
                <<"Got: "<<out.cols()<<"\n";
        return;
    }

    std::cout<<"passed\n";

}


void UnitTest_LinearLayer::forwardTest(){

    std::cout<<"Forward test: ";

    int inDim = 3;
    int outDim = 4;

    Linear linear(inDim,outDim);

    Eigen::MatrixXf w = Eigen::MatrixXf::Identity(inDim,outDim);
    Eigen::MatrixXf b = Eigen::MatrixXf::Zero(1,outDim);

    linear._set_weights_bias(w,b);

    Eigen::MatrixXf x{{1.f, 2.f, 3.f}};
    Eigen::MatrixXf target{{1.f, 2.f, 3.f, 0.f}};
    
    Eigen::MatrixXf out;

    linear.forward(out,x);

    if(!(out.isApprox(target))){
        std::cout<<"Result error:\n"
                <<"Expected: "<<x<<"\n"
                <<"GOt: "<<out<<"\n";
        return;
    }

    std::cout<<"passed\n";
}

}; // namespace UnitTests
}; // namespace Layers
}; // namespace neuralNets