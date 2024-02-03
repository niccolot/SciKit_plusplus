#ifndef UNIT_TEST_LINEAR_LAYER_H
#define UNIT_TEST_LINEAR_LAYER_H

namespace neuralNets{
namespace Layers{
namespace UnitTests{

class UnitTest_LinearLayer{

public:

UnitTest_LinearLayer() = delete;
~UnitTest_LinearLayer() = delete;

static void shapeTest();
static void forwardTest();
static void backwardTest();

}; // class UnitTest_LinearLayer

}; // namespace UnitTests
}; // namespace Layers
}; // namespace neuralNets
#endif