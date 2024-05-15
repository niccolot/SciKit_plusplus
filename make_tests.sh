#! /bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake ..
make
cd test/neural_nets

echo ""

echo "Activation functions unit tests:"
./UnitTests_activations

echo "Layers unit tests:"
./UnitTests_layers

echo "Loss functions unit tests:"
./UnitTests_losses

echo "Sequential model unit tests:"
./UnitTests_sequential

echo "Sequential model integration tests"
./IntegrationTest_sequential