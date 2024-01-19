#ifndef LAYERS_H
#define LAYERS_H

#include <eigen3/Eigen/Dense>

namespace neuralNets{
namespace Layers{

class Layers{

private:

int in_nodes, out_nodes;
Eigen::MatrixXf weigths;
Eigen::VectorXf bias;
    
public:

Layers() = delete;
Layers(int in_nodes, int out_nodes, std::string_view init="random", bool bias=true);
   
}; // class Layers
}; // namespace Layers
}; // namespace neuralNets
#endif