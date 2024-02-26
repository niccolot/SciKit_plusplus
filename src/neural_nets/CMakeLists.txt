file(GLOB neural_nets_sources "*.cpp")

add_library(NeuralNets ${neural_nets_sources})

target_include_directories(NeuralNets PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/neural_nets
)
