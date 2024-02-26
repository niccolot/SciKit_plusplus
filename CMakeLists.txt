cmake_minimum_required(VERSION 3.8)

project(SciKit_plusplus
    VERSION 1.0
    DESCRIPTION "C++ machine learning library"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Catch2 3 REQUIRED)

add_subdirectory(src/svm)
add_subdirectory(src/neural_nets)
add_subdirectory(src/k_means_clustering)
add_subdirectory(test/neural_nets)

include_directories(include)
