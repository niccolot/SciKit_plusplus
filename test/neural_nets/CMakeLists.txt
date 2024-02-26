add_executable(UnitTest main.cpp UnitTest_LinearLayer.cpp)
target_link_libraries(UnitTest NeuralNets) 
target_link_libraries(UnitTest Catch2::Catch2)

target_include_directories(NeuralNets PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/
)

target_compile_features(UnitTest PUBLIC cxx_std_17)
target_compile_options(UnitTest PUBLIC -Wall -Wextra -Wpedantic)