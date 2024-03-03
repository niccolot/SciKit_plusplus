add_executable(UnitTests_layers unitTests_layers.cpp)
add_executable(UnitTests_losses unitTests_losses.cpp)

target_link_libraries(UnitTests_layers NeuralNets) 
target_link_libraries(UnitTests_layers Catch2::Catch2)
target_link_libraries(UnitTests_losses NeuralNets) 
target_link_libraries(UnitTests_losses Catch2::Catch2)

target_include_directories(NeuralNets PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/
)

target_compile_features(UnitTests_layers PUBLIC cxx_std_17)
target_compile_options(UnitTests_layers PUBLIC -Wall -Wextra -Wpedantic)
target_compile_features(UnitTests_losses PUBLIC cxx_std_17)
target_compile_options(UnitTests_losses PUBLIC -Wall -Wextra -Wpedantic)