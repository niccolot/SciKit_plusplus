add_executable(UnitTests_layers unitTests_layers.cpp)
add_executable(UnitTests_losses unitTests_losses.cpp)
add_executable(UnitTests_sequential unitTests_sequential.cpp)
add_executable(IntegrationTest_sequential integrationTest_sequential.cpp)

target_link_libraries(UnitTests_layers NeuralNets Catch2::Catch2) 
target_link_libraries(UnitTests_losses NeuralNets Catch2::Catch2) 
target_link_libraries(UnitTests_sequential NeuralNets Catch2::Catch2) 
target_link_libraries(IntegrationTest_sequential NeuralNets)

target_include_directories(NeuralNets PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/
)

target_compile_features(UnitTests_layers PUBLIC cxx_std_17)
target_compile_options(UnitTests_layers PUBLIC -Wall -Wextra -Wpedantic)
target_compile_features(UnitTests_losses PUBLIC cxx_std_17)
target_compile_options(UnitTests_losses PUBLIC -Wall -Wextra -Wpedantic)
target_compile_features(UnitTests_sequential PUBLIC cxx_std_17)
target_compile_options(UnitTests_sequential PUBLIC -Wall -Wextra -Wpedantic)
target_compile_features(IntegrationTest_sequential PUBLIC cxx_std_17)
target_compile_options(IntegrationTest_sequential PUBLIC -Wall -Wextra -Wpedantic)