file(GLOB svm_sources "*.cpp")

add_library(svm ${svm_sources})

target_include_directories(svm PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/svm
)