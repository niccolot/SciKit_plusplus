file(GLOB k_means_sources "*.cpp")

add_library(KMeansClustering ${k_means_sources})

target_include_directories(KMeansClustering PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/k_means_clustering
)