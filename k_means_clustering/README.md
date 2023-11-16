# k-means clustering

Implementation of a k means clustering model as a C++ class. 

The model works with a .csv file containing the datapoints along the rows and the features along the columns. It produces a similar .csv file with an extra column containing the cluster index for each datapoint.

The cluster initialization methods implemented are the random (i.e. Forgy's) and kmeans++ algorithm.

The actual clusterization algorithms implementes are the Lloyd's and Elkan's, the latter based on the algorithm presented in ['Using the tringle inequality to accelerate K-means'](https://www.researchgate.net/publication/2480121_Using_the_Triangle_Inequality_to_Accelerate_K-Means).
## Contents

* `point.h`: declaration of the `Point` class implementing the datapoint
* `point.cpp`: definition of the `Point` class
* `k_means.h`: declariation of the `K_means` class implementing the model
* `k_means.cpp`: definitions of the class's member functions
* `utils.h`: declarations of utility functions (file reading, initializations and trainig steps)
* `utils.cpp`: definitions of the hel functions
