README

1) KMEANS:
   Running the code:
        python3 KMEANS.py <dataset> <k> <list of initial centroids, no spaces in between> <maxIterations(optional, default: 100)> <(optional) metricOrder, to calculate distance (Manhattan, Euclidiean(Default if nothing is specified), Minkowski, etc., eg:"2" for Euclidean)>
        example: python3 KMEANS.py new_dataset_1.txt 3 [3,5,9] 2 <-Last 2 is optional
        
2) Hierarchical Agglomerative Clustering
   Running the code:
        python3 HAC.py <dataset> <number of clusters(optional)>
   
3) DBSCAN:
   Running the code:
        python3 DBSCAN.py <dataset> <eps> <minPts>
        python3 DBSCAN.py cho.txt 1.03 4

4) MAP REDUCE KMEANS:
   Running the code:
        python3 mapreduce_controller.py <dataset> <k> <list of initial centroids, no spaces in between>  <maxIterations> <(optional) metricOrder, to calculate distance (Manhattan, Euclidiean(Default if nothing is specified), Minkowski, etc., eg:"2" for Euclidean)>
        example: python3 KMEANS.py new_dataset_1.txt 3 [3,5,9] 10 2 <-Last 2 is optional
