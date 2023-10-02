Description:
	The program's main purpose is to explore and compare the performance of different clustering algorithms on a given dataset, providing insights into the optimal number of clusters and the quality of the clustering results. It serves as a valuable tool for unsupervised machine learning and data analysis tasks.

Inference:
1.	Data Preprocessing: The code reads data from an Excel file, handles missing values by dropping corresponding rows, and standardizes the data using StandardScaler.
2.	Optimal Cluster Number: It finds the optimal number of clusters for KMeans using Silhouette Scores and creates a plot to visualize these scores.
3.	KMeans Clustering: KMeans clustering is performed using the optimal number of clusters, and cluster labels are assigned to each data point.
4.	Agglomerative Clustering: Agglomerative Clustering is applied with the same optimal number of clusters as KMeans.
5.	DBSCAN and MeanShift: DBSCAN and MeanShift clustering algorithms are applied without specifying the optimal number of clusters.
6.	Evaluation Metrics: The code calculates and prints evaluation metrics (Silhouette Score, Davies-Bouldin Score, and Calinski-Harabasz Score) for each clustering algorithm, providing insights into the quality of the clusters generated.
