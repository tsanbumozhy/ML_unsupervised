import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

file_path = "C:/Users/Anbumozhy/Desktop/data.xlsx"
data = pd.read_excel(file_path)

data.dropna(inplace=True)# drops samples with missing values

selected_columns = data.columns.tolist()

selected_columns.remove('S.NO')

X = data[selected_columns]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find the optimal number of clusters for KMeans using Silhouette Score
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=0)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, kmeans_labels)
    silhouette_scores.append(silhouette_avg)

# Plot Silhouette Scores for KMeans
plt.plot(range(2, 11), silhouette_scores)
plt.title('KMeans: Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Choose the optimal number of clusters for KMeans based on the plot
optimal_kmeans_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 to get the optimal cluster count

# Perform clustering with KMeans using the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_kmeans_clusters, n_init=10, random_state=0)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Perform clustering with Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=optimal_kmeans_clusters)  # Use the same optimal clusters for Agglomerative
agg_labels = agg_clustering.fit_predict(X_scaled)

# Perform clustering with DBSCAN (no need for optimal clusters)
dbscan = DBSCAN(eps=2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Perform clustering with MeanShift
meanshift = MeanShift()
meanshift_labels = meanshift.fit_predict(X_scaled)

data['KMeans_Cluster'] = kmeans_labels
data['Agglomerative_Cluster'] = agg_labels
data['DBSCAN_Cluster'] = dbscan_labels
data['MeanShift_Cluster'] = meanshift_labels

# Calculate evaluation metrics for KMeans cluster
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_davies_bouldin = davies_bouldin_score(X_scaled, kmeans_labels)
kmeans_calinski_harabasz = calinski_harabasz_score(X_scaled, kmeans_labels)

# Print the evaluation metrics for KMeans
print("KMeans Silhouette Score:", kmeans_silhouette)
print("KMeans Davies-Bouldin Score:", kmeans_davies_bouldin)
print("KMeans Calinski-Harabasz Score:", kmeans_calinski_harabasz)

# Calculate evaluation metrics for Agglomerative Clustering
agg_silhouette = silhouette_score(X_scaled, agg_labels)
agg_davies_bouldin = davies_bouldin_score(X_scaled, agg_labels)
agg_calinski_harabasz = calinski_harabasz_score(X_scaled, agg_labels)

# Print the evaluation metrics for each algorithm
print("Agglomerative Clustering Silhouette Score:", agg_silhouette)
print("Agglomerative Clustering Davies-Bouldin Score:", agg_davies_bouldin)
print("Agglomerative Clustering Calinski-Harabasz Score:", agg_calinski_harabasz)

# Calculate evaluation metrics for DBSCAN if there are at least two unique labels
unique_labels = set(dbscan_labels)
if len(unique_labels) > 1:
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    dbscan_davies_bouldin = davies_bouldin_score(X_scaled, dbscan_labels)
    dbscan_calinski_harabasz = calinski_harabasz_score(X_scaled, dbscan_labels)
else:
    dbscan_silhouette = None
    dbscan_davies_bouldin = None
    dbscan_calinski_harabasz = None

# Print the evaluation metrics for DBSCAN
if dbscan_silhouette is not None:
    print("DBSCAN Silhouette Score:", dbscan_silhouette)
    print("DBSCAN Davies-Bouldin Score:", dbscan_davies_bouldin)
    print("DBSCAN Calinski-Harabasz Score:", dbscan_calinski_harabasz)
else:
    print("DBSCAN did not produce multiple clusters.")

# Calculate evaluation metrics for MeanShift if there are at least two unique labels
unique_labels_meanshift = set(meanshift_labels)
if len(unique_labels_meanshift) > 1:
    meanshift_silhouette = silhouette_score(X_scaled, meanshift_labels)
    meanshift_davies_bouldin = davies_bouldin_score(X_scaled, meanshift_labels)
    meanshift_calinski_harabasz = calinski_harabasz_score(X_scaled, meanshift_labels)
else:
    meanshift_silhouette = None
    meanshift_davies_bouldin = None
    meanshift_calinski_harabasz = None

# Print the evaluation metrics for MeanShift
if meanshift_silhouette is not None:
    print("MeanShift Silhouette Score:", meanshift_silhouette)
    print("MeanShift Davies-Bouldin Score:", meanshift_davies_bouldin)
    print("MeanShift Calinski-Harabasz Score:", meanshift_calinski_harabasz)
else:
    print("MeanShift produced only one cluster.")


# Visualize the clusters (you may need to adjust this based on the number of clusters)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels)
plt.title('KMeans Clustering')
plt.show()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agg_labels)
plt.title('Agglomerative Clustering')
plt.show()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels)
plt.title('DBSCAN Clustering')
plt.show()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=meanshift_labels)
plt.title('MeanShift Clustering')
plt.show()
