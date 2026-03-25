import sklearn.preprocessing
from sklearn.decomposition import PCA

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import create_rfm
from src.clustering.kmeans import run_kmeans
from src.clustering.hierarchical import run_hierarchical
from src.clustering.dbscan import run_dbscan
from src.clustering.gmm import run_gmm
from src.evaluation import evaluate_clustering
from src.utils import plot_clusters

# Step 1: Load Data
df = load_data("data/raw/online_retail.csv")

# Step 2: Clean Data
df = clean_data(df)

# Step 3: Feature Engineering
rfm = create_rfm(df)

# Step 4: Scaling
scaler = sklearn.preprocessing.StandardScaler()
rfm_scaled = scaler.fit_transform(rfm.drop(columns=['CustomerID']))

# Step 5: PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(rfm_scaled)

# Step 6: Run Models
print("\n--- KMEANS ---")
k_labels, _ = run_kmeans(rfm_scaled, 4)
print(evaluate_clustering(rfm_scaled, k_labels))

print("\n--- HIERARCHICAL ---")
h_labels = run_hierarchical(rfm_scaled, 4)
print(evaluate_clustering(rfm_scaled, h_labels))

print("\n--- DBSCAN ---")
d_labels = run_dbscan(rfm_scaled)
print(evaluate_clustering(rfm_scaled, d_labels))

print("\n--- GMM ---")
g_labels, _ = run_gmm(rfm_scaled, 4)
print(evaluate_clustering(rfm_scaled, g_labels))

# Step 7: Visualization
plot_clusters(pca_data, k_labels, "KMeans Clusters")
plot_clusters(pca_data, h_labels, "Hierarchical Clusters")
plot_clusters(pca_data, d_labels, "DBSCAN Clusters")
plot_clusters(pca_data, g_labels, "GMM Clusters")