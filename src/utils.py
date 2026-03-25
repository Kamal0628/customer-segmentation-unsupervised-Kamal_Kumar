import matplotlib.pyplot as plt

def plot_clusters(data, labels, title="Clusters"):
    plt.figure(figsize=(8,6))
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar()
    plt.show()