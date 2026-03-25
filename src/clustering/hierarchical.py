import sklearn.cluster

def run_hierarchical(data, k=4):
    model = sklearn.cluster.AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(data)
    return labels