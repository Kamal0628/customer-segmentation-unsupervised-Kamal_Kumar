import sklearn.metrics

def evaluate_clustering(data, labels):
    results = {}

    if len(set(labels)) > 1:
        results['silhouette'] = sklearn.metrics.silhouette_score(data, labels)
        results['davies_bouldin'] = sklearn.metrics.davies_bouldin_score(data, labels)
    else:
        results['silhouette'] = -1
        results['davies_bouldin'] = -1

    return results