import sklearn.mixture

def run_gmm(data, k=4):
    model = sklearn.mixture.GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(data)
    return labels, model