import pandas as pd
import numpy as np

class my_KMeans:

    def __init__(self, n_clusters=8, init="k-means++", n_init=10, max_iter=300, tol=1e-4):
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = n_init
        self.max_iter = int(max_iter)
        self.tol = tol

        self.classes_ = range(n_clusters)
        self.cluster_centers_ = None
        self.inertia_ = None

    def dist(self, a, b):
        return np.sum((np.array(a) - np.array(b)) ** 2) ** 0.5

    def initiate(self, X):
        if self.init == "random":
            cluster_centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.init == "k-means++":
            cluster_centers = [X[np.random.choice(X.shape[0])]] 
            for _ in range(1, self.n_clusters):
                distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in cluster_centers]) for x in X])
                cluster_centers.append(X[np.random.choice(X.shape[0], p=distances / distances.sum())])
        else:
            raise Exception("Unknown value of self.init.")
        return cluster_centers

    def fit_once(self, X):
        cluster_centers = self.initiate(X)
        last_inertia = None
        for i in range(self.max_iter):
            clusters = [[] for _ in range(self.n_clusters)]
            inertia = 0
            for x in X:
                dists = [self.dist(x, center) for center in cluster_centers]
                inertia += min(dists) ** 2
                cluster_id = np.argmin(dists)
                clusters[cluster_id].append(x)

            if last_inertia is not None and last_inertia - inertia < self.tol:
                break
            for j in range(self.n_clusters):
                if len(clusters[j]) > 0:
                    cluster_centers[j] = np.mean(clusters[j], axis=0)

            last_inertia = inertia

        return cluster_centers, inertia

    def fit(self, X):
        X_feature = X.to_numpy()
        best_inertia = None
        best_cluster_centers = None

        for _ in range(self.n_init):
            cluster_centers, inertia = self.fit_once(X_feature)
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_cluster_centers = cluster_centers

        self.inertia_ = best_inertia
        self.cluster_centers_ = best_cluster_centers


    def transform(self, X):
        X_array = X.to_numpy()
        cluster_centers_array = np.array(self.cluster_centers_)
        dists = np.linalg.norm(X_array[:, np.newaxis] - cluster_centers_array, axis=2)
    
        return dists.tolist()


    def predict(self, X):
        predictions = [np.argmin(dist) for dist in self.transform(X)]
        return predictions

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



