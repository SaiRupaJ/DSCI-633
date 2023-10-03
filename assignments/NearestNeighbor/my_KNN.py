from turtle import distance
import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p
    
    def _distance(self, x1, x2):
        if self.metric == "minkowski":
            return np.linalg.norm(np.array(x1) - np.array(x2), ord=self.p)
        elif self.metric == "manhattan":
            return np.sum(np.abs(np.array(x1) - np.array(x2)))
        elif self.metric == "euclidean":
            return np.linalg.norm(np.array(x1) - np.array(x2))
        elif self.metric == "cosine":
            return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))



    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        self.X_train=X
        self.y_train=y
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        

    def predict(self, X):
        predictions = []
        for x in X.values: 
            # Iterate through rows of X as numpy arrays
            K_distances = [self._distance(x, x_train) for x_train in self.X_train.values]
            I = np.argsort(K_distances)
            K_nearest_indices = I[:self.n_neighbors]
            K_nearest_labels = [self.y_train.iloc[idx] for idx in K_nearest_indices]
            most_common = Counter(K_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return predictions


    def predict_proba(self, X):
        probs = []
        for i in X.values:
            K_distances = [self._distance(i, i_train) for i_train in self.X_train.values]
            I = np.argsort(K_distances)
            K_nearest_indices = I[:self.n_neighbors]
            K_nearest_labels = [self.y_train.iloc[idx] for idx in K_nearest_indices]
            No_of_counts = Counter(K_nearest_labels)
            prob_dict = {cls: No_of_counts[cls] / self.n_neighbors for cls in self.classes_}
            probs.append(prob_dict)
        return pd.DataFrame(probs, columns=self.classes_)