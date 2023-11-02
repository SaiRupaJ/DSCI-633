import pandas as pd
import numpy as np

class my_Logistic:

    def __init__(self, learning_rate=0.1, batch_size=10, max_iter=100, shuffle=False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.shuffle = shuffle

    def fit(self, X, y):
        data = X.to_numpy()
        d = data.shape[1]
        self.w = np.array([0.0] * d)
        self.w0 = 0.0
        n = len(y)
        num_batches = n // self.batch_size 
        if self.shuffle:
            indices = np.random.permutation(n)
            data = data[indices]
            y = y[indices]

        for _ in range(self.max_iter):
            for i in range(num_batches):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size

                X_train = data[start:end]
                y_train = y[start:end]

                self.w, self.w0 = self.sgd(X_train, y_train, self.w, self.w0)

    def generate_batches(self, n):
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)
        batches = []
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_indices = indices[start:end]
            batches.append(batch_indices)
        return batches


    def sgd(self, X, y, w, w0):
        y_pred = 1.0 / (1 + np.exp(-(np.dot(X, w) + w0)))
        error = y - y_pred
        w_grad = -np.dot(X.T, error) / len(y)
        w0_grad = -np.mean(error)
        w -= self.learning_rate * w_grad
        w0 -= self.learning_rate * w0_grad
        return w, w0

    def predict_proba(self, X):
        data = X.to_numpy()
        wx = np.dot(self.w, data.transpose()) + self.w0
        fx = 1.0 / (1 + np.exp(-wx))
        return fx

    def predict(self, X):
        probs = self.predict_proba(X)
        predictions = [1 if prob >= 0.5 else 0 for prob in probs]
        return predictions
