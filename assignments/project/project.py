import imp
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import sys

from sklearn.linear_model import SGDClassifier

sys.path.insert(0,'../..')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm

class my_model():
    def fit(self, data_X, data_y):
        # do not exceed 29 mins

        # Preprocessing text data
        data_X['final'] = data_X['description'] + data_X['requirements'] + data_X['title']
        data_X['final'] = data_X['final'].str.replace('[^a-zA-Z]', ' ')

        # Vectorizing
        self.tfidf = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
        final_vectorized = self.tfidf.fit_transform(data_X['final'].astype('U'))

        params = {
            "loss": ["hinge", "log", "squared_hinge", "modified_huber"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "penalty": ["l2", "l1"],
        }

        model = SGDClassifier(max_iter=10000, class_weight="balanced")
        self.clf = GridSearchCV(model, param_grid=params)
        self.clf.fit(final_vectorized, data_y)

        return

    def predict(self, test_X):
        # Remember to apply the same preprocessing in fit() on test data before making predictions
        test_X['final'] = test_X['description'] + test_X['requirements'] + test_X['title']
        test_X['final'] = test_X['final'].str.replace('[^a-zA-Z]', ' ')
        final_vectorized = self.tfidf.transform(test_X['final'].astype('U'))
        predictions = self.clf.predict(final_vectorized)
        return predictions
