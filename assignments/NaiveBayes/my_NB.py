import pandas as pd
import numpy as np
from collections import Counter

class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha
        #The probability of y dictionary
        self.y_dict = {} 
        #The probability of xi and yi dictionary
        self.xi_yj_dict = {}


    def fit(self, X, y):
        # Calculate P(yj)
        self.classes_ = np.unique(y)
        total_samples = len(y)
        
        for cls in self.classes_:
            count_cls = np.sum(y == cls)
            self.y_dict[cls] = (count_cls + self.alpha) / (total_samples + len(self.classes_) * self.alpha)
        
        # Calculate P(xi|yj)
        for col in X.columns:
            self.xi_yj_dict[col] = {}
            unique_values = X[col].unique()
            num_unique_values = len(unique_values)
            
            for cls in self.classes_:
                self.xi_yj_dict[col][cls] = {}
                count_cls = np.sum(y == cls)
                
                for val in unique_values:
                    count_xi_yj = np.sum((y == cls) & (X[col] == val))
                    self.xi_yj_dict[col][cls][val] = (count_xi_yj + self.alpha) / (count_cls + num_unique_values * self.alpha)
        return 

    def predict(self, X):
        # Initialize predictions list
        predictions = []

        for _, row in X.iterrows():
            class_scores = {}

            for cls in self.classes_:
                class_score = np.log(self.y_dict[cls])  # Start with the log probability of the class

                for feature, value in row.items():  # Use items() for Series
                    if value in self.xi_yj_dict[feature][cls]:
                        class_score += np.log(self.xi_yj_dict[feature][cls][value])
                    else:
                        # Handle Laplace smoothing when the feature value is not seen in training data
                        class_score += np.log(self.alpha / (len(self.classes_) + len(self.xi_yj_dict[feature][cls]) * self.alpha + 1))

                class_scores[cls] = class_score

            # Predict the class with the highest score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        return predictions



    def predict_proba(self, X):
        # Initialize the probabilities DataFrame
        probs = pd.DataFrame(columns=self.classes_)

        for _, row in X.iterrows():
            class_probs = {}

            for cls in self.classes_:
                class_prob = self.y_dict[cls]

                for feature, value in row.items():
                    if value in self.xi_yj_dict[feature][cls]:
                        class_prob *= self.xi_yj_dict[feature][cls][value]

                class_probs[cls] = class_prob

            total_prob = sum(class_probs.values())
            
            if total_prob != 0:
                class_probs_normalized = {cls: prob / total_prob for cls, prob in class_probs.items()}
            else:
                # Handle the case where total_prob is 0 (all probabilities are zero)
                class_probs_normalized = {cls: 0.0 for cls in self.classes_}

            probs.loc[_]=class_probs_normalized

        return probs