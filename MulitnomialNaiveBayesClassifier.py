from collections import Counter, defaultdict
from math import log
import numpy as np

    
class MNBC:
    def __init__(self):
        self.feature_count = defaultdict(lambda: defaultdict(lambda: Counter()))
        self.class_count = Counter()
        self.priors = defaultdict(dict)

    def fit(self, X, Y):
        Y = np.array(Y)
        X = np.array(X) 
        n, n_features = X.shape
        for i in range(n):
            c = Y[i]
            self.class_count[c] +=1#++
            for j in range(n_features):
                feature = j
                self.feature_count[c][feature][X[i, j]] +=1#++

        self.class_count = dict(Counter(Y))
        for c in set(Y):
            self.priors[c] = self.class_count[c] / len(X)

    def predict(self, X):
        X = np.array(X) 
        predictions = []
        for i in range(len(X)):
            c_hat = self.predict_proba(X[i])
            predictions.append(c_hat)
        return predictions

    def predict_proba(self, x):
        probabilities = {k: log(v) for k, v in self.priors.items()}

        for c in self.class_count:
            for feature in range(len(x)):
                feature_val = x[feature]
                nominator = self.feature_count[c][feature][feature_val] + 1
                denominator = sum(self.feature_count[c][feature].values()) + len(self.feature_count[c][feature])
                probabilities[c] += log(nominator / denominator)

        return max(probabilities, key=probabilities.get)