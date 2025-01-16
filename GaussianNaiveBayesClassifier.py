import numpy as np
import pandas as pd


class GNBC:
    def __init__(self):
        self.class_priors = {}
        self.class_means = {}
        self.class_vars = {}
        self.classes = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.classes = np.unique(y)

        # Calculate prior probabilities
        n_samples = len(y)
        for c in self.classes:
            class_samples = X[y == c]
            self.class_priors[c] = len(class_samples) / n_samples
            self.class_means[c] = np.mean(class_samples, axis=0)
            self.class_vars[c] = np.var(class_samples, axis=0) + 1e-9  # Add small value to avoid division by zero

        return self

    def _calculate_gaussian_probability(self, x, mean, var):
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return exponent / np.sqrt(2 * np.pi * var)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        proba = []
        for x in X:
            sample_probs = {}
            for c in self.classes:
                prior = np.log(self.class_priors[c])
                likelihood = np.sum(np.log(self._calculate_gaussian_probability(
                    x, self.class_means[c], self.class_vars[c])))
                sample_probs[c] = prior + likelihood

            # Convert log probabilities to normal probabilities
            max_log_prob = max(sample_probs.values())
            probs_exp = {k: np.exp(v - max_log_prob) for k, v in sample_probs.items()}
            total = sum(probs_exp.values())
            proba.append({k: v / total for k, v in probs_exp.items()})

        return proba[0] if len(proba) == 1 else proba

    def predict(self, X):
        proba = self.predict_proba(X)
        if isinstance(proba, dict):
            return max(proba, key=proba.get)
        return [max(p, key=p.get) for p in proba]