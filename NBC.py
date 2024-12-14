import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict

class NaiveBayesDiscrete(BaseEstimator, ClassifierMixin):
    def __init__(self, laplace_smoothing=True, use_logarithms=False):
        self.laplace_smoothing = laplace_smoothing
        self.use_logarithms = use_logarithms
        self.class_probs = None
        self.feature_probs = None
        self.feature_values = None

    def fit(self, X, y):
        y = y.ravel()
        self.classes = np.unique(y)
        if self.use_logarithms:
            self.class_probs = {cls: np.log(np.sum(y == cls) / len(y)) for cls in self.classes}
        else:
            self.class_probs = {cls: np.sum(y == cls) / len(y) for cls in self.classes}
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.feature_values = [np.unique(X[:, i]) for i in range(X.shape[1])]

        for cls in self.classes:
            X_cls = X[y == cls]
            for i in range(X.shape[1]):
                unique_vals = self.feature_values[i]
                for val in unique_vals:
                    count = np.sum(X_cls[:, i] == val)
                    if self.laplace_smoothing:
                        prob = (count + 1) / (len(X_cls) + len(unique_vals))
                    else:
                        prob = count / len(X_cls)
                    if self.use_logarithms:
                        self.feature_probs[i][cls][val] = np.log(prob) if prob > 0 else -np.inf
                    else:
                        self.feature_probs[i][cls][val] = prob

        return self

    def predict(self, X):
        predictions = [self._predict_instance(x) for x in X]
        return np.array(predictions)

    def predict_proba(self, X):
        return np.array([self._predict_proba_instance(x) for x in X])

    def _predict_instance(self, x):
        class_probs = {cls: self.class_probs[cls] for cls in self.classes}
        for i, val in enumerate(x):
            for cls in self.classes:
                if self.use_logarithms:
                    class_probs[cls] += self.feature_probs[i][cls].get(val, -np.inf)
                else:
                    class_probs[cls] *= self.feature_probs[i][cls].get(val, 1e-6)
        return max(class_probs, key=class_probs.get)

    def _predict_proba_instance(self, x):
        class_probs = {cls: self.class_probs[cls] for cls in self.classes}
        for i, val in enumerate(x):
            for cls in self.classes:
                if self.use_logarithms:
                    class_probs[cls] += self.feature_probs[i][cls].get(val, -np.inf)
                else:
                    class_probs[cls] *= self.feature_probs[i][cls].get(val, 1e-6)
        if self.use_logarithms:
            log_sum_exp = np.log(np.sum(np.exp(list(class_probs.values()))))
            probabilities = {cls: np.exp(log_prob - log_sum_exp) for cls, log_prob in class_probs.items()}
        else:
            total = sum(class_probs.values())
            probabilities = {cls: prob / total for cls, prob in class_probs.items()}
        return probabilities

def discretize(data, bins):
    return pd.qcut(data, q=bins, labels=False, duplicates="drop").astype(int)

# Load dataset from file
file_path = "C:/ZUT/spambase.data"   # Path to the dataset file
column_names = [f"feature_{i}" for i in range(1, 58)] + ["class"]
data = pd.read_csv(file_path, header=None, names=column_names)

# Split features and target
X = data.drop("class", axis=1).values
y = data["class"].values

# Normalize data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Discrete data experiments
bins_list = [20]  # Experiment with different bin sizes
laplace_smoothing_values = [True, False]

for bins in bins_list:
    for laplace_smoothing in laplace_smoothing_values:
        X_discretized = np.apply_along_axis(discretize, axis=0, arr=X_normalized, bins=bins)
        X_train, X_test, y_train, y_test = train_test_split(X_discretized, y, test_size=0.3, random_state=42)

        # Add rare values to test set to test Laplace smoothing
        X_test[0, 0] = np.max(X_train[:, 0]) + 1
        X_test[1, 1] = np.max(X_train[:, 1]) + 1

        nb_discrete = NaiveBayesDiscrete(laplace_smoothing=laplace_smoothing, use_logarithms=False)
        nb_discrete.fit(X_train, y_train)

        train_predictions = nb_discrete.predict(X_train)
        test_predictions = nb_discrete.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_predictions) * 100
        test_accuracy = accuracy_score(y_test, test_predictions) * 100

        print(f"\nWyniki dla Naive Bayes z dyskretyzacją (Liczba przedziałów = {bins}, Wygładzanie Laplace'a = {laplace_smoothing}):")
        print(f"Dokładność na zbiorze uczącym: {train_accuracy:.2f}%")
        print(f"Dokładność na zbiorze testowym: {test_accuracy:.2f}%")

# Numerical stability experiment with duplicated features
X_dangerous = np.hstack([X_normalized] * 10)
X_discretized_dangerous = np.apply_along_axis(discretize, axis=0, arr=X_dangerous, bins=5)
X_train, X_test, y_train, y_test = train_test_split(X_discretized_dangerous, y, test_size=0.3, random_state=42)

nb_safe = NaiveBayesDiscrete(laplace_smoothing=True, use_logarithms=True)
nb_safe.fit(X_train, y_train)

test_predictions_safe = nb_safe.predict(X_test)
test_accuracy_safe = accuracy_score(y_test, test_predictions_safe) * 100

print("\nEksperyment z bezpieczeństwem numerycznym z użyciem logarytmów:")
print(f"Dokładność na zbiorze testowym z bezpiecznym modelem logarytmicznym: {test_accuracy_safe:.2f}%")
