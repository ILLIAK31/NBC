import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from collections import defaultdict

# Дискретизатор
def discretize(data, bins, min_val=None, max_val=None):
    """
    Дискретизує безперервні дані у рівні інтервали.
    """
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)
    
    bin_width = (max_val - min_val) / bins
    discretized_data = np.floor((data - min_val) / bin_width).astype(int)
    discretized_data[discretized_data == bins] = bins - 1
    
    return discretized_data

# Класифікатор Naive Bayes
class NaiveBayesDiscrete(BaseEstimator, ClassifierMixin):
    def __init__(self, laplace_smoothing=True):
        self.class_probs = None
        self.feature_probs = None
        self.laplace_smoothing = laplace_smoothing
    
    def fit(self, X, y):
        """
        Навчання класифікатора Naive Bayes.
        """
        self.classes = np.unique(y)
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
                        self.feature_probs[i][cls][val] = (count + 1) / (len(X_cls) + len(unique_vals))
                    else:
                        self.feature_probs[i][cls][val] = count / len(X_cls)
        return self
    
    def predict(self, X):
        """
        Передбачення класів для заданих даних.
        """
        predictions = [self._predict_instance(x) for x in X]
        return np.array(predictions)
    
    def _predict_instance(self, x):
        """
        Передбачення класу для одного зразка.
        """
        class_probs = {cls: self.class_probs[cls] for cls in self.classes}
        for i, val in enumerate(x):
            for cls in self.classes:
                class_probs[cls] *= self.feature_probs[i][cls].get(val, 1e-6)
        return max(class_probs, key=class_probs.get)

# Завантаження набору даних Wine
data = load_wine()
X = data.data
y = data.target

# Розділення даних на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Дискретизація даних
bins = 5  # Кількість інтервалів
X_train_discretized = np.apply_along_axis(discretize, 0, X_train, bins)
X_test_discretized = np.apply_along_axis(discretize, 0, X_test, bins)

# Навчання класифікатора
nb = NaiveBayesDiscrete(laplace_smoothing=True)
nb.fit(X_train_discretized, y_train)

# Передбачення
train_predictions = nb.predict(X_train_discretized)
test_predictions = nb.predict(X_test_discretized)

# Обчислення точності
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

# Виведення результатів
print(f"Train : {train_accuracy * 100:.4f}%")
print(f"Test : {test_accuracy * 100:.4f}%")
