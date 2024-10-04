import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class KNN:
  def __init__(self, k=3):
    self.k = k
    self.X_train = None
    self.y_train = None

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X):
    predictions = [self._predict(x) for x in X]
    return np.array(predictions)

  def _predict(self, x):
    distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
    k_indices = np.argsort(distances)[:self.k]
    k_nearest_labels = [self.y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# Load the Iris dataset
iris = load_iris()
print(iris)

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = [1, 2, 3, 4, 5]
for k in k_values:
  knn = KNN(k=k)
  knn.fit(X_train, y_train)
  predictions = knn.predict(X_test)

  # Count the number of correctly classified samples
  correct_predictions = np.sum(predictions == y_test)
  total_predictions = len(y_test)

  print(f"For k = {k}: {correct_predictions} out of {total_predictions} test samples were correctly classified.")