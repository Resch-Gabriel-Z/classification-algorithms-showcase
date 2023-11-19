import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter


def euclidean_distance(x1, x2):
    """Compute euclidean distance between two vectors.

    Args:
        x1 (array): vector 1
        x2 (array): vector 2

    Returns:
        float: euclidean distance
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    """Compute manhattan distance between two vectors.

    Args:
        x1 (array): vector 1
        x2 (array): vector 2

    Returns:
        float: manhattan distance
    """
    return np.sum(np.abs(x1 - x2))


def cosine_distance(x1, x2):
    """Compute cosine distance between two vectors.

    Args:
        x1 (array): vector 1
        x2 (array): vector 2

    Returns:
        float: cosine distance
    """
    return 1 - np.dot(x1, x2) / (np.sqrt(np.dot(x1, x1)) * np.sqrt(np.dot(x2, x2)))


class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5, metric="euclidean"):
        """KNN classifier. init method.

        Args:
            k (int, optional): value for k, how many neighbours we want to take into account. Defaults to 5.
            metric (function, optional): the distance functions. Defaults to euclidean_distance.
        """
        self.k = k
        if metric == "euclidean":
            self.metric = euclidean_distance
        elif metric == "manhattan":
            self.metric = manhattan_distance
        elif metric == "cosine":
            self.metric = cosine_distance
        else:
            raise ValueError("Invalid metric name.")

    def fit(self, X, y):
        """Fit KNN classifier. KNN is simple, thus we only need to store the data.

        Args:
            X (array): data
            y (array): labels
        """
        self.X_train = X
        self.y_train = y

        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """Predict labels for new data. With help of the _predict method, we can predict the labels for many data points at once.

        Args:
            X (Array): data

        Returns:
            array: predicted labels
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """Predict label for a single data point.

        Args:
            x (point): data point

        Returns:
            int: predicted label
        """
        distances = [self.metric(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[: self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        most_common = Counter(k_neighbor_labels).most_common(1)[0][0]
        return most_common
