import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


# Add BaseEstimator and ClassifierMixin to your class for compatibility with sklearn #
class SVM(BaseEstimator, ClassifierMixin):
    """Support Vector Machine classifier."""

    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-4, lambda_param=0.01):
        """SVM classifier.

        Args:
            learning_rate (float, optional): learning rate. Defaults to 0.01.
            max_iters (int, optional): maximum number of iterations. Defaults to 1000.
            tol (_type_, optional): tolerance. Defaults to 1e-4.
            lambda_param (float, optional): lambda parameter. Defaults to 0.01.
        """
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.lambda_param = lambda_param

        self.weights = None
        self.bias = None

    def compute_gradient_svm(self, X, y):
        """Compute gradient for SVM.

        Args:
            X (array): data
            y (array): labels

        Returns:
            loss_weight, loss_bias: gradient for weights and bias
        """
        n_samples, n_features = X.shape
        loss_weight = np.zeros(n_features)
        loss_bias = 0

        for i in range(n_samples):
            if y[i] * (np.dot(X[i], self.weights) + self.bias) < 1:
                loss_weight += -y[i] * X[i]
                loss_bias -= y[i]

        return loss_weight, loss_bias

    def fit(self, X, y):
        """Fit SVM classifier.

        Args:
            X (array): data
            y (array): labels
        """

        n_features = X.shape[1]
        loss_weight_dif = float("inf")

        # Data preprocessing
        # Add self.classes_ attribute for compatibility with sklearn #
        y_ = np.where(y <= 0, -1, 1)
        self.classes_ = np.unique(y_)

        # init weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.max_iters):
            loss_weight, loss_bias = self.compute_gradient_svm(X, y_)

            self.weights += self.learning_rate * (
                loss_weight - self.lambda_param * self.weights
            )

            self.bias -= self.learning_rate * loss_bias

            if np.linalg.norm(loss_weight_dif - loss_weight) < self.tol:
                break

            loss_weight_dif = loss_weight

        # return self for compatibility with sklearn #
        return self

    def predict(self, X):
        """Predict labels.

        Args:
            X (array): data to predict

        Returns:
            array: predicted labels
        """
        prediction = np.dot(X, self.weights) + self.bias
        prediction = np.where(prediction <= 0, 1, 0)
        return prediction
