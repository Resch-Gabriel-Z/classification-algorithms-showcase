import numpy as np


class SVM:
    def __init__(
        self, C=1.0, learning_rate=0.01, max_iters=1000, tol=1e-4, lambda_param=0.01
    ):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.lambda_param = lambda_param

        self.weights = None
        self.bias = None

    def compute_gradient_svm(self, X, y):
        loss_weight_c = 0
        loss_bias_c = 0
        for idx, x_i in enumerate(X):
            if (y[idx] * (np.dot(x_i, self.weights) + self.bias)) >= 1:
                loss_weight_c += 0
                loss_bias_c += 0
            else:
                loss_weight_c += self.C * (y[idx] * x_i)
                loss_bias_c += self.C * y[idx]

        loss_weight = self.weights - loss_weight_c
        loss_bias = -loss_bias_c
        return loss_weight, loss_bias

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Data preprocessing
        y_ = np.where(y <= 0, -1, 1)

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iters):
            loss_weight, loss_bias = self.compute_gradient_svm(X, y_)
            self.weights -= self.learning_rate * (
                loss_weight + self.lambda_param * self.weights
            )
            self.bias -= self.learning_rate * loss_bias

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
