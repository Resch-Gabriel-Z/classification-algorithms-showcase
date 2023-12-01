from decisiontree import DecisionTree
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class Randomforest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, max_features=None):
        self.n_estimators = n_estimators  # number of trees
        self.max_depth = max_depth  # max depth of each tree
        self.max_features = (
            max_features  # max number of features to consider for each split
        )
        self.trees = [
            DecisionTree(max_depth=self.max_depth) for _ in range(self.n_estimators)
        ]
        self.feature_per_tree = []  # store the features used for each tree

    def fit(self, X, y):
        for tree in self.trees:
            # choose subset of features
            features = np.random.choice(X.shape[1], self.max_features, replace=True)
            self.feature_per_tree.append(features)
            X_tree = X[:, features]
            tree.fit(X_tree, y)
        return self

    def predict(self, X):
        predictions = []
        for tree in self.trees:
            features = self.feature_per_tree[self.trees.index(tree)]
            X_tree = X[:, features]
            predictions.append(tree.predict(X_tree))
        predictions = np.array(predictions)
        # return the most common class for each sample
        # this works by applying the function to each column
        # we use lambda to address each column
        # np.argmax(np.bincount(x)) returns the most common element in x
        return np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions
        )
