import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class DecisionNode:
    def __init__(
        self, feature_index=None, threshold=None, value=None, left=None, right=None
    ):
        self.feature_index = feature_index  # feature index
        self.threshold = threshold  # threshold
        self.value = value  # value in the leaf node. The value is the class that the node predicts if the node is a leaf node.
        self.left = left  # left subtree
        self.right = right  # right subtree


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        # build the tree recursively, depth = 0 because we start from the root node
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """Build the tree recursively.

        Args:
            X (data): data, may be a subset of the original data
            y (labels): labels, may be a subset of the original labels
            depth (int): depth of the tree, starts from 0, increases by 1 each time the function is called recursively.

        Returns:
            DecisionNode: a decision node representing the split, or a leaf node if the split is not found.
        """
        unique_classes = len(np.unique(y))  # number of unique classes

        # Create variable to store the count of each class
        class_counts = np.bincount(y)

        # if the Data is empty then return None
        # because we cannot split the data if there is no data
        # that is okay, because in the parent node, the other child will have data.
        if len(class_counts) == 0:
            return None

        # Get the index of the class with the highest count
        # we calculate it no matter what, because even if we only have one class,
        # we still want to return a leaf node with that class
        majority_class = np.argmax(class_counts)

        if unique_classes == 1 or (
            self.max_depth is not None and depth == self.max_depth
        ):
            # If all labels are the same or max depth is reached, return a leaf node
            return DecisionNode(
                value=majority_class
            )  # return a leaf node with the majority class

        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            # If no split is found, return a leaf node with the majority class
            return DecisionNode(value=majority_class)

        # Split the data, by using the best feature and best threshold.
        # Luckily numpy supports boolean indexing, so we can use boolean indexing to split the data.

        # : -> all rows, best_feature -> column with index best_feature
        # <= best_threshold -> boolean array with True if the value is less than or equal to the threshold, False otherwise
        # That means that left_indices is an array with True if the value of the best feature is less than or equal to the threshold, False otherwise
        left_indices = X[:, best_feature] <= best_threshold

        # ~ -> invert the boolean array, so right_indices is an array with True if the value of the best feature is greater than the threshold,
        # False otherwise
        right_indices = ~left_indices

        # Recursively build the left and right subtrees. We need to increase the depth by 1 each time we call the function recursively.
        # X and y are subsets of the original data, specficially the data that is less than or equal to the threshold
        # and the data that is greater than the threshold.
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Return a decision node representing the split
        # feature_index is the index of the best feature
        # threshold is the best threshold of the best feature
        # left is the left subtree
        # right is the right subtree

        return DecisionNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def _find_best_split(self, X, y):
        """Find the best split.

        Args:
            X (Array): data
            y (Array): labels

        Returns:
            feature and threshold: best feature and best threshold
        """

        # Find the best split
        # num_samples is the number of samples, and used to calculate the weighted Gini impurity,
        # and to check if there are enough samples to split
        # num_features is the number of features, and used to loop through all features
        num_samples, num_features = X.shape

        if num_samples <= 1:
            # If there is only one sample, we cannot split the data, so we return None, None.
            # This will be handled in the _build_tree function.
            return None, None

        # Initialize variables
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        # Loop through all features and all possible thresholds to find the best split
        for feature_index in range(num_features):
            # Get all possible thresholds for the current feature,
            # since continuous features can have infinite thresholds, we need to approximate the thresholds
            # by using the unique values of the feature.
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                # Split the data, like above in the _build_tree function.
                left_indices = X[:, feature_index] <= threshold
                right_indices = ~left_indices

                # Calculate the Gini impurity for the left and right nodes
                left_gini = self._gini_impurity(y[left_indices])
                right_gini = self._gini_impurity(y[right_indices])

                # Calculate the weighted Gini impurity
                # The weighted Gini impurity is the sum of the Gini impurities of the left and right nodes,
                # weighted by the number of samples in each node.
                # in latex: \sum_{i=1}^{n} p_i * G_i
                # where n is the number of nodes, p_i is the number of samples in node i divided by the total number of samples,
                # and G_i is the Gini impurity of node i.
                weighted_gini = (len(y[left_indices]) / num_samples) * left_gini + (
                    len(y[right_indices]) / num_samples
                ) * right_gini

                # If the weighted Gini impurity is lower than the current best, update the variables
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_index
                    best_threshold = threshold

        # after the loop, return the best feature and best threshold
        return best_feature, best_threshold

    def _gini_impurity(self, y):
        """Calculate the Gini impurity.

        Args:
            y (array): labels

        Returns:
            int: Gini impurity
        """
        # Calculate the Gini impurity

        # classes is an array of all unique classes, and counts is an array of the count of each class.
        classes, counts = np.unique(y, return_counts=True)

        # print(f'Y: {y}\nclasses: {classes}\ncounts: {counts}\n')

        # Calculate the probabilities. This can look like this: [0.5, 0.3, 0.2]
        probabilities = counts / len(y)
        # Calculate the Gini impurity. This can look like this: [0.5, 0.3, 0.2] -> 1 - (0.5^2 + 0.3^2 + 0.2^2)
        # = 1 - (0.25 + 0.09 + 0.04)
        # = 1 - 0.38
        # = 0.62
        gini = 1 - np.sum(probabilities**2)
        return gini

    def predict(self, X):
        """Predict the class of each sample in X.

        Args:
            X (array): data

        Returns:
            array: predicted classes
        """
        # Predict the class of each sample in X
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        """Predict the class of a sample recursively.

        Args:
            x (sample): sample
            node (DecisionNode): node

        Returns:
            node value: predicted class, in form of the value of the node.
        """

        # If the node is a leaf node, return the value of the node
        # the node.value of course contains the class that the node predicts
        if node.value is not None:
            return node.value

        # If the node only has one child, go to that child
        if node.left is None and node.right is not None:
            return self._predict_tree(x, node.right)
        if node.right is None and node.left is not None:
            return self._predict_tree(x, node.left)

        # If the feature value of the sample is less than or equal to the threshold, go to the left subtree, otherwise go to the right subtree
        # Basically it goes like this, the x asks the tree if arrived at a leaf node,
        # if not, it asks the tree if the feature value of the sample is less than or equal to the threshold,
        # if yes, it goes to the left subtree,
        # if not, it goes to the right subtree,
        # and so on, until it arrives at a leaf node.
        if x[node.feature_index] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)
