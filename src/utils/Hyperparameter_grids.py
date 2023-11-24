param_grid_linear_svm = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "learning_rate": [0.1, 0.01, 0.001, 0.0001],
    "max_iters": [100, 500, 1000],
    "tol": [1e-4, 1e-3, 1e-2],
    "lambda_param": [0.001, 0.01, 0.1, 1, 10, 100],
}

param_grid_KNN = {"n_neighbors": [1, 3, 5, 7, 9]}

param_grid_decision_tree = {
    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8],
}

param_grid_random_forest = {
    "n_estimators": [100, 200, 300],
    "criterion": ["gini", "entropy"],
    "max_depth": [1, 2, 3, 4, 5],
    "min_samples_split": [2, 3, 4, 5],
    "min_samples_leaf": [1, 2, 3, 4, 5],
}
