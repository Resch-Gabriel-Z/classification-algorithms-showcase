from algorithms.svm import SVM
from algorithms.knn import KNN
from algorithms.decisiontree import DecisionTree
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from utils.loaddataset import (
    load_synthetic_blobs_dataset,
    load_synthetic_moons_dataset,
    load_synthetic_circles_dataset,
    load_synthetic_classification_dataset,
)
from utils.Hyperparameter_grids import (
    param_grid_linear_svm,
    param_grid_KNN,
    param_grid_decision_tree,
)

import argparse


# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic_blobs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model", type=str, default="SVM")
    parser.add_argument("--search", type=str, default="RandomizedSearchCV")
    parser.add_argument("--n_iter", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.dataset == "synthetic_blobs":
        X_train, X_test, y_train, y_test = load_synthetic_blobs_dataset(
            test_size=args.test_size, random_state=args.random_state
        )
    elif args.dataset == "synthetic_moons":
        X_train, X_test, y_train, y_test = load_synthetic_moons_dataset(
            test_size=args.test_size, random_state=args.random_state
        )
    elif args.dataset == "synthetic_circles":
        X_train, X_test, y_train, y_test = load_synthetic_circles_dataset(
            test_size=args.test_size, random_state=args.random_state
        )
    elif args.dataset == "synthetic_classification":
        X_train, X_test, y_train, y_test = load_synthetic_classification_dataset(
            test_size=args.test_size, random_state=args.random_state
        )
        raise ValueError("Invalid dataset name.")

    if args.model == "SVM":
        hyperparameters_grid = param_grid_linear_svm
        model = SVM()
    elif args.model == "KNN":
        hyperparameters_grid = param_grid_KNN
        model = KNN()
    elif args.model == "DecisionTree":
        hyperparameters_grid = param_grid_decision_tree
        model = DecisionTree()
    else:
        raise ValueError("Invalid model name.")

    if args.search == "GridSearchCV":
        search = GridSearchCV(
            model, param_grid=hyperparameters_grid, cv=5, verbose=1, scoring="accuracy"
        )
    elif args.search == "RandomizedSearchCV":
        search = RandomizedSearchCV(
            model,
            param_distributions=hyperparameters_grid,
            cv=5,
            verbose=1,
            n_iter=args.n_iter,
            scoring="accuracy",
        )
    else:
        raise ValueError("Invalid search name.")

    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # return several objects for information purposes
    return (
        accuracy,
        search.best_params_,
        search.best_estimator_,
        search.best_score_,
        args,
    )


if __name__ == "__main__":
    accuracy, best_params, best_estimator, best_score, args = main()
    print("Accuracy:", accuracy)
    print("Best parameters:", best_params)
    print("Best estimator:", best_estimator)
    print("Best score:", best_score)
    print("Arguments:", args)
