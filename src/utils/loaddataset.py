from sklearn.datasets import (
    make_classification,
    make_blobs,
    make_moons,
    make_circles,
)
from sklearn.model_selection import train_test_split


def load_synthetic_classification_dataset(test_size=0.2, random_state=42):
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_synthetic_blobs_dataset(test_size=0.2, random_state=42):
    X, y = make_blobs(n_samples=1000, centers=2, random_state=random_state)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_synthetic_moons_dataset(test_size=0.2, random_state=42):
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=random_state)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_synthetic_circles_dataset(test_size=0.2, random_state=42):
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=random_state)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    (
        synthetic_class_train,
        synthetic_class_test,
        synthetic_class_labels_train,
        synthetic_class_labels_test,
    ) = load_synthetic_classification_dataset()
    (
        synthetic_blobs_train,
        synthetic_blobs_test,
        synthetic_blobs_labels_train,
        synthetic_blobs_labels_test,
    ) = load_synthetic_blobs_dataset()
    (
        synthetic_moons_train,
        synthetic_moons_test,
        synthetic_moons_labels_train,
        synthetic_moons_labels_test,
    ) = load_synthetic_moons_dataset()
    (
        synthetic_circles_train,
        synthetic_circles_test,
        synthetic_circles_labels_train,
        synthetic_circles_labels_test,
    ) = load_synthetic_circles_dataset()
    print(
        "Synthetic Classification dataset shapes:",
        synthetic_class_train.shape,
        synthetic_class_test.shape,
    )
    print(
        "Synthetic Blobs dataset shapes:",
        synthetic_blobs_train.shape,
        synthetic_blobs_test.shape,
    )
