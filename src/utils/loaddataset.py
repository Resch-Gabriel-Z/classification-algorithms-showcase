from sklearn.datasets import load_iris, load_digits, make_classification, make_blobs
from sklearn.model_selection import train_test_split


def load_iris_dataset():
    iris = load_iris()
    X, y = iris.data, iris.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


def load_digits_dataset():
    digits = load_digits()
    X, y = digits.data, digits.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


def load_synthetic_classification_dataset():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_clusters_per_class=2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


def load_synthetic_blobs_dataset():
    X, y = make_blobs(n_samples=1000, centers=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    iris_train, iris_test, iris_labels_train, iris_labels_test = load_iris_dataset()
    (
        digits_train,
        digits_test,
        digits_labels_train,
        digits_labels_test,
    ) = load_digits_dataset()
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

    print("Iris dataset shapes:", iris_train.shape, iris_test.shape)
    print("Digits dataset shapes:", digits_train.shape, digits_test.shape)
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
