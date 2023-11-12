from algorithms.svm import SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.loaddataset import load_synthetic_blobs_dataset

def main():
    a,b,c,d = load_synthetic_blobs_dataset()
    print(a)
    pass


if __name__ == "__main__":
    main()
