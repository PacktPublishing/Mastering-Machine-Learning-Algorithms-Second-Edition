import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == "__main__":
    # Create a dummy classification dataset
    X, Y = make_classification(n_samples=500, n_classes=5, n_features=50, n_informative=10,
                               n_redundant=5, n_clusters_per_class=3, random_state=1000)

    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=1000)

    print(X_train.shape)
    print(Y_train.shape)

    print(X_test.shape)
    print(Y_test.shape)


