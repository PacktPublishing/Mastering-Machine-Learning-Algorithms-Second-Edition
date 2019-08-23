import numpy as np

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.utils import shuffle


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == "__main__":
    # Load the dataset
    iris = load_iris()
    X, Y = shuffle(iris['data'], iris['target'], random_state=1000)

    nb_samples = X.shape[0]
    nb_labeled = 20
    nb_unlabeled = nb_samples - nb_labeled
    nb_unlabeled_samples = 20

    X_train = X[:nb_labeled]
    Y_train = Y[:nb_labeled]
    X_unlabeled = X[nb_labeled:]

    # Train a test Naive-Bayes classifier
    nb0 = GaussianNB()
    nb0.fit(X, Y)

    # Single NB classification report
    print(classification_report(Y, nb0.predict(X), target_names=iris['target_names']))

    # Perform the self-training procedure
    nt = 0
    nb = None

    while X_train.shape[0] <= nb_samples:
        nt += 1

        nb = GaussianNB()
        nb.fit(X_train, Y_train)

        if X_train.shape[0] == nb_samples:
            break

        probs = nb.predict_proba(X_unlabeled)
        top_confidence_idxs = np.argsort(np.max(probs, axis=1)).astype(np.int64)[::-1]
        selected_idxs = top_confidence_idxs[0:nb_unlabeled_samples]

        X_new_train = X_unlabeled[selected_idxs]
        Y_new_train = nb.predict(X_new_train)

        X_train = np.concatenate((X_train, X_new_train), axis=0)
        Y_train = np.concatenate((Y_train, Y_new_train), axis=0)

        X_unlabeled = np.delete(X_unlabeled, selected_idxs, axis=0)

    # Self-training classification report
    print(classification_report(Y, nb.predict(X), target_names=iris['target_names']))