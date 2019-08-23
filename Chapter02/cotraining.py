import numpy as np

from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.utils import shuffle


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == "__main__":
    # Load the dataset
    wine = load_wine()
    X, Y = shuffle(wine['data'], wine['target'], random_state=1000)

    nb_samples = X.shape[0]
    nb_labeled = 20
    nb_unlabeled = nb_samples - nb_labeled
    nb_unlabeled_samples = 2
    feature_cut = 7

    X_unlabeled = X[-nb_unlabeled:]
    X_labeled = X[:nb_labeled]
    Y_labeled = Y[:nb_labeled]
    X_labeled_1 = X_labeled[:, 0:feature_cut]
    X_labeled_2 = X_labeled[:, feature_cut:]

    # Train a test Naive-Bayes classifier
    nb0 = GaussianNB()
    nb0.fit(X_labeled, Y_labeled)

    # Single NB classification report
    print(classification_report(Y, nb0.predict(X), target_names=wine['target_names']))

    # Perform the Cotraining procedure
    nb1 = None
    nb2 = None

    while X_labeled_1.shape[0] <= nb_samples:
        nb1 = GaussianNB()
        nb1.fit(X_labeled_1, Y_labeled)

        nb2 = GaussianNB()
        nb2.fit(X_labeled_2, Y_labeled)

        if X_labeled_1.shape[0] == nb_samples:
            break

        probs1 = nb1.predict_proba(X_unlabeled[:, 0:feature_cut])
        top_confidence_idxs1 = np.argsort(np.max(probs1, axis=1))[::-1]
        selected_idxs1 = top_confidence_idxs1[0:nb_unlabeled_samples]

        probs2 = nb2.predict_proba(X_unlabeled[:, feature_cut:])
        top_confidence_idxs2 = np.argsort(np.max(probs2, axis=1))[::-1]
        selected_idxs2 = top_confidence_idxs2[0:nb_unlabeled_samples]

        selected_idxs = list(selected_idxs1) + list(selected_idxs2)

        X_new_labeled = X_unlabeled[selected_idxs]
        X_new_labeled_1 = X_unlabeled[selected_idxs1, 0:feature_cut]
        X_new_labeled_2 = X_unlabeled[selected_idxs2, feature_cut:]

        Y_new_labeled_1 = nb1.predict(X_new_labeled_1)
        Y_new_labeled_2 = nb2.predict(X_new_labeled_2)

        X_labeled_1 = np.concatenate((X_labeled_1, X_new_labeled[:, 0:feature_cut]), axis=0)
        X_labeled_2 = np.concatenate((X_labeled_2, X_new_labeled[:, feature_cut:]), axis=0)
        Y_labeled = np.concatenate((Y_labeled, Y_new_labeled_1, Y_new_labeled_2), axis=0)

        X_unlabeled = np.delete(X_unlabeled, selected_idxs, axis=0)

    # Print the Cotraining classification reports
    print(classification_report(Y, nb1.predict(X[:, 0:feature_cut]), target_names=wine['target_names']))
    print(classification_report(Y, nb2.predict(X[:, feature_cut:]), target_names=wine['target_names']))

