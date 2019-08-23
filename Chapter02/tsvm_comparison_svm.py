import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Set random seed for reproducibility
np.random.seed(1000)

nb_samples = 100
nb_unlabeled = 90


# Create dataset
X, Y = make_classification(n_samples=nb_samples, n_features=2, n_redundant=0, random_state=100)
Y[Y==0] = -1
Y[nb_samples - nb_unlabeled:nb_samples] = 0


if __name__ == '__main__':
    # Show the initial dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], marker='o', s=100, label='Class 0')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='^', s=100, label='Class 1')
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], facecolor='none', edgecolor='#003200', marker='o', s=80, label='Unlabeled')

    ax.set_xlabel(r'$x_0$', fontsize=16)
    ax.set_ylabel(r'$x_1$', fontsize=16)
    ax.grid(True)
    ax.legend(fontsize=16)

    plt.show()

    # Train a SVM
    svc = SVC(kernel='linear', C=1.0)
    svc.fit(X[Y != 0], Y[Y != 0])

    Xu_svc = X[nb_samples - nb_unlabeled:nb_samples]
    yu_svc = svc.predict(Xu_svc)

    # Show the final plots
    fig, ax = plt.subplots(1, 2, figsize=(22, 9), sharey=True)

    ax[0].scatter(X[Y == -1, 0], X[Y == -1, 1], marker='o', s=100, label='Class 0')
    ax[0].scatter(X[Y == 1, 0], X[Y == 1, 1], marker='^', s=100, label='Class 1')
    ax[0].scatter(X[Y == 0, 0], X[Y == 0, 1], facecolor='none', edgecolor='#003200', s=100, label='Unlabeled')

    ax[0].set_xlabel(r'$x_0$', fontsize=16)
    ax[0].set_ylabel(r'$x_1$', fontsize=16)
    ax[0].set_title('Dataset', fontsize=18)
    ax[0].grid(True)
    ax[0].legend(fontsize=16)

    ax[1].scatter(X[Y == -1, 0], X[Y == -1, 1], c='r', marker='o', s=100, label='Labeled class 0')
    ax[1].scatter(X[Y == 1, 0], X[Y == 1, 1], c='b', marker='^', s=100, label='Labeled class 1')

    ax[1].scatter(Xu_svc[yu_svc == -1, 0], Xu_svc[yu_svc == -1, 1], c='r', marker='s', s=150, label='Unlabeled class 0')
    ax[1].scatter(Xu_svc[yu_svc == 1, 0], Xu_svc[yu_svc == 1, 1], c='b', marker='v', s=150, label='Unlabeled class 1')

    ax[1].set_xlabel(r'$x_0$', fontsize=16)
    ax[1].set_title('Linear SVM', fontsize=18)
    ax[1].grid(True)
    ax[1].legend(fontsize=16)

    plt.show()