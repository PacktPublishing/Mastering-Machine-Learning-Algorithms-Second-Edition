import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph

# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 2000
nb_unlabeled = 1950
nb_classes = 2



if __name__ == '__main__':
    X, Y = make_blobs(n_samples=nb_samples,
                      n_features=2,
                      centers=nb_classes,
                      cluster_std=2.5,
                      random_state=1000)

    Y[Y == 0] = -1
    Y[nb_samples - nb_unlabeled:nb_samples] = 0

    # Show the original dataset
    sns.set()
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], color='r', marker='s', s=150, label="Class 0")
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], color='b', marker='o', s=150, label="Class 1")
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], marker='o', facecolor='none', edgecolor='#003200', s=20, label="Unlabeled")

    ax.set_xlabel(r'$x_0$', fontsize=18)
    ax.set_ylabel(r'$x_1$', fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True)

    plt.show()

    # Compute W
    W = kneighbors_graph(X, n_neighbors=15, mode='connectivity', include_self=True).toarray()

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    Luu = L[nb_samples - nb_unlabeled:, nb_samples - nb_unlabeled:]
    Wul = W[nb_samples - nb_unlabeled:, 0:nb_samples - nb_unlabeled, ]
    Yl = Y[0:nb_samples - nb_unlabeled]

    Yu = np.round(np.linalg.solve(Luu, np.dot(Wul, Yl)))
    Y_final = Y.copy()
    Y_final[nb_samples - nb_unlabeled:] = Yu.copy()

    # Show the final result
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    ax[0].scatter(X[Y == -1, 0], X[Y == -1, 1], color='r', marker='s', s=100, label="Class 0")
    ax[0].scatter(X[Y == 1, 0], X[Y == 1, 1], color='b', marker='o', s=100, label="Class 1")
    ax[0].scatter(X[Y == 0, 0], X[Y == 0, 1], marker='o', facecolor='none', edgecolor='#003200', s=20,
                  label="Unlabeled")

    ax[0].set_xlabel(r'$x_0$')
    ax[0].set_ylabel(r'$x_1$')
    ax[0].set_title('Dataset')
    ax[0].legend(fontsize=16)
    ax[0].grid(True)

    ax[1].scatter(X[Y_final == -1, 0], X[Y_final == -1, 1], color='r', marker='s', s=100, label="Class 0")
    ax[1].scatter(X[Y_final == 1, 0], X[Y_final == 1, 1], color='b', marker='o', s=100, label="Class 1")

    ax[1].set_xlabel(r'$x_0$')
    ax[1].set_ylabel(r'$x_1$')
    ax[1].set_title('Markov Random Walk')
    ax[1].legend(fontsize=16)
    ax[1].grid(True)

    plt.show()

