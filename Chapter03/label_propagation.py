import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(1000)


nb_samples=100
nb_unlabeled = 75
tolerance = 0.01


def rbf(x1, x2, gamma=10.0):
    n = np.linalg.norm(x1 - x2, ord=1)
    return np.exp(-gamma * np.power(n, 2))


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0, random_state=1000)
    Y[Y == 0] = -1
    Y[nb_samples - nb_unlabeled:nb_samples] = 0

    # Show the original dataset
    sns.set()
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], color='r', marker='s', s=150, label="Class 0")
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], color='b', marker='o', s=150, label="Class 1")
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], marker='o', facecolor='none', edgecolor='#003200', s=200, label="Unlabeled")

    ax.set_xlabel(r'$x_0$', fontsize=18)
    ax.set_ylabel(r'$x_1$', fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True)

    plt.show()

    # Compute W
    W_rbf = np.zeros((nb_samples, nb_samples))

    for i in range(nb_samples):
        for j in range(nb_samples):
            W_rbf[i, j] = rbf(X[i], X[j])

    # Compute D and its inverse
    D_rbf = np.diag(np.sum(W_rbf, axis=1))
    D_rbf_inv = np.linalg.inv(D_rbf)

    # Perform the label propagation
    Yt = Y.copy()
    Y_prev = np.zeros((nb_samples,))
    iterations = 0

    while np.linalg.norm(Yt - Y_prev, ord=1) > tolerance:
        P = np.dot(D_rbf_inv, W_rbf)
        Y_prev = Yt.copy()
        Yt = np.dot(P, Yt)
        #Yt[0:nb_samples - nb_unlabeled] = Y[0:nb_samples - nb_unlabeled]

    Y_final = np.sign(Yt)

    # Show the final result
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    ax[0].scatter(X[Y == -1, 0], X[Y == -1, 1], color='r', marker='s', s=100, label="Class 0")
    ax[0].scatter(X[Y == 1, 0], X[Y == 1, 1], color='b', marker='o', s=100, label="Class 1")
    ax[0].scatter(X[Y == 0, 0], X[Y == 0, 1], marker='o', facecolor='none', edgecolor='#003200', s=50, label="Unlabeled")

    ax[0].set_xlabel(r'$x_0$')
    ax[0].set_ylabel(r'$x_1$')
    ax[0].set_title('Dataset')
    ax[0].legend(fontsize=16)
    ax[0].grid(True)

    ax[1].scatter(X[Y_final == -1, 0], X[Y_final == -1, 1], color='r', marker='s', s=100, label="Class 0")
    ax[1].scatter(X[Y_final == 1, 0], X[Y_final == 1, 1], color='b', marker='o', s=100, label="Class 1")

    ax[1].set_xlabel(r'$x_0$')
    ax[1].set_ylabel(r'$x_1$')
    ax[1].set_title('Label Propagation')
    ax[1].legend(fontsize=16)
    ax[1].grid(True)

    plt.show()
