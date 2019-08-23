import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import make_classification
from scipy.optimize import minimize


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples=200
nb_unlabeled=150
k = 50


def rbf(x1, x2, gamma=0.1):
    n = np.linalg.norm(x1 - x2, ord=1)
    return np.exp(-gamma * np.power(n, 2))


def objective(t):
    return np.sum(np.power(Y - np.dot(t, V.T), 2))


if __name__ == "__main__":
    # Create dataset
    X, Y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0, random_state=1000)
    Y[Y == 0] = -1
    Y[nb_samples - nb_unlabeled:nb_samples] = 0

    # Show initial dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], color='r', marker='s', s=150, label="Class 0")
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], color='b', marker='o', s=150, label="Class 1")
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], facecolor='none', edgecolor='#003200', marker='o', s=50, label="Unlabeled")

    ax.set_xlabel(r'$x_0$', fontsize=18)
    ax.set_ylabel(r'$x_1$', fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True)

    plt.show()

    # Compute the Laplacian
    W_rbf = np.zeros((nb_samples, nb_samples))

    for i in range(nb_samples):
        for j in range(nb_samples):
            if i == j:
                W_rbf[i, j] = 0.0
            else:
                W_rbf[i, j] = rbf(X[i], X[j])

    D_rbf = np.diag(np.sum(W_rbf, axis=1))
    L_rbf = D_rbf - W_rbf

    # Eigendecompose the Laplacian
    w, v = np.linalg.eig(L_rbf)
    sw = np.argsort(w)[0:k]

    V = v[:, sw]
    theta = np.random.normal(0.0, 0.1, size=(1, k))
    Yu = np.zeros(shape=(nb_unlabeled,))

    # Optimize the objective
    result = minimize(objective,
                      theta,
                      method="BFGS",
                      options={
                          "maxiter": 500000,
                      })

    # Compute the final labeling
    Y_final = np.sign(np.dot(result["x"], V.T))

    # Show the final result
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    ax[0].scatter(X[Y == -1, 0], X[Y == -1, 1], color='r', marker='s', s=100, label="Class 0")
    ax[0].scatter(X[Y == 1, 0], X[Y == 1, 1], color='b', marker='o', s=100, label="Class 1")
    ax[0].scatter(X[Y == 0, 0], X[Y == 0, 1], facecolor='none', edgecolor='#003200', marker='o', s=50,
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
    ax[1].set_title('Laplacian Regularization')
    ax[1].legend(fontsize=16)
    ax[1].grid(True)

    plt.show()

