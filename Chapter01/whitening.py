import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 200
mu = [1.0, 1.0]
covm = [[2.0, 0.0],[0.0, 0.8]]


def zero_center(X):
    return X - np.mean(X, axis=0)


def whiten(X, correct=True):
    Xc = zero_center(X)
    _, L, V = np.linalg.svd(Xc)
    W = np.dot(V.T, np.diag(1.0 / L))
    return np.dot(Xc, W) * np.sqrt(X.shape[0]) if correct else 1.0


if __name__ == "__main__":
    # Create the dataset
    X = np.random.multivariate_normal(mean=mu, cov=covm, size=nb_samples)

    # Perform whitening
    X_whiten = whiten(X)

    # Show the results
    sns.set()

    fig, ax = plt.subplots(1, 2, figsize=(22, 8), sharey=True)

    ax[0].scatter(X[:, 0], X[:, 1], s=50)
    ax[0].set_xlim([-6, 6])
    ax[0].set_ylim([-6, 6])
    ax[0].set_xlabel(r'$x_0$', fontsize=18)
    ax[0].set_ylabel(r'$x_1$', fontsize=18)
    ax[0].set_title('Original dataset', fontsize=18)

    ax[1].scatter(X_whiten[:, 0], X_whiten[:, 1], s=50)
    ax[1].set_xlim([-6, 6])
    ax[1].set_ylim([-6, 6])
    ax[1].set_xlabel(r'$x_0$', fontsize=18)
    ax[1].set_title(r'Whitened dataset', fontsize=18)

    plt.show()

    # Show original and whitened covariance matrices
    print(np.cov(X.T))
    print(np.cov(X_whiten.T))

