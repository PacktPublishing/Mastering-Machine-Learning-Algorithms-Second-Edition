import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 1000


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples,
                      n_features=2,
                      centers=3, cluster_std=1.5,
                      random_state=1000)

    # Show the unclustered dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.scatter(X[:, 0], X[:, 1], s=40)
    ax.grid(True)
    ax.set_xlabel(r'$x_0$', fontsize=22)
    ax.set_ylabel(r'$x_1$', fontsize=22)

    plt.show()

    # Find the optimal number of components
    nb_components = [2, 3, 4, 5, 6, 7, 8]

    aics = []
    bics = []

    for n in nb_components:
        gm = GaussianMixture(n_components=n,
                             max_iter=1000,
                             random_state=1000)
        gm.fit(X)
        aics.append(gm.aic(X))
        bics.append(gm.bic(X))

    fig, ax = plt.subplots(2, 1, figsize=(15, 8))

    ax[0].plot(nb_components, aics)
    ax[0].set_ylabel('AIC', fontsize=22)
    ax[0].grid(True)

    ax[1].plot(nb_components, bics)
    ax[1].set_xlabel('Number of components', fontsize=22)
    ax[1].set_ylabel('BIC', fontsize=22)
    ax[1].grid(True)

    plt.show()

    # Create and fit a Gaussian Mixture
    gm = GaussianMixture(n_components=3)
    gm.fit(X)

    # Show the Gaussian parameters
    print('Weights:')
    print(gm.weights_)

    print('\nMeans:')
    print(gm.means_)

    print('\nCovariances:')
    print(gm.covariances_)

    # Show the clustered dataset
    Yp = gm.predict(X)

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.scatter(X[Yp == 0, 0], X[Yp == 0, 1], c='red', marker='o', s=80, label="Gaussian 1")
    ax.scatter(X[Yp == 1, 0], X[Yp == 1, 1], c='blue', marker='x', s=80, label="Gaussian 2")
    ax.scatter(X[Yp == 2, 0], X[Yp == 2, 1], c='green', marker='d', s=80, label="Gaussian 3")
    ax.grid(True)
    ax.set_xlabel(r'$x_0$', fontsize=22)
    ax.set_ylabel(r'$x_1$', fontsize=22)
    ax.legend(fontsize=22)

    plt.show()