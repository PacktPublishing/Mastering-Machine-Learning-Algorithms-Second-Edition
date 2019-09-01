import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 1000


if __name__ == '__main__':
    # Create the dataset
    X = np.zeros(shape=(nb_samples, 2))

    for i in range(nb_samples):
        X[i, 0] = float(i)

        if i % 2 == 0:
            X[i, 1] = 1.0 + (np.random.uniform(0.65, 1.0) *
                             np.sin(float(i) / 100.0))
        else:
            X[i, 1] = 0.1 + (np.random.uniform(0.5, 0.85) *
                             np.sin(float(i) / 100.0))

    ss = StandardScaler()
    Xs = ss.fit_transform(X)

    # Show the dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.scatter(Xs[:, 0], Xs[:, 1], marker="o", s=50)

    ax.set_xlabel(r'$x_0$', fontsize=18)
    ax.set_ylabel(r'$x_1$', fontsize=18)
    ax.grid(True)
    plt.show()

    # Test K-Means
    km = KMeans(n_clusters=2, random_state=1000)
    Y_km = km.fit_predict(Xs)

    # Plot the result
    sns.set()

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.scatter(Xs[Y_km == 0, 0], Xs[Y_km == 0, 1], marker="o", s=100, label="Class 0")
    ax.scatter(Xs[Y_km == 1, 0], Xs[Y_km == 1, 1], marker="d", s=100, label="Class 1")

    ax.set_xlabel(r'$x_0$', fontsize=18)
    ax.set_ylabel(r'$x_1$', fontsize=18)
    ax.grid(True)
    ax.legend(fontsize=18)
    plt.show()

    # Apply Spectral clustering
    sc = SpectralClustering(n_clusters=2,
                            affinity='nearest_neighbors',
                            n_neighbors=20,
                            random_state=1000)
    Y_sc = sc.fit_predict(Xs)

    # Plot the result
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.scatter(Xs[Y_sc == 0, 0], Xs[Y_sc == 0, 1], marker="o", s=100, label="Class 0")
    ax.scatter(Xs[Y_sc == 1, 0], Xs[Y_sc == 1, 1], marker="s", s=100, label="Class 1")

    ax.set_xlabel(r'$x_0$', fontsize=18)
    ax.set_ylabel(r'$x_1$', fontsize=18)
    ax.legend(fontsize=18)
    ax.grid(True)
    plt.show()




