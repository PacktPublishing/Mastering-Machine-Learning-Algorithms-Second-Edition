import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == "__main__":
    # Create the dataset
    mus = [[-5, -3], [-5, 3], [-1, -4], [1, -4], [-2, 0],
           [0, 1], [4, 2], [6, 4], [5, 1], [6, -3], [-5, 3]]
    Xts = []

    for mu in mus:
        n = np.random.randint(100, 1000)
        covm = np.diag(np.random.uniform(0.2, 3.5, size=(2,)))
        Xt = np.random.multivariate_normal(mu, covm,
                                           size=(n,))
        Xts.append(Xt)

    X = np.concatenate(Xts)

    # Show the original dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(25, 18))

    ax.scatter(X[:, 0], X[:, 1], facecolor='none', edgecolor='b', marker='o', s=30, label="Density group")
    ax.set_xlabel("Longitudinal extension (km)", fontsize=18)
    ax.set_ylabel("Latitudinal extension (km)", fontsize=18)
    ax.set_title("Territory", fontsize=18)
    ax.legend(fontsize=18)
    ax.grid(True)

    plt.show()

    # Compute the scores
    ch = []
    db = []
    no = []

    for e in np.arange(0.1, 0.5, 0.02):
        dbscan = DBSCAN(eps=e, min_samples=8, leaf_size=50)
        Y = dbscan.fit_predict(X)
        ch.append(calinski_harabasz_score(X, Y))
        db.append(davies_bouldin_score(X, Y))
        no.append(np.sum(Y == -1))

    # Plot the results
    fig, ax = plt.subplots(1, 3, figsize=(25, 8))

    x = np.arange(0.1, 0.5, 0.02)

    ax[0].plot(x, ch, "o-")
    ax[0].set_xlabel(r"$\epsilon$", fontsize=20)
    ax[0].set_title("Calinski-Harabasz score", fontsize=20)

    ax[1].plot(x, db, "o-")
    ax[1].set_xlabel(r"$\epsilon$", fontsize=20)
    ax[1].set_title("Davies-Bouldin score", fontsize=20)

    ax[2].plot(x, no, "o-")
    ax[2].set_xlabel(r"$\epsilon$", fontsize=20)
    ax[2].set_title("Number of noisy points", fontsize=20)

    plt.show()

    # Perform the clustering
    dbscan = DBSCAN(eps=0.2, min_samples=8, leaf_size=50)
    Y = dbscan.fit_predict(X)

    print("No. clusters: {}".format(np.unique(dbscan.labels_).shape))
    print("No. noisy points: {}".format(np.sum(Y == -1)))
    print("CH = {}".format(calinski_harabasz_score(X, Y)))
    print("DB = {}".format(davies_bouldin_score(X, Y)))

    # Show the final result
    fig, ax = plt.subplots(figsize=(25, 18))

    ms = ['o', 'd', '^', 'v']

    for i, y in enumerate(np.unique(dbscan.labels_)):
        label = "C{}".format(y + 1) if y != -1 else "Noisy"
        m = ms[i % 4]
        if y != -1:
            ax.scatter(X[Y == y, 0], X[Y == y, 1], marker=m, facecolor='none', edgecolor=cm.nipy_spectral((y + 1) * 10),
                       s=100, label=label)
        else:
            ax.scatter(X[Y == y, 0], X[Y == y, 1], marker='o', color="r", s=20, label=label)

    ax.set_xlabel("Longitudinal extension (km)", fontsize=22)
    ax.set_ylabel("Latitudinal extension (km)", fontsize=22)
    ax.set_title(r"DBSCAN with $\epsilon=0.2$ and leaf size = 50 ({} classes)".format(len(np.unique(dbscan.labels_))),
                 fontsize=22)
    ax.legend(fontsize=12)
    ax.grid(True)

    plt.show()