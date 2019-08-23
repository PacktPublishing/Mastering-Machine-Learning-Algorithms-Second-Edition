import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal

sns.set()

# Set random seed for reproducibility
np.random.seed(1000)

nb_samples = 250
nb_unlabeled = 200
threshold = 1e-4

# First Gaussian
m1 = np.array([-2.0, -2.5])
c1 = np.array([[1.0, 1.0],
               [1.0, 2.0]])
q1 = 0.5

# Second Gaussian
m2 = np.array([1.0, 3.0])
c2 = np.array([[2.0, -1.0],
               [-1.0, 3.5]])
q2 = 0.5

m1_old = np.zeros((2,))
c1_old = np.zeros((2, 2))
q1_old = 0

m2_old = np.zeros((2,))
c2_old = np.zeros((2, 2))
q2_old = 0


def total_norm():
    global m1, m1_old, m2, m2_old, c1, c1_old, c2, c2_old, q1, q1_old, q2, q2_old
    return np.linalg.norm(m1 - m1_old) + \
           np.linalg.norm(m2 - m2_old) + \
           np.linalg.norm(c1 - c1_old) + \
           np.linalg.norm(c2 - c2_old) + \
           np.linalg.norm(q1 - q1_old) + \
           np.linalg.norm(q2 - q2_old)


def show_dataset():
    w1, v1 = np.linalg.eigh(c1)
    w2, v2 = np.linalg.eigh(c2)

    nv1 = v1 / np.linalg.norm(v1)
    nv2 = v2 / np.linalg.norm(v2)

    a1 = np.arccos(np.dot(nv1[:, 1], [1.0, 0.0]) / np.linalg.norm(nv1[:, 1])) * 180.0 / np.pi
    a2 = np.arccos(np.dot(nv2[:, 1], [1.0, 0.0]) / np.linalg.norm(nv2[:, 1])) * 180.0 / np.pi

    fig, ax = plt.subplots(figsize=(20, 15))

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=120, marker='o', label='Class 1')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=120, marker='d', label='Class 2')
    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], s=120, marker='x', label='Unlabeled')

    g1 = Ellipse(xy=m1, width=w1[1] * 3, height=w1[0] * 3, fill=False, linestyle='dashed', angle=a1, color='black',
                 linewidth=1)
    g1_1 = Ellipse(xy=m1, width=w1[1] * 2, height=w1[0] * 2, fill=False, linestyle='dashed', angle=a1, color='black',
                   linewidth=2)
    g1_2 = Ellipse(xy=m1, width=w1[1] * 1.4, height=w1[0] * 1.4, fill=False, linestyle='dashed', angle=a1,
                   color='black', linewidth=3)

    g2 = Ellipse(xy=m2, width=w2[1] * 3, height=w2[0] * 3, fill=False, linestyle='dashed', angle=a2, color='black',
                 linewidth=1)
    g2_1 = Ellipse(xy=m2, width=w2[1] * 2, height=w2[0] * 2, fill=False, linestyle='dashed', angle=a2, color='black',
                   linewidth=2)
    g2_2 = Ellipse(xy=m2, width=w2[1] * 1.4, height=w2[0] * 1.4, fill=False, linestyle='dashed', angle=a2,
                   color='black', linewidth=3)

    ax.set_xlabel(r'$x_0$', fontsize=18)
    ax.set_ylabel(r'$x_1$', fontsize=18)

    ax.add_artist(g1)
    ax.add_artist(g1_1)
    ax.add_artist(g1_2)
    ax.add_artist(g2)
    ax.add_artist(g2_1)
    ax.add_artist(g2_2)

    ax.legend(fontsize=18)

    plt.show()


if __name__ == '__main__':
    # Generate dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=2, cluster_std=1.25, random_state=100)

    unlabeled_idx = np.random.choice(np.arange(0, nb_samples, 1), replace=False, size=nb_unlabeled)
    Y[unlabeled_idx] = -1

    # Show the dataset with the initial Gaussians
    show_dataset()

    # Training process
    while total_norm() > threshold:
        m1_old = m1.copy()
        c1_old = c1.copy()
        q1_old = q1

        m2_old = m2.copy()
        c2_old = c2.copy()
        q2_old = q2

        Pij = np.zeros((nb_samples, 2))

        # E Step
        for i in range(nb_samples):
            if Y[i] == -1:
                p1 = multivariate_normal.pdf(X[i], m1, c1, allow_singular=True) * q1
                p2 = multivariate_normal.pdf(X[i], m2, c2, allow_singular=True) * q2
                Pij[i] = [p1, p2] / (p1 + p2)

            else:
                Pij[i, :] = [1.0, 0.0] if Y[i] == 0 else [0.0, 1.0]

        # M Step
        n = np.sum(Pij, axis=0)
        m = np.sum(np.dot(Pij.T, X), axis=0)

        m1 = np.dot(Pij[:, 0], X) / n[0]
        m2 = np.dot(Pij[:, 1], X) / n[1]

        q1 = n[0] / float(nb_samples)
        q2 = n[1] / float(nb_samples)

        c1 = np.zeros((2, 2))
        c2 = np.zeros((2, 2))

        for t in range(nb_samples):
            c1 += Pij[t, 0] * np.outer(X[t] - m1, X[t] - m1)
            c2 += Pij[t, 1] * np.outer(X[t] - m2, X[t] - m2)

        c1 /= n[0]
        c2 /= n[1]

    # Show the final Gaussians
    show_dataset()

    # Check some points
    print('The first 10 unlabeled samples:')
    print(np.round(X[Y == -1][0:10], 3))

    print('\nCorresponding Gaussian assigments:')
    print(np.round(Pij[Y == -1][0:10], 3))