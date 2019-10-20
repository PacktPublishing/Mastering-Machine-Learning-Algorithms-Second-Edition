import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA

# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the dataset
    X, _ = make_circles(n_samples=1000,
                        factor=0.25,
                        noise=0.1)

    # Show the dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(X[:, 0], X[:, 1])
    ax.set_xlabel(r"$x_0$", fontsize=22)
    ax.set_ylabel(r"$x_1$", fontsize=22)
    ax.grid(True)

    plt.show()

    # Perform a kernel PCA (with radial basis function)
    kpca = KernelPCA(n_components=2,
                     kernel='rbf',
                     fit_inverse_transform=True,
                     gamma=5,
                     random_state=1000)
    X_kpca = kpca.fit_transform(X)

    # Plot the dataset after PCA
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(kpca.X_transformed_fit_[:, 0],
               kpca.X_transformed_fit_[:, 1])
    ax.set_xlabel(r"$z_0$", fontsize=22)
    ax.set_ylabel(r"$z_1$", fontsize=22)
    ax.grid(True)

    plt.show()




