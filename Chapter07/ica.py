import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.decomposition import FastICA

# Set random seed for reproducibility
np.random.seed(1000)


def zero_center(Xd):
    return Xd - np.mean(Xd, axis=0)


if __name__ == '__main__':
    # Load the dataset
    digits = fetch_openml("mnist_784")
    X = zero_center(digits['data'].
                    astype(np.float64) / 255.0)
    np.random.shuffle(X)

    # Peform Fast ICA with 64 components
    fastica = FastICA(n_components=64,
                      max_iter=5000,
                      random_state=1000)
    fastica.fit(X)

    # Plot the indipendent components
    sns.set()

    fig, ax = plt.subplots(8, 8, figsize=(11, 11))

    for i in range(8):
        for j in range(8):
            ax[i, j].imshow(fastica.components_[(i * 8) + j].reshape((28, 28)), cmap='gray')
            ax[i, j].axis('off')

    plt.show()

    # Peform Fast ICA with 640 components
    fastica = FastICA(n_components=640,
                      max_iter=5000,
                      random_state=1000)
    fastica.fit(X)

    # Plot the indipendent components
    fig, ax = plt.subplots(10, 10, figsize=(11, 11))

    for i in range(10):
        for j in range(10):
            ax[i, j].imshow(fastica.components_[(i * 10) + j].reshape((28, 28)), cmap='gray')
            ax[i, j].axis('off')

    plt.show()

