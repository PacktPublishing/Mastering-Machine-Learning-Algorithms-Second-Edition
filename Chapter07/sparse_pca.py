import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib

from sklearn.datasets import fetch_openml
from sklearn.decomposition import SparsePCA

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

    # Perform a PCA on the digits dataset
    spca = SparsePCA(n_components=10,
                     alpha=0.1,
                     normalize_components=True,
                     n_jobs=joblib.cpu_count(),
                     random_state=1000)
    X_spca = spca.fit_transform(X[0:10, :])

    print('SPCA components shape:')
    print(spca.components_.shape)

    # Show the components
    sns.set()
    fig, ax = plt.subplots(1, 10, figsize=(22, 5))

    for i in range(10):
        ax[i].set_axis_off()
        ax[i].imshow(spca.components_[i].reshape((28, 28)), cmap='gray')

    plt.show()



