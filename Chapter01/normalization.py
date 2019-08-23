import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import Normalizer


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 200
mu = [1.0, 1.0]
covm = [[2.0, 0.0],[0.0, 0.8]]


if __name__ == "__main__":
    # Create the dataset
    X = np.random.multivariate_normal(mean=mu, cov=covm, size=nb_samples)

    # Perform normalization
    nz = Normalizer(norm='l2')
    X_nz = nz.fit_transform(X)

    # Show the results
    sns.set()

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(X_nz[:, 0], X_nz[:, 1], s=50)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_xlabel(r'$x_0$', fontsize=16)
    ax.set_ylabel(r'$x_1$', fontsize=16)
    ax.set_title(r'Normalized dataset ($L_2$ norm = 1)', fontsize=16)

    plt.show()

    # Compute a test example
    X_test = [
        [-4., 0.],
        [-1., 3.]
    ]

    Y_test = nz.transform(X_test)

    # Print the degree (in radians)
    print(np.arccos(np.dot(Y_test[0], Y_test[1])))