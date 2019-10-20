import numpy as np

from sklearn.datasets import make_blobs
from sklearn.mixture import BayesianGaussianMixture

# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 1000


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples,
                      n_features=2,
                      centers=3, cluster_std=1.5,
                      random_state=1000)

    # Create and fit a Bayesian Gaussian Mixture
    gm = BayesianGaussianMixture(n_components=8,
                                 max_iter=10000,
                                 weight_concentration_prior=1,
                                 random_state=1000)
    gm.fit(X)

    # Show the Gaussian parameters
    print('Weights (wc = 1):')
    for w in gm.weights_:
        print("{:.2f}".format(w))


