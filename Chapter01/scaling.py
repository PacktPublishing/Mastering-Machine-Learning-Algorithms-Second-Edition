import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 200
mu = [1.0, 1.0]
covm = [[2.0, 0.0],[0.0, 0.8]]


if __name__ == "__main__":
    # Create the dataset
    X = np.random.multivariate_normal(mean=mu, cov=covm, size=nb_samples)

    # Perform scaling
    ss = StandardScaler()
    X_ss = ss.fit_transform(X)

    rs = RobustScaler(quantile_range=(10, 90))
    X_rs = rs.fit_transform(X)

    mms = MinMaxScaler(feature_range=(-1, 1))
    X_mms = mms.fit_transform(X)

    # Show the results
    sns.set()

    fig, ax = plt.subplots(2, 2, figsize=(22, 15), sharex=True, sharey=True)

    ax[0, 0].scatter(X[:, 0], X[:, 1], s=50)
    ax[0, 0].set_xlim([-6, 6])
    ax[0, 0].set_ylim([-6, 6])
    ax[0, 0].set_ylabel(r'$x_1$', fontsize=16)
    ax[0, 0].set_title('Original dataset', fontsize=18)

    ax[0, 1].scatter(X_mms[:, 0], X_mms[:, 1], s=50)
    ax[0, 1].set_xlim([-6, 6])
    ax[0, 1].set_ylim([-6, 6])
    ax[0, 1].set_title(r'Min-Max scaling (-1, 1)', fontsize=18)

    ax[1, 0].scatter(X_ss[:, 0], X_ss[:, 1], s=50)
    ax[1, 0].set_xlim([-6, 6])
    ax[1, 0].set_ylim([-6, 6])
    ax[1, 0].set_xlabel(r'$x_0$', fontsize=16)
    ax[1, 0].set_ylabel(r'$x_1$', fontsize=16)
    ax[1, 0].set_title(r'Standard scaling ($\mu=0$ and $\sigma=1$)', fontsize=18)

    ax[1, 1].scatter(X_rs[:, 0], X_rs[:, 1], s=50)
    ax[1, 1].set_xlim([-6, 6])
    ax[1, 1].set_ylim([-6, 6])
    ax[1, 1].set_xlabel(r'$x_0$', fontsize=16)
    ax[1, 1].set_title(r'Robust scaling based on ($10^{th}, 90^{th}$) quantiles', fontsize=18)

    plt.show()