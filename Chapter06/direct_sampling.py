import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(1000)


Nsamples = 10000


def X1_sample():
    return np.random.normal(0.1, 2.0)


def X2_sample(x1):
    return np.random.normal(x1, 0.5 + np.sqrt(np.abs(x1)))


if __name__ == '__main__':
    X = np.zeros((Nsamples,))
    Y = np.zeros((Nsamples,))

    for i, t in enumerate(range(Nsamples)):
        x1 = X1_sample()
        x2 = X2_sample(x1)

        X[i] = x1
        Y[i] = x2

    # Show the density estimation
    sns.set()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.kdeplot(X, Y, shade=True, shade_lowest=True, kernel="gau", ax=ax)
    ax.set_xlabel(r"$x_1$", fontsize=18)
    ax.set_ylabel(r"$x_2$", fontsize=18)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-6, 6])

    plt.show()
