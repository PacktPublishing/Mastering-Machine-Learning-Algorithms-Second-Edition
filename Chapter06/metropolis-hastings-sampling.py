import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(1000)


nb_iterations = 100000
x = 1.0
samples = []


def prior(x):
    return 0.1 * np.exp(-0.1 * x)


def likelihood(x):
    a = np.sqrt(0.2 / (2.0 * np.pi * np.power(x, 3)))
    b = - (0.2 * np.power(x - 1.0, 2)) / (2.0 * x)
    return a * np.exp(b)


def g(x):
    return likelihood(x) * prior(x)


def q(xp):
    return np.random.normal(xp)


if __name__ == '__main__':
    # Main loop
    for i in range(nb_iterations):
        xc = q(x)

        alpha = g(xc) / g(x)
        if np.isnan(alpha):
            continue

        if alpha >= 1:
            samples.append(xc)
            x = xc
        else:
            if np.random.uniform(0.0, 1.0) < alpha:
                samples.append(xc)
                x = xc

    # Show the histogram
    sns.set()

    fig, ax = plt.subplots(1, 2, figsize=(22, 10))
    sns.kdeplot(samples, shade=True, shade_lowest=True, kernel="gau", ax=ax[0])
    sns.kdeplot(samples, shade=True, shade_lowest=True, cumulative=True, kernel="gau", ax=ax[1])

    ax[0].set_xlabel('x', fontsize=22)
    ax[0].set_title('Probability density function', fontsize=22)

    ax[1].set_xlabel('x', fontsize=22)
    ax[1].set_title('Cumulative distribution function', fontsize=22)

    plt.show()