import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import fetch_olivetti_faces
from sklearn.manifold import TSNE

# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the dataset
    faces = fetch_olivetti_faces()

    # Train TSNE
    tsne = TSNE(n_components=2, perplexity=2, random_state=1000)
    X_tsne = tsne.fit_transform(faces['data'])

    print("Final KL divergence: {}".format(tsne.kl_divergence_))

    # Plot the result
    sns.set()
    fig, ax = plt.subplots(figsize=(18, 10))

    for i in range(400):
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], color=cm.rainbow(faces['target'] * 10), marker='o', s=20)
        ax.annotate('%d' % faces['target'][i], xy=(X_tsne[i, 0] + 1, X_tsne[i, 1] + 1))

    ax.set_xlabel(r'$x_0$', fontsize=18)
    ax.set_ylabel(r'$x_1$', fontsize=18)
    ax.grid(True)

    plt.show()