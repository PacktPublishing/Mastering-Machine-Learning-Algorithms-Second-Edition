import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import tensorflow as tf

# To install the DBN package: pip install git+git://github.com/albertbup/deep-belief-network.git
# Further information: https://github.com/albertbup/deep-belief-network
from dbn import UnsupervisedDBN

from sklearn.manifold import TSNE
from sklearn.utils import shuffle


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 400


if __name__ == '__main__':
    # Load the dataset
    (X_train, Y_train), (_, _) = \
        tf.keras.datasets.mnist.load_data()
    X_train, Y_train = shuffle(X_train, Y_train,
                               random_state=1000)

    width = X_train.shape[1]
    height = X_train.shape[2]

    X = X_train[0:nb_samples].reshape(
        (nb_samples, width * height)).\
            astype(np.float32) / 255.0
    Y = Y_train[0:nb_samples]

    # Train the unsupervised DBN
    unsupervised_dbn = UnsupervisedDBN(
        hidden_layers_structure=[512, 256, 64],
        learning_rate_rbm=0.05,
        n_epochs_rbm=100,
        batch_size=64,
        activation_function='sigmoid')

    X_dbn = unsupervised_dbn.fit_transform(X)

    # Perform t-SNE
    tsne = TSNE(n_components=2,
                perplexity=10,
                random_state=1000)
    X_tsne = tsne.fit_transform(X_dbn)

    # Show the result
    sns.set()
    fig, ax = plt.subplots(figsize=(18, 12))

    colors = [cm.tab10(i) for i in Y]

    for i in range(nb_samples):
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, marker='o', s=50)
        ax.annotate('%d' % Y[i], xy=(X_tsne[i, 0] + 1, X_tsne[i, 1] + 1), fontsize=15)

    ax.set_xlabel(r'$x_0$', fontsize=20)
    ax.set_ylabel(r'$x_1$', fontsize=20)
    ax.grid(True)

    plt.show()

