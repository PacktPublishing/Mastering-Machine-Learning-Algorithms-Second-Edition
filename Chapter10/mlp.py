import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# To install Keras: pip install -U tensorflow (or tensorflow-gpu)
# Further information: https://www.tensorflow.org/
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 1000
nsb = int(nb_samples / 4)


if __name__ == '__main__':
    # Create dataset
    X = np.zeros((nb_samples, 2))
    Y = np.zeros((nb_samples,))

    X[0:nsb, :] = np.random.multivariate_normal([1.0, -1.0], np.diag([0.1, 0.1]), size=nsb)
    Y[0:nsb] = 0.0

    X[nsb:(2 * nsb), :] = np.random.multivariate_normal([1.0, 1.0], np.diag([0.1, 0.1]), size=nsb)
    Y[nsb:(2 * nsb)] = 1.0

    X[(2 * nsb):(3 * nsb), :] = np.random.multivariate_normal([-1.0, 1.0], np.diag([0.1, 0.1]), size=nsb)
    Y[(2 * nsb):(3 * nsb)] = 0.0

    X[(3 * nsb):, :] = np.random.multivariate_normal([-1.0, -1.0], np.diag([0.1, 0.1]), size=nsb)
    Y[(3 * nsb):] = 1.0

    ss = StandardScaler()
    X = ss.fit_transform(X)

    X, Y = shuffle(X, Y, random_state=1000)

    # Create an MLP
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, input_dim=2,
                              activation='tanh'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Create train and test sets
    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size=0.3,
                         random_state=1000)

    # Train the model
    model.fit(X_train,
              tf.keras.utils.to_categorical(
                  Y_train, num_classes=2),
              epochs=100,
              batch_size=32,
              validation_data=
              (X_test,
               tf.keras.utils.to_categorical(
                   Y_test, num_classes=2)))

    # Plot the classification result
    Y_pred = model.predict(X)
    Y_pred_mlp = np.argmax(Y_pred, axis=1)

    sns.set()
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(X[Y_pred_mlp == 0, 0], X[Y_pred_mlp == 0, 1], marker='o', s=100, label="Class 0")
    ax.scatter(X[Y_pred_mlp == 1, 0], X[Y_pred_mlp == 1, 1], marker='s', s=100, label="Class 1")
    ax.set_xlabel(r'$x_0$', fontsize=22)
    ax.set_ylabel(r'$x_1$', fontsize=22)
    ax.grid(True)
    ax.legend(fontsize=20)
    plt.show()

