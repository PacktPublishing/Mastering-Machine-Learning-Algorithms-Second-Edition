import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    (X_train, Y_train), (X_test, Y_test) = \
        tf.keras.datasets.mnist.load_data()

    width = height = X_train.shape[1]

    X_train = X_train.reshape(
        (X_train.shape[0], width, height, 1)).\
                  astype(np.float32) / 255.0
    X_test = X_test.reshape(
        (X_test.shape[0], width, height, 1)).\
                 astype(np.float32) / 255.0

    Y_train = tf.keras.utils.to_categorical(
        Y_train, num_classes=10)
    Y_test = tf.keras.utils.to_categorical(
        Y_test, num_classes=10)

    # Create the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dropout(0.25,
                                input_shape=(width, height, 1),
                                seed=1000),

        tf.keras.layers.Conv2D(16,
                               kernel_size=(3, 3),
                               padding='same',
                               activation='relu'),
        tf.keras.layers.Dropout(0.5, seed=1000),

        tf.keras.layers.Conv2D(32,
                               kernel_size=(3, 3),
                               padding='same',
                               activation='relu'),
        tf.keras.layers.Dropout(0.5, seed=1000),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                         padding='same'),

        tf.keras.layers.Conv2D(64,
                               kernel_size=(3, 3),
                               padding='same',
                               activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                         padding='same'),

        tf.keras.layers.Conv2D(64,
                               kernel_size=(3, 3),
                               padding='same',
                               activation='relu'),
        tf.keras.layers.Dropout(0.5, seed=1000),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                         padding='same'),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1024,
                              activation='relu'),
        tf.keras.layers.Dropout(0.5, seed=1000),

        tf.keras.layers.Dense(10,
                              activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        epochs=200,
                        batch_size=256,
                        validation_data=(X_test, Y_test))

    # Show the results
    sns.set()

    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    ax[0].plot(history.history['accuracy'], label='Training accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation accuracy')
    ax[0].set_xlabel('Epoch', fontsize=20)
    ax[0].set_ylabel('Accuracy', fontsize=20)
    ax[0].legend(fontsize=20)
    ax[0].grid(True)

    ax[1].plot(history.history['loss'], label='Training loss')
    ax[1].plot(history.history['val_loss'], label='Validation loss')
    ax[1].set_xlabel('Epoch', fontsize=20)
    ax[1].set_ylabel('Loss', fontsize=20)
    ax[1].set_yticks(np.linspace(0.0, 1.0, 10))
    ax[1].legend(fontsize=20)
    ax[1].grid(True)
    plt.show()