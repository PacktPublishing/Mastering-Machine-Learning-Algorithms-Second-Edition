import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf


# Set random seed for reproducibility
np.random.seed(1000)


nb_classes = 10
train_batch_size = 256
test_batch_size = 100
nb_epochs = 100
steps_per_epoch = 1500


if __name__ == '__main__':
    # Load the dataset
    (X_train, Y_train), (X_test, Y_test) = \
        tf.keras.datasets.fashion_mnist.load_data()

    # Create the augmented data generators
    train_idg = tf.keras.preprocessing.image.\
        ImageDataGenerator(
        rescale=1.0 / 255.0,
        samplewise_center=True,
        samplewise_std_normalization=True,
        horizontal_flip=True,
        rotation_range=10.0,
        shear_range=np.pi / 12.0,
        zoom_range=0.25)

    train_dg = train_idg.flow(
        x=np.expand_dims(X_train, axis=3),
        y=tf.keras.utils.to_categorical(
            Y_train, num_classes=nb_classes),
        batch_size=train_batch_size,
        shuffle=True,
        seed=1000)

    test_idg = tf.keras.preprocessing.image.\
        ImageDataGenerator(
        rescale=1.0 / 255.0,
        samplewise_center=True,
        samplewise_std_normalization=True)

    test_dg = train_idg.flow(
        x=np.expand_dims(X_test, axis=3),
        y=tf.keras.utils.to_categorical(
            Y_test, num_classes=nb_classes),
        shuffle=False,
        batch_size=test_batch_size,
        seed=1000)

    # Create the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,
                               kernel_size=(3, 3),
                               padding='same',
                               input_shape=(X_train.shape[1],
                                            X_train.shape[2],
                                            1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(64,
                               kernel_size=(3, 3),
                               padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(128,
                               kernel_size=(3, 3),
                               padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(128,
                               kernel_size=(3, 3),
                               padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1024),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Dense(1024),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Dense(nb_classes,
                              activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(
                      lr=0.0001, decay=1e-5),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit_generator(
        generator=train_dg,
        epochs=nb_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dg,
        validation_steps=int(X_test.shape[0] /
                             test_batch_size),
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.1, patience=1,
                cooldown=1, min_lr=1e-6)
        ])

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

    