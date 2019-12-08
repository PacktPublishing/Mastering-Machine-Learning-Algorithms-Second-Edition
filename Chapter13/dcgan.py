import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf


# Set random seed for reproducibility
np.random.seed(1000)
tf.random.set_seed(1000)


nb_samples = 5000
code_length = 100
nb_epochs = 100
batch_size = 128


generator = tf.keras.models.Sequential([
    tf.keras.layers.Conv2DTranspose(
        input_shape=(1, 1, code_length),
        filters=1024,
        kernel_size=(4, 4),
        padding='valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Conv2DTranspose(
        filters=512,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Conv2DTranspose(
        filters=256,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        activation='tanh')
])


discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        input_shape=(64, 64, 1),
        filters=128,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same'),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Conv2D(
        filters=1024,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(4, 4),
        padding='valid')
])


optimizer_generator = \
    tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
optimizer_discriminator = \
    tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

train_loss_generator = \
    tf.keras.metrics.Mean(name='train_loss')
train_loss_discriminator = \
    tf.keras.metrics.Mean(name='train_loss')


def run_generator(z, training=False):
    zg = tf.reshape(z, (-1, 1, 1, code_length))
    return generator(zg, training=training)


def run_discriminator(x, training=False):
    xd = tf.image.resize(x, (64, 64))
    return discriminator(xd, training=training)


@tf.function
def train(xi):
    zn = tf.random.uniform(
        (batch_size, code_length), -1.0, 1.0)

    with tf.GradientTape() as tape_generator, \
            tf.GradientTape() as tape_discriminator:
        xg = run_generator(zn, training=True)
        zd1 = run_discriminator(xi, training=True)
        zd2 = run_discriminator(xg, training=True)

        loss_d1 = tf.keras.losses.\
            BinaryCrossentropy(from_logits=True)\
            (tf.ones_like(zd1), zd1)
        loss_d2 = tf.keras.losses.\
            BinaryCrossentropy(from_logits=True)\
            (tf.zeros_like(zd2), zd2)
        loss_discriminator = loss_d1 + loss_d2

        loss_generator = tf.keras.losses.\
            BinaryCrossentropy(from_logits=True)\
            (tf.ones_like(zd2), zd2)

    gradients_generator = \
        tape_generator.gradient(
        loss_generator,
        generator.trainable_variables)
    gradients_discriminator = \
        tape_discriminator.gradient(
        loss_discriminator,
        discriminator.trainable_variables)

    optimizer_discriminator.apply_gradients(
        zip(gradients_discriminator,
            discriminator.trainable_variables))
    optimizer_generator.apply_gradients(
        zip(gradients_generator,
            generator.trainable_variables))

    train_loss_discriminator(loss_discriminator)
    train_loss_generator(loss_generator)


if __name__ == '__main__':
    # Load the dataset
    (X_train, _), (_, _) = \
        tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train.astype(np.float32)[0:nb_samples] / 255.0
    X_train = (2.0 * X_train) - 1.0

    width = X_train.shape[1]
    height = X_train.shape[2]

    # Train the model
    x_train_g = tf.data.Dataset.from_tensor_slices(
        np.expand_dims(X_train, axis=3)).\
        shuffle(1000).batch(batch_size)

    for e in range(nb_epochs):
        for xi in x_train_g:
            train(xi)

        print("Epoch {}: "
              "Discriminator Loss: {:.3f}, "
              "Generator Loss: {:.3f}".
              format(e + 1,
                     train_loss_discriminator.result(),
                     train_loss_generator.result()))

        train_loss_discriminator.reset_states()
        train_loss_generator.reset_states()


    # Show some results
    Z = np.random.uniform(-1.0, 1.0,
                          size=(50, code_length)).\
        astype(np.float32)
    Ys = run_generator(Z, training=False)
    Ys = np.squeeze((Ys + 1.0) * 0.5 * 255.0).\
        astype(np.uint8)

    sns.set()
    fig, ax = plt.subplots(5, 10, figsize=(15, 8))

    fig, ax = plt.subplots(5, 10, figsize=(22, 8))

    for i in range(5):
        for j in range(10):
            ax[i, j].imshow(Ys[(i * 10) + j], cmap='gray')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    plt.show()


