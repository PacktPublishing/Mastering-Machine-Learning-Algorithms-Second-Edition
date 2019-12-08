import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf


# Set random seed for reproducibility
np.random.seed(1000)
tf.random.set_seed(1000)


nb_samples = 10240
nb_epochs = 100
nb_critic = 5
batch_size = 64
code_length = 256


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


critic = tf.keras.models.Sequential([
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
    tf.keras.optimizers.Adam(
        0.00005, beta_1=0.5, beta_2=0.9)
optimizer_critic = \
    tf.keras.optimizers.Adam(
        0.00005, beta_1=0.5, beta_2=0.9)

train_loss_generator = \
    tf.keras.metrics.Mean(name='train_loss')
train_loss_critic = \
    tf.keras.metrics.Mean(name='train_loss')


def run_generator(z, training=False):
    zg = tf.reshape(z, (-1, 1, 1, code_length))
    return generator(zg, training=training)


def run_critic(x, training=False):
    xc = tf.image.resize(x, (64, 64))
    return critic(xc, training=training)


def run_model(xi, zn, training=True):
    xg = run_generator(zn, training=training)
    zc1 = run_critic(xi, training=training)
    zc2 = run_critic(xg, training=training)

    loss_critic = tf.reduce_mean(zc2 - zc1)
    loss_generator = tf.reduce_mean(-zc2)

    return loss_critic, loss_generator


@tf.function
def train_critic(xi):
    zn = tf.random.uniform(
        (batch_size, code_length), -1.0, 1.0)

    with tf.GradientTape() as tape:
        loss_critic, _ = run_model(xi, zn,
                                   training=True)

    gradients_critic = tape.gradient(
        loss_critic,
        critic.trainable_variables)
    optimizer_critic.apply_gradients(
        zip(gradients_critic,
            critic.trainable_variables))

    for v in critic.trainable_variables:
        v.assign(tf.clip_by_value(v, -0.01, 0.01))

    train_loss_critic(loss_critic)


@tf.function
def train_generator():
    zn = tf.random.uniform(
        (batch_size, code_length), -1.0, 1.0)
    xg = tf.zeros((batch_size, width, height, 1))

    with tf.GradientTape() as tape:
        _, loss_generator = run_model(xg, zn,
                                      training=True)

    gradients_generator = tape.gradient(
        loss_generator,
        generator.trainable_variables)
    optimizer_generator.apply_gradients(
        zip(gradients_generator,
            generator.trainable_variables))

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
    x_train = tf.data.Dataset.from_tensor_slices(
        np.expand_dims(X_train, axis=3)).\
        shuffle(1000).batch(nb_critic * batch_size)

    for e in range(nb_epochs):
        for xi in x_train:
            for i in range(nb_critic):
                train_critic(xi[i * batch_size:
                                (i + 1) * batch_size])

            train_generator()

        print("Epoch {}: "
              "Critic Loss: {:.3f}, "
              "Generator Loss: {:.3f}".
              format(e + 1,
                     train_loss_critic.result(),
                     train_loss_generator.result()))

        train_loss_critic.reset_states()
        train_loss_generator.reset_states()

    # Show some results
    Z = np.random.uniform(-1.0, 1.0, size=(50, code_length)).astype(np.float32)
    Ys = run_generator(Z, training=False)
    Ys = np.squeeze((Ys + 1.0) * 0.5 * 255.0).astype(np.uint8)

    sns.set()
    fig, ax = plt.subplots(5, 10, figsize=(15, 8))

    for i in range(5):
        for j in range(10):
            ax[i, j].imshow(Ys[(i * 10) + j], cmap='gray')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    plt.show()
