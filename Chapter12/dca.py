import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 1000
nb_epochs = 400
batch_size = 200
code_length = 256


class DAC(tf.keras.Model):
    def __init__(self):
        super(DAC, self).__init__()

        # Encoder layers
        self.c1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=tf.keras.activations.relu,
            padding='same')

        self.c2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation=tf.keras.activations.relu,
            padding='same')

        self.c3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation=tf.keras.activations.relu,
            padding='same')

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(
            units=code_length,
            activation=tf.keras.activations.sigmoid)

        # Decoder layers
        self.dc0 = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=tf.keras.activations.relu,
            padding='same')

        self.dc1 = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=(3, 3),
            activation=tf.keras.activations.relu,
            padding='same')

        self.dc2 = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=(3, 3),
            activation=tf.keras.activations.relu,
            padding='same')

        self.dc3 = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(3, 3),
            activation=tf.keras.activations.sigmoid,
            padding='same')

    def r_images(self, x):
        return tf.image.resize(x, (32, 32))

    def encoder(self, x):
        c1 = self.c1(self.r_images(x))
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        code_input = self.flatten(c3)
        z = self.dense(code_input)
        return z

    def decoder(self, z):
        decoder_input = tf.reshape(z, (-1, 16, 16, 1))
        dc0 = self.dc0(decoder_input)
        dc1 = self.dc1(dc0)
        dc2 = self.dc2(dc1)
        dc3 = self.dc3(dc2)
        return dc3

    def call(self, x):
        code = self.encoder(x)
        xhat = self.decoder(code)
        return xhat


# Create the model
model = DAC()

# Define the optimizer and the train loss function
optimizer = tf.keras.optimizers.Adam(0.001)
train_loss = tf.keras.metrics.Mean(name='train_loss')


@tf.function
def train(images):
    with tf.GradientTape() as tape:
        reconstructions = model(images)
        loss = tf.keras.losses.MSE(
            model.r_images(images), reconstructions)
    gradients = tape.gradient(
        loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables))
    train_loss(loss)


if __name__ == '__main__':
    # Load the dataset
    (X_train, _), (_, _) = \
        tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train.astype(np.float32)[0:nb_samples] \
              / 255.0

    width = X_train.shape[1]
    height = X_train.shape[2]

    X_train_g = tf.data.Dataset.\
        from_tensor_slices(np.expand_dims(X_train, axis=3)).\
        shuffle(1000).batch(batch_size)

    # Train the model
    for e in range(nb_epochs):
        for xi in X_train_g:
            train(xi)
        print("Epoch {}: Loss: {:.3f}".
              format(e + 1, train_loss.result()))
        train_loss.reset_states()

    # Compute the mean of the codes
    codes = model.encoder(np.expand_dims(X_train, axis=3))
    print("Code mean: {:.3f}".format(np.mean(codes)))
    print("Code STD: {:.3f}".format(np.std(codes)))

    # Show some examples
    Xs = np.reshape(X_train[0:batch_size],
                    (batch_size, width, height, 1))
    Ys = model(Xs)
    Ys = np.squeeze(Ys * 255.0)

    # Show the results
    sns.set()

    fig, ax = plt.subplots(2, 10, figsize=(18, 4))

    for i in range(10):
        ax[0, i].imshow(np.squeeze(Xs[i]), cmap='gray')
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])

        ax[1, i].imshow(Ys[i + 10], cmap='gray')
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])

    plt.show()


