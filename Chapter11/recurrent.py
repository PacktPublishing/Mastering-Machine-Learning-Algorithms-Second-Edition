import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler


# Set random seed for reproducibility
np.random.seed(1000)

# Download the dataset from: http://sidc.be/silso/infossntotmonthly
dataset_filename = '<FILENAME>'

n_samples = 3175
sequence_length = 15


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv(dataset_filename, header=None).dropna()
    data =  df[3].values[:n_samples - sequence_length].\
        astype(np.float32)

    # Scale the dataset between -1 and 1
    mmscaler = MinMaxScaler((-1.0, 1.0))
    data = mmscaler.fit_transform(data.reshape(-1, 1))

    # Show the dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(data)
    ax.grid(True)
    ax.set_xlabel('Time steps', fontsize=20)
    ax.set_ylabel('Monthly sunspots numbers', fontsize=20)
    plt.show()

    # Create the train and test sets (rounding to 2800 samples)
    X_ts = np.zeros(shape=(n_samples - sequence_length,
                           sequence_length, 1),
                    dtype=np.float32)
    Y_ts = np.zeros(shape=(n_samples - sequence_length, 1),
                    dtype=np.float32)

    for i in range(0, data.shape[0] - sequence_length):
        X_ts[i] = data[i:i + sequence_length]
        Y_ts[i] = data[i + sequence_length]

    X_ts_train = X_ts[0:2600, :]
    Y_ts_train = Y_ts[0:2600]

    X_ts_test = X_ts[2600:n_samples, :]
    Y_ts_test = Y_ts[2600:n_samples]

    # Create the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(4,
                             stateful=True,
                             batch_input_shape=
                             (20, sequence_length, 1)),
        tf.keras.layers.Dense(1,
                              activation='tanh')
        ])

    # Compile the model
    model.compile(optimizer=
                  tf.keras.optimizers.Adam(
                      lr=0.001, decay=0.0001),
                  loss='mse',
                  metrics=['mse'])

    # Train the model
    model.fit(X_ts_train, Y_ts_train,
              batch_size=20,
              epochs=100,
              shuffle=False,
              validation_data=(X_ts_test, Y_ts_test))

    # Show the result
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(Y_ts_test, label='True values')
    ax.plot(model.predict(X_ts_test, batch_size=20), label='Predicted values')
    ax.grid(True)
    ax.set_xlabel('Time steps', fontsize=20)
    ax.set_ylabel('Monthly sunspots numbers',fontsize=20)
    ax.legend(fontsize=20)
    plt.show()





