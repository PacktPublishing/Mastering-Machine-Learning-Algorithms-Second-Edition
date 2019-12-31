import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Install it using pip install hmmlearn
from hmmlearn import hmm

# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create a Multinomial HMM
    hmm_model = hmm.MultinomialHMM(n_components=2,
                                   n_iter=100,
                                   random_state=1000)

    # Define a list of observations
    observations = np.array([[0], [1], [1],
                             [0], [1], [1],
                             [1], [0], [1],
                             [0], [0], [0],
                             [1], [0], [1],
                             [1], [0], [1],
                             [0], [0], [1],
                             [0], [1], [0],
                             [0], [0], [1],
                             [0], [1], [0],
                             [1], [0], [0],
                             [0], [0], [0]],
                            dtype=np.int32)

    # Fit the model using the Forward-Backward algorithm
    hmm_model.fit(observations)

    # Check the convergence
    print('Converged: {}'.format(hmm_model.monitor_.converged))

    # Print the transition probability matrix
    print('\nTransition probability matrix:')
    print(hmm_model.transmat_)

    # Create a test sequence
    sequence = np.array([[1], [1], [1],
                         [0], [1], [1],
                         [1], [0], [1],
                         [0], [1], [0],
                         [1], [0], [1],
                         [1], [0], [1],
                         [1], [0], [1],
                         [0], [1], [0],
                         [1], [0], [1],
                         [1], [1], [0],
                         [0], [1], [1],
                         [0], [1], [1]],
                        dtype=np.int32)

    # Find the the most likely hidden states using the Viterbi algorithm
    lp, hs = hmm_model.decode(sequence)

    print('\nMost likely hidden state sequence:')
    print(hs)

    print('\nLog-propability:')
    print(lp)

    # Compute the posterior probabilities
    pp = hmm_model.predict_proba(sequence)

    print('\nPosterior probabilities:')
    print(pp)

    sns.set()
    fig, ax = plt.subplots(figsize=(22, 10))

    ax.plot(pp[:, 0], "o-", linewidth=3.0, label="On-time")
    ax.plot(pp[:, 1], "o-", linewidth=3.0, linestyle="dashed", label="Delayed")

    ax.set_xlabel("Time", fontsize=22)
    ax.set_ylabel("State", fontsize=22)
    ax.legend(fontsize=22)

    plt.show()

    # Repeat the prediction with a sequence of On-time (0) flights
    sequence0 = np.array([[0], [0], [0],
                          [0], [0], [0],
                          [0], [0], [0],
                          [0], [0], [0],
                          [0], [0], [0],
                          [0], [0], [0],
                          [0], [0], [0],
                          [0], [0], [0],
                          [0], [0], [0],
                          [0], [0], [0],
                          [0], [0], [0]],
                         dtype=np.int32)
    pp0 = hmm_model.predict_proba(sequence0)

    fig, ax = plt.subplots(figsize=(22, 10))

    ax.plot(pp0[:, 0], "o-", linewidth=3.0, label="On-time")
    ax.plot(pp0[:, 1], "o-", linewidth=3.0, linestyle="dashed", label="Delayed")

    ax.set_xlabel("Time", fontsize=22)
    ax.set_ylabel("State", fontsize=22)
    ax.legend(fontsize=22)

    plt.show()
