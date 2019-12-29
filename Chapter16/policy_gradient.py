import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gym
import tensorflow as tf

from scipy.signal import savgol_filter

# To force the usage of CPU
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Set random seed for reproducibility
np.random.seed(1000)
tf.random.set_seed(1000)


nb_episodes = 2000
max_length = 200
batch_size = 5
gamma = 0.99


# Create environment
env = gym.make('CartPole-v0')
env.seed(1000)

# Create policy
policy = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32,
                          activation='relu',
                          input_dim=4),
    tf.keras.layers.Dense(32,
                          activation='relu'),
    tf.keras.layers.Dense(2,
                          activation='relu')
])

optimizer = tf.keras.optimizers.Adam()


def policy_step(s, grads=False):
    with tf.GradientTape() as tape:
        actions = policy(s, training=True)
        action = tf.random.categorical(
            actions, 1)
        action = tf.keras.utils.to_categorical(action, 2)
        loss = tf.squeeze(
            tf.nn.softmax_cross_entropy_with_logits(
                action, actions))

    if grads:
        gradients = tape.gradient(
            loss, policy.trainable_variables)
        return np.argmax(action), gradients

    return np.argmax(action)


def create_gradients():
    gradients = policy.trainable_variables
    for i, g in enumerate(gradients):
        gradients[i] = 0
    return gradients


def discounted_rewards(r):
    dr = []
    da = 0.0
    for t in range(len(r)-1, -1, -1):
        da *= gamma
        da += r[t]
        dr.append(da)
    return dr[::-1]


if __name__ == "__main__":
    gradients = create_gradients()
    global_rewards = []

    for e in range(nb_episodes):
        state = env.reset()

        e_gradients = []
        e_rewards = []
        done = False
        total_reward = 0.0
        t = 0

        while not done and t < max_length:
            # Uncomment to show the GUI
            env.render()

            state = np.reshape(state, (1, 4)).\
                astype(np.float32)
            action, grads = policy_step(state,
                                        grads=True)
            state, reward, done, _ = env.step(action)

            total_reward += reward
            e_rewards.append(
                reward if not done else -5)

            grads = np.array(grads)
            e_gradients.append(grads)
            t += 1

        global_rewards.append(total_reward)

        d_rewards = discounted_rewards(e_rewards)
        for i, g in enumerate(e_gradients):
            gradients += g * d_rewards[i]

        if e > 1 and e % batch_size == 0:
            optimizer.apply_gradients(
                zip(gradients / batch_size,
                    policy.trainable_variables))
            gradients = create_gradients()

        print("Finished episode: {}. "
              "Total reward: {:.2f}".
              format(e + 1, total_reward))

    env.close()

    # Show the reward plot
    sns.set()
    fig, ax = plt.subplots(figsize=(22, 10))

    ax.plot(global_rewards, linewidth=0.35)
    ax.plot(savgol_filter(global_rewards, 51, 8), linewidth=2.5, c='r')
    ax.set_xlabel('Episode', fontsize=20)
    ax.set_ylabel('Total Reward', fontsize=20)
    ax.set_title('Total Rewards (t={})'.format(nb_episodes), fontsize=20)
    ax.grid(True)

    plt.show()