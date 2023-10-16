!pip install gym
!pip install tensorflow

import numpy as np
import gym

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Environment setup
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# Hyperparameters
learning_rate = 0.01
gamma = 0.99
num_episodes = 1000

# Neural network for policy
model = tf.keras.Sequential([
    Dense(24, input_dim=state_dim, activation='relu'),
    Dense(24, activation='relu'),
    Dense(num_actions, activation='softmax')
])

optimizer = Adam(learning_rate)

def get_action(state):
    state = state.reshape([1, state_dim])
    policy = model.predict(state, batch_size=1).flatten()
    return np.random.choice(num_actions, 1, p=policy)

def compute_discounted_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    mean = np.mean(discounted_rewards)
    std = np.std(discounted_rewards)
    discounted_rewards = (discounted_rewards - mean) / (std + 1e-8)
    return discounted_rewards

episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_states, episode_actions, episode_rewards = [], [], []

    while not done:
        action = get_action(state)
        next_state, reward, done, _ = env.step(action[0])
        episode_states.append(state)
        episode_actions.append(action[0])
        episode_rewards.append(reward)
        state = next_state

    discounted_rewards = compute_discounted_rewards(episode_rewards)

    with tf.GradientTape() as tape:
        action_masks = tf.one_hot(episode_actions, num_actions)
        logits = model(np.vstack(episode_states))
        action_prob = tf.reduce_sum(action_masks * logits, axis=1)
        loss = -tf.reduce_sum(tf.math.log(action_prob) * discounted_rewards)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward_sum = sum(episode_rewards)
    episode_rewards.append(episode_reward_sum)
    print(f"Episode: {episode + 1}, Total Reward: {episode_reward_sum}")

# Display the average reward at the end
average_reward = np.mean(episode_rewards)
print(f"Average Reward: {average_reward}")

env.close()
