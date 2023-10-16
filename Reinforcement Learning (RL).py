!pip install gym

import numpy as np
import gym
import random

env = gym.make("FrozenLake-v1")
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

learning_rate = 0.1
discount_factor = 0.99
num_episodes = 10000
max_steps_per_episode = 100
exploration_prob = 1.0
min_exploration_prob = 0.01
exploration_decay = 0.995

# Define Q-learning function
def q_learning():
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        for step in range(max_steps_per_episode):
            if random.uniform(0, 1) < exploration_prob:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)

            # Q-value update
            Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

            total_reward += reward
            state = next_state

            if done:
                break

        # Exploration probability decay
        exploration_prob = max(exploration_prob * exploration_decay, min_exploration_prob)

# Define testing function
def test_agent():
    num_test_episodes = 100
    total_test_rewards = 0

    for episode in range(num_test_episodes):
        state = env.reset()
        done = False

        for step in range(max_steps_per_episode):
            action = np.argmax(Q[state, :])
            next_state, reward, done, _ = env.step(action)

            total_test_rewards += reward
            state = next_state

            if done:
                break

    average_test_reward = total_test_rewards / num_test_episodes
    print("Average test reward: ", average_test_reward)

# Training and testing the agent
print("Training the agent...")

print("Training complete!")

print("Testing the agent...")
test_agent()