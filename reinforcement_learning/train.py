import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from reinforcement_learning.dqn import DQN

def train_fn(env):
    """
    Train a DQN to play Flappy Bird.
    """
    # Hyperparameters
    num_episodes = 1000
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Exploration probability
    epsilon_decay = 0.999
    min_epsilon = 0.01
    learning_rate = 1e-3
    batch_size = 64
    replay_memory_size = 100000

    # Initialize replay memory
    replay_memory = deque(maxlen=replay_memory_size)

    # Initialize DQN and optimizer
    input_size = env._get_state().shape[0]
    action_size = 2  # Flap or no flap
    dqn = DQN(input_size, action_size)
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            # Choose action
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)  # Random action
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = dqn(state_tensor)
                    action = torch.argmax(q_values).item()

            # Take action and observe reward
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Store experience in replay memory
            replay_memory.append((state, action, reward, next_state, done))

            state = next_state

            # Perform training
            if len(replay_memory) >= batch_size:
                batch = random.sample(replay_memory, batch_size)
                train_batch(dqn, optimizer, loss_fn, batch, gamma)

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    # Save the trained model
    torch.save(dqn.state_dict(), "flappy_bird_dqn.pth")

def train_batch(dqn, optimizer, loss_fn, batch, gamma):
    """
    Train DQN on a single batch of experience.
    """
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Compute current Q values
    q_values = dqn(states).gather(1, actions).squeeze()

    # Compute target Q values
    with torch.no_grad():
        next_q_values = dqn(next_states).max(1)[0]
        targets = rewards + (1 - dones) * gamma * next_q_values

    # Compute loss
    loss = loss_fn(q_values, targets)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
