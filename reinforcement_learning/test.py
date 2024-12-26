import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from reinforcement_learning.dqn import DQN
def test_fn(env):
    """
    Test a trained DQN on Flappy Bird.
    """
    # Load the trained model
    input_size = env._get_state().shape[0]
    action_size = 2
    dqn = DQN(input_size, action_size)
    dqn.load_state_dict(torch.load("flappy_bird_dqn.pth"))
    print("Model loaded successfully!")
    dqn.eval()

    for episode in range(2):  # Number of test episodes
        print(f"Starting Test Episode {episode + 1}")
        state = env.reset()
        total_reward = 0

        while True:
            print(f"State: {state}, Total Reward: {total_reward}")
            env.render()

            # Choose the best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = dqn(state_tensor)
                action = torch.argmax(q_values).item()
            print(f"Action taken: {action}")

            # Take action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done:
                print(f"Test Episode {episode + 1}, Total Reward: {total_reward}")
                break

    env.close()
