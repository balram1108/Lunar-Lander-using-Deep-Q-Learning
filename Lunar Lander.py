import gymnasium as gym
import torch
import torch.nn as nn


# -----------------------------
# 1. Create the DQN neural network
# -----------------------------

class DQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.network(x)


# Create one DQN object
dqn = DQN()


# -----------------------------
# 2. Create Lunar Lander
# -----------------------------

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset()


# -----------------------------
# 3. Run the environment
# -----------------------------

for step_number in range(500):

    # Convert observation from NumPy array to PyTorch tensor
    state = torch.tensor(observation, dtype=torch.float32)

    # Pass the 8 observation values through the neural network
    q_values = dqn(state)

    # Choose the action with the highest Q-value
    action = torch.argmax(q_values).item()

    # Take that action in Lunar Lander
    observation, reward, terminated, truncated, info = env.step(action)

    print("Step:", step_number)
    print("Action:", action)
    print("Q-values:", q_values)
    print("Observation:", observation)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("----------------------")

    if terminated or truncated:
        print("Episode finished.")
        break


env.close()