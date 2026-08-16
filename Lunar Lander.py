import gymnasium as gym
import torch
import torch.nn as nn

from collections import deque


# -----------------------------------
# 1. Create DQN neural network
# -----------------------------------

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


dqn = DQN()


# -----------------------------------
# 2. Create Replay Buffer
# -----------------------------------

replay_buffer = deque(maxlen=10000)


# -----------------------------------
# 3. Create Lunar Lander
# -----------------------------------

env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

state, info = env.reset()


# -----------------------------------
# 4. Run one episode
# -----------------------------------

for step_number in range(500):

    # Convert current state to PyTorch tensor
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    )


    # Neural network predicts 4 Q-values
    q_values = dqn(state_tensor)


    # Choose action with highest Q-value
    action = torch.argmax(q_values).item()


    # Take action in environment
    next_state, reward, terminated, truncated, info = env.step(action)


    # Check if episode ended
    done = terminated or truncated


    # -----------------------------------
    # 5. Save experience in Replay Buffer
    # -----------------------------------

    experience = (
        state,
        action,
        reward,
        next_state,
        done
    )

    replay_buffer.append(experience)


    # -----------------------------------
    # 6. Print what happened
    # -----------------------------------

    print("Step:", step_number)

    print("State:")
    print(state)

    print("Action:")
    print(action)

    print("Reward:")
    print(reward)

    print("Next State:")
    print(next_state)

    print("Done:")
    print(done)

    print("Replay Buffer Size:")
    print(len(replay_buffer))

    print("------------------------")


    # Stop if episode finished
    if done:
        print("Episode finished.")
        break


    # Move to next state
    state = next_state


env.close()