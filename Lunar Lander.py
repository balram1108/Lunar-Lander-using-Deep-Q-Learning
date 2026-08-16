import gymnasium as gym
import torch
import torch.nn as nn


# -------------------------
# Neural Network
# -------------------------

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


# -------------------------
# Lunar Lander
# -------------------------

env = gym.make("LunarLander-v3", render_mode="human")

# Starting state
state, info = env.reset()


# -------------------------
# Run one episode
# -------------------------

for step_number in range(500):

    # 1. Convert current state so PyTorch can use it
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # 2. Neural network gives 4 Q-values
    q_values = dqn(state_tensor)

    # 3. Choose the action with the highest Q-value
    action = torch.argmax(q_values).item()

    # 4. Perform the action
    # Gymnasium gives us:
    # next state + reward
    next_state, reward, terminated, truncated, info = env.step(action)

    # 5. Check whether the episode finished
    done = terminated or truncated


    # 6. This is ONE complete experience
    transition = (
        state,
        action,
        reward,
        next_state,
        done
    )


    # Show what happened
    print("STATE:", state)
    print("ACTION:", action)
    print("REWARD:", reward)
    print("NEXT STATE:", next_state)
    print("DONE:", done)
    print("------------------------")


    if done:
        break


    # 7. Move forward:
    # next_state now becomes our new current state
    state = next_state


env.close()