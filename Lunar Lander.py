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
# 2. Replay Buffer
# -----------------------------------

replay_buffer = deque(maxlen=10000)


# -----------------------------------
# 3. Discount factor
# -----------------------------------

gamma = 0.99


# -----------------------------------
# 4. Create Lunar Lander
# -----------------------------------

env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

state, info = env.reset()


# -----------------------------------
# 5. Run one episode
# -----------------------------------

for step_number in range(500):

    # -------------------------------
    # CURRENT STATE
    # -------------------------------

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    )


    # -------------------------------
    # GET CURRENT Q-VALUES
    # -------------------------------

    q_values = dqn(state_tensor)


    # -------------------------------
    # CHOOSE ACTION
    # -------------------------------

    action = torch.argmax(q_values).item()


    # -------------------------------
    # TAKE ACTION
    # -------------------------------

    next_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated


    # -------------------------------
    # SAVE EXPERIENCE
    # -------------------------------

    experience = (
        state,
        action,
        reward,
        next_state,
        done
    )

    replay_buffer.append(experience)


    # ===================================
    # 6. CALCULATE THE TARGET
    # ===================================

    next_state_tensor = torch.tensor(
        next_state,
        dtype=torch.float32
    )


    # We do not want to update weights yet.
    # We only want the next state's Q-values.
    with torch.no_grad():

        next_q_values = dqn(next_state_tensor)

        best_next_q_value = torch.max(next_q_values).item()


    # If episode ended, there is no future reward
    if done:

        target = reward

    else:

        target = reward + gamma * best_next_q_value


    # -----------------------------------
    # PRINT EVERYTHING
    # -----------------------------------

    print("Step:", step_number)

    print("State:")
    print(state)

    print("Q-values:")
    print(q_values)

    print("Action:")
    print(action)

    print("Reward:")
    print(reward)

    print("Next State:")
    print(next_state)

    print("Next Q-values:")
    print(next_q_values)

    print("Best Next Q-value:")
    print(best_next_q_value)

    print("TARGET:")
    print(target)

    print("Done:")
    print(done)

    print("------------------------")


    if done:
        print("Episode finished.")
        break


    # next state becomes current state
    state = next_state


env.close()