import gymnasium as gym
import torch
import torch.nn as nn
import random

from collections import deque


# -----------------------------------
# 1. CREATE DQN NEURAL NETWORK
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
# 2. CREATE REPLAY BUFFER
# -----------------------------------

replay_buffer = deque(maxlen=10000)


# -----------------------------------
# 3. DISCOUNT FACTOR
# -----------------------------------

gamma = 0.99


# -----------------------------------
# 4. BATCH SIZE
# -----------------------------------

batch_size = 32


# -----------------------------------
# 5. CREATE LUNAR LANDER
# -----------------------------------

env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

state, info = env.reset()


# -----------------------------------
# 6. RUN ONE EPISODE
# -----------------------------------

for step_number in range(500):


    # --------------------------------
    # CURRENT STATE -> TENSOR
    # --------------------------------

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    )


    # --------------------------------
    # GET Q-VALUES
    # --------------------------------

    q_values = dqn(state_tensor)


    # --------------------------------
    # CHOOSE ACTION
    # --------------------------------

    action = torch.argmax(q_values).item()


    # --------------------------------
    # TAKE ACTION IN ENVIRONMENT
    # --------------------------------

    next_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated


    # --------------------------------
    # STORE EXPERIENCE
    # --------------------------------

    experience = (
        state,
        action,
        reward,
        next_state,
        done
    )

    replay_buffer.append(experience)


    # =========================================
    # 7. SAMPLE RANDOM EXPERIENCES
    # =========================================

    if len(replay_buffer) >= batch_size:

        batch = random.sample(
            replay_buffer,
            batch_size
        )


        # =====================================
        # 8. SPLIT BATCH INTO 5 GROUPS
        # =====================================

        states, actions, rewards, next_states, dones = zip(*batch)


        print("BATCH STATES:")
        print(states)

        print("BATCH ACTIONS:")
        print(actions)

        print("BATCH REWARDS:")
        print(rewards)

        print("BATCH NEXT STATES:")
        print(next_states)

        print("BATCH DONES:")
        print(dones)


    # --------------------------------
    # CALCULATE TARGET FOR CURRENT STEP
    # --------------------------------

    next_state_tensor = torch.tensor(
        next_state,
        dtype=torch.float32
    )


    with torch.no_grad():

        next_q_values = dqn(next_state_tensor)

        best_next_q_value = torch.max(
            next_q_values
        ).item()


    if done:

        target = reward

    else:

        target = reward + gamma * best_next_q_value


    # --------------------------------
    # PRINT CURRENT STEP
    # --------------------------------

    print("Step:")
    print(step_number)

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

    print("Target:")
    print(target)

    print("Replay Buffer Size:")
    print(len(replay_buffer))

    print("------------------------")


    # --------------------------------
    # END EPISODE
    # --------------------------------

    if done:

        print("Episode finished.")
        break


    # --------------------------------
    # NEXT STATE BECOMES CURRENT STATE
    # --------------------------------

    state = next_state


env.close()