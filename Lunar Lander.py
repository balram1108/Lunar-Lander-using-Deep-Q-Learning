import gymnasium as gym
import torch
import torch.nn as nn
import random
import numpy as np

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
# 2. CREATE OPTIMIZER
# -----------------------------------
# NEW
# Adam will update the weights

optimizer = torch.optim.Adam(
    dqn.parameters(),
    lr=0.001
)


# -----------------------------------
# 3. CREATE REPLAY BUFFER
# -----------------------------------

replay_buffer = deque(maxlen=10000)


# -----------------------------------
# 4. DISCOUNT FACTOR
# -----------------------------------

gamma = 0.99


# -----------------------------------
# 5. BATCH SIZE
# -----------------------------------

batch_size = 32


# -----------------------------------
# 6. LOSS FUNCTION
# -----------------------------------

loss_function = nn.MSELoss()


# -----------------------------------
# 7. CREATE LUNAR LANDER
# -----------------------------------

env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

state, info = env.reset()


# -----------------------------------
# 8. RUN ONE EPISODE
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
    # STATE -> NEURAL NETWORK
    # --------------------------------

    q_values = dqn(state_tensor)


    # --------------------------------
    # CHOOSE HIGHEST Q-VALUE ACTION
    # --------------------------------

    action = torch.argmax(q_values).item()


    # --------------------------------
    # TAKE ACTION
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
    # 9. SAMPLE FROM REPLAY BUFFER
    # =========================================

    if len(replay_buffer) >= batch_size:

        batch = random.sample(
            replay_buffer,
            batch_size
        )


        # -----------------------------------
        # SPLIT BATCH
        # -----------------------------------

        states, actions, rewards, next_states, dones = zip(*batch)


        # -----------------------------------
        # CONVERT EVERYTHING TO TENSORS
        # -----------------------------------

        states_tensor = torch.tensor(
            np.array(states),
            dtype=torch.float32
        )

        actions_tensor = torch.tensor(
            actions,
            dtype=torch.long
        )

        rewards_tensor = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states_tensor = torch.tensor(
            np.array(next_states),
            dtype=torch.float32
        )

        dones_tensor = torch.tensor(
            dones,
            dtype=torch.float32
        )


        # ===================================
        # 10. PREDICT Q-VALUES
        # ===================================

        all_q_values = dqn(states_tensor)


        # Get Q-value for the action
        # that was actually taken

        predicted_q_values = all_q_values.gather(
            1,
            actions_tensor.unsqueeze(1)
        ).squeeze(1)


        # ===================================
        # 11. CALCULATE TARGET
        # ===================================

        with torch.no_grad():

            next_q_values = dqn(next_states_tensor)

            best_next_q_values = next_q_values.max(
                dim=1
            ).values


            targets = (
                rewards_tensor
                +
                gamma
                * best_next_q_values
                * (1 - dones_tensor)
            )


        # ===================================
        # 12. CALCULATE LOSS
        # ===================================

        loss = loss_function(
            predicted_q_values,
            targets
        )


        # ===================================
        # 13. BACKPROPAGATION
        # ===================================

        # NEW
        # Clear old gradients
        optimizer.zero_grad()


        # NEW
        # Calculate gradients
        loss.backward()


        # NEW
        # Update the weights
        optimizer.step()


        # -----------------------------------
        # PRINT TRAINING INFORMATION
        # -----------------------------------

        print("Predicted Q-values:")
        print(predicted_q_values)

        print("Targets:")
        print(targets)

        print("Loss:")
        print(loss.item())

        print("WEIGHTS UPDATED")


    # -----------------------------------
    # PRINT CURRENT STEP
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

    print("Replay Buffer Size:")
    print(len(replay_buffer))

    print("------------------------")


    # -----------------------------------
    # END EPISODE
    # -----------------------------------

    if done:

        print("Episode finished.")
        break


    # -----------------------------------
    # NEXT STATE BECOMES CURRENT STATE
    # -----------------------------------

    state = next_state


env.close()