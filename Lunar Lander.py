import gymnasium as gym
import torch
import torch.nn as nn
import random
import numpy as np
import csv

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

optimizer = torch.optim.Adam(
    dqn.parameters(),
    lr=0.001
)


# -----------------------------------
# 3. CREATE REPLAY BUFFER
# -----------------------------------

replay_buffer = deque(
    maxlen=10000
)


# -----------------------------------
# 4. SETTINGS
# -----------------------------------

gamma = 0.99

batch_size = 32


# ===================================
# 5. EPSILON SETTINGS
# ===================================

# Start with lots of exploration
epsilon = 1.0

# Never go below 5% random actions
epsilon_min = 0.05

# Decrease epsilon after every episode
epsilon_decay = 0.995


# -----------------------------------
# 6. LOSS FUNCTION
# -----------------------------------

loss_function = nn.MSELoss()


# -----------------------------------
# 7. CREATE ENVIRONMENT
# -----------------------------------

env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)


# ===================================
# 8. CREATE CSV REPORT
# ===================================

report_file = open(
    "training_report_epsilon_decay.csv",
    "w",
    newline=""
)

csv_writer = csv.writer(
    report_file
)


csv_writer.writerow([

    "Episode",
    "Epsilon",
    "Total Reward",
    "Steps",
    "Final X",
    "Final Y",
    "Closest X To Center",
    "Left Leg",
    "Right Leg",
    "Final Reward",
    "Average Loss"

])


# ===================================
# 9. RUN MANY EPISODES
# ===================================

for episode in range(500):


    # --------------------------------
    # RESET ENVIRONMENT
    # --------------------------------

    state, info = env.reset()


    # --------------------------------
    # REPORT VARIABLES
    # --------------------------------

    episode_reward = 0

    episode_losses = []

    closest_x_to_center = abs(
        state[0]
    )

    final_reward = 0


    # Save epsilon used for THIS episode
    episode_epsilon = epsilon


    # =================================
    # 10. RUN STEPS
    # =================================

    for step_number in range(500):


        # -----------------------------
        # STATE -> TENSOR
        # -----------------------------

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )


        # -----------------------------
        # GET Q VALUES
        # -----------------------------

        q_values = dqn(
            state_tensor
        )


        # =================================
        # 11. EPSILON-GREEDY ACTION
        # =================================

        if random.random() < epsilon:

            # EXPLORE
            action = env.action_space.sample()

        else:

            # EXPLOIT
            action = torch.argmax(
                q_values
            ).item()


        # -----------------------------
        # TAKE ACTION
        # -----------------------------

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated


        # =================================
        # UPDATE REPORT INFORMATION
        # =================================

        episode_reward += reward

        final_reward = reward


        current_x_distance = abs(
            next_state[0]
        )


        if current_x_distance < closest_x_to_center:

            closest_x_to_center = current_x_distance


        # -----------------------------
        # STORE EXPERIENCE
        # -----------------------------

        experience = (

            state,
            action,
            reward,
            next_state,
            done

        )

        replay_buffer.append(
            experience
        )


        # =================================
        # 12. SAMPLE REPLAY BUFFER
        # =================================

        if len(replay_buffer) >= batch_size:

            batch = random.sample(
                replay_buffer,
                batch_size
            )


            # -----------------------------
            # SPLIT BATCH
            # -----------------------------

            states, actions, rewards, next_states, dones = zip(
                *batch
            )


            # -----------------------------
            # CONVERT TO TENSORS
            # -----------------------------

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


            # =================================
            # 13. PREDICTED Q VALUES
            # =================================

            all_q_values = dqn(
                states_tensor
            )


            predicted_q_values = all_q_values.gather(

                1,

                actions_tensor.unsqueeze(1)

            ).squeeze(1)


            # =================================
            # 14. TARGET Q VALUES
            # =================================

            with torch.no_grad():

                next_q_values = dqn(
                    next_states_tensor
                )


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


            # =================================
            # 15. LOSS
            # =================================

            loss = loss_function(

                predicted_q_values,

                targets

            )


            episode_losses.append(
                loss.item()
            )


            # =================================
            # 16. BACKPROPAGATION
            # =================================

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


        # =================================
        # END EPISODE
        # =================================

        if done:

            break


        # -----------------------------
        # NEXT STATE BECOMES STATE
        # -----------------------------

        state = next_state


    # ===================================
    # 17. EPISODE REPORT
    # ===================================

    final_x = next_state[0]

    final_y = next_state[1]

    left_leg = next_state[6]

    right_leg = next_state[7]


    if len(episode_losses) > 0:

        average_loss = (
            sum(episode_losses)
            /
            len(episode_losses)
        )

    else:

        average_loss = 0


    # ===================================
    # 18. PRINT REPORT
    # ===================================

    print()

    print("==============================")

    print("EPISODE REPORT:", episode)

    print("==============================")


    print("Epsilon:")
    print(episode_epsilon)


    print("Total Reward:")
    print(episode_reward)


    print("Steps:")
    print(step_number + 1)


    print("Final X:")
    print(final_x)


    print("Final Y:")
    print(final_y)


    print("Closest X To Center:")
    print(closest_x_to_center)


    print("Left Leg:")
    print(left_leg)


    print("Right Leg:")
    print(right_leg)


    print("Final Reward:")
    print(final_reward)


    print("Average Loss:")
    print(average_loss)


    print("==============================")

    print()


    # ===================================
    # 19. SAVE REPORT TO CSV
    # ===================================

    csv_writer.writerow([

        episode,

        episode_epsilon,

        episode_reward,

        step_number + 1,

        final_x,

        final_y,

        closest_x_to_center,

        left_leg,

        right_leg,

        final_reward,

        average_loss

    ])


    report_file.flush()


    # ===================================
    # 20. DECREASE EPSILON
    # ===================================

    epsilon = max(

        epsilon_min,

        epsilon * epsilon_decay

    )


# -----------------------------------
# CLOSE EVERYTHING
# -----------------------------------

report_file.close()

env.close()