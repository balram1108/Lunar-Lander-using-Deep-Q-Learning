import gymnasium as gym
import torch
import torch.nn as nn
import random
import numpy as np
import csv

from collections import deque


# ===================================
# 1. CREATE DQN NEURAL NETWORK
# ===================================

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


# ===================================
# 2. CREATE MAIN DQN
# ===================================

dqn = DQN()


# ===================================
# 3. CREATE TARGET DQN
# ===================================

target_dqn = DQN()

# Start both networks with the SAME weights
target_dqn.load_state_dict(
    dqn.state_dict()
)

# Target network is only used for targets
target_dqn.eval()


# ===================================
# 4. CREATE OPTIMIZER
# ===================================

optimizer = torch.optim.Adam(
    dqn.parameters(),
    lr=0.001
)


# ===================================
# 5. CREATE REPLAY BUFFER
# ===================================

replay_buffer = deque(
    maxlen=10000
)


# ===================================
# 6. SETTINGS
# ===================================

gamma = 0.99

batch_size = 32


# ===================================
# 7. EPSILON SETTINGS
# ===================================

epsilon = 1.0

epsilon_min = 0.05

epsilon_decay = 0.995


# ===================================
# 8. TARGET NETWORK SETTINGS
# ===================================

# Copy main network weights into
# target network every 1000 training updates

target_update_frequency = 1000

training_step = 0


# ===================================
# 9. LOSS FUNCTION
# ===================================

loss_function = nn.MSELoss()


# ===================================
# 10. CREATE ENVIRONMENT
# ===================================

env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)


# ===================================
# 11. CREATE CSV REPORT
# ===================================

report_file = open(
    "training_report_target_network.csv",
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
# 12. RUN EPISODES
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

    episode_epsilon = epsilon


    # ===================================
    # 13. RUN STEPS
    # ===================================

    for step_number in range(500):


        # -----------------------------
        # STATE -> TENSOR
        # -----------------------------

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )


        # -----------------------------
        # MAIN NETWORK Q VALUES
        # -----------------------------

        q_values = dqn(
            state_tensor
        )


        # =================================
        # 14. EPSILON-GREEDY ACTION
        # =================================

        if random.random() < epsilon:

            # EXPLORE
            action = env.action_space.sample()

        else:

            # EXPLOIT
            action = torch.argmax(
                q_values
            ).item()


        # =================================
        # 15. TAKE ACTION
        # =================================

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated


        # =================================
        # 16. UPDATE REPORT
        # =================================

        episode_reward += reward

        final_reward = reward


        current_x_distance = abs(
            next_state[0]
        )


        if current_x_distance < closest_x_to_center:

            closest_x_to_center = current_x_distance


        # =================================
        # 17. STORE EXPERIENCE
        # =================================

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
        # 18. TRAIN FROM REPLAY BUFFER
        # =================================

        if len(replay_buffer) >= batch_size:


            batch = random.sample(
                replay_buffer,
                batch_size
            )


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
            # 19. CURRENT PREDICTION
            #
            # MAIN DQN
            # =================================

            all_q_values = dqn(
                states_tensor
            )


            predicted_q_values = all_q_values.gather(

                1,

                actions_tensor.unsqueeze(1)

            ).squeeze(1)


            # =================================
            # 20. CALCULATE TD TARGET
            #
            # TARGET DQN
            # =================================

            with torch.no_grad():

                next_q_values = target_dqn(
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
            # 21. CALCULATE LOSS
            # =================================

            loss = loss_function(

                predicted_q_values,

                targets

            )


            episode_losses.append(
                loss.item()
            )


            # =================================
            # 22. UPDATE MAIN DQN
            # =================================

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            # =================================
            # 23. COUNT TRAINING UPDATES
            # =================================

            training_step += 1


            # =================================
            # 24. UPDATE TARGET NETWORK
            # =================================

            if training_step % target_update_frequency == 0:

                target_dqn.load_state_dict(
                    dqn.state_dict()
                )

                print(
                    "TARGET NETWORK UPDATED"
                )


        # =================================
        # 25. END EPISODE IF DONE
        # =================================

        if done:

            break


        # -----------------------------
        # STATE 2 BECOMES NEW STATE
        # -----------------------------

        state = next_state


    # ===================================
    # 26. EPISODE REPORT
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
    # 27. PRINT REPORT
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
    # 28. SAVE REPORT TO CSV
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
    # 29. DECAY EPSILON
    # ===================================

    epsilon = max(

        epsilon_min,

        epsilon * epsilon_decay

    )


# ===================================
# 30. CLOSE EVERYTHING
# ===================================

report_file.close()

env.close()