import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import csv
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import pygame


# ============================================================
# 1. DQN NEURAL NETWORK
# ============================================================

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


# ============================================================
# 2. MAIN DQN + TARGET DQN
# ============================================================

dqn = DQN()
target_dqn = DQN()

target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()


# ============================================================
# 3. OPTIMIZER + LOSS
# ============================================================

optimizer = optim.Adam(
    dqn.parameters(),
    lr=0.001
)

loss_function = nn.MSELoss()


# ============================================================
# 4. REPLAY BUFFER
# ============================================================

replay_buffer = deque(maxlen=10000)

batch_size = 32


# ============================================================
# 5. DQN SETTINGS
# ============================================================

gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

number_of_episodes = 500

training_step = 0

target_update_frequency = 1000


# ============================================================
# 6. SOFT-LANDING SETTINGS
# ============================================================

# Start caring more about vertical speed
# when the lander is close to the ground.

soft_landing_height = 0.30

# Falling faster than this near the ground
# gets an extra penalty.

safe_vertical_speed = -0.20

# Strength of the extra penalty.

soft_landing_penalty_strength = 20.0


# ============================================================
# 7. CREATE ENVIRONMENT
# ============================================================

env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)


# ============================================================
# NEW: FONT FOR EPISODE NUMBER
# ============================================================

pygame.font.init()

episode_font = pygame.font.SysFont(
    "Arial",
    24,
    bold=True
)


# ============================================================
# 8. TRAINING REPORT
# ============================================================

training_file = open(
    "training_report_soft_landing.csv",
    "w",
    newline=""
)

training_writer = csv.writer(training_file)

training_writer.writerow([
    "Episode",
    "Epsilon",
    "Gym Reward",
    "Training Reward",
    "Steps",
    "Final X",
    "Final Y",
    "Final Vertical Velocity",
    "Closest X To Center",
    "Left Leg",
    "Right Leg",
    "Final Reward",
    "Average Loss",
    "Soft Landing Penalty"
])


# ============================================================
# 9. TRAINING
# ============================================================

training_rewards = []
training_losses = []


for episode in range(number_of_episodes):

    state, info = env.reset()

    gym_total_reward = 0
    training_total_reward = 0

    episode_losses = []

    total_soft_penalty = 0

    closest_x = abs(state[0])

    final_reward = 0

    step_number = 0


    # --------------------------------------------------------
    # RUN ONE EPISODE
    # --------------------------------------------------------

    for step_number in range(500):


        # ====================================================
        # NEW: DISPLAY EPISODE NUMBER
        # ====================================================

        if env.unwrapped.screen is not None:

            episode_text = episode_font.render(
                f"Training Episode: {episode + 1} / {number_of_episodes}",
                True,
                (255, 255, 255)
            )

            env.unwrapped.screen.blit(
                episode_text,
                (20, 20)
            )

            pygame.display.update()


        # ----------------------------------------------------
        # STATE → TENSOR
        # ----------------------------------------------------

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)


        # ----------------------------------------------------
        # EPSILON-GREEDY ACTION
        # ----------------------------------------------------

        if random.random() < epsilon:

            action = env.action_space.sample()

        else:

            with torch.no_grad():

                q_values = dqn(state_tensor)

                action = torch.argmax(
                    q_values
                ).item()


        # ----------------------------------------------------
        # ENVIRONMENT STEP
        # ----------------------------------------------------

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        final_reward = reward


        # ====================================================
        # SOFT-LANDING PENALTY
        # ====================================================

        training_reward = reward

        height = next_state[1]

        vertical_velocity = next_state[3]

        extra_soft_penalty = 0


        # Lander is close to ground
        # AND falling too quickly

        if (
            height < soft_landing_height
            and
            vertical_velocity < safe_vertical_speed
        ):

            excess_speed = (
                abs(vertical_velocity)
                -
                abs(safe_vertical_speed)
            )

            extra_soft_penalty = (
                soft_landing_penalty_strength
                *
                excess_speed
            )

            training_reward -= extra_soft_penalty


        total_soft_penalty += extra_soft_penalty


        # ====================================================
        # STORE EXPERIENCE
        # ====================================================

        replay_buffer.append(
            (
                state,
                action,
                training_reward,
                next_state,
                done
            )
        )


        # Original Gym reward
        gym_total_reward += reward

        # Reward the DQN actually learns from
        training_total_reward += training_reward


        closest_x = min(
            closest_x,
            abs(next_state[0])
        )


        # ====================================================
        # 10. TRAIN DQN
        # ====================================================

        if len(replay_buffer) >= batch_size:

            batch = random.sample(
                replay_buffer,
                batch_size
            )


            states, actions, rewards, next_states, dones = zip(*batch)


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


            # ------------------------------------------------
            # MAIN DQN PREDICTION
            # ------------------------------------------------

            all_q_values = dqn(
                states_tensor
            )


            predicted_q_values = all_q_values.gather(
                1,
                actions_tensor.unsqueeze(1)
            ).squeeze(1)


            # ------------------------------------------------
            # TARGET DQN
            # ------------------------------------------------

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
                    *
                    best_next_q_values
                    *
                    (1 - dones_tensor)
                )


            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            loss = loss_function(
                predicted_q_values,
                targets
            )


            # ------------------------------------------------
            # UPDATE MAIN DQN
            # ------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            episode_losses.append(
                loss.item()
            )


            # ------------------------------------------------
            # TARGET NETWORK UPDATE
            # ------------------------------------------------

            training_step += 1


            if training_step % target_update_frequency == 0:

                target_dqn.load_state_dict(
                    dqn.state_dict()
                )


        # ----------------------------------------------------
        # MOVE TO NEXT STATE
        # ----------------------------------------------------

        state = next_state


        if done:
            break


    # ========================================================
    # 11. END OF EPISODE
    # ========================================================

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


    if len(episode_losses) > 0:

        average_loss = np.mean(
            episode_losses
        )

    else:

        average_loss = 0


    training_rewards.append(
        training_total_reward
    )

    training_losses.append(
        average_loss
    )


    training_writer.writerow([
        episode + 1,
        epsilon,
        gym_total_reward,
        training_total_reward,
        step_number + 1,
        state[0],
        state[1],
        state[3],
        closest_x,
        state[6],
        state[7],
        final_reward,
        average_loss,
        total_soft_penalty
    ])


    print(
        f"Episode {episode + 1} | "
        f"Gym Reward: {gym_total_reward:.2f} | "
        f"Training Reward: {training_total_reward:.2f} | "
        f"Epsilon: {epsilon:.3f} | "
        f"Final VY: {state[3]:.3f} | "
        f"Soft Penalty: {total_soft_penalty:.2f}"
    )


training_file.close()


# ============================================================
# 12. SAVE TRAINED MODEL
# ============================================================

torch.save(
    dqn.state_dict(),
    "lunar_lander_soft_landing_dqn.pth"
)


# ============================================================
# 13. PURE EVALUATION
# ============================================================

dqn.eval()

number_of_test_episodes = 100

evaluation_rewards = []

successful_landings = 0


evaluation_file = open(
    "evaluation_report_soft_landing.csv",
    "w",
    newline=""
)

evaluation_writer = csv.writer(
    evaluation_file
)

evaluation_writer.writerow([
    "Test Episode",
    "Total Reward",
    "Steps",
    "Final X",
    "Final Y",
    "Final Vertical Velocity",
    "Left Leg",
    "Right Leg",
    "Final Reward"
])


for test_episode in range(number_of_test_episodes):

    state, info = env.reset()

    total_reward = 0

    final_reward = 0


    for step_number in range(500):


        # ====================================================
        # NEW: DISPLAY TEST EPISODE
        # ====================================================

        if env.unwrapped.screen is not None:

            test_text = episode_font.render(
                f"Evaluation: {test_episode + 1} / {number_of_test_episodes}",
                True,
                (255, 255, 255)
            )

            env.unwrapped.screen.blit(
                test_text,
                (20, 20)
            )

            pygame.display.update()


        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)


        # ----------------------------------------------------
        # PURE GREEDY ACTION
        # ----------------------------------------------------

        with torch.no_grad():

            q_values = dqn(
                state_tensor
            )

            action = torch.argmax(
                q_values
            ).item()


        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated


        # IMPORTANT:
        # Evaluation uses ORIGINAL Gym reward.
        # No custom penalty here.

        total_reward += reward

        final_reward = reward

        state = next_state


        if done:
            break


    if final_reward == 100:

        successful_landings += 1


    evaluation_rewards.append(
        total_reward
    )


    evaluation_writer.writerow([
        test_episode + 1,
        total_reward,
        step_number + 1,
        state[0],
        state[1],
        state[3],
        state[6],
        state[7],
        final_reward
    ])


    print(
        f"TEST {test_episode + 1} | "
        f"Reward: {total_reward:.2f} | "
        f"Final VY: {state[3]:.3f}"
    )


evaluation_file.close()


# ============================================================
# 14. EVALUATION SUMMARY
# ============================================================

average_test_reward = np.mean(
    evaluation_rewards
)

success_rate = (
    successful_landings
    /
    number_of_test_episodes
) * 100


print("\n==============================")
print("EVALUATION RESULTS")
print("==============================")


print(
    f"Average Test Reward: "
    f"{average_test_reward:.2f}"
)


print(
    f"Successful Landings: "
    f"{successful_landings}/"
    f"{number_of_test_episodes}"
)


print(
    f"Success Rate: "
    f"{success_rate:.1f}%"
)


# ============================================================
# 15. CREATE GRAPH
# ============================================================

training_series = pd.Series(
    training_rewards
)

training_average = training_series.rolling(
    window=25
).mean()


evaluation_series = pd.Series(
    evaluation_rewards
)

evaluation_average = evaluation_series.rolling(
    window=10
).mean()


plt.figure(figsize=(11, 6))


plt.plot(
    range(1, number_of_episodes + 1),
    training_average,
    label="Training - 25 Episode Average"
)


plt.plot(
    range(
        number_of_episodes + 1,
        number_of_episodes + number_of_test_episodes + 1
    ),
    evaluation_average,
    label="Evaluation - 10 Episode Average"
)


plt.axvline(
    x=number_of_episodes,
    linestyle="--"
)


plt.xlabel("Episode")

plt.ylabel("Reward")


plt.title(
    "Lunar Lander DQN - Soft Landing Experiment"
)


plt.legend()

plt.tight_layout()


plt.savefig(
    "soft_landing_reward_graph.png",
    dpi=300
)


plt.show()


# ============================================================
# 16. CREATE EXCEL FILE
# ============================================================

training_dataframe = pd.read_csv(
    "training_report_soft_landing.csv"
)

evaluation_dataframe = pd.read_csv(
    "evaluation_report_soft_landing.csv"
)


training_dataframe[
    "25 Episode Average Reward"
] = training_dataframe[
    "Training Reward"
].rolling(
    25
).mean()


training_dataframe[
    "25 Episode Average Loss"
] = training_dataframe[
    "Average Loss"
].rolling(
    25
).mean()


evaluation_dataframe[
    "10 Episode Average Reward"
] = evaluation_dataframe[
    "Total Reward"
].rolling(
    10
).mean()


with pd.ExcelWriter(
    "lunar_lander_soft_landing_results.xlsx"
) as writer:

    training_dataframe.to_excel(
        writer,
        sheet_name="Training",
        index=False
    )

    evaluation_dataframe.to_excel(
        writer,
        sheet_name="Evaluation",
        index=False
    )


# ============================================================
# 17. CLOSE ENVIRONMENT
# ============================================================

env.close()