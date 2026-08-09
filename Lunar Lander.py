import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset()

for step_number in range(500):

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print("Step:", step_number)
    print("Action:", action)
    print("Observation:", observation)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("----------------------")

    if terminated or truncated:
        print("Episode finished.")
        break

env.close()