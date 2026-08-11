import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset()

for step_number in range(500):

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    #print(f"Step:{step_number}, Action:{action}, Observation:{observation} , Reward:{reward}, Terminated:{terminated},Truncated:{truncated},Info: {info} ")

    print(f"Step:{step_number}, Action:{action}, Observation:{observation}, "
          f"Reward:{reward}, Terminated:{terminated}, Truncated:{truncated}, Info:{info}")

    if terminated or truncated:
        print("Episode finished.")
        break

env.close()