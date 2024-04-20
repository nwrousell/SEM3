import gymnasium as gym

env = gym.make("Amidar-v4", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    # print(env.action_space)
    
    action = env.action_space.sample()  # agent policy that uses the observation and info
    
    observation, reward, terminated, truncated, info = env.step(action)

    # print(observation.shape) # (210, 160, 3)

    if terminated or truncated:
        observation, info = env.reset()

env.close()



# TODO - setup audio with ALE
# TODO - set up replay buffer, n-step temporal difference learning

# TODO - BBF's resets / weight-interpolation
# TODO - weight decay 
# TODO - receding update horizon: “the use of an update horizon (n-step) that decreases exponentially from 10 to 3 over the first 10K gradient steps following each network reset”
# TODO - increasing discount factor