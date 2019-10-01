import gym
import time
import numpy as np

env = gym.make('MsPacman-v0')
print ("*" * 50)
#(0 = center, 1 = up, 2 = right, 3 = left, 4 = down, 5 = upper-right, 6 = upper-left, 7 = lower-right, 8 = lower-left)
print ("Observation space: ", env.observation_space)
print ("Action space: ", env.action_space)
print ("env.spec.timestep_limit: ", env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
print ("*" * 50)
time.sleep(5)

for i_episode in range(20):
    observation = env.reset()

    # #skip the start
    # [env.step(0) for _ in range(90)]

    for t in range(1000000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        print ("Reward: ", reward)
        time.sleep(0.05)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()