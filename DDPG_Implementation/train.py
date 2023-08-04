import sys
import torch
import gym
from gymnasium.wrappers import RescaleAction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPG_Agent import DDPGagent
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

env = NormalizedEnv(gym.make("Pendulum-v1"))
#env = RescaleAction(env, min_action=env.action_space.low , max_action=env.action_space.high)

agent = DDPGagent(env, device)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []

episodes= 100

for episode in range(episodes):
    state = env.reset()
    noise.reset()
    episode_reward = 0
    
    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step) # M: add noise for exploration vs exploitation
        new_state, reward, done, trunc, _ = env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
        
        if len(agent.memory) > batch_size:
            agent.learn(batch_size)        
        
        state = new_state
        episode_reward += reward

        if done or trunc:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

env.close()

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()