import torch
import gymnasium as gym
from agent import Acrobot_Agent
import datetime
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from utils import plot_durations
from itertools import count


plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# M: Action Space: (3,)
# M: Observation Space: (6,)
# M: Reward: -1 for each step, when reaching goal target height (-cos(theta1) - cos(theta2 + theta1) > 1.0): 0, Threshold -100
env = gym.make("Acrobot-v1") # Acrobot-v1, CartPole-v1
env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"Observation Space: {next_state.shape}, \n Action Space: {env.action_space.n}, \n Reward: {reward},\n Truncated: {done},\n Info: {info}")

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

state_dim = len(next_state)
action_dim = env.action_space.n

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

BATCH_SIZE = 128
CAPACITY = 10000
episode_durations = []

acrobot_agent = Acrobot_Agent(state_dim, action_dim, save_dir, CAPACITY, BATCH_SIZE)


episodes = 600
for e in range(episodes):
    state = env.reset()
    
    for t in count():
        # Run agent on the state
        action = acrobot_agent.act(state)
        #action = acrobot_agent.select_action(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)
        
        # Remember
        acrobot_agent.store_replay_buffer(state, action, reward, next_state, done)
        
        # Learn
        acrobot_agent.learn()

        # Move to the next state
        state = next_state

        finished = done or trunc
        if finished:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break


print('Training Complete!')
acrobot_agent.save_model()
plot_durations(episode_durations, show_result=True)
plt.ioff()
plt.show()