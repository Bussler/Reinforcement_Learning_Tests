from collections import namedtuple, deque
import random
import torch
import math
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    

# M: Single transition in our environment. It essentially maps (state, action) pairs to their (next_state, reward) result
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# M: A cyclic buffer of bounded size that holds the transitions observed recently. 
# -> Reuse this data later to sample from randomly; thereby decorrelating the transitions that build up a batch
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
    
    
# will select an action accordingly to an epsilon greedy policy.
# Simply put, we’ll sometimes use our model for choosing the action, and sometimes we’ll just sample one uniformly.
# The probability of choosing a random action will start at EPS_START and will decay exponentially towards EPS_END.
# EPS_DECAY controls the rate of the decay.
def select_action(state, device, policy_net, env, steps_done):
    #global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done[0] / EPS_DECAY)
    steps_done[0] += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


# a helper for plotting the duration of episodes, along with an average over the last 100 episodes
# (the measure used in the official evaluations). The plot will be underneath the cell containing
# the main training loop, and will update after every episode.
def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())