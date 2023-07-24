import torch
import torch.nn as nn
import torch.nn.functional as F

# M: network is trying to predict the expected return Q(s,a) of taking each action given the current input
class DQN(nn.Module):

    # M: policy net: in state: from 4 observations, out 2 (return of q value from action 1 or 2)
    def __init__(self, in_observations, out_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(in_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, out_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor of Q values for each action of each observation.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        res = self.layer3(x)
        return res