import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from model import Value_Network, Policy_Network
from utils import *
from storage import ReplayMemory

class DDPGagent:
    def __init__(self, env, device, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=50000):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        
        self.device = device

        # Networks
        self.actor = Policy_Network(self.num_states, hidden_size, self.num_actions).to(device)
        self.actor_target = Policy_Network(self.num_states, hidden_size, self.num_actions).to(device)
        self.critic = Value_Network(self.num_states + self.num_actions, hidden_size, self.num_actions).to(device)
        self.critic_target = Value_Network(self.num_states + self.num_actions, hidden_size, self.num_actions).to(device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = ReplayMemory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    def get_action(self, state):
        # M: look if we rly need the Variable here!
        #state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        #with torch.no_grad():
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).float().unsqueeze(0)
        action = self.actor.forward(state)
        action = action.detach().cpu().numpy()[0,0]
        return action
    
    def learn(self, batch_size):
        # M: get data
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.stack(states).to(self.device).float()
        actions = torch.stack(actions).to(self.device).float()
        rewards = torch.stack(rewards).to(self.device).float()
        next_states = torch.stack(next_states).to(self.device).float()
    
        # Critic loss (Q Value net loss) According to paper: 
        # M: yi(Bellman iterative Qval from next state and current reward) - Qval (pred from nw with states and actions)
        # M: yi = r + gamma * (1-d) * Qprime (Qval on next state and next best actions)      
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss According to paper:
        # Policiy function is differentiable (since contiuous): We want to max the expected return of each state: Q(s, mü(s))
        # -> Q is calculated from the q value network and mü from the policy nw
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks with soft updates: polyak averaging
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
