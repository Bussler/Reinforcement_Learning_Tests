import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import numpy as np



class PPO_Agent(nn.Module):
    
    def __init__(self, envs, save_dir, device, lr) -> None:
        super(PPO_Agent, self).__init__()
        
        self.save_dir = save_dir
        self.device = device
        
        # M: Gives the actor feedback on how good the current state is: Value function for state
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        ).to(self.device)
        
        # M: defines the policy: 
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)
    
    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    
    def critic_value(self, x):
        return self.critic(x)
    
    # M: bundle actor and critic inference
    def get_action_and_value(self, x, action=None):
        actor_values = self.actor(x)
        actor_probabs = Categorical(logits=actor_values)  # M: softmax to get action probab dist
        if action is None:
            action = actor_probabs.sample()
        value = self.critic_value(x)
        return action, actor_probabs.log_prob(action), actor_probabs.entropy(), value
    
    
    def learn():
        pass
    
    
    def save_model(self):
        save_path = (
            self.save_dir / f"ppo_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(critic=self.critic.state_dict(), actor=self.actor.state_dict()),
            save_path,
        )
        print(f"Acrobot_DQN saved to {save_path} at step {self.curr_step}")