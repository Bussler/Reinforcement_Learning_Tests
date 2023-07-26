import torch


# M: holds experiences during training
class ReplayMemory(object):
    
    def __init__(self, args, envs, device) -> None:
        self.device = device
        self.envs = envs
        
        self._obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        self._actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        self._logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device) # M: log probabs of actions in state
        self._rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self._advantages = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self._returns = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self._dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self._values = torch.zeros((args.num_steps, args.num_envs)).to(device) # M: value of action in state
        
    
    def flatten_batches(self):
        return self.obs.reshape((-1,) + self.envs.single_observation_space.shape), \
        self.logprobs.reshape(-1), \
        self.actions.reshape((-1,) + self.envs.single_action_space.shape), \
        self.advantages.reshape(-1), \
        self.returns.reshape(-1), \
        self.values.reshape(-1), 
    
       
    @property
    def obs(self):
        return self._obs
    
    @property
    def actions(self):
        return self._actions
    
    @property
    def logprobs(self):
        return self._logprobs
    
    @property
    def rewards(self):
        return self._rewards
    
    @property
    def advantages(self):
        return self._advantages
    
    @advantages.setter
    def advantages(self, new_advantages):
        self._advantages = new_advantages
    
    @property
    def returns(self):
        return self._returns
    
    @returns.setter
    def returns(self, new_returns):
        self._returns = new_returns
    
    @property
    def dones(self):
        return self._dones
    
    @property
    def values(self):
        return self._values

