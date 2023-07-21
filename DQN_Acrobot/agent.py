import torch
from collections import namedtuple, deque
import random
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from model import DQN
import math


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class Acrobot_Agent():

    def __init__(self, state_dim, action_dim, save_dir, capacity, batch_size) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # M: First nw for est Q values 
        self.online_net = DQN(state_dim, action_dim)
        self.online_net = self.online_net.to(self.device)
        
        # M: Use second nw that is updated less frequenly: Used for evaluating the Q values of actions:
        # Reduce maximization bias of greedy action selection strategy from online net
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        #self.replay_buffer = deque([], maxlen=capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = batch_size
        
        self.save_every = 1e4
        self.sync_every = 1
        
        self.exploration_rate = 0.9
        self.exploration_rate_decay = 1000#0.99999975
        self.exploration_rate_min = 0.05
        self.curr_step = 0
        
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.SmoothL1Loss()

    
    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.
        """
        
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        # EXPLOIT
        else:
            with torch.no_grad():
                state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
                state = torch.tensor(state, device=self.device).unsqueeze(0)
                action_values = self.online_net(state)
                action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        #self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate *= math.exp(-1. * self.curr_step / self.exploration_rate_decay)
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
    
    def store_replay_buffer(self, state, action, reward, next_state, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()
        
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        self.replay_buffer.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))
    
    def sample_replay_buffer(self):
        batch = self.replay_buffer.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def sync_Q_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    def sync_Q_target_interpolated(self):
        TAU = 0.005
        
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.online_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)
    
    def calculate_online_Q_values(self, states, actions):
        # M: given the sampled states, what are the Q values that we are predicting
        Q_values_online = self.online_net(states)
        
        # M: we are only interested in the Q values of the action that was taken in the end for that sample
        Q_values_max_action = Q_values_online[np.arange(0, self.batch_size), actions]
        
        return Q_values_max_action
        
    @torch.no_grad()
    def calculate_target_Q_values(self, next_states, rewards, done):
        Q_values_next_state = self.online_net(next_states)
        best_actions = torch.argmax(Q_values_next_state, axis=1)
        
        Q_values_next_state_target = self.target_net(next_states)
        Q_values_next_state_max_action = Q_values_next_state_target[np.arange(0, self.batch_size), best_actions]
    
        return (rewards + (1 - done.float()) * self.gamma * Q_values_next_state_max_action).float()
        
    
    def learn(self):
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        if self.curr_step % self.save_every == 0:
            self.save_model()
            
        # M: update target net less frequently
        if self.curr_step % self.sync_every == 0:
            #self.sync_Q_target()
            self.sync_Q_target_interpolated()
            
        # M: Get samples for training and calculate the Q value of the policy net and the Expected Q value of the next state with target net
        # according to bellmann equation. Then use the difference to iteratively update the approximator of the Q function
        state, next_state, action, reward, done = self.sample_replay_buffer()
        
        q_values_online = self.calculate_online_Q_values(state, action)
        
        q_values_target = self.calculate_target_Q_values(next_state, reward, done)
        
        loss = self.loss_fn(q_values_online, q_values_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        pass
        
    
    def save_model(self):
        save_path = (
            self.save_dir / f"acrobot_dqn_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.online_net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Acrobot_DQN saved to {save_path} at step {self.curr_step}")