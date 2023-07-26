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
    
    def set_optim_lr(self, lr):
        self.optimizer.param_groups[0]["lr"] = lr
    
    # M: get state value function from criti nw
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
    
    
    # M: calculate advantage with general advantage calculation according to ppo paper
    def calc_gae_advantages(self, num_update_steps, gamma, gae_lambda, done, next_value, storage):
        with torch.no_grad():
            advantages = torch.zeros_like(storage.rewards).to(self.device)
            lastgaelam = 0
                
            for t in reversed(range(num_update_steps)):
                if t == num_update_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - storage.dones[t + 1]
                    nextvalues = storage.values[t + 1]
                delta = storage.rewards[t] + gamma * nextvalues * nextnonterminal - storage.values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            return advantages
    
    
    def learn(self, args, storage):            
        # M: flatten the batches for better batch processing
        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = storage.flatten_batches()
        
        # M: training
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            
            # M: loop through all batch data in minibatches
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                newaction, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds] # M: prob of taking action in new nw vs in old nw
                ratio = logratio.exp()
                
                # M: Advantage normalization
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # M: clipped policy objective: calculate clipped policy loss
                policy_loss_1 = -ratio * mb_advantages
                policy_loss_2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coeff, 1 + args.clip_coeff)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
        
                # M: value loss clipping
                # M: normally: MSE of q value we predict vs q value that was recorded: ((newvalue - b_returns[mb_inds]) ** 2).mean()
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2 #b_returns, b_values
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coeff,
                    args.clip_coeff,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2 #b_returns, b_values
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                
                # M: entropy loss: to enable better exploration in actor network
                entropy_loss = entropy.mean()
                
                training_loss = policy_loss + args.value_loss_coeff * v_loss - args.entropy_loss_coeff * entropy_loss
                
                self.optimizer.zero_grad()
                training_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                self.optimizer.step()
                
        return self.optimizer.param_groups[0]["lr"], v_loss, policy_loss, entropy_loss, training_loss
    
    
    def save_model(self, curr_step):
        save_path = (
            self.save_dir / f"ppo_{int(curr_step)}.chkpt"
        )
        torch.save(
            dict(critic=self.critic.state_dict(), actor=self.actor.state_dict()),
            save_path,
        )
        print(f"Acrobot_DQN saved to {save_path} at step {curr_step}")