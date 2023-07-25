import gym
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from agent import PPO_Agent


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger = lambda x: x%1000 == 0)
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def training(args, run_name):
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # M: Setup vector environment: Stack multiple environments into single environment -> Lets us explore in parallel
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_name, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    print("Observation Space: ", envs.single_observation_space.shape) # M: 4
    print("Action Space: ", envs.single_action_space.n) # M: 2
    
    save_dir = Path("checkpoints") / run_name
    save_dir.mkdir(parents=True)
    
    agent = PPO_Agent(envs, save_dir, device, args.lr)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    
    
    # M: storage for training
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device) # M: log probabs of actions in state
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device) # M: value of action in state
    
    global_step = 0
    start_time = time.time()
    
    # M: train in the environment
    state = envs.reset()
    state = torch.Tensor(state[0]).to(device)
    done = torch.zeros(args.num_envs).to(device)
    num_updates = args.timesteps // args.batch_size
    
    
    # M: training loop: each update is one iteration of the training loop: use policy to get data -> train policy and update agent
    for update in range(1, num_updates+1):
        
        if args.lr_decay:
            # M: linearly decrease lr from start to 0
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.lr
            optimizer.param_groups[0]["lr"] = lr_now
            
        # M: policy rollout: use policy to generate data in env and store them for training
        for step in range(0, args.num_steps):
            global_step += 1* args.num_envs
            
            obs[step] = state
            dones[step] = done

            with torch.no_grad():
                # M: generate action. Do not need gradients here, since we are optimizing later fromt he stored info
                action, logprob, entropy, value = agent.get_action_and_value(state)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            # M: use generated action to step in environment
            state, reward, done, trunc, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            state, done = torch.Tensor(state).to(device), torch.Tensor(done).to(device)
            
            # M: episode truncated or done
            if info:
                for item in info["final_info"]:
                    if item and "episode" in item.keys():
                        # ?Note: Vector environments automatically reset
                        er = item["episode"]["r"]
                        el = item["episode"]["l"]
                        print(f"global step = {global_step}, episodic return = {er}, episodic length = {el}")
                        writer.add_scalar("charts/episodic_return", er, global_step)
                        writer.add_scalar("charts/episodic_length", el, global_step)
                        break
    
        # M: calculate advantage with general advantage calculation according to ppo paper
        with torch.no_grad():
            next_value = agent.critic_value(state).reshape(1, -1)
            def calc_gae_advantages():
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                return advantages
            
            advantages = calc_gae_advantages()
            returns = advantages + values
            
        # M: flatten the batches for better batch processing
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # M: training
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            
            # M: loop through all batch data in minibatches
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                newaction, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds] # M: prob of taking action in new nw vs in old nw
                ratio = logratio.exp()
                
                # M: Advantage normalization
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # M: clipped policy objective: calculate clipped policy loss
                policy_loss_1 = -ratio * mb_advantages
                policy_loss_2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coeff, 1 + args.clip_coeff)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                
                #pg_loss1 = mb_advantages * ratio
                #pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - args.clip_coeff, 1 + args.clip_coeff)
                #policy_loss = torch.min(pg_loss1, pg_loss2).mean()
        
                # M: value loss clipping
                # M: normally: MSE of q value we predict vs q value that was recorded: ((newvalue - b_returns[mb_inds]) ** 2).mean()
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coeff,
                    args.clip_coeff,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                
                # M: entropy loss: to enable better exploration in actor network
                entropy_loss = entropy.mean()
                
                training_loss = policy_loss + args.value_loss_coeff * v_loss - args.entropy_loss_coeff * entropy_loss
                
                optimizer.zero_grad()
                training_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) #agent.parameters()
                optimizer.step()
                
        # M: end of training epochs, logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/training_loss", training_loss.item(), global_step)
        
        steps_per_second = int(global_step / (time.time() - start_time))
        print("SPS:", steps_per_second)
        writer.add_scalar("charts/SPS", steps_per_second, global_step)
        
    
    envs.close()
    writer.close()
    pass