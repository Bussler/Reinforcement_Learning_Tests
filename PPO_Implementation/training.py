import gym
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from agent import PPO_Agent
from storage import ReplayMemory


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
    #optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    
    
    # M: storage for training    
    storage = ReplayMemory(args, envs, device)
    
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
            #optimizer.param_groups[0]["lr"] = lr_now
            agent.set_optim_lr(lr_now)
            
        # M: policy rollout: use policy to generate data in env and store them for training
        for step in range(0, args.num_steps):
            global_step += 1* args.num_envs
            
            storage.obs[step] = state
            storage.dones[step] = done

            with torch.no_grad():
                # M: generate action. Do not need gradients here, since we are optimizing later fromt he stored info
                action, logprob, entropy, value = agent.get_action_and_value(state)
            storage.values[step] = value.flatten()
            storage.actions[step] = action
            storage.logprobs[step] = logprob
            
            # M: use generated action to step in environment
            state, reward, done, trunc, info = envs.step(action.cpu().numpy())
            storage.rewards[step] = torch.tensor(reward).to(device)
            storage.rewards[step] = torch.tensor(reward).to(device).view(-1)
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
            storage.advantages = agent.calc_gae_advantages(args.num_steps, args.gamma, args.gae_lambda, 
                                                           done, next_value, storage)
            storage.returns = storage.advantages + storage.values
            
        # M: train and update agent with stored info and calculated advantag from policy rollout
        lr, v_loss, policy_loss, entropy_loss, training_loss = agent.learn(args, storage)
                
        # M: end of training epochs, logging
        writer.add_scalar("charts/learning_rate", lr, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/training_loss", training_loss.item(), global_step)
        
        steps_per_second = int(global_step / (time.time() - start_time))
        print("SPS:", steps_per_second)
        writer.add_scalar("charts/SPS", steps_per_second, global_step)
        
    agent.save_model(global_step)
    envs.close()
    writer.close()
    pass