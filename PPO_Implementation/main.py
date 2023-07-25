import argparse
import os
import time
import random
import torch
import numpy as np
from distutils.util import strtobool

from training import training


def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='PPO')
    parser.add_argument('--gym_name', type=str, default='CartPole-v1')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--lr_decay', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,)
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument('--gae_lambda', type=float, default=0.95, help="Lambda for general advantage estimation")
    parser.add_argument('--clip_coeff', type=float, default=0.2, help="Coeff for PPO clipped policy objective")
    parser.add_argument('--value_loss_coeff', type=float, default=0.5, help="Coeff for value loss in PPO training")
    parser.add_argument('--entropy_loss_coeff', type=float, default=0.01, help="Coeff for entropy loss in PPO training")
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help="Global gradient clipping during training")
    
    parser.add_argument('--timesteps', type=int, default=25000)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=128, help = "num of steps to run in each env per policy rollout (store data before training)")
    parser.add_argument('--num_minibatches', type=int, default=4)
    parser.add_argument('--update_epochs', type=int, default=4)
    
    parser.add_argument('--capture_video', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,)

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


if __name__ == "__main__":
    args = setup_argparse()
    run_name = f"{args.gym_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    print(args)
    
    # M: Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    training(args, run_name)
    pass