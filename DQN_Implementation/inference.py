import torch
import gymnasium as gym
from model import DQN
from itertools import count


#CHECKPOINT_PATH = "DQN_Implementation\checkpoints/2023-07-21T15-37-16/acrobot_dqn_10.chkpt"
CHECKPOINT_PATH = "DQN_Implementation/checkpoints/2023-07-23T16-26-34/acrobot_dqn_3.chkpt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def policy_take_step(model, state):
    with torch.no_grad():
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=device).unsqueeze(0)
        action_values = model(state)
        action_idx = torch.argmax(action_values, axis=1).item()
        
        return action_idx


def ddqn_inference():
    
    env = gym.make("CartPole-v1", render_mode = "human") # Acrobot-v1, CartPole-v1
    obs_state, info = env.reset()
    env.render()
    
    state_dim = len(obs_state)
    action_dim = env.action_space.n
    
    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH)['model'])
    
    state = env.reset()
    for t in count():
        # Run agent on the state
        action = policy_take_step(model, state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Move to the next state
        state = next_state

        finished = done or trunc
        if finished:
            print(f"Used {t} steps.")
            break


if __name__ == "__main__":
    ddqn_inference()
    print ("Finished")