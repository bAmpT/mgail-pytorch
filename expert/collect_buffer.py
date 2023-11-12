# Environment
import gym

import argparse
import numpy as np
import torch
import pickle
from collections import namedtuple
from tqdm import trange
from mgail.algo.sac import SAC

# Agent
from mgail.buffer import ER

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='./logs/*.pth')
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    args = p.parse_args()

    use_cuda = False
    use_random_actions = False
    show_visualization = False
    model_path = args.model

    env = gym.make(args.env_id)

    model = SAC(
        state_shape=(18,),
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if use_cuda else "cpu"),
        seed=12345,
    )
    model.actor.load_state_dict(torch.load(model_path))
    model.actor.eval()

    num_episodes = 1000
    buffer = ER(num_episodes * env.config["duration"], 3*6, 2, 1, 32)
        
    obs, done = env.reset(), False
    EnvTransition = namedtuple("EnvTransition", ["actions", "rewards", "next_states", "terminals"])
    transitions = []
    episode_steps = 0
    collected_episodes = 0
    while collected_episodes <= num_episodes:
        #action, _ = model.predict(obs, deterministic=True)
        if use_random_actions:
            action = env.action_space.sample()
        else:
            action = model.exploit(obs)
        obs, reward, done, info = env.step(action)
        if show_visualization:
            env.render()

        transitions.append(EnvTransition([action], [reward], [obs], [done]))
        episode_steps += 1
        if done:
            if use_random_actions or episode_steps <= 50:
                for t in transitions:
                    buffer.add(
                        actions=t.actions, 
                        rewards=t.rewards, 
                        next_states=t.next_states, # add latest observed state to buffer
                        terminals=t.terminals, 
                    )
                collected_episodes += 1
            episode_steps = 0
            transitions = []
            obs = env.reset()
    
    env.close()

    print("Saving buffer...", end="")
    with open(f'er_{num_episodes}.bin', 'wb') as f:
        pickle.dump(buffer, f)
    print(" [DONE].")