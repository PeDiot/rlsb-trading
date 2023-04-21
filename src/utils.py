import yaml
from yaml import Loader

import matplotlib.pyplot as plt 

from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Dict, Tuple, Optional
from src.environment import MyStocksEnv

def display_env(env: MyStocksEnv, fig_dims: Tuple=(10, 4)): 
    plt.figure(figsize=fig_dims, facecolor="w")
    plt.cla()
    env.render_all()
    plt.show()

def evaluate_episode(env: MyStocksEnv, model: Optional[BaseAlgorithm]=None) -> float: 

    episode_reward = 0    
    observation = env.reset()

    while True:
        if model is None: 
            action = env.action_space.sample()
        else:
            action, _ = model.predict(observation)

        observation, reward, done, info = env.step(action)
        episode_reward += reward

        if done:
            print("info:", info)
            break

    return episode_reward

def load_config(cfg_path: str) -> Dict:
    with open(cfg_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader)

    return cfg
