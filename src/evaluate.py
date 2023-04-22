import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Dict, Tuple, Optional, List
from src.environment import MyStocksEnv


def display_env(env: MyStocksEnv, fig_dims: Tuple=(8, 4)): 
    plt.figure(figsize=fig_dims, facecolor="w")
    plt.cla()
    env.render_all()
    plt.show()

def plot_results(rewards: List, profits: List, fig: Optional[plt.Figure]=None):

    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor="w") 

    n_actions = [t+1 for t in range(len(rewards))]
    cum_rewards = np.cumsum(rewards)

    sns.lineplot(x=n_actions, y=cum_rewards, ax=axes[0])
    sns.lineplot(x=n_actions, y=profits, ax=axes[1], color="orange")

    axes[0].set_title("Cumulative Rewards")
    axes[1].set_title("Cumulative Profits")    

def evaluate_episode(env: MyStocksEnv, model: Optional[BaseAlgorithm]=None, plot: bool=False) -> Dict: 

    rewards, profits, positions = [], [], []
    observation = env.reset()

    while True:
        if model is None: 
            action = env.action_space.sample()
        else:
            action, _ = model.predict(observation)

        observation, reward, done, info = env.step(action)
        
        positions.append(action)
        rewards.append(reward) 
        profits.append(info["total_profit"])

        if done:
            avg_position = np.mean(positions)
            info["avg_position"] = avg_position
            print("info:", info)
            break

    if plot:
        plot_results(rewards, profits)

    return info